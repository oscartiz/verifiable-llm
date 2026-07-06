//! Quantized llama, vendored from candle-transformers 0.11
//! `models/quantized_llama.rs` (Apache-2.0 OR MIT) and reduced to the llama
//! GGUF path (no MoE, no ggml v1/v2 files, no tracing spans). Vendored so we
//! can add what the verification layers need without forking behavior:
//!
//! - [`ModelWeights::forward_traced`]: forward pass that hands back the
//!   hidden states entering every block (and exiting the last) — trace
//!   capture for v0.2 commitments.
//! - [`ModelWeights::forward_block`]: re-execute one block over a revealed
//!   prefix of hidden states — the v0.2 verifier's single-layer check.
//! - [`ModelWeights::lm_head`]: final norm + output projection, so a
//!   verifier can tie last-block activations to committed logits.
//!
//! All tensor operations are kept identical to upstream so that generation
//! through this module reproduces upstream logits bit-for-bit (validated by
//! the parity of transcript hash chains).

use std::collections::HashMap;

use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::{build_causal_mask, repeat_kv};

pub const MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// RoPE convention: true = NEOX (pairs i with i+d/2), false = NORM
    /// (interleaved pairs 2i, 2i+1). Wrong convention corrupts attention.
    rope_is_neox: bool,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let x = x.contiguous()?;
        if self.rope_is_neox {
            candle_nn::rotary_emb::rope(&x, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope_i(&x, &cos, &sin)
        }
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            // No-op except on the initial prompt; enables the fast kernel.
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let y = if q.device().is_metal() && seq_len == 1 {
            // SDPA handles MQA for us.
            candle_nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                false,
                1. / (self.head_dim as f32).sqrt(),
                1.,
            )?
        } else {
            let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.attention_wo.forward(&y)
    }

    /// One full transformer block: x + attn(norm(x)), then h + mlp(norm(h)).
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.attention_norm.forward(x)?;
        let attn = self.forward_attn(&x, mask, index_pos)?;
        let x = (attn + residual)?;
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<(usize, usize), Tensor>,
    hidden_dim: usize,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        if n_expert > 1 {
            candle_core::bail!("MoE llama models are not supported by the vendored model");
        }
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        // RoPE convention by architecture, matching llama.cpp (see upstream).
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let rope_is_neox = matches!(
            arch.as_str(),
            "qwen"
                | "qwen2"
                | "qwen2moe"
                | "qwen3"
                | "qwen3moe"
                | "falcon"
                | "grok"
                | "dbrx"
                | "phi2"
                | "phi3"
                | "phimoe"
                | "stablelm"
                | "starcoder2"
                | "bert"
                | "nomic-bert"
                | "jina-bert-v2"
                | "olmo2"
                | "olmoe"
                | "codeshell"
                | "plamo"
        );

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp = Mlp {
                feed_forward_w1: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_gate.weight"),
                    device,
                )?)?,
                feed_forward_w2: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_down.weight"),
                    device,
                )?)?,
                feed_forward_w3: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_up.weight"),
                    device,
                )?)?,
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                rope_is_neox,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            hidden_dim: embedding_length,
        })
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }

    /// Standard forward: logits for the last position, `[b, vocab]`.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward_traced(x, index_pos, &mut |_, _| Ok(()))
    }

    /// Forward pass that reports intermediate hidden states: `on_hidden(j, h)`
    /// is called with the `[b, seq, dim]` tensor *entering* block `j` for
    /// each `j in 0..n_layers`, and finally with `(n_layers, h)` for the
    /// tensor exiting the last block (entering the final norm). The hook only
    /// observes — the computation is identical to [`Self::forward`].
    pub fn forward_traced(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        on_hidden: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for (j, layer) in self.layers.iter_mut().enumerate() {
            on_hidden(j, &layer_in)?;
            layer_in = layer.forward(&layer_in, mask.as_ref(), index_pos)?;
        }
        on_hidden(self.layers.len(), &layer_in)?;
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }

    /// Re-execute a single block over a prefix of hidden states, from
    /// position 0, with a fresh KV cache and a causal mask — the verifier's
    /// single-layer check. `h` is `[1, seq, dim]` (hidden states entering
    /// `block_idx` at positions `0..seq`); returns the block's outputs at all
    /// positions, `[1, seq, dim]`.
    pub fn forward_block(
        &mut self,
        block_idx: usize,
        h: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        let layer = self
            .layers
            .get_mut(block_idx)
            .ok_or_else(|| candle_core::Error::Msg(format!("no block {block_idx}")))?;
        layer.kv_cache = None;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(build_causal_mask(seq_len, 0, h.device())?)
        };
        let out = layer.forward(h, mask.as_ref(), 0)?;
        layer.kv_cache = None;
        Ok(out)
    }

    /// Final RMS norm + output projection over `[b, seq, dim]` hidden states;
    /// returns `[b, seq, vocab]` logits. Lets a verifier tie last-block
    /// activations to committed logits.
    pub fn lm_head(&self, h: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(h)?;
        self.output.forward(&x)
    }

    /// Token embeddings `[1, n, dim]` — the exact inputs of block 0, so a
    /// verifier can check revealed layer-0 cells against committed tokens.
    pub fn embed(&self, tokens: &[u32], device: &Device) -> Result<Tensor> {
        let x = Tensor::new(tokens, device)?.unsqueeze(0)?;
        self.tok_embeddings.forward(&x)
    }
}
