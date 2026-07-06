//! Deterministic CPU llama forward: bit-identical results on every run and
//! every platform, so v0.2 challenges can be verified with EXACT equality
//! (tolerance zero) and the bounded-drift attack of REPORT.md is closed.
//!
//! Determinism comes from three commitments, spelled out in DECISIONS.md #16:
//!
//! 1. **Fixed evaluation order.** All reductions (matvec rows, softmax sums,
//!    norms, attention) are sequential loops in this file; rayon parallelism
//!    is only ever over *independent* outputs, which no scheduler can
//!    reassociate. IEEE-754 f32/f64 +,-,*,/ and sqrt are exactly specified,
//!    so fixed order ⇒ fixed bits. (Rust never contracts a*b+c into FMA.)
//! 2. **No libm.** exp/sin/cos come from `det_math` (fixed polynomials).
//! 3. **Hook requantization.** The hidden state is snapped to the trace
//!    grid (q = round(x·2^frac_bits), then back) at every commitment hook,
//!    so a committed cell IS the computation state: the verifier recomputes
//!    a block from dequantized committed inputs and gets bit-identical
//!    outputs. Activation cells use 2^-8 (the f32 grid is coarser than
//!    2^-16 beyond |x| = 128, which llama outlier channels exceed; at 2^-8
//!    the quantize∘dequantize roundtrip is exact up to |x| = 32768).
//!    Logits are requantized at 2^-16 before they leave the model.
//!
//! Weights are dequantized once via candle's scalar CPU kernels
//! (per-element, no reductions ⇒ order-independent ⇒ deterministic; pinned
//! to candle 0.11). ~5 GB resident for the 1B model. Everything downstream
//! of dequantization is this file.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::Device;
use candle_core::quantized::gguf_file;
use rayon::prelude::*;

use crate::det_math;

/// Backend tag recorded in transcripts; bump on ANY change to the
/// computation defined in this file or `det_math`.
pub const DET_BACKEND: &str = "det-cpu-v1";
/// Trace-grid precision for activation cells in deterministic mode.
pub const DET_ACT_FRAC_BITS: u8 = 8;

/// Snap to the fixed-point grid: dequantize(quantize(x)). Identical to the
/// vllm-core quantize/dequantize pair, kept here as a hot-path scalar op.
#[inline]
pub fn requantize(x: f32, frac_bits: u8) -> f32 {
    let scale = (1u64 << frac_bits) as f64;
    let q = (x as f64 * scale)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64);
    (q / scale) as f32
}

fn requantize_all(xs: &mut [f32], frac_bits: u8) {
    for x in xs.iter_mut() {
        *x = requantize(*x, frac_bits);
    }
}

/// A row-major [rows, cols] f32 matrix.
struct Mat {
    w: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Mat {
    /// out[i] = Σ_j w[i,j]·x[j], sequential over j (fixed order), parallel
    /// over rows (independent outputs).
    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &self.w[i * self.cols..(i + 1) * self.cols];
            let mut acc = 0f32;
            for (w, xv) in row.iter().zip(x) {
                acc += w * xv;
            }
            *o = acc;
        });
    }
}

struct DetLayer {
    attn_norm: Vec<f32>,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    ffn_norm: Vec<f32>,
    w_gate: Mat,
    w_up: Mat,
    w_down: Mat,
}

/// Per-layer KV cache: k/v rows per position, [n_kv_heads * head_dim] each.
#[derive(Default, Clone)]
struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

pub struct DetModel {
    embed: Mat, // [vocab, dim]; also the LM head when weights are tied
    output: Option<Mat>,
    layers: Vec<DetLayer>,
    final_norm: Vec<f32>,
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rms_eps: f64,
    /// cos/sin[pos * head_dim/2 + i], from det_math.
    cos: Vec<f32>,
    sin: Vec<f32>,
    max_seq: usize,
    kv: Vec<KvCache>,
    act_frac_bits: u8,
    logit_frac_bits: u8,
}

impl DetModel {
    pub fn load(path: &Path, logit_frac_bits: u8) -> Result<Self> {
        let device = Device::Cpu;
        let mut file = File::open(path).with_context(|| format!("opening {path:?}"))?;
        let ct = gguf_file::Content::read(&mut file)?;
        let md = |name: &str| -> Result<u32> {
            ct.metadata
                .get(name)
                .and_then(|v| v.to_u32().ok())
                .with_context(|| format!("metadata {name}"))
        };
        let n_heads = md("llama.attention.head_count")? as usize;
        let n_kv_heads = md("llama.attention.head_count_kv")? as usize;
        let n_layers = md("llama.block_count")? as usize;
        let dim = md("llama.embedding_length")? as usize;
        let rms_eps = ct
            .metadata
            .get("llama.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.to_f32().ok())
            .context("rms epsilon")? as f64;
        let rope_base = ct
            .metadata
            .get("llama.rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000.0) as f64;
        let head_dim = dim / n_heads;

        let mut mat = |name: &str| -> Result<Mat> {
            let t = ct.tensor(&mut file, name, &device)?.dequantize(&device)?;
            let dims = t.dims().to_vec();
            let (rows, cols) = match dims.len() {
                2 => (dims[0], dims[1]),
                1 => (1, dims[0]),
                _ => bail!("{name}: unexpected rank {dims:?}"),
            };
            Ok(Mat {
                w: t.flatten_all()?.to_vec1()?,
                rows,
                cols,
            })
        };

        let embed = mat("token_embd.weight")?;
        // Absent output.weight means the LM head is tied with the embedding.
        let output = mat("output.weight").ok();
        let final_norm = mat("output_norm.weight")?.w;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            layers.push(DetLayer {
                attn_norm: mat(&format!("{p}.attn_norm.weight"))?.w,
                wq: mat(&format!("{p}.attn_q.weight"))?,
                wk: mat(&format!("{p}.attn_k.weight"))?,
                wv: mat(&format!("{p}.attn_v.weight"))?,
                wo: mat(&format!("{p}.attn_output.weight"))?,
                ffn_norm: mat(&format!("{p}.ffn_norm.weight"))?.w,
                w_gate: mat(&format!("{p}.ffn_gate.weight"))?,
                w_up: mat(&format!("{p}.ffn_up.weight"))?,
                w_down: mat(&format!("{p}.ffn_down.weight"))?,
            });
        }

        // RoPE tables from the deterministic sin/cos, matching the
        // interleaved (NORM) convention of the float path.
        let max_seq = crate::model::MAX_SEQ_LEN;
        let half = head_dim / 2;
        let ln_base = det_math_ln(rope_base);
        let mut cos = vec![0f32; max_seq * half];
        let mut sin = vec![0f32; max_seq * half];
        for p in 0..max_seq {
            for i in 0..half {
                let theta = 1.0 / det_math::exp((2.0 * i as f64 / head_dim as f64) * ln_base);
                let angle = p as f64 * theta;
                cos[p * half + i] = det_math::cos(angle) as f32;
                sin[p * half + i] = det_math::sin(angle) as f32;
            }
        }

        Ok(DetModel {
            embed,
            output,
            layers,
            final_norm,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
            cos,
            sin,
            max_seq,
            kv: vec![KvCache::default(); n_layers],
            act_frac_bits: DET_ACT_FRAC_BITS,
            logit_frac_bits,
        })
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn hidden_dim(&self) -> usize {
        self.dim
    }

    pub fn vocab(&self) -> usize {
        self.embed.rows
    }

    pub fn act_frac_bits(&self) -> u8 {
        self.act_frac_bits
    }

    pub fn clear_kv_cache(&mut self) {
        for c in self.kv.iter_mut() {
            c.k.clear();
            c.v.clear();
        }
    }

    fn rmsnorm(&self, x: &[f32], weight: &[f32], out: &mut [f32]) {
        let mut ss = 0f64;
        for &v in x {
            ss += (v as f64) * (v as f64);
        }
        let scale = (1.0 / (ss / x.len() as f64 + self.rms_eps).sqrt()) as f32;
        for i in 0..x.len() {
            out[i] = x[i] * scale * weight[i];
        }
    }

    /// Interleaved RoPE on one [n_heads_x * head_dim] vector at position p.
    fn rope(&self, x: &mut [f32], n_heads_x: usize, pos: usize) {
        let half = self.head_dim / 2;
        for h in 0..n_heads_x {
            let base = h * self.head_dim;
            for i in 0..half {
                let (c, s) = (self.cos[pos * half + i], self.sin[pos * half + i]);
                let (x0, x1) = (x[base + 2 * i], x[base + 2 * i + 1]);
                x[base + 2 * i] = x0 * c - x1 * s;
                x[base + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }

    /// The embedding row for a token, requantized — cell (pos, 0).
    pub fn embed_row(&self, token: u32) -> Vec<f32> {
        let row = &self.embed.w[token as usize * self.dim..(token as usize + 1) * self.dim];
        let mut out = row.to_vec();
        requantize_all(&mut out, self.act_frac_bits);
        out
    }

    /// One transformer block over the state entering it, using/extending
    /// `cache` with this position's K/V. Output is requantized (the next
    /// cell). `pos` is the absolute position of `x`.
    fn block(&self, layer: &DetLayer, cache: &mut KvCache, x: &[f32], pos: usize) -> Vec<f32> {
        assert!(pos < self.max_seq, "position beyond RoPE table");
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut h = vec![0f32; self.dim];
        self.rmsnorm(x, &layer.attn_norm, &mut h);

        let mut q = vec![0f32; self.dim];
        let mut k = vec![0f32; kv_dim];
        let mut v = vec![0f32; kv_dim];
        layer.wq.matvec(&h, &mut q);
        layer.wk.matvec(&h, &mut k);
        layer.wv.matvec(&h, &mut v);
        self.rope(&mut q, self.n_heads, pos);
        self.rope(&mut k, self.n_kv_heads, pos);
        cache.k.push(k);
        cache.v.push(v);

        // Attention, head-parallel (independent outputs), sequential over
        // cached positions inside each head.
        let group = self.n_heads / self.n_kv_heads;
        let inv_sqrt = 1.0 / (self.head_dim as f32).sqrt();
        let n_pos = cache.k.len();
        let mut attn = vec![0f32; self.dim];
        attn.par_chunks_mut(self.head_dim)
            .enumerate()
            .for_each(|(head, out)| {
                let kv_head = head / group;
                let q_h = &q[head * self.head_dim..(head + 1) * self.head_dim];
                let mut scores = vec![0f32; n_pos];
                for (t, score) in scores.iter_mut().enumerate() {
                    let k_t = &cache.k[t][kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                    let mut dot = 0f32;
                    for (a, b) in q_h.iter().zip(k_t) {
                        dot += a * b;
                    }
                    *score = dot * inv_sqrt;
                }
                // Softmax: max is order-independent; sums are sequential.
                let mut m = f32::NEG_INFINITY;
                for &s in &scores {
                    m = m.max(s);
                }
                let mut denom = 0f64;
                for s in scores.iter_mut() {
                    let e = det_math::exp((*s - m) as f64);
                    *s = e as f32;
                    denom += e;
                }
                let inv_denom = (1.0 / denom) as f32;
                for o in out.iter_mut() {
                    *o = 0.0;
                }
                for (t, s) in scores.iter().enumerate() {
                    let p = s * inv_denom;
                    let v_t = &cache.v[t][kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                    for (o, vv) in out.iter_mut().zip(v_t) {
                        *o += p * vv;
                    }
                }
            });

        let mut attn_out = vec![0f32; self.dim];
        layer.wo.matvec(&attn, &mut attn_out);
        let mut x1 = vec![0f32; self.dim];
        for i in 0..self.dim {
            x1[i] = x[i] + attn_out[i];
        }

        let mut h2 = vec![0f32; self.dim];
        self.rmsnorm(&x1, &layer.ffn_norm, &mut h2);
        let ffn = layer.w_gate.rows;
        let mut gate = vec![0f32; ffn];
        let mut up = vec![0f32; ffn];
        layer.w_gate.matvec(&h2, &mut gate);
        layer.w_up.matvec(&h2, &mut up);
        for i in 0..ffn {
            gate[i] = det_math::silu(gate[i]) * up[i];
        }
        let mut down = vec![0f32; self.dim];
        layer.w_down.matvec(&gate, &mut down);

        let mut out = vec![0f32; self.dim];
        for i in 0..self.dim {
            out[i] = x1[i] + down[i];
        }
        requantize_all(&mut out, self.act_frac_bits);
        out
    }

    /// Final norm + LM head over one hidden state; logits requantized.
    pub fn lm_head(&self, x: &[f32]) -> Vec<f32> {
        let mut h = vec![0f32; self.dim];
        self.rmsnorm(x, &self.final_norm, &mut h);
        let head = self.output.as_ref().unwrap_or(&self.embed);
        let mut logits = vec![0f32; head.rows];
        head.matvec(&h, &mut logits);
        requantize_all(&mut logits, self.logit_frac_bits);
        logits
    }

    /// Feed one token at absolute position `pos`; returns the logits and,
    /// when `capture`, the L+1 requantized hook states (cells of `pos`).
    pub fn step(
        &mut self,
        token: u32,
        pos: usize,
        capture: bool,
    ) -> (Vec<f32>, Option<Vec<Vec<f32>>>) {
        let mut hooks = capture.then(|| Vec::with_capacity(self.layers.len() + 1));
        let mut x = self.embed_row(token);
        // The KV caches are advanced layer by layer for this position.
        let mut kv = std::mem::take(&mut self.kv);
        for (layer, cache) in self.layers.iter().zip(kv.iter_mut()) {
            if let Some(h) = hooks.as_mut() {
                h.push(x.clone());
            }
            x = self.block(layer, cache, &x, pos);
        }
        self.kv = kv;
        if let Some(h) = hooks.as_mut() {
            h.push(x.clone());
        }
        (self.lm_head(&x), hooks)
    }

    /// Verifier: re-execute block `j` over a prefix of committed input
    /// states (positions 0..len); returns the requantized output at the
    /// last position — must equal the committed output cell EXACTLY.
    pub fn forward_block(&self, j: usize, prefix: &[Vec<f32>]) -> Vec<f32> {
        let layer = &self.layers[j];
        let mut cache = KvCache::default();
        let mut out = Vec::new();
        for (pos, x) in prefix.iter().enumerate() {
            out = self.block(layer, &mut cache, x, pos);
        }
        out
    }
}

/// ln via the deterministic exp is not available; rope_base is a constant
/// per model, so a deterministic natural log for that one value is computed
/// with a fixed Newton iteration on exp.
fn det_math_ln(x: f64) -> f64 {
    // Newton: y' = y + x·e^{-y} − 1, seeded from the exponent bits.
    let mut y = ((x.to_bits() >> 52) as i64 - 1023) as f64 * core::f64::consts::LN_2;
    for _ in 0..40 {
        y = y + x * det_math::exp(-y) - 1.0;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_matches_std() {
        for &x in &[10000.0f64, 500000.0, 2.5, 1.0, 0.7] {
            assert!((det_math_ln(x) - x.ln()).abs() < 1e-12, "ln({x})");
        }
    }

    #[test]
    fn requantize_is_idempotent_on_grid() {
        for i in -100000..100000i32 {
            let x = i as f32 * 0.017;
            let once = requantize(x, DET_ACT_FRAC_BITS);
            assert_eq!(once, requantize(once, DET_ACT_FRAC_BITS), "x={x}");
        }
        // Large magnitudes (outlier channels) stay exact up to 2^15.
        for &x in &[300.5f32, 1017.33, 20000.1, 32000.9] {
            let once = requantize(x, DET_ACT_FRAC_BITS);
            assert_eq!(once, requantize(once, DET_ACT_FRAC_BITS), "x={x}");
        }
    }
}
