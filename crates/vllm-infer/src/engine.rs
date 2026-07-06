//! Quantized llama inference (candle) with commitment hooks: every decode
//! step hashes the logit vector and folds (token, logits_hash) into the
//! vllm-core hash chain, producing a Transcript alongside the text.

use std::fs::File;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::model::ModelWeights;
use vllm_core::Hash32;
use vllm_core::chain::{Chain, SamplerConfig, SamplerMode, hash_logits, hash_logits_q};
use vllm_core::commit::ModelCommitment;
use vllm_core::trace::{TraceBuilder, quantize};
use vllm_core::transcript::{EnvInfo, StepRecord, TRANSCRIPT_VERSION, TraceInfo, Transcript};

/// Default fixed-point precision for logit hashing; see DECISIONS.md.
/// 16 fractional bits ≈ 1.5e-5 resolution: far coarser than same-backend
/// reproducibility noise (measured zero on Metal), far finer than the
/// cross-backend CPU↔Metal deltas (up to ~0.5) that make cross-backend hash
/// equality impossible regardless of precision.
pub const DEFAULT_LOGIT_FRAC_BITS: u8 = 16;

pub enum Prompt {
    /// Tokenized with the provided tokenizer, wrapped in the llama3 chat
    /// template unless `raw` is set.
    Text(String),
    /// Pre-tokenized ids; no tokenizer needed (used by tests).
    Tokens(Vec<u32>),
}

/// A per-step circuit-friendly logits commitment produced by the Layer-3
/// hook (vllm-zk supplies the implementation; vllm-infer stays free of the
/// proof-system dependency).
pub struct ZkCommitment {
    pub salt: [u64; 4],
    pub digest: [u8; 32],
}

pub type ZkCommitFn = dyn Fn(&[i32]) -> Result<ZkCommitment, String> + Send;

pub struct GenerateRequest {
    pub model_path: PathBuf,
    /// Required for `Prompt::Text`; also used to detokenize output.
    pub tokenizer_path: Option<PathBuf>,
    pub prompt: Prompt,
    /// Skip the llama3-instruct chat template.
    pub raw: bool,
    pub max_new_tokens: usize,
    pub sampler: SamplerConfig,
    pub logit_frac_bits: u8,
    pub force_cpu: bool,
    /// Write an activation trace here (v0.2 challenge protocol). Uses
    /// `logit_frac_bits` for activation cells too.
    pub trace_path: Option<PathBuf>,
    /// Layer-3 hook: commit each step's quantized logits with a salted,
    /// circuit-friendly hash, folded into the chain. Requires `trace_path`
    /// (salts and logits are stored in the trace file for later proving).
    pub zk_commit: Option<Box<ZkCommitFn>>,
    /// Run the deterministic CPU backend (bit-exact re-execution; v0.2
    /// challenges verify with tolerance zero). Slower; see det.rs.
    pub deterministic: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub model_load: Duration,
    pub prompt_eval: Duration,
    /// Whole decode loop, commitment work included.
    pub decode: Duration,
    /// Time spent hashing logits and extending the chain (subset of
    /// prompt_eval + decode).
    pub commit: Duration,
    pub tokens_generated: usize,
}

impl Timing {
    /// Commitment overhead as a fraction of end-to-end inference time.
    pub fn commit_overhead(&self) -> f64 {
        let total = self.prompt_eval + self.decode;
        if total.is_zero() {
            return 0.0;
        }
        self.commit.as_secs_f64() / total.as_secs_f64()
    }
}

pub struct GenerateOutput {
    pub text: String,
    pub transcript: Transcript,
    pub timing: Timing,
}

pub fn pick_device(force_cpu: bool) -> Result<(Device, &'static str)> {
    #[cfg(feature = "metal")]
    if !force_cpu {
        return Ok((Device::new_metal(0)?, "metal"));
    }
    let _ = force_cpu;
    Ok((Device::Cpu, "cpu"))
}

fn sampling_for(config: &SamplerConfig) -> Sampling {
    match config.mode {
        SamplerMode::Greedy => Sampling::ArgMax,
        SamplerMode::TopP => Sampling::TopP {
            p: config.top_p as f64,
            temperature: config.temperature as f64,
        },
    }
}

const LLAMA3_TEMPLATE: (&str, &str) = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
);

struct PromptSetup {
    token_ids: Vec<u32>,
    tokenizer: Option<tokenizers::Tokenizer>,
    stop_tokens: Vec<u32>,
}

fn prepare_prompt(req: &GenerateRequest) -> Result<PromptSetup> {
    let tokenizer = match &req.tokenizer_path {
        Some(path) => Some(
            tokenizers::Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("loading tokenizer {path:?}: {e}"))?,
        ),
        None => None,
    };

    let token_ids = match &req.prompt {
        Prompt::Tokens(ids) => ids.clone(),
        Prompt::Text(text) => {
            let tokenizer = tokenizer
                .as_ref()
                .context("a --tokenizer is required for text prompts")?;
            let full = if req.raw {
                text.clone()
            } else {
                format!("{}{}{}", LLAMA3_TEMPLATE.0, text, LLAMA3_TEMPLATE.1)
            };
            tokenizer
                .encode(full, false)
                .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
                .get_ids()
                .to_vec()
        }
    };
    if token_ids.is_empty() {
        bail!("empty prompt");
    }

    let stop_tokens = tokenizer
        .as_ref()
        .map(|t| {
            ["<|eot_id|>", "<|end_of_text|>"]
                .iter()
                .filter_map(|s| t.token_to_id(s))
                .collect()
        })
        .unwrap_or_default();

    Ok(PromptSetup {
        token_ids,
        tokenizer,
        stop_tokens,
    })
}

/// The engine's view of an inference backend: feed tokens, get logits and
/// (optionally) the per-position commitment-hook states. Keeping the chain,
/// trace, and zk logic in `generate` — single copy — means the transcript
/// format cannot drift between the float and deterministic paths.
trait InferBackend {
    fn n_layers(&self) -> usize;
    fn hidden_dim(&self) -> usize;
    fn name(&self) -> &'static str;
    /// Activation-cell fixed-point precision this backend commits at.
    fn cell_frac_bits(&self, logit_frac_bits: u8) -> u8;
    /// Feed `tokens` at absolute positions `index_pos..`; return the logits
    /// of the last fed position and, when `capture`, the hook states as
    /// hooks[pos_in_batch][hook 0..=n_layers] = Vec<f32> of hidden_dim.
    #[allow(clippy::type_complexity)]
    fn step(
        &mut self,
        tokens: &[u32],
        index_pos: usize,
        capture: bool,
    ) -> Result<(Vec<f32>, Option<Vec<Vec<Vec<f32>>>>)>;
}

struct CandleBackend {
    model: ModelWeights,
    device: Device,
    name: &'static str,
}

impl InferBackend for CandleBackend {
    fn n_layers(&self) -> usize {
        self.model.n_layers()
    }

    fn hidden_dim(&self) -> usize {
        self.model.hidden_dim()
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn cell_frac_bits(&self, logit_frac_bits: u8) -> u8 {
        logit_frac_bits
    }

    fn step(
        &mut self,
        tokens: &[u32],
        index_pos: usize,
        capture: bool,
    ) -> Result<(Vec<f32>, Option<Vec<Vec<Vec<f32>>>>)> {
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let to_vec = |t: Tensor| -> Result<Vec<f32>> {
            Ok(t.squeeze(0)?.to_dtype(candle_core::DType::F32)?.to_vec1()?)
        };
        if capture {
            let mut layer_rows: Vec<Vec<Vec<f32>>> = Vec::new(); // [hook][pos]
            let logits = self.model.forward_traced(&input, index_pos, &mut |_j, h| {
                layer_rows.push(h.squeeze(0)?.to_dtype(candle_core::DType::F32)?.to_vec2()?);
                Ok(())
            })?;
            let hooks = (0..tokens.len())
                .map(|p| layer_rows.iter().map(|rows| rows[p].clone()).collect())
                .collect();
            Ok((to_vec(logits)?, Some(hooks)))
        } else {
            Ok((to_vec(self.model.forward(&input, index_pos)?)?, None))
        }
    }
}

struct DetBackend {
    model: crate::det::DetModel,
}

impl InferBackend for DetBackend {
    fn n_layers(&self) -> usize {
        self.model.n_layers()
    }

    fn hidden_dim(&self) -> usize {
        self.model.hidden_dim()
    }

    fn name(&self) -> &'static str {
        crate::det::DET_BACKEND
    }

    fn cell_frac_bits(&self, _logit_frac_bits: u8) -> u8 {
        self.model.act_frac_bits()
    }

    fn step(
        &mut self,
        tokens: &[u32],
        index_pos: usize,
        capture: bool,
    ) -> Result<(Vec<f32>, Option<Vec<Vec<Vec<f32>>>>)> {
        // Deterministic mode processes positions one at a time even for the
        // prompt: there is no separate batch path to diverge from.
        let mut hooks = capture.then(Vec::new);
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let (l, h) = self.model.step(token, index_pos + i, capture);
            logits = l;
            if let (Some(all), Some(h)) = (hooks.as_mut(), h) {
                all.push(h);
            }
        }
        Ok((logits, hooks))
    }
}

/// Run generation, committing every decode step into the hash chain.
/// `on_text` receives detokenized output incrementally (streaming).
pub fn generate(
    req: &GenerateRequest,
    model_commitment: &ModelCommitment,
    mut on_text: Option<&mut dyn FnMut(&str)>,
) -> Result<GenerateOutput> {
    let PromptSetup {
        token_ids: prompt_ids,
        tokenizer,
        stop_tokens,
    } = prepare_prompt(req)?;
    if req.zk_commit.is_some() && req.trace_path.is_none() {
        bail!("--prove-decode requires --trace (salts and logits live in the trace file)");
    }
    let mut timing = Timing::default();
    let t0 = Instant::now();
    let mut backend: Box<dyn InferBackend> = if req.deterministic {
        Box::new(DetBackend {
            model: crate::det::DetModel::load(&req.model_path, req.logit_frac_bits)?,
        })
    } else {
        let (device, name) = pick_device(req.force_cpu)?;
        let mut file = File::open(&req.model_path)
            .with_context(|| format!("opening model {:?}", req.model_path))?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;
        Box::new(CandleBackend {
            model,
            device,
            name,
        })
    };
    timing.model_load = t0.elapsed();

    let mut chain = Chain::seed(&model_commitment.root, &prompt_ids, &req.sampler);
    let chain_seed = chain.value();
    let mut logits_processor =
        LogitsProcessor::from_sampling(req.sampler.rng_seed, sampling_for(&req.sampler));

    let n_layers = backend.n_layers();
    let cells_per_pos = n_layers + 1;
    let cell_frac_bits = backend.cell_frac_bits(req.logit_frac_bits);
    let mut tracer: Option<TraceBuilder> = req.trace_path.as_ref().map(|_| {
        TraceBuilder::new(
            n_layers as u32,
            backend.hidden_dim() as u32,
            cell_frac_bits,
            req.logit_frac_bits,
            prompt_ids.len() as u32 - 1,
        )
    });

    // Prompt evaluation; the returned logits (last position) are what the
    // first generated token is sampled from, so they are committed as
    // step 0.
    let t0 = Instant::now();
    let (mut logits_vec, prompt_hooks) = backend.step(&prompt_ids, 0, tracer.is_some())?;
    if let (Some(tb), Some(hooks)) = (tracer.as_mut(), prompt_hooks) {
        let t_commit = Instant::now();
        for per_pos in &hooks {
            for h in per_pos {
                tb.push_cell(h)?;
            }
        }
        timing.commit += t_commit.elapsed();
    }
    timing.prompt_eval = t0.elapsed();

    // Fold the cell hashes of prompt positions 0..P-2 into the chain; the
    // last prompt position's cells belong to step 0 (they computed its
    // logits) and are folded there.
    let mut prompt_layer_hashes: Option<Vec<Hash32>> = None;
    if let Some(tb) = tracer.as_ref() {
        let split = (prompt_ids.len() - 1) * cells_per_pos;
        let hashes = tb.hashes()[..split].to_vec();
        chain.absorb_layer_hashes(&hashes);
        prompt_layer_hashes = Some(hashes);
    }

    let vocab_size = logits_vec.len() as u32;
    let mut steps: Vec<StepRecord> = Vec::with_capacity(req.max_new_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(req.max_new_tokens);
    let mut text = String::new();
    let mut detok_consumed = 0;

    let t_decode = Instant::now();
    for step in 0..req.max_new_tokens {
        // Commit the logits, then sample from those same logits.
        let t_commit = Instant::now();
        let mut zk_digest: Option<Hash32> = None;
        let logits_hash = if let Some(tb) = tracer.as_mut() {
            let q = quantize(&logits_vec, req.logit_frac_bits)?;
            let h = hash_logits_q(&q, req.logit_frac_bits);
            if let Some(zk) = &req.zk_commit {
                let commitment =
                    zk(&q).map_err(|e| anyhow::anyhow!("zk commit failed at step {step}: {e}"))?;
                tb.push_zk_salt(commitment.salt);
                zk_digest = Some(Hash32(commitment.digest));
            }
            tb.push_logits_row(q);
            h
        } else {
            hash_logits(&logits_vec, req.logit_frac_bits, step)?
        };
        timing.commit += t_commit.elapsed();

        let logits_tensor = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
        let token_id = logits_processor.sample(&logits_tensor)?;

        let t_commit = Instant::now();
        let chain_value = chain.absorb_step(token_id, &logits_hash);
        // Fold the cells of the position that computed this step's logits.
        let (layer_hashes, mut chain_value) = match tracer.as_ref() {
            Some(tb) => {
                let pos = prompt_ids.len() - 1 + step;
                let hashes = tb.hashes()[pos * cells_per_pos..(pos + 1) * cells_per_pos].to_vec();
                let value = chain.absorb_layer_hashes(&hashes);
                (Some(hashes), value)
            }
            None => (None, chain_value),
        };
        if let Some(digest) = &zk_digest {
            chain_value = chain.absorb_zk_digest(digest);
        }
        steps.push(StepRecord {
            token_id,
            logits_hash,
            chain: chain_value,
            layer_hashes,
            zk_digest,
        });
        timing.commit += t_commit.elapsed();

        generated.push(token_id);
        if let Some(tokenizer) = &tokenizer {
            let decoded = tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("detokenizing: {e}"))?;
            // Hold back while the tail could still be an incomplete UTF-8
            // sequence from a partial token.
            if !decoded.ends_with('\u{FFFD}') && decoded.len() > detok_consumed {
                let fresh = &decoded[detok_consumed..];
                if let Some(cb) = on_text.as_deref_mut() {
                    cb(fresh);
                }
                text.push_str(fresh);
                detok_consumed = decoded.len();
            }
        }

        if stop_tokens.contains(&token_id) || step + 1 == req.max_new_tokens {
            break;
        }

        let index_pos = prompt_ids.len() + step;
        let (l, hooks) = backend.step(&[token_id], index_pos, tracer.is_some())?;
        logits_vec = l;
        if let (Some(tb), Some(hooks)) = (tracer.as_mut(), hooks) {
            let t_commit = Instant::now();
            for h in &hooks[0] {
                tb.push_cell(h)?;
            }
            timing.commit += t_commit.elapsed();
        }
    }
    timing.decode = t_decode.elapsed();
    timing.tokens_generated = generated.len();

    // The builder holds cells for exactly the processed positions
    // (prompt + generated - 1): the final sampled token is never fed back,
    // so it has no cells.
    let trace = match (tracer, &req.trace_path) {
        (Some(tb), Some(path)) => {
            let t_commit = Instant::now();
            let meta = tb.write(path)?;
            timing.commit += t_commit.elapsed();
            Some(TraceInfo {
                root: meta.root,
                n_positions: meta.n_positions,
                n_layers: meta.n_layers,
                hidden_dim: meta.hidden_dim,
                frac_bits: meta.frac_bits,
            })
        }
        _ => None,
    };

    let transcript = Transcript {
        version: TRANSCRIPT_VERSION.into(),
        model_root: model_commitment.root,
        model_file: model_commitment.file_name.clone(),
        prompt_token_ids: prompt_ids,
        sampler: req.sampler.clone(),
        logit_frac_bits: req.logit_frac_bits,
        vocab_size,
        chain_seed,
        prompt_layer_hashes,
        trace,
        steps,
        final_chain: chain.value(),
        text: (!text.is_empty()).then(|| text.clone()),
        env: EnvInfo {
            backend: backend.name().into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
        },
    };

    Ok(GenerateOutput {
        text,
        transcript,
        timing,
    })
}
