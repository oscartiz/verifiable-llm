//! CPU verifier for the v0.2 spot-check protocol.
//!
//! Given (transcript, challenge, response, GGUF file), this checks — always
//! on CPU, so a second machine without Metal can verify:
//!
//! 1. the transcript's hash chain replays and binds the trace root;
//! 2. the GGUF file matches the committed model root;
//! 3. the challenge is the Fiat–Shamir derivation from the transcript;
//! 4. every revealed cell Merkle-verifies against the trace root;
//! 5. every challenged block, re-executed over the revealed input prefix
//!    with the committed weights, reproduces the committed output within a
//!    numerical tolerance (float re-execution across backends is not
//!    bit-exact; see DECISIONS.md #4 and #9);
//! 6. layer-0 inputs equal the token embeddings of the committed tokens;
//! 7. head challenges: the revealed logits hash exactly to the chain's
//!    logits_hash, and the recomputed LM head output matches them within
//!    tolerance.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{Device, IndexOp, Tensor};

use vllm_core::chain::hash_logits_q;
use vllm_core::commit;
use vllm_core::protocol::{
    Challenge, ChallengeSpace, Response, b64_decode_i32, check_revealed_cell, derive_challenges,
};
use vllm_core::trace::{TraceMeta, dequantize, quantize};
use vllm_core::transcript::{ChainCheck, Transcript};
use vllm_infer::det::{DET_BACKEND, DetModel};
use vllm_infer::model::ModelWeights;

/// Default max |Δ| per activation element when re-executing a block on CPU
/// against a trace produced on Metal. Calibrated on Llama-3.2-1B Q4_K_M —
/// see DECISIONS.md #9 — with headroom; tighten with --tolerance for
/// same-backend verification.
pub const DEFAULT_TOLERANCE: f32 = 0.5;

/// Default max mean |Δ| per challenged cell; see [`VerifyConfig`] and
/// DECISIONS.md #15 for the calibration.
pub const DEFAULT_MEAN_TOLERANCE: f32 = 0.05;

#[derive(Debug, Clone, Copy)]
pub struct VerifyConfig {
    /// Max per-element |Δ| for block outputs and embeddings.
    pub tolerance: f32,
    /// Max per-element |Δ| for recomputed logits (head challenges).
    pub logits_tolerance: f32,
    /// Max MEAN |Δ| per challenged cell. Honest backend drift concentrates
    /// far below the per-element tolerance, while the bounded-drift attack
    /// (REPORT.md) must spend the whole budget at nearly every coordinate —
    /// the mean separates the two by ~2 orders of magnitude.
    pub mean_tolerance: f32,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        VerifyConfig {
            tolerance: DEFAULT_TOLERANCE,
            logits_tolerance: DEFAULT_TOLERANCE,
            mean_tolerance: DEFAULT_MEAN_TOLERANCE,
        }
    }
}

#[derive(Debug)]
pub struct ItemReport {
    pub pos: u32,
    pub layer: u32,
    /// Max |Δ| between recomputation and committed values.
    pub max_dev: f32,
    /// Mean |Δ| over the compared vector.
    pub mean_dev: f32,
}

#[derive(Debug)]
pub struct VerifyReport {
    pub items: Vec<ItemReport>,
    /// Largest deviation across all checks (for tolerance calibration).
    pub max_dev: f32,
    /// Largest per-cell mean deviation across all checks.
    pub max_mean_dev: f32,
}

/// Reconstruct the TraceMeta the transcript commits to.
fn trace_meta_of(t: &Transcript) -> Result<TraceMeta> {
    let info = t
        .trace
        .as_ref()
        .context("transcript has no trace commitment")?;
    Ok(TraceMeta {
        version: vllm_core::trace::TRACE_VERSION.into(),
        n_positions: info.n_positions,
        n_layers: info.n_layers,
        hidden_dim: info.hidden_dim,
        frac_bits: info.frac_bits,
        root: info.root,
        vocab_size: t.vocab_size,
        logit_frac_bits: t.logit_frac_bits,
        first_logit_pos: t.prompt_token_ids.len() as u32 - 1,
        n_logit_rows: t.steps.len() as u32,
        zk_salts: None,
    })
}

/// Token at absolute position p (prompt, then generated tokens in order).
fn token_at(t: &Transcript, pos: u32) -> Result<u32> {
    let p = pos as usize;
    let prompt = &t.prompt_token_ids;
    if p < prompt.len() {
        Ok(prompt[p])
    } else {
        t.steps
            .get(p - prompt.len())
            .map(|s| s.token_id)
            .with_context(|| format!("no token at position {pos}"))
    }
}

pub fn verify(
    model_path: &Path,
    transcript: &Transcript,
    challenge: &Challenge,
    response: &Response,
    config: &VerifyConfig,
) -> Result<VerifyReport> {
    // 1. Chain + trace binding.
    match transcript.replay_chain() {
        ChainCheck::Ok => {}
        bad => bail!("transcript chain does not replay: {bad:?}"),
    }
    transcript
        .check_trace_binding()
        .map_err(|e| anyhow::anyhow!("trace binding: {e}"))?;
    let meta = trace_meta_of(transcript)?;

    // 2. Model root.
    let model_commitment = commit::commit_gguf(model_path)?;
    if model_commitment.root != transcript.model_root {
        bail!(
            "model file root {} does not match transcript's committed root {}",
            model_commitment.root,
            transcript.model_root
        );
    }

    // 3. Challenge authenticity (Fiat–Shamir re-derivation).
    if challenge.final_chain != transcript.final_chain || challenge.trace_root != meta.root {
        bail!("challenge does not reference this transcript");
    }
    let space = ChallengeSpace::from_transcript(transcript)?;
    let expected = derive_challenges(
        &challenge.final_chain,
        &challenge.trace_root,
        challenge.k,
        &challenge.nonce,
        &space,
    );
    if expected != challenge.cells {
        bail!("challenge cells are not the Fiat–Shamir derivation for this transcript");
    }
    if response.trace_root != meta.root {
        bail!("response references a different trace root");
    }
    if response.items.len() != challenge.cells.len() {
        bail!(
            "response has {} items for {} challenges",
            response.items.len(),
            challenge.cells.len()
        );
    }

    // 4-7. Load the model on CPU and check every item. Deterministic
    // transcripts get EXACT verification (tolerance zero, i32 equality);
    // float transcripts get tolerance-based verification.
    let exact = transcript.env.backend == DET_BACKEND;
    let mut det_model = None;
    let mut float_model = None;
    let (n_layers, hidden_dim) = if exact {
        let m = DetModel::load(model_path, transcript.logit_frac_bits)?;
        if m.act_frac_bits() != meta.frac_bits {
            bail!(
                "deterministic transcript committed at 2^-{} but this det backend uses 2^-{}",
                meta.frac_bits,
                m.act_frac_bits()
            );
        }
        let dims = (m.n_layers(), m.hidden_dim());
        det_model = Some(m);
        dims
    } else {
        let device = Device::Cpu;
        let mut file = File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let m = ModelWeights::from_gguf(content, &mut file, &device)?;
        let dims = (m.n_layers(), m.hidden_dim());
        float_model = Some(m);
        dims
    };
    if n_layers as u32 != meta.n_layers || hidden_dim as u32 != meta.hidden_dim {
        bail!(
            "model shape ({n_layers} layers, dim {hidden_dim}) does not match trace ({}, {})",
            meta.n_layers,
            meta.hidden_dim
        );
    }

    let mut items = Vec::with_capacity(response.items.len());
    let mut max_dev = 0f32;
    let mut max_mean_dev = 0f32;
    for (item, &cell) in response.items.iter().zip(&challenge.cells) {
        if item.cell != cell {
            bail!(
                "response item for ({}, {}) out of order",
                cell.pos,
                cell.layer
            );
        }
        let (dev, mean) = match (&mut det_model, &mut float_model) {
            (Some(m), _) => {
                if cell.layer < meta.n_layers {
                    verify_block_det(m, transcript, &meta, item)?
                } else {
                    verify_head_det(m, transcript, &meta, item)?
                }
            }
            (None, Some(m)) => {
                let device = Device::Cpu;
                if cell.layer < meta.n_layers {
                    verify_block(m, transcript, &meta, item, config, &device)?
                } else {
                    verify_head(m, transcript, &meta, item, config, &device)?
                }
            }
            _ => unreachable!(),
        };
        max_dev = max_dev.max(dev);
        max_mean_dev = max_mean_dev.max(mean);
        items.push(ItemReport {
            pos: cell.pos,
            layer: cell.layer,
            max_dev: dev,
            mean_dev: mean,
        });
    }
    Ok(VerifyReport {
        items,
        max_dev,
        max_mean_dev,
    })
}

/// Check one block challenge; returns (max, mean) deviation observed.
fn verify_block(
    model: &mut ModelWeights,
    transcript: &Transcript,
    meta: &TraceMeta,
    item: &vllm_core::protocol::ResponseItem,
    config: &VerifyConfig,
    device: &Device,
) -> Result<(f32, f32)> {
    let cell = item.cell;
    let seq = cell.pos as usize + 1;
    if item.inputs.len() != seq {
        bail!(
            "challenge ({}, {}): expected {seq} input cells, got {}",
            cell.pos,
            cell.layer,
            item.inputs.len()
        );
    }

    // Merkle-check and decode the revealed input prefix.
    let mut prefix = Vec::with_capacity(seq * meta.hidden_dim as usize);
    for (p, revealed) in item.inputs.iter().enumerate() {
        if revealed.pos != p as u32 || revealed.layer != cell.layer {
            bail!(
                "challenge ({}, {}): input cell at wrong coordinates",
                cell.pos,
                cell.layer
            );
        }
        let data = check_revealed_cell(revealed, meta, &meta.root)?;
        prefix.extend(dequantize(&data, meta.frac_bits));
    }

    let output = item.output.as_ref().with_context(|| {
        format!(
            "challenge ({}, {}): missing output cell",
            cell.pos, cell.layer
        )
    })?;
    if output.pos != cell.pos || output.layer != cell.layer + 1 {
        bail!(
            "challenge ({}, {}): output cell at wrong coordinates",
            cell.pos,
            cell.layer
        );
    }
    let committed_out = dequantize(
        &check_revealed_cell(output, meta, &meta.root)?,
        meta.frac_bits,
    );

    let mut max_dev = 0f32;

    // Layer-0 inputs must equal the embeddings of the committed tokens
    // (otherwise nothing ties the trace to the actual token sequence).
    if cell.layer == 0 {
        let tokens: Vec<u32> = (0..=cell.pos)
            .map(|p| token_at(transcript, p))
            .collect::<Result<_>>()?;
        let emb: Vec<f32> = model
            .embed(&tokens, device)?
            .squeeze(0)?
            .flatten_all()?
            .to_vec1()?;
        for (a, b) in emb.iter().zip(&prefix) {
            max_dev = max_dev.max((a - b).abs());
        }
        if max_dev > config.tolerance {
            bail!(
                "challenge ({}, 0): layer-0 inputs deviate from token embeddings by {max_dev}",
                cell.pos
            );
        }
    }

    // Re-execute the block over the prefix and compare the final position.
    let h = Tensor::from_vec(prefix, (1, seq, meta.hidden_dim as usize), device)?;
    let out = model.forward_block(cell.layer as usize, &h, seq)?;
    let recomputed: Vec<f32> = out.i((0, seq - 1, ..))?.to_vec1()?;
    let mut dev = 0f32;
    let mut sum = 0f64;
    for (a, b) in recomputed.iter().zip(&committed_out) {
        let d = (a - b).abs();
        dev = dev.max(d);
        sum += d as f64;
    }
    let mean = (sum / recomputed.len() as f64) as f32;
    if dev > config.tolerance {
        bail!(
            "challenge ({}, {}): recomputed block output deviates by {dev} (tolerance {})",
            cell.pos,
            cell.layer,
            config.tolerance
        );
    }
    if mean > config.mean_tolerance {
        bail!(
            "challenge ({}, {}): mean deviation {mean} exceeds {} — committed outputs are \
             uniformly displaced from the recomputation (bounded-drift attack signature)",
            cell.pos,
            cell.layer,
            config.mean_tolerance
        );
    }
    Ok((max_dev.max(dev), mean))
}

/// Check one head challenge; returns (max, mean) deviation observed.
fn verify_head(
    model: &ModelWeights,
    transcript: &Transcript,
    meta: &TraceMeta,
    item: &vllm_core::protocol::ResponseItem,
    config: &VerifyConfig,
    device: &Device,
) -> Result<(f32, f32)> {
    let cell = item.cell;
    let step = (cell.pos - meta.first_logit_pos) as usize;
    let record = transcript
        .steps
        .get(step)
        .with_context(|| format!("head challenge at pos {} has no step", cell.pos))?;

    let [input] = item.inputs.as_slice() else {
        bail!(
            "head challenge at pos {}: expected exactly one input cell",
            cell.pos
        );
    };
    if input.pos != cell.pos || input.layer != meta.n_layers {
        bail!(
            "head challenge at pos {}: input cell at wrong coordinates",
            cell.pos
        );
    }
    let hidden = dequantize(
        &check_revealed_cell(input, meta, &meta.root)?,
        meta.frac_bits,
    );

    // The revealed logits must hash to the chain's committed logits_hash —
    // an exact, binding check.
    let logits_b64 = item
        .logits
        .as_ref()
        .with_context(|| format!("head challenge at pos {}: missing logits", cell.pos))?;
    let revealed_q = b64_decode_i32(logits_b64)
        .with_context(|| format!("head challenge at pos {}: bad logits encoding", cell.pos))?;
    if revealed_q.len() != transcript.vocab_size as usize {
        bail!(
            "head challenge at pos {}: logits have wrong dimension",
            cell.pos
        );
    }
    if hash_logits_q(&revealed_q, transcript.logit_frac_bits) != record.logits_hash {
        bail!(
            "head challenge at pos {}: revealed logits do not hash to the committed logits_hash",
            cell.pos
        );
    }
    let revealed = dequantize(&revealed_q, transcript.logit_frac_bits);

    // Recompute the LM head from the committed hidden state.
    let h = Tensor::from_vec(hidden, (1, 1, meta.hidden_dim as usize), device)?;
    let recomputed: Vec<f32> = model.lm_head(&h)?.flatten_all()?.to_vec1()?;
    let mut dev = 0f32;
    let mut sum = 0f64;
    for (a, b) in recomputed.iter().zip(&revealed) {
        let d = (a - b).abs();
        dev = dev.max(d);
        sum += d as f64;
    }
    let mean = (sum / recomputed.len() as f64) as f32;
    if dev > config.logits_tolerance {
        bail!(
            "head challenge at pos {}: recomputed logits deviate by {dev} (tolerance {})",
            cell.pos,
            config.logits_tolerance
        );
    }
    if mean > config.mean_tolerance {
        bail!(
            "head challenge at pos {}: mean logit deviation {mean} exceeds {} (bounded-drift \
             attack signature)",
            cell.pos,
            config.mean_tolerance
        );
    }
    Ok((dev, mean))
}

/// Exact block check for deterministic transcripts: recompute block `j`
/// over the dequantized committed inputs, requantize, and require i32
/// equality with the committed output cell. Any nonzero difference —
/// a single quantum — is a cheat.
fn verify_block_det(
    model: &DetModel,
    transcript: &Transcript,
    meta: &TraceMeta,
    item: &vllm_core::protocol::ResponseItem,
) -> Result<(f32, f32)> {
    let cell = item.cell;
    let seq = cell.pos as usize + 1;
    if item.inputs.len() != seq {
        bail!(
            "challenge ({}, {}): expected {seq} input cells, got {}",
            cell.pos,
            cell.layer,
            item.inputs.len()
        );
    }
    let mut prefix = Vec::with_capacity(seq);
    for (p, revealed) in item.inputs.iter().enumerate() {
        if revealed.pos != p as u32 || revealed.layer != cell.layer {
            bail!(
                "challenge ({}, {}): input cell at wrong coordinates",
                cell.pos,
                cell.layer
            );
        }
        let data = check_revealed_cell(revealed, meta, &meta.root)?;
        prefix.push(dequantize(&data, meta.frac_bits));
    }
    let output = item.output.as_ref().with_context(|| {
        format!(
            "challenge ({}, {}): missing output cell",
            cell.pos, cell.layer
        )
    })?;
    if output.pos != cell.pos || output.layer != cell.layer + 1 {
        bail!(
            "challenge ({}, {}): output cell at wrong coordinates",
            cell.pos,
            cell.layer
        );
    }
    let committed_q = check_revealed_cell(output, meta, &meta.root)?;

    // Layer-0 inputs must BE the committed tokens' embeddings, exactly.
    if cell.layer == 0 {
        for (p, x) in prefix.iter().enumerate() {
            let expected = model.embed_row(token_at(transcript, p as u32)?);
            if &expected != x {
                bail!(
                    "challenge ({}, 0): layer-0 input at position {p} is not the token embedding",
                    cell.pos
                );
            }
        }
    }

    let recomputed = model.forward_block(cell.layer as usize, &prefix);
    let recomputed_q = quantize(&recomputed, meta.frac_bits)
        .map_err(|e| anyhow::anyhow!("recomputed activations unquantizable: {e}"))?;
    exact_compare(&recomputed_q, &committed_q, cell.pos, cell.layer)
}

/// Exact head check: recomputed logits must equal the revealed (chain-bound)
/// quantized logits, i32 for i32.
fn verify_head_det(
    model: &DetModel,
    transcript: &Transcript,
    meta: &TraceMeta,
    item: &vllm_core::protocol::ResponseItem,
) -> Result<(f32, f32)> {
    let cell = item.cell;
    let step = (cell.pos - meta.first_logit_pos) as usize;
    let record = transcript
        .steps
        .get(step)
        .with_context(|| format!("head challenge at pos {} has no step", cell.pos))?;
    let [input] = item.inputs.as_slice() else {
        bail!(
            "head challenge at pos {}: expected exactly one input cell",
            cell.pos
        );
    };
    if input.pos != cell.pos || input.layer != meta.n_layers {
        bail!(
            "head challenge at pos {}: input cell at wrong coordinates",
            cell.pos
        );
    }
    let hidden = dequantize(
        &check_revealed_cell(input, meta, &meta.root)?,
        meta.frac_bits,
    );
    let logits_b64 = item
        .logits
        .as_ref()
        .with_context(|| format!("head challenge at pos {}: missing logits", cell.pos))?;
    let revealed_q = b64_decode_i32(logits_b64)
        .with_context(|| format!("head challenge at pos {}: bad logits encoding", cell.pos))?;
    if hash_logits_q(&revealed_q, transcript.logit_frac_bits) != record.logits_hash {
        bail!(
            "head challenge at pos {}: revealed logits do not hash to the committed logits_hash",
            cell.pos
        );
    }
    let recomputed = model.lm_head(&hidden);
    let recomputed_q = quantize(&recomputed, transcript.logit_frac_bits)
        .map_err(|e| anyhow::anyhow!("recomputed logits unquantizable: {e}"))?;
    exact_compare(&recomputed_q, &revealed_q, cell.pos, cell.layer)
}

/// i32-exact comparison; reports the max quantum difference on failure.
fn exact_compare(
    recomputed: &[i32],
    committed: &[i32],
    pos: u32,
    layer: u32,
) -> Result<(f32, f32)> {
    if recomputed.len() != committed.len() {
        bail!("challenge ({pos}, {layer}): dimension mismatch");
    }
    let max_dq = recomputed
        .iter()
        .zip(committed)
        .map(|(a, b)| (*a as i64 - *b as i64).unsigned_abs())
        .max()
        .unwrap_or(0);
    if max_dq != 0 {
        bail!(
            "challenge ({pos}, {layer}): EXACT verification failed — recomputation differs from \
             the committed cell by up to {max_dq} quanta (deterministic transcripts admit zero)"
        );
    }
    Ok((0.0, 0.0))
}
