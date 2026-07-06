//! Running hash chain over inference state.
//!
//! seed    = H(seed-domain || model_root || prompt_tokens || sampler_config || rng_seed)
//! chain_i = H(step-domain || chain_{i-1} || i || token_id_i || H(quantized logits_i))
//!
//! Logits are committed at a fixed precision: q = round(logit * 2^frac_bits)
//! as i32, because float outputs are not bit-stable across backends. The
//! precision is part of the transcript so a verifier knows exactly what was
//! hashed. All integers are little-endian.

use serde::{Deserialize, Serialize};

use crate::{Error, Hash32};

const SEED_DOMAIN: &[u8] = b"vllm/chain-seed/v1";
const STEP_DOMAIN: &[u8] = b"vllm/chain-step/v1";
const LOGITS_DOMAIN: &[u8] = b"vllm/logits/v1";
const LAYERS_DOMAIN: &[u8] = b"vllm/chain-layers/v1";
const ZK_DOMAIN: &[u8] = b"vllm/chain-zk/v1";

/// Sampling configuration, committed byte-exactly into the chain seed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplerConfig {
    /// "greedy" or "top-p"; greedy ignores the remaining fields.
    pub mode: SamplerMode,
    pub temperature: f32,
    pub top_p: f32,
    pub rng_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SamplerMode {
    Greedy,
    TopP,
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        SamplerConfig {
            mode: SamplerMode::Greedy,
            temperature: 0.0,
            top_p: 1.0,
            rng_seed: 0,
        }
    }

    /// Canonical byte encoding hashed into the chain seed. Floats are encoded
    /// as raw IEEE-754 bits, so the commitment is to the exact configuration.
    fn canonical_bytes(&self) -> [u8; 17] {
        let mut out = [0u8; 17];
        out[0] = match self.mode {
            SamplerMode::Greedy => 0,
            SamplerMode::TopP => 1,
        };
        out[1..5].copy_from_slice(&self.temperature.to_bits().to_le_bytes());
        out[5..9].copy_from_slice(&self.top_p.to_bits().to_le_bytes());
        out[9..17].copy_from_slice(&self.rng_seed.to_le_bytes());
        out
    }
}

/// Commit to a logit vector at fixed precision. `frac_bits` is the number of
/// fractional bits kept: q = round(x * 2^frac_bits) as i32, little-endian.
/// ±inf, and finite values whose scaled magnitude exceeds i32 range, saturate
/// at the i32 bounds (unreachable for real logits at frac_bits ≤ 16). NaN is
/// an error — never silently committed.
///
/// This runs once per generated token in the decode hot loop, so it converts
/// in stack-sized chunks with no heap allocation.
pub fn hash_logits(logits: &[f32], frac_bits: u8, step: usize) -> Result<Hash32, Error> {
    let scale = (1u64 << frac_bits) as f64;
    let mut hasher = blake3::Hasher::new();
    hasher.update(LOGITS_DOMAIN);
    hasher.update(&[frac_bits]);
    hasher.update(&(logits.len() as u32).to_le_bytes());
    let mut buf = [0u8; 4 * 4096];
    for (chunk_index, chunk) in logits.chunks(4096).enumerate() {
        for (i, &x) in chunk.iter().enumerate() {
            if x.is_nan() {
                return Err(Error::NonFiniteLogit {
                    step,
                    index: chunk_index * 4096 + i,
                });
            }
            let q = (x as f64 * scale)
                .round()
                .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            buf[4 * i..4 * i + 4].copy_from_slice(&q.to_le_bytes());
        }
        hasher.update(&buf[..4 * chunk.len()]);
    }
    Ok(hasher.finalize().into())
}

/// Same digest as [`hash_logits`], from already-quantized values. Used by the
/// verifier to bind logits revealed in a challenge response to the chain.
pub fn hash_logits_q(quantized: &[i32], frac_bits: u8) -> Hash32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(LOGITS_DOMAIN);
    hasher.update(&[frac_bits]);
    hasher.update(&(quantized.len() as u32).to_le_bytes());
    let mut buf = [0u8; 4 * 4096];
    for chunk in quantized.chunks(4096) {
        for (i, &q) in chunk.iter().enumerate() {
            buf[4 * i..4 * i + 4].copy_from_slice(&q.to_le_bytes());
        }
        hasher.update(&buf[..4 * chunk.len()]);
    }
    hasher.finalize().into()
}

/// The running chain. Create with [`Chain::seed`], then absorb one step per
/// generated token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chain {
    state: Hash32,
    steps: u32,
}

impl Chain {
    pub fn seed(model_root: &Hash32, prompt_tokens: &[u32], sampler: &SamplerConfig) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(SEED_DOMAIN);
        h.update(&model_root.0);
        h.update(&(prompt_tokens.len() as u32).to_le_bytes());
        for &t in prompt_tokens {
            h.update(&t.to_le_bytes());
        }
        h.update(&sampler.canonical_bytes());
        Chain {
            state: h.finalize().into(),
            steps: 0,
        }
    }

    /// Absorb one decode step: the sampled token and the (already hashed)
    /// logit vector it was sampled from. Returns the new chain value.
    pub fn absorb_step(&mut self, token_id: u32, logits_hash: &Hash32) -> Hash32 {
        let mut h = blake3::Hasher::new();
        h.update(STEP_DOMAIN);
        h.update(&self.state.0);
        h.update(&self.steps.to_le_bytes());
        h.update(&token_id.to_le_bytes());
        h.update(&logits_hash.0);
        self.state = h.finalize().into();
        self.steps += 1;
        self.state
    }

    /// Optionally fold per-layer activation hashes for the step that was just
    /// absorbed (used by --trace-layers). Order of `layer_hashes` is layer 0
    /// upward; the count is included so truncation is detectable.
    pub fn absorb_layer_hashes(&mut self, layer_hashes: &[Hash32]) -> Hash32 {
        let mut h = blake3::Hasher::new();
        h.update(LAYERS_DOMAIN);
        h.update(&self.state.0);
        h.update(&(layer_hashes.len() as u32).to_le_bytes());
        for lh in layer_hashes {
            h.update(&lh.0);
        }
        self.state = h.finalize().into();
        self.state
    }

    /// Fold a circuit-friendly (Rescue-Prime) logits commitment for the
    /// step that was just absorbed (used by --prove-decode).
    pub fn absorb_zk_digest(&mut self, digest: &Hash32) -> Hash32 {
        let mut h = blake3::Hasher::new();
        h.update(ZK_DOMAIN);
        h.update(&self.state.0);
        h.update(&digest.0);
        self.state = h.finalize().into();
        self.state
    }

    pub fn value(&self) -> Hash32 {
        self.state
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> Hash32 {
        Hash32(*blake3::hash(b"model").as_bytes())
    }

    #[test]
    fn seed_binds_every_input() {
        let base = Chain::seed(&root(), &[1, 2, 3], &SamplerConfig::greedy());
        let other_root = Hash32(*blake3::hash(b"other").as_bytes());
        assert_ne!(
            base,
            Chain::seed(&other_root, &[1, 2, 3], &SamplerConfig::greedy())
        );
        assert_ne!(
            base,
            Chain::seed(&root(), &[1, 2], &SamplerConfig::greedy())
        );
        assert_ne!(
            base,
            Chain::seed(&root(), &[1, 2, 4], &SamplerConfig::greedy())
        );
        let mut sampler = SamplerConfig::greedy();
        sampler.rng_seed = 7;
        assert_ne!(base, Chain::seed(&root(), &[1, 2, 3], &sampler));
        let topp = SamplerConfig {
            mode: SamplerMode::TopP,
            temperature: 0.7,
            top_p: 0.9,
            rng_seed: 42,
        };
        assert_ne!(base, Chain::seed(&root(), &[1, 2, 3], &topp));
    }

    #[test]
    fn prompt_length_prefix_prevents_boundary_shifts() {
        // [1, 2] ++ [] must differ from [1] ++ [2]-style ambiguity: the
        // length prefix makes token framing unambiguous.
        let a = Chain::seed(&root(), &[1, 2], &SamplerConfig::greedy());
        let b = Chain::seed(&root(), &[1], &SamplerConfig::greedy());
        assert_ne!(a, b);
    }

    #[test]
    fn steps_are_order_and_index_sensitive() {
        let lh1 = hash_logits(&[0.1, -0.2, 3.0], 16, 0).unwrap();
        let lh2 = hash_logits(&[1.0, 2.0, -3.0], 16, 1).unwrap();

        let mut a = Chain::seed(&root(), &[1], &SamplerConfig::greedy());
        a.absorb_step(5, &lh1);
        a.absorb_step(6, &lh2);

        let mut b = Chain::seed(&root(), &[1], &SamplerConfig::greedy());
        b.absorb_step(6, &lh2);
        b.absorb_step(5, &lh1);

        assert_ne!(a.value(), b.value());
        assert_eq!(a.steps(), 2);
    }

    #[test]
    fn logits_hash_respects_precision_and_dim() {
        let l = [0.5f32, -1.25, 2.0];
        assert_eq!(
            hash_logits(&l, 16, 0).unwrap(),
            hash_logits(&l, 16, 9).unwrap()
        );
        assert_ne!(
            hash_logits(&l, 16, 0).unwrap(),
            hash_logits(&l, 8, 0).unwrap()
        );
        assert_ne!(
            hash_logits(&[0.5, -1.25], 16, 0).unwrap(),
            hash_logits(&[0.5, -1.25, 0.0], 16, 0).unwrap()
        );
        // Sub-precision perturbations quantize away…
        assert_eq!(
            hash_logits(&[0.5], 8, 0).unwrap(),
            hash_logits(&[0.5 + 1e-4], 8, 0).unwrap()
        );
        // …supra-precision perturbations do not.
        assert_ne!(
            hash_logits(&[0.5], 8, 0).unwrap(),
            hash_logits(&[0.5 + 0.01], 8, 0).unwrap()
        );
    }

    #[test]
    fn nan_logits_are_rejected() {
        assert!(matches!(
            hash_logits(&[0.0, f32::NAN], 16, 3),
            Err(Error::NonFiniteLogit { step: 3, index: 1 })
        ));
        // Infinities saturate rather than erroring.
        assert!(hash_logits(&[f32::INFINITY, f32::NEG_INFINITY], 16, 0).is_ok());
    }

    #[test]
    fn layer_folding_changes_state_and_binds_count() {
        let lh = hash_logits(&[1.0], 16, 0).unwrap();
        let mut a = Chain::seed(&root(), &[1], &SamplerConfig::greedy());
        a.absorb_step(5, &lh);
        let mut b = a;
        let layer = Hash32(*blake3::hash(b"layer0").as_bytes());
        b.absorb_layer_hashes(&[layer]);
        assert_ne!(a.value(), b.value());
        let mut c = a;
        c.absorb_layer_hashes(&[]);
        assert_ne!(b.value(), c.value());
    }

    #[test]
    fn hash_logits_q_matches_float_path() {
        let logits = [0.5f32, -1.25, 2.0, 7.125, -0.0004];
        let q: Vec<i32> = logits
            .iter()
            .map(|&x| (x as f64 * 65536.0).round() as i32)
            .collect();
        assert_eq!(hash_logits(&logits, 16, 0).unwrap(), hash_logits_q(&q, 16));
    }

    /// Golden vector pinning the whole chain construction.
    #[test]
    fn golden_chain_is_stable() {
        let sampler = SamplerConfig {
            mode: SamplerMode::TopP,
            temperature: 0.7,
            top_p: 0.95,
            rng_seed: 42,
        };
        let mut chain = Chain::seed(&root(), &[128000, 9906], &sampler);
        let lh = hash_logits(&[-1.5, 0.0, 2.25, 7.125], 16, 0).unwrap();
        chain.absorb_step(2, &lh);
        assert_eq!(
            chain.value().to_hex(),
            golden_expected(),
            "chain construction changed; this breaks existing transcripts"
        );
    }

    fn golden_expected() -> String {
        // Computed once with blake3 1.8; must never change for v1 domains.
        "8628a76950170f8550bf6ffed19463670870ff6435e1f2f384e2c9001712bf42".into()
    }
}
