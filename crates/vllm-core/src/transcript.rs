//! Generation transcript: everything a verifier needs to check the hash
//! chain and (in v0.2) issue challenges. Serialized as JSON next to the
//! generated text.

use serde::{Deserialize, Serialize};

use crate::Hash32;
use crate::chain::{Chain, SamplerConfig};

pub const TRANSCRIPT_VERSION: &str = "vllm/transcript/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub token_id: u32,
    pub logits_hash: Hash32,
    /// Chain value after absorbing this step (and its layer hashes, if any).
    pub chain: Hash32,
    /// Per-layer activation hashes folded into the chain at this step
    /// (--trace-layers, sampled every N steps). Absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_hashes: Option<Vec<Hash32>>,
    /// Rescue-Prime commitment to this step's quantized logits, folded into
    /// the chain (--prove-decode). The Layer-3 proof's public input must
    /// match this value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zk_digest: Option<Hash32>,
}

/// Environment facts that affect reproducibility but are not commitments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvInfo {
    /// "metal" or "cpu".
    pub backend: String,
    pub crate_version: String,
}

/// Commitment to the activation trace (present when generated with --trace).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceInfo {
    /// Merkle root over all trace cell hashes in index order.
    pub root: Hash32,
    pub n_positions: u32,
    /// Number of transformer blocks L (L+1 cells per position).
    pub n_layers: u32,
    pub hidden_dim: u32,
    /// Fixed-point precision of the trace cells.
    pub frac_bits: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub version: String,
    pub model_root: Hash32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_file: Option<String>,
    pub prompt_token_ids: Vec<u32>,
    pub sampler: SamplerConfig,
    /// Fixed-point precision used for logit hashing: q = round(x * 2^bits).
    pub logit_frac_bits: u8,
    pub vocab_size: u32,
    /// Chain value right after seeding, before any step.
    pub chain_seed: Hash32,
    /// With --trace: cell hashes of prompt positions 0..prompt_len-2, folded
    /// into the chain right after seeding (position-major, layer-minor).
    /// Cells of position prompt_len-1+s live in steps[s].layer_hashes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_layer_hashes: Option<Vec<Hash32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceInfo>,
    pub steps: Vec<StepRecord>,
    pub final_chain: Hash32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub env: EnvInfo,
}

/// Internal-consistency failures a verifier can detect without the model.
#[derive(Debug, PartialEq, Eq)]
pub enum ChainCheck {
    Ok,
    BadSeed { expected: Hash32 },
    BadStep { index: usize, expected: Hash32 },
    BadFinal { expected: Hash32 },
}

impl Transcript {
    /// Recompute the chain from the recorded inputs and compare it with the
    /// recorded per-step values. This proves the transcript is internally
    /// consistent — it does NOT prove the logits came from the model; that is
    /// what Layers 2 and 3 add.
    pub fn replay_chain(&self) -> ChainCheck {
        let mut chain = Chain::seed(&self.model_root, &self.prompt_token_ids, &self.sampler);
        if chain.value() != self.chain_seed {
            return ChainCheck::BadSeed {
                expected: chain.value(),
            };
        }
        if let Some(prompt_hashes) = &self.prompt_layer_hashes {
            chain.absorb_layer_hashes(prompt_hashes);
        }
        for (index, step) in self.steps.iter().enumerate() {
            chain.absorb_step(step.token_id, &step.logits_hash);
            if let Some(layer_hashes) = &step.layer_hashes {
                chain.absorb_layer_hashes(layer_hashes);
            }
            if let Some(zk_digest) = &step.zk_digest {
                chain.absorb_zk_digest(zk_digest);
            }
            if chain.value() != step.chain {
                return ChainCheck::BadStep {
                    index,
                    expected: chain.value(),
                };
            }
        }
        if chain.value() != self.final_chain {
            return ChainCheck::BadFinal {
                expected: chain.value(),
            };
        }
        ChainCheck::Ok
    }

    /// The generated token ids, in order.
    pub fn token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.steps.iter().map(|s| s.token_id)
    }

    /// Check that the trace commitment is consistent with the chain-folded
    /// cell hashes: the Merkle root over (prompt_layer_hashes ++ per-step
    /// layer_hashes) — which replay_chain has bound into final_chain — must
    /// equal trace.root. Returns an error message on any inconsistency.
    pub fn check_trace_binding(&self) -> Result<(), String> {
        let trace = self
            .trace
            .as_ref()
            .ok_or("transcript has no trace commitment")?;
        let cells_per_pos = trace.n_layers as usize + 1;
        let prompt_positions = self.prompt_token_ids.len() - 1;
        let expected_positions = prompt_positions + self.steps.len();
        if trace.n_positions as usize != expected_positions {
            return Err(format!(
                "trace has {} positions, transcript implies {expected_positions}",
                trace.n_positions
            ));
        }
        let prompt_hashes = self
            .prompt_layer_hashes
            .as_ref()
            .ok_or("traced transcript is missing prompt_layer_hashes")?;
        if prompt_hashes.len() != prompt_positions * cells_per_pos {
            return Err(format!(
                "prompt_layer_hashes has {} entries, expected {}",
                prompt_hashes.len(),
                prompt_positions * cells_per_pos
            ));
        }
        let mut leaves = prompt_hashes.clone();
        for (i, step) in self.steps.iter().enumerate() {
            let lh = step
                .layer_hashes
                .as_ref()
                .ok_or_else(|| format!("step {i} is missing layer_hashes"))?;
            if lh.len() != cells_per_pos {
                return Err(format!(
                    "step {i} has {} layer hashes, expected {cells_per_pos}",
                    lh.len()
                ));
            }
            leaves.extend_from_slice(lh);
        }
        match crate::merkle::root(&leaves) {
            Some(root) if root == trace.root => Ok(()),
            Some(root) => Err(format!(
                "trace root mismatch: chain-folded cells give {root}, transcript claims {}",
                trace.root
            )),
            None => Err("no trace cells".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::hash_logits;

    fn build(tamper: impl FnOnce(&mut Transcript)) -> Transcript {
        let model_root = Hash32(*blake3::hash(b"model").as_bytes());
        let sampler = SamplerConfig::greedy();
        let prompt = vec![10, 20];
        let mut chain = Chain::seed(&model_root, &prompt, &sampler);
        let chain_seed = chain.value();

        let mut steps = Vec::new();
        for (i, token) in [7u32, 8, 9].into_iter().enumerate() {
            let lh = hash_logits(&[i as f32, 1.0, -1.0], 16, i).unwrap();
            let value = chain.absorb_step(token, &lh);
            steps.push(StepRecord {
                token_id: token,
                logits_hash: lh,
                chain: value,
                layer_hashes: None,
                zk_digest: None,
            });
        }

        let mut t = Transcript {
            version: TRANSCRIPT_VERSION.into(),
            model_root,
            model_file: None,
            prompt_token_ids: prompt,
            sampler,
            logit_frac_bits: 16,
            vocab_size: 3,
            chain_seed,
            prompt_layer_hashes: None,
            trace: None,
            steps,
            final_chain: chain.value(),
            text: None,
            env: EnvInfo {
                backend: "cpu".into(),
                crate_version: "test".into(),
            },
        };
        tamper(&mut t);
        t
    }

    #[test]
    fn honest_transcript_replays_clean() {
        assert_eq!(build(|_| {}).replay_chain(), ChainCheck::Ok);
    }

    #[test]
    fn tampering_is_caught() {
        let swapped_token = build(|t| t.steps[1].token_id = 99);
        assert!(matches!(
            swapped_token.replay_chain(),
            ChainCheck::BadStep { index: 1, .. }
        ));

        let swapped_logits = build(|t| t.steps[2].logits_hash.0[0] ^= 1);
        assert!(matches!(
            swapped_logits.replay_chain(),
            ChainCheck::BadStep { index: 2, .. }
        ));

        let bad_prompt = build(|t| t.prompt_token_ids.push(1));
        assert!(matches!(
            bad_prompt.replay_chain(),
            ChainCheck::BadSeed { .. }
        ));

        let truncated = build(|t| {
            t.steps.pop();
        });
        assert!(matches!(
            truncated.replay_chain(),
            ChainCheck::BadFinal { .. }
        ));
    }

    #[test]
    fn json_roundtrip() {
        let t = build(|_| {});
        let json = serde_json::to_string_pretty(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(back.replay_chain(), ChainCheck::Ok);
        assert_eq!(back.final_chain, t.final_chain);
    }
}
