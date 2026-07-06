//! Layer 3: a STARK proof that the emitted token is the argmax of a
//! committed logit vector, without revealing the logits.
//!
//! The commitment is a Rescue-Prime sponge (winterfell's Rp64_256
//! permutation over the 64-bit Goldilocks field) with the salt as a private
//! capacity IV: digest = Sponge_{cap=salt}(quantized logits). The AIR
//! recomputes the sponge from the witness logits *inside the proof* and
//! simultaneously enforces, for the public token index c:
//!
//!   - m − xᵢ ∈ [0, 2²⁷) for every i (27-bit decomposition), and
//!   - x_c = m,
//!
//! where m is a private "claimed maximum" column. Together: logits[c] is a
//! maximum of the committed vector. Sampler modes other than greedy are out
//! of scope (documented roadmap).
//!
//! ## Security statement (be precise about what this is!)
//!
//! This is a *succinct, transparent argument of knowledge* with a binding,
//! salted commitment. It is NOT formal zero-knowledge: winterfell 0.13 has
//! no trace randomization, so each proof reveals evaluations of the trace
//! polynomials at ~O(num queries) coset points — random linear projections
//! of the (salt, logits) vector. No individual logit is recoverable from
//! them (the system is massively underdetermined), and the salt prevents
//! digest-level guess-testing, but a party that already knows a candidate
//! for the *entire* logit vector could confirm it. See DECISIONS.md #13.

mod air;
mod prover;
mod rescue;

use winterfell::math::{FieldElement, StarkField, fields::f64::BaseElement as Felt};
use winterfell::{
    AcceptableOptions, BatchingMethod, FieldExtension, Proof, ProofOptions, VerifierError,
    crypto::{DefaultRandomCoin, MerkleTree, hashers::Blake3_256},
};

pub use air::{LogitsArgmaxAir, PublicInputs};
pub use prover::LogitsArgmaxProver;
pub use rescue::salted_digest;

/// Fixed-point logits per sponge block (= sponge rate).
pub const RATE: usize = 8;
/// Rows per block: 7 Rescue rounds + 1 absorption row.
pub const CYCLE: usize = 8;
/// Bit width of the range check on m - x_i. Logit spreads above
/// 2^27 / 2^16 = 2048 logit units are unprovable (never happens in practice).
pub const DIFF_BITS: usize = 27;

type Hasher = Blake3_256<Felt>;
type RandCoin = DefaultRandomCoin<Hasher>;
type VC = MerkleTree<Hasher>;

/// ~100-bit conjectured security: blowup 8 (required by the degree-7 Rescue
/// constraints), 27 queries, 16 bits of grinding, quadratic extension field.
pub fn proof_options() -> ProofOptions {
    ProofOptions::new(
        27,
        8,
        16,
        FieldExtension::Quadratic,
        8,
        127,
        BatchingMethod::Linear,
        BatchingMethod::Linear,
    )
}

#[derive(Debug)]
pub enum ZkError {
    BadInput(String),
    Prover(String),
    Verifier(VerifierError),
}

impl std::fmt::Display for ZkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZkError::BadInput(m) => write!(f, "invalid input: {m}"),
            ZkError::Prover(m) => write!(f, "proving failed: {m}"),
            ZkError::Verifier(e) => write!(f, "proof rejected: {e}"),
        }
    }
}

impl std::error::Error for ZkError {}

/// Encode a fixed-point logit as a field element (signed embedding mod p).
pub(crate) fn felt_of(q: i32) -> Felt {
    felt_of_i64(q as i64)
}

pub(crate) fn felt_of_i64(q: i64) -> Felt {
    if q >= 0 {
        Felt::new(q as u64)
    } else {
        Felt::new(Felt::MODULUS - (-q) as u64)
    }
}

/// Draw a fresh commitment salt (4 field elements) from the OS RNG.
pub fn random_salt() -> [u64; 4] {
    let mut salt = [0u64; 4];
    for s in salt.iter_mut() {
        *s = rand::random::<u64>() % Felt::MODULUS;
    }
    salt
}

/// The salted commitment digest as 32 bytes (4 canonical u64, little-endian).
pub fn commit_logits(quantized: &[i32], salt: [u64; 4]) -> Result<[u8; 32], ZkError> {
    let digest = salted_digest(quantized, salt)?;
    let mut out = [0u8; 32];
    for (i, d) in digest.iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&d.as_int().to_le_bytes());
    }
    Ok(out)
}

/// Prove that `token` is an argmax of the committed `quantized` logits.
pub fn prove_argmax(quantized: &[i32], salt: [u64; 4], token: u32) -> Result<Vec<u8>, ZkError> {
    if token as usize >= quantized.len() {
        return Err(ZkError::BadInput(format!(
            "token {token} out of range for vocab {}",
            quantized.len()
        )));
    }
    let max = *quantized.iter().max().expect("non-empty");
    if quantized[token as usize] != max {
        return Err(ZkError::BadInput(format!(
            "token {token} is not an argmax: logit {} < max {max}",
            quantized[token as usize]
        )));
    }
    let prover = LogitsArgmaxProver::new(proof_options()).with_claim(token, quantized.len() as u32);
    let trace = prover.build_trace(quantized, salt, token)?;
    let proof =
        winterfell::Prover::prove(&prover, trace).map_err(|e| ZkError::Prover(e.to_string()))?;
    Ok(proof.to_bytes())
}

/// Verify an argmax proof against the committed digest (32 bytes as
/// produced by [`commit_logits`]), the emitted token, and the vocab size.
pub fn verify_argmax(
    digest: &[u8; 32],
    token: u32,
    vocab: u32,
    proof_bytes: &[u8],
) -> Result<(), ZkError> {
    let proof = Proof::from_bytes(proof_bytes)
        .map_err(|e| ZkError::BadInput(format!("malformed proof: {e}")))?;
    let mut digest_felts = [Felt::ZERO; 4];
    for (i, d) in digest_felts.iter_mut().enumerate() {
        let raw = u64::from_le_bytes(digest[8 * i..8 * i + 8].try_into().unwrap());
        if raw >= Felt::MODULUS {
            return Err(ZkError::BadInput("digest element out of field".into()));
        }
        *d = Felt::new(raw);
    }
    let pub_inputs = PublicInputs {
        digest: digest_felts,
        token,
        vocab,
    };
    let min_security = AcceptableOptions::OptionSet(vec![proof_options()]);
    winterfell::verify::<LogitsArgmaxAir, Hasher, RandCoin, VC>(proof, pub_inputs, &min_security)
        .map_err(ZkError::Verifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(n: usize, argmax_at: usize) -> Vec<i32> {
        let mut v: Vec<i32> = (0..n)
            .map(|i| ((i as f32 * 0.7).sin() * 30_000.0) as i32 - 40_000)
            .collect();
        let max = *v.iter().max().unwrap();
        v[argmax_at] = max + 5_000;
        v
    }

    #[test]
    fn prove_and_verify_roundtrip() {
        let q = logits(64, 17);
        let salt = [1, 2, 3, 4];
        let digest = commit_logits(&q, salt).unwrap();
        let proof = prove_argmax(&q, salt, 17).unwrap();
        verify_argmax(&digest, 17, 64, &proof).unwrap();
    }

    #[test]
    fn wrong_claims_are_rejected() {
        let q = logits(64, 17);
        let salt = [9, 9, 9, 9];
        let digest = commit_logits(&q, salt).unwrap();
        let proof = prove_argmax(&q, salt, 17).unwrap();

        // Wrong token index.
        assert!(verify_argmax(&digest, 18, 64, &proof).is_err());
        // Wrong digest (different salt).
        let other = commit_logits(&q, [1, 1, 1, 1]).unwrap();
        assert!(verify_argmax(&other, 17, 64, &proof).is_err());
        // Wrong digest (different logits).
        let mut q2 = q.clone();
        q2[3] += 1;
        let other = commit_logits(&q2, salt).unwrap();
        assert!(verify_argmax(&other, 17, 64, &proof).is_err());
        // Proving a non-argmax token is refused outright.
        assert!(prove_argmax(&q, salt, 5).is_err());
    }

    #[test]
    fn salt_hides_digest() {
        let q = logits(64, 0);
        let a = commit_logits(&q, [1, 2, 3, 4]).unwrap();
        let b = commit_logits(&q, [5, 6, 7, 8]).unwrap();
        assert_ne!(
            a, b,
            "same logits, different salt must give different digests"
        );
    }

    #[test]
    fn rejects_bad_shapes() {
        let q = logits(60, 0); // not a multiple of 8
        assert!(commit_logits(&q, [0; 4]).is_err());
        let q = logits(64, 0);
        assert!(prove_argmax(&q, [0; 4], 64).is_err()); // token out of range
    }
}
