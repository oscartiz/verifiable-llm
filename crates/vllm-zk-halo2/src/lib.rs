//! Layer 3, formally zero-knowledge variant: a halo2 proof (transparent inner
//! product argument over the Pasta curves — no trusted setup) that an emitted
//! token is the argmax of a committed logit vector, *without revealing the
//! logits* and with formal zero-knowledge.
//!
//! This is the sibling of `vllm-zk` (the winterfell STARK). Both prove the
//! same statement — "the committed token is a maximum of the committed logit
//! vector" — but where winterfell 0.13 has no trace randomization (so its
//! openings leak random linear projections of the witness; see DECISIONS.md
//! #13), halo2's IPA prover blinds its committed witness polynomials, so the
//! proof is *formally* zero-knowledge. The trade is proving cost: the
//! commitment is an in-circuit Poseidon hash chain over the whole vocab, which
//! is the millions of rows anticipated in DECISIONS.md #13.
//!
//! ## What is proved
//!
//! For a public token index `c` and a public digest `d`, the prover knows a
//! logit vector `x` (length `vocab`) and a salt `s` such that
//!
//!   * `d = PoseidonChain_s(x)` — the salted hash chain of `commit.rs`, and
//!   * `x[c] ≥ x[i]` for every `i` — argmax.
//!
//! Sampler modes other than greedy are out of scope (documented roadmap),
//! exactly as for the STARK.

mod circuit;
mod commit;

use group::ff::PrimeField;
use halo2_proofs::{
    pasta::{EqAffine, Fp},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, SingleVerifier},
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use rand::rngs::OsRng;

use circuit::ArgmaxCircuit;
use commit::{chain_digest, felt_of, salt_fp};

pub use commit::commit_logits;

/// Bit width of the range check on `x[c] − x[i]`. Matches the STARK
/// (`vllm-zk`): logit spreads above `2²⁷ / 2¹⁶ = 2048` logit units are
/// unprovable, far above real spreads (~40 units).
pub const DIFF_BITS: usize = 27;

#[derive(Debug)]
pub enum ZkError {
    BadInput(String),
    Prover(String),
    Verifier(String),
}

impl std::fmt::Display for ZkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZkError::BadInput(m) => write!(f, "invalid input: {m}"),
            ZkError::Prover(m) => write!(f, "proving failed: {m}"),
            ZkError::Verifier(m) => write!(f, "proof rejected: {m}"),
        }
    }
}

impl std::error::Error for ZkError {}

/// Draw a fresh commitment salt (a uniform field element) as 32 little-endian
/// bytes.
pub fn random_salt() -> [u8; 32] {
    use group::ff::Field;
    Fp::random(rand::rngs::OsRng).to_repr()
}

/// The token index `c` as the field element the circuit exposes for it.
pub(crate) fn token_felt(token: u32) -> Fp {
    Fp::from(token as u64)
}

/// Parse a 32-byte little-endian digest as a canonical field element.
pub(crate) fn digest_fp(digest: &[u8; 32]) -> Result<Fp, ZkError> {
    Option::from(Fp::from_repr(*digest))
        .ok_or_else(|| ZkError::BadInput("digest is not a canonical field element".into()))
}

/// Circuit size (log₂ rows) for a given vocab. The dominant cost is the
/// Poseidon hash chain (~40 rows per logit); the estimate carries generous
/// headroom for the argmax region and IPA blinding rows.
pub(crate) fn circuit_k(vocab: usize) -> u32 {
    let rows = 64u64 * vocab as u64 + 256;
    let mut k = 9u32;
    while (1u64 << k) < rows {
        k += 1;
    }
    k
}

fn digest_felt(quantized: &[i32], salt: Fp) -> Fp {
    let logits: Vec<Fp> = quantized.iter().map(|&q| felt_of(q)).collect();
    chain_digest(salt, &logits)
}

/// Prove that `token` is an argmax of the committed `quantized` logits, in
/// formal zero-knowledge. `salt` is the same 32-byte value passed to
/// [`commit_logits`]. Returns the serialized IPA proof.
pub fn prove_argmax(quantized: &[i32], salt: &[u8; 32], token: u32) -> Result<Vec<u8>, ZkError> {
    let vocab = quantized.len();
    if vocab == 0 {
        return Err(ZkError::BadInput("empty logit vector".into()));
    }
    if token as usize >= vocab {
        return Err(ZkError::BadInput(format!(
            "token {token} out of range for vocab {vocab}"
        )));
    }
    let max = *quantized.iter().max().expect("non-empty");
    if quantized[token as usize] != max {
        return Err(ZkError::BadInput(format!(
            "token {token} is not an argmax: logit {} < max {max}",
            quantized[token as usize]
        )));
    }
    let min = *quantized.iter().min().expect("non-empty");
    if (max as i64 - min as i64) >= (1i64 << DIFF_BITS) {
        return Err(ZkError::BadInput(format!(
            "logit spread {} exceeds the provable range 2^{DIFF_BITS}",
            max as i64 - min as i64
        )));
    }

    let salt = salt_fp(salt)?;
    let digest = digest_felt(quantized, salt);
    let instance: [Fp; 2] = [digest, token_felt(token)];

    let k = circuit_k(vocab);
    let params = Params::<EqAffine>::new(k);
    let vk = keygen_vk(&params, &ArgmaxCircuit::keygen(vocab))
        .map_err(|e| ZkError::Prover(format!("keygen_vk: {e:?}")))?;
    let pk = keygen_pk(&params, vk, &ArgmaxCircuit::keygen(vocab))
        .map_err(|e| ZkError::Prover(format!("keygen_pk: {e:?}")))?;

    let circuit = ArgmaxCircuit::prover(quantized, salt, token);
    let inst_col: &[Fp] = &instance;
    let per_circuit: &[&[Fp]] = &[inst_col];
    let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
    create_proof(
        &params,
        &pk,
        &[circuit],
        &[per_circuit],
        OsRng,
        &mut transcript,
    )
    .map_err(|e| ZkError::Prover(format!("create_proof: {e:?}")))?;
    Ok(transcript.finalize())
}

/// Verify an argmax proof against the committed `digest` (32 bytes from
/// [`commit_logits`]), the emitted `token`, and the `vocab` size. Needs no
/// model, trace, or GPU — and reconstructs the verifying key deterministically
/// from `(vocab)`.
pub fn verify_argmax(
    digest: &[u8; 32],
    token: u32,
    vocab: u32,
    proof: &[u8],
) -> Result<(), ZkError> {
    let vocab = vocab as usize;
    if vocab == 0 || token as usize >= vocab {
        return Err(ZkError::BadInput("token out of range for vocab".into()));
    }
    let digest = digest_fp(digest)?;
    let instance: [Fp; 2] = [digest, token_felt(token)];

    let k = circuit_k(vocab);
    let params = Params::<EqAffine>::new(k);
    let vk = keygen_vk(&params, &ArgmaxCircuit::keygen(vocab))
        .map_err(|e| ZkError::Verifier(format!("keygen_vk: {e:?}")))?;

    let inst_col: &[Fp] = &instance;
    let per_circuit: &[&[Fp]] = &[inst_col];
    let strategy = SingleVerifier::new(&params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);
    verify_proof(&params, &vk, strategy, &[per_circuit], &mut transcript)
        .map_err(|e| ZkError::Verifier(format!("{e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(n: usize, argmax_at: usize) -> Vec<i32> {
        let mut v: Vec<i32> = (0..n)
            .map(|i| ((i as f32 * 0.7).sin() * 3000.0) as i32)
            .collect();
        let max = *v.iter().max().unwrap();
        v[argmax_at] = max + 500;
        v
    }

    #[test]
    fn prove_and_verify_roundtrip() {
        let q = logits(32, 11);
        let salt = random_salt();
        let digest = commit_logits(&q, &salt).unwrap();
        let proof = prove_argmax(&q, &salt, 11).unwrap();
        verify_argmax(&digest, 11, 32, &proof).unwrap();
    }

    #[test]
    fn wrong_claims_are_rejected() {
        let q = logits(32, 11);
        let salt = random_salt();
        let digest = commit_logits(&q, &salt).unwrap();
        let proof = prove_argmax(&q, &salt, 11).unwrap();

        // Wrong token index.
        assert!(verify_argmax(&digest, 12, 32, &proof).is_err());
        // Wrong digest (different logits).
        let mut q2 = q.clone();
        q2[3] += 1;
        let other = commit_logits(&q2, &salt).unwrap();
        assert!(verify_argmax(&other, 11, 32, &proof).is_err());
        // Proving a non-argmax token is refused outright.
        assert!(prove_argmax(&q, &salt, 5).is_err());
    }

    #[test]
    fn salt_hides_digest() {
        let q = logits(32, 0);
        let a = commit_logits(&q, &random_salt()).unwrap();
        let b = commit_logits(&q, &random_salt()).unwrap();
        assert_ne!(
            a, b,
            "same logits, different salt must give different digests"
        );
    }
}
