//! Native side of the Poseidon commitment. The digest is a salted Poseidon
//! *hash chain* over the fixed-point logits:
//!
//!   acc₀ = salt (a private, uniformly random field element)
//!   accᵢ₊₁ = Poseidon2(accᵢ, xᵢ)          for each logit xᵢ
//!   digest = acc_V
//!
//! where `Poseidon2(a, b)` is the width-3 rate-2 Poseidon hash of the two
//! elements `[a, b]` under the `ConstantLength<2>` domain (the same primitive
//! the halo2 gadget realizes in-circuit — see `circuit.rs`). Chaining a
//! 2-to-1 compression is binding by collision resistance; seeding with a
//! secret `salt` makes the digest hiding.
//!
//! Using `halo2_gadgets::poseidon::primitives` here (rather than a private
//! re-implementation) guarantees the native digest equals what the circuit
//! recomputes: the gadget crate's own tests pin `Pow5Chip` output to this
//! exact `Hash` primitive.

use group::ff::PrimeField;
use halo2_gadgets::poseidon::primitives::{ConstantLength, Hash, P128Pow5T3};
use halo2_proofs::pasta::Fp;

use crate::ZkError;

/// Encode a fixed-point (i32) logit as a field element via signed embedding:
/// a negative `q` maps to `p − |q|`. Differences of embedded logits computed
/// in the field then equal the true integer differences whenever they stay
/// well below the field modulus, which the range check relies on.
pub(crate) fn felt_of(q: i32) -> Fp {
    if q >= 0 {
        Fp::from(q as u64)
    } else {
        -Fp::from((-(q as i64)) as u64)
    }
}

/// Parse a 32-byte little-endian salt as a canonical field element.
pub(crate) fn salt_fp(salt: &[u8; 32]) -> Result<Fp, ZkError> {
    Option::from(Fp::from_repr(*salt))
        .ok_or_else(|| ZkError::BadInput("salt is not a canonical field element".into()))
}

/// One width-3 rate-2 Poseidon compression of two field elements.
pub(crate) fn poseidon2(a: Fp, b: Fp) -> Fp {
    Hash::<Fp, P128Pow5T3, ConstantLength<2>, 3, 2>::init().hash([a, b])
}

/// The salted Poseidon hash-chain digest of the field-encoded logits.
pub(crate) fn chain_digest(salt: Fp, logits: &[Fp]) -> Fp {
    let mut acc = salt;
    for &x in logits {
        acc = poseidon2(acc, x);
    }
    acc
}

/// The commitment digest of a quantized logit vector as 32 little-endian
/// bytes (a canonical field element).
pub fn commit_logits(quantized: &[i32], salt: &[u8; 32]) -> Result<[u8; 32], ZkError> {
    if quantized.is_empty() {
        return Err(ZkError::BadInput("empty logit vector".into()));
    }
    let salt = salt_fp(salt)?;
    let logits: Vec<Fp> = quantized.iter().map(|&q| felt_of(q)).collect();
    Ok(chain_digest(salt, &logits).to_repr())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Vec<i32> {
        (0..32).map(|i| i * 100 - 1600).collect()
    }

    #[test]
    fn digest_is_deterministic_and_sensitive() {
        let q = logits();
        let salt = [7u8; 32];
        let a = commit_logits(&q, &salt).unwrap();
        assert_eq!(a, commit_logits(&q, &salt).unwrap(), "deterministic");

        let mut q2 = q.clone();
        q2[31] += 1;
        assert_ne!(a, commit_logits(&q2, &salt).unwrap(), "sensitive to logits");

        let mut salt2 = salt;
        salt2[0] ^= 1;
        assert_ne!(a, commit_logits(&q, &salt2).unwrap(), "sensitive to salt");
    }

    #[test]
    fn felt_encoding_is_signed() {
        use group::ff::Field;
        assert_eq!(felt_of(0), Fp::ZERO);
        assert_eq!(felt_of(5) - felt_of(3), Fp::from(2));
        // negative embedding: felt_of(-3) + 3 == 0
        assert_eq!(felt_of(-3) + Fp::from(3), Fp::ZERO);
        // a difference that stays small is a genuine small field element
        assert_eq!(felt_of(10) - felt_of(-10), Fp::from(20));
    }
}
