//! Native side of the salted Rescue-Prime sponge. Uses winterfell's
//! Rp64_256 permutation (whose MDS/ARK constants are public), with our own
//! sponge layout chosen to keep the AIR simple:
//!
//!   state[0..8]  = rate    (logit blocks are added here)
//!   state[8..12] = capacity, initialized with the PRIVATE salt
//!
//! Absorption cadence exactly mirrors the AIR's row layout: the state is
//! permuted first (7 rounds = rows of a cycle), then the next block is
//! added (the cycle's absorb row). After the last block, one final
//! permutation; digest = state[0..4].
//!
//! digest = P(... P(P([0;8] ‖ salt) + b₀) + b₁ ...) + b_{n-1}) [0..4]

use winterfell::crypto::hashers::Rp64_256;
use winterfell::math::fields::f64::BaseElement as Felt;
use winterfell::math::{FieldElement, StarkField};

use crate::{RATE, ZkError, felt_of};

pub const STATE_WIDTH: usize = 12;
pub const NUM_ROUNDS: usize = 7;

pub fn apply_round(state: &mut [Felt; STATE_WIDTH], round: usize) {
    Rp64_256::apply_round(state, round);
}

fn apply_permutation(state: &mut [Felt; STATE_WIDTH]) {
    for round in 0..NUM_ROUNDS {
        apply_round(state, round);
    }
}

/// Initial sponge state for a given salt.
pub fn initial_state(salt: [u64; 4]) -> Result<[Felt; STATE_WIDTH], ZkError> {
    let mut state = [Felt::ZERO; STATE_WIDTH];
    for (i, &s) in salt.iter().enumerate() {
        if s >= Felt::MODULUS {
            return Err(ZkError::BadInput(format!("salt element {i} out of field")));
        }
        state[RATE + i] = Felt::new(s);
    }
    Ok(state)
}

/// The salted digest of a quantized logit vector (length must be a multiple
/// of the rate; true for llama vocab sizes — otherwise pad upstream).
pub fn salted_digest(quantized: &[i32], salt: [u64; 4]) -> Result<[Felt; 4], ZkError> {
    if quantized.is_empty() || !quantized.len().is_multiple_of(RATE) {
        return Err(ZkError::BadInput(format!(
            "logit vector length {} is not a positive multiple of {RATE}",
            quantized.len()
        )));
    }
    let mut state = initial_state(salt)?;
    for block in quantized.chunks_exact(RATE) {
        apply_permutation(&mut state);
        for (lane, &q) in block.iter().enumerate() {
            state[lane] += felt_of(q);
        }
    }
    apply_permutation(&mut state);
    Ok([state[0], state[1], state[2], state[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_deterministic_and_sensitive() {
        let q: Vec<i32> = (0..32).map(|i| i * 100 - 1600).collect();
        let a = salted_digest(&q, [1, 2, 3, 4]).unwrap();
        assert_eq!(a, salted_digest(&q, [1, 2, 3, 4]).unwrap());
        let mut q2 = q.clone();
        q2[31] += 1;
        assert_ne!(a, salted_digest(&q2, [1, 2, 3, 4]).unwrap());
        assert_ne!(a, salted_digest(&q, [1, 2, 3, 5]).unwrap());
        assert!(salted_digest(&q[..7], [0; 4]).is_err());
        assert!(salted_digest(&q, [u64::MAX, 0, 0, 0]).is_err());
    }
}
