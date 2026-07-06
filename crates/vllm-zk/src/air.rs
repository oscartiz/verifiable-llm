//! AIR for the salted-sponge argmax statement. One Rescue round per row;
//! every 8th row absorbs a block of 8 logits; every row carries one logit
//! (column x) plus the range-checked difference m − x.
//!
//! Column layout (51 main trace columns):
//!   0..12   sponge state (capacity 8..12 starts as the private salt)
//!   12..19  acc_0..acc_6 — lane accumulators for the next absorbed block
//!           (lane 7 is taken from x directly on the absorb row)
//!   19      x   — the logit assigned to this row (pad rows: x = m)
//!   20      m   — claimed maximum, constant across the trace
//!   21..48  b_0..b_26 — bit decomposition of m − x
//!   48      sel — 1 exactly at row `token`, else 0
//!   49      idx — row counter
//!   50      acc_sel — running sum of sel

use winterfell::crypto::hashers::Rp64_256;
use winterfell::math::{FieldElement, ToElements, fields::f64::BaseElement as Felt};
use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};

use crate::rescue::{NUM_ROUNDS, STATE_WIDTH};
use crate::{CYCLE, DIFF_BITS, RATE};

pub const TRACE_WIDTH: usize = STATE_WIDTH + 7 + 1 + 1 + DIFF_BITS + 3; // 51

// Column indices.
const STATE: usize = 0;
const ACC: usize = STATE + STATE_WIDTH; // 12..19
const X: usize = ACC + 7; // 19
const M: usize = X + 1; // 20
const BITS: usize = M + 1; // 21..48
const SEL: usize = BITS + DIFF_BITS; // 48
const IDX: usize = SEL + 1; // 49
const ACC_SEL: usize = IDX + 1; // 50

// Periodic column indices (period 8).
const P_ARK1: usize = 0; // 12 columns
const P_ARK2: usize = P_ARK1 + STATE_WIDTH; // 12 columns
const P_ROUND: usize = P_ARK2 + STATE_WIDTH; // round flag
const P_ABSORB: usize = P_ROUND + 1; // absorb flag
const P_PICK: usize = P_ABSORB + 1; // 7 lane-pick flags
pub const NUM_PERIODIC: usize = P_PICK + 7; // 33

/// Row holding the digest for a vocab of n logits (n % 8 == 0):
/// after n/8 absorb cycles plus one final permutation cycle.
pub fn digest_row(vocab: u32) -> usize {
    CYCLE * (vocab as usize / RATE + 1) - 1
}

#[derive(Debug, Clone)]
pub struct PublicInputs {
    pub digest: [Felt; 4],
    pub token: u32,
    pub vocab: u32,
}

impl ToElements<Felt> for PublicInputs {
    fn to_elements(&self) -> Vec<Felt> {
        let mut out = self.digest.to_vec();
        out.push(Felt::new(self.token as u64));
        out.push(Felt::new(self.vocab as u64));
        out
    }
}

pub struct LogitsArgmaxAir {
    context: AirContext<Felt>,
    pub_inputs: PublicInputs,
}

impl Air for LogitsArgmaxAir {
    type BaseField = Felt;
    type PublicInputs = PublicInputs;

    fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {
        assert_eq!(TRACE_WIDTH, trace_info.width());
        assert!(pub_inputs.vocab > 0 && (pub_inputs.vocab as usize).is_multiple_of(RATE));
        assert!((pub_inputs.token) < pub_inputs.vocab);
        assert!(digest_row(pub_inputs.vocab) < trace_info.length());

        let mut degrees = Vec::new();
        // 12 state constraints: Rescue round (degree 7, one period-8 flag)
        // merged with the absorb transition.
        for _ in 0..STATE_WIDTH {
            degrees.push(TransitionConstraintDegree::with_cycles(7, vec![CYCLE]));
        }
        // 7 accumulator constraints: degree 1 in trace, two periodic flags.
        for _ in 0..7 {
            degrees.push(TransitionConstraintDegree::with_cycles(
                1,
                vec![CYCLE, CYCLE],
            ));
        }
        degrees.push(TransitionConstraintDegree::new(1)); // m constant
        degrees.push(TransitionConstraintDegree::new(1)); // idx increment
        degrees.push(TransitionConstraintDegree::new(1)); // acc_sel accumulation
        degrees.push(TransitionConstraintDegree::new(2)); // sel boolean
        degrees.push(TransitionConstraintDegree::new(2)); // sel picks row `token`
        degrees.push(TransitionConstraintDegree::new(2)); // sel forces x = m
        for _ in 0..DIFF_BITS {
            degrees.push(TransitionConstraintDegree::new(2)); // bit boolean
        }
        degrees.push(TransitionConstraintDegree::new(1)); // bit recomposition

        let num_assertions = 8 + 7 + 2 + 4 + 1;
        LogitsArgmaxAir {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            pub_inputs,
        }
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        periodic_values: &[E],
        result: &mut [E],
    ) {
        let cur = frame.current();
        let next = frame.next();
        let round_flag = periodic_values[P_ROUND];
        let absorb_flag = periodic_values[P_ABSORB];

        let pow7 = |v: E| {
            let v2 = v * v;
            let v4 = v2 * v2;
            v4 * v2 * v
        };

        // --- state: Rescue round on round rows, block addition on absorb
        // rows. Round: MDS(s^7) + ark1 == (MDS^{-1}(s' - ark2))^7.
        let mut fwd = [E::ZERO; STATE_WIDTH];
        let mut bwd = [E::ZERO; STATE_WIDTH];
        for i in 0..STATE_WIDTH {
            let si7 = pow7(cur[STATE + i]);
            let ti = next[STATE + i] - periodic_values[P_ARK2 + i];
            for j in 0..STATE_WIDTH {
                fwd[j] += E::from(Rp64_256::MDS[j][i]) * si7;
                bwd[j] += E::from(Rp64_256::INV_MDS[j][i]) * ti;
            }
        }
        for i in 0..STATE_WIDTH {
            let round_expr = fwd[i] + periodic_values[P_ARK1 + i] - pow7(bwd[i]);
            let absorbed = if i < 7 {
                cur[STATE + i] + cur[ACC + i]
            } else if i == 7 {
                cur[STATE + i] + cur[X]
            } else {
                cur[STATE + i]
            };
            let absorb_expr = next[STATE + i] - absorbed;
            result[i] = round_flag * round_expr + absorb_flag * absorb_expr;
        }

        // --- lane accumulators: pick up x on this lane's phase, reset on
        // absorb rows.
        let one = E::ONE;
        for j in 0..7 {
            let picked = cur[ACC + j] + periodic_values[P_PICK + j] * cur[X];
            result[STATE_WIDTH + j] = next[ACC + j] - (one - absorb_flag) * picked;
        }

        let r = STATE_WIDTH + 7;
        // m is constant.
        result[r] = next[M] - cur[M];
        // idx increments by one.
        result[r + 1] = next[IDX] - cur[IDX] - one;
        // acc_sel accumulates sel.
        result[r + 2] = next[ACC_SEL] - cur[ACC_SEL] - cur[SEL];
        // sel is boolean.
        result[r + 3] = cur[SEL] * cur[SEL] - cur[SEL];
        // sel may be 1 only at row `token`.
        result[r + 4] = cur[SEL] * (cur[IDX] - E::from(Felt::new(self.pub_inputs.token as u64)));
        // where sel is 1, x equals the claimed maximum.
        result[r + 5] = cur[SEL] * (cur[X] - cur[M]);
        // bits are boolean and recompose to m - x.
        let mut recomposed = E::ZERO;
        let mut pow = E::ONE;
        for k in 0..DIFF_BITS {
            let b = cur[BITS + k];
            result[r + 6 + k] = b * b - b;
            recomposed += b * pow;
            pow += pow; // pow *= 2
        }
        result[r + 6 + DIFF_BITS] = recomposed - (cur[M] - cur[X]);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let mut assertions = Vec::new();
        // Rate part of the state starts at zero; the capacity (8..12) is the
        // UNCONSTRAINED private salt.
        for i in 0..RATE {
            assertions.push(Assertion::single(STATE + i, 0, Felt::ZERO));
        }
        for j in 0..7 {
            assertions.push(Assertion::single(ACC + j, 0, Felt::ZERO));
        }
        assertions.push(Assertion::single(IDX, 0, Felt::ZERO));
        assertions.push(Assertion::single(ACC_SEL, 0, Felt::ZERO));
        let d = digest_row(self.pub_inputs.vocab);
        for i in 0..4 {
            assertions.push(Assertion::single(STATE + i, d, self.pub_inputs.digest[i]));
        }
        let last = self.trace_length() - 1;
        assertions.push(Assertion::single(ACC_SEL, last, Felt::ONE));
        assertions
    }

    fn get_periodic_column_values(&self) -> Vec<Vec<Self::BaseField>> {
        let mut columns = vec![vec![Felt::ZERO; CYCLE]; NUM_PERIODIC];
        #[allow(clippy::needless_range_loop)]
        for round in 0..NUM_ROUNDS {
            for i in 0..STATE_WIDTH {
                columns[P_ARK1 + i][round] = Rp64_256::ARK1[round][i];
                columns[P_ARK2 + i][round] = Rp64_256::ARK2[round][i];
            }
            columns[P_ROUND][round] = Felt::ONE;
        }
        columns[P_ABSORB][CYCLE - 1] = Felt::ONE;
        for j in 0..7 {
            columns[P_PICK + j][j] = Felt::ONE;
        }
        columns
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}
