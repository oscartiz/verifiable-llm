//! Trace construction and the winterfell Prover impl for the argmax AIR.

use winterfell::crypto::{DefaultRandomCoin, MerkleTree, hashers::Blake3_256};
use winterfell::math::{FieldElement, fields::f64::BaseElement as Felt};
use winterfell::matrix::ColMatrix;
use winterfell::{
    AuxRandElements, CompositionPoly, CompositionPolyTrace, DefaultConstraintCommitment,
    DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, ProofOptions, Prover,
    StarkDomain, TraceInfo, TracePolyTable, TraceTable,
};

use crate::air::{LogitsArgmaxAir, PublicInputs, TRACE_WIDTH, digest_row};
use crate::rescue::{STATE_WIDTH, apply_round, initial_state, salted_digest};
use crate::{CYCLE, DIFF_BITS, RATE, ZkError, felt_of_i64};

pub struct LogitsArgmaxProver {
    options: ProofOptions,
    token: u32,
    vocab: u32,
}

impl LogitsArgmaxProver {
    pub fn new(options: ProofOptions) -> Self {
        LogitsArgmaxProver {
            options,
            token: 0,
            vocab: 0,
        }
    }

    /// Simulate the row machine and lay out all 51 columns.
    pub fn build_trace(
        &self,
        quantized: &[i32],
        salt: [u64; 4],
        token: u32,
    ) -> Result<TraceTable<Felt>, ZkError> {
        let n = quantized.len();
        if n == 0 || !n.is_multiple_of(RATE) {
            return Err(ZkError::BadInput(format!(
                "vocab {n} is not a positive multiple of {RATE}"
            )));
        }
        let m_int = *quantized.iter().max().expect("non-empty") as i64;
        let rows_used = CYCLE * (n / RATE + 1);
        // At least one padding row past the digest row: the second-to-last
        // row carries a full-range difference (2^27 - 1) so that every bit
        // column is a non-degenerate polynomial regardless of the input
        // (winterfell's debug mode asserts declared == actual degrees).
        let trace_len = (rows_used + 1).next_power_of_two().max(CYCLE * 2);

        // Integer logit per row: real logits, then m on padding rows,
        // except the degree-pinning row. Padding rows are only ever
        // absorbed after the digest row, so they do not affect the digest.
        let x_int = move |r: usize| -> i64 {
            if r < n {
                quantized[r] as i64
            } else if r == trace_len - 2 {
                m_int - ((1 << DIFF_BITS) - 1)
            } else {
                m_int
            }
        };
        for (i, &q) in quantized.iter().enumerate() {
            let diff = m_int - q as i64;
            if !(0..1 << DIFF_BITS).contains(&diff) {
                return Err(ZkError::BadInput(format!(
                    "logit spread too large at index {i}: max - logit = {diff} >= 2^{DIFF_BITS}"
                )));
            }
        }

        let m = felt_of_i64(m_int);
        let mut columns: Vec<Vec<Felt>> = (0..TRACE_WIDTH)
            .map(|_| Vec::with_capacity(trace_len))
            .collect();
        let mut state = initial_state(salt)?;
        let mut acc = [Felt::ZERO; 7];
        let mut acc_sel = Felt::ZERO;

        for r in 0..trace_len {
            let x = felt_of_i64(x_int(r));
            let sel = if r as u32 == token {
                Felt::ONE
            } else {
                Felt::ZERO
            };
            let diff = (m_int - x_int(r)) as u64;

            for i in 0..STATE_WIDTH {
                columns[i].push(state[i]);
            }
            for j in 0..7 {
                columns[STATE_WIDTH + j].push(acc[j]);
            }
            columns[STATE_WIDTH + 7].push(x); // x
            columns[STATE_WIDTH + 8].push(m); // m
            for k in 0..DIFF_BITS {
                columns[STATE_WIDTH + 9 + k].push(Felt::new(diff >> k & 1));
            }
            columns[STATE_WIDTH + 9 + DIFF_BITS].push(sel);
            columns[STATE_WIDTH + 10 + DIFF_BITS].push(Felt::new(r as u64));
            columns[STATE_WIDTH + 11 + DIFF_BITS].push(acc_sel);

            // Advance the machine to the next row.
            let phase = r % CYCLE;
            if phase < CYCLE - 1 {
                apply_round(&mut state, phase);
                acc[phase] += x;
            } else {
                for (lane, a) in acc.iter().enumerate() {
                    state[lane] += *a;
                }
                state[7] += x;
                acc = [Felt::ZERO; 7];
            }
            acc_sel += sel;
        }

        // Cross-check the in-trace sponge against the native digest.
        let digest = salted_digest(quantized, salt)?;
        let d = digest_row(n as u32);
        for i in 0..4 {
            debug_assert_eq!(columns[i][d], digest[i], "trace/native digest divergence");
        }

        Ok(TraceTable::init(columns))
    }

    /// Bind the (token, vocab) public inputs for the next proof.
    pub fn with_claim(mut self, token: u32, vocab: u32) -> Self {
        self.token = token;
        self.vocab = vocab;
        self
    }
}

impl Prover for LogitsArgmaxProver {
    type BaseField = Felt;
    type Air = LogitsArgmaxAir;
    type Trace = TraceTable<Felt>;
    type HashFn = Blake3_256<Felt>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Felt>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintCommitment<E: FieldElement<BaseField = Felt>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Felt>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> PublicInputs {
        let d = digest_row(self.vocab);
        PublicInputs {
            digest: [
                trace.get(0, d),
                trace.get(1, d),
                trace.get(2, d),
                trace.get(3, d),
            ],
            token: self.token,
            vocab: self.vocab,
        }
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Felt>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
    }

    fn build_constraint_commitment<E: FieldElement<BaseField = Felt>>(
        &self,
        composition_poly_trace: CompositionPolyTrace<E>,
        num_constraint_composition_columns: usize,
        domain: &StarkDomain<Felt>,
        partition_options: PartitionOptions,
    ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
        DefaultConstraintCommitment::new(
            composition_poly_trace,
            num_constraint_composition_columns,
            domain,
            partition_options,
        )
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Felt>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }
}
