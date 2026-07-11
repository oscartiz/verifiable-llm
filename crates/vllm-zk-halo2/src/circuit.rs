//! The argmax circuit.
//!
//! One region (`argmax`) lays the logits out one per row and runs, in
//! lock-step down the rows, a running-selector construction that proves the
//! committed token is a maximum; a sequence of Poseidon-gadget hashes then
//! recomputes the commitment digest over the *same* logit cells. Formal
//! zero-knowledge comes for free from halo2's blinded IPA prover.
//!
//! ## The running-selector argmax
//!
//! Per logit row `i` we assign `x[i]` (the logit), a boolean `pick[i]`, its
//! bit-decomposed difference from a broadcast claimed-maximum `xc`, and four
//! running accumulators carried in adjacent rows:
//!
//! | column   | meaning                              | pinned to        |
//! |----------|--------------------------------------|------------------|
//! | `rowidx` | `0,1,2,…` (row counter)              | `rowidx[0]=0`    |
//! | `sumsel` | `Σ pick`                             | final `=1`       |
//! | `selval` | `Σ pick·x`                           | final `=xc`      |
//! | `selidx` | `Σ pick·rowidx`                      | final `=c` (pub) |
//!
//! `sumsel=1` + booleans force exactly one `pick`; `selidx=c` forces it to lie
//! at the public index `c`; `selval=xc` then makes `xc = x[c]`. The per-row
//! range check `xc − x[i] ∈ [0, 2^DIFF_BITS)` finally gives `x[c] ≥ x[i]` for
//! every `i`. `c` is read from the instance column via a copy constraint, so a
//! single verifying key serves every token index.

use group::ff::Field;
use halo2_gadgets::poseidon::{
    primitives::{ConstantLength, P128Pow5T3},
    Hash as PoseidonHash, Pow5Chip, Pow5Config,
};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    pasta::Fp,
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Constraints, Error, Expression, Instance,
        Selector,
    },
    poly::Rotation,
};

use crate::{commit::felt_of, DIFF_BITS};

const WIDTH: usize = 3;
const RATE: usize = 2;

#[derive(Clone, Debug)]
pub struct ArgmaxConfig {
    x: Column<Advice>,
    pick: Column<Advice>,
    rowidx: Column<Advice>,
    sumsel: Column<Advice>,
    selval: Column<Advice>,
    selidx: Column<Advice>,
    xc: Column<Advice>,
    bits: [Column<Advice>; DIFF_BITS],
    salt: Column<Advice>,
    s_arg: Selector,
    instance: Column<Instance>,
    poseidon: Pow5Config<Fp, WIDTH, RATE>,
}

/// Prover witness. Absent during key generation (`without_witnesses`).
#[derive(Clone)]
struct Witness {
    salt: Fp,
    logits: Vec<i32>,
    token: u32,
}

/// Concrete per-row values, computed once from a [`Witness`].
struct Prepared {
    x: Vec<Fp>,
    pick: Vec<Fp>,
    rowidx: Vec<Fp>,
    sumsel: Vec<Fp>,
    selval: Vec<Fp>,
    selidx: Vec<Fp>,
    xc: Fp,
    bits: Vec<[Fp; DIFF_BITS]>,
}

impl Prepared {
    fn from(w: &Witness) -> Self {
        let v = w.logits.len();
        let c = w.token as usize;
        let x: Vec<Fp> = w.logits.iter().map(|&q| felt_of(q)).collect();
        let mut pick = vec![Fp::ZERO; v];
        pick[c] = Fp::ONE;
        let rowidx: Vec<Fp> = (0..=v).map(|i| Fp::from(i as u64)).collect();
        let (mut sumsel, mut selval, mut selidx) = (
            vec![Fp::ZERO; v + 1],
            vec![Fp::ZERO; v + 1],
            vec![Fp::ZERO; v + 1],
        );
        for i in 0..v {
            sumsel[i + 1] = sumsel[i] + pick[i];
            selval[i + 1] = selval[i] + pick[i] * x[i];
            selidx[i + 1] = selidx[i] + pick[i] * rowidx[i];
        }
        let xc_int = w.logits[c] as i64;
        let bits: Vec<[Fp; DIFF_BITS]> = (0..v)
            .map(|i| {
                let diff = (xc_int - w.logits[i] as i64) as u64;
                core::array::from_fn(|k| Fp::from((diff >> k) & 1))
            })
            .collect();
        Prepared {
            x,
            pick,
            rowidx,
            sumsel,
            selval,
            selidx,
            xc: felt_of(w.logits[c]),
            bits,
        }
    }
}

/// The argmax circuit, parameterized by vocab size.
#[derive(Clone)]
pub struct ArgmaxCircuit {
    vocab: usize,
    witness: Option<Witness>,
}

impl ArgmaxCircuit {
    /// A witness-bearing circuit for proving.
    pub fn prover(quantized: &[i32], salt: Fp, token: u32) -> Self {
        ArgmaxCircuit {
            vocab: quantized.len(),
            witness: Some(Witness {
                salt,
                logits: quantized.to_vec(),
                token,
            }),
        }
    }

    /// The witness-free circuit of the same shape, for key generation.
    pub fn keygen(vocab: usize) -> Self {
        ArgmaxCircuit {
            vocab,
            witness: None,
        }
    }
}

fn to_val(o: Option<Fp>) -> Value<Fp> {
    o.map(Value::known).unwrap_or_else(Value::unknown)
}

impl Circuit<Fp> for ArgmaxCircuit {
    type Config = ArgmaxConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        ArgmaxCircuit {
            vocab: self.vocab,
            witness: None,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> ArgmaxConfig {
        let x = meta.advice_column();
        let pick = meta.advice_column();
        let rowidx = meta.advice_column();
        let sumsel = meta.advice_column();
        let selval = meta.advice_column();
        let selidx = meta.advice_column();
        let xc = meta.advice_column();
        let salt = meta.advice_column();
        let bits: [Column<Advice>; DIFF_BITS] = core::array::from_fn(|_| meta.advice_column());
        let instance = meta.instance_column();
        let constants = meta.fixed_column();

        // Columns that participate in copy constraints (to the Poseidon gadget,
        // to the instance column, or to fixed constants) must be equality-enabled.
        for col in [x, salt, rowidx, sumsel, selval, selidx, xc] {
            meta.enable_equality(col);
        }
        meta.enable_equality(instance);
        meta.enable_constant(constants);

        let s_arg = meta.selector();
        meta.create_gate("argmax step", |meta| {
            let s = meta.query_selector(s_arg);
            let x = meta.query_advice(x, Rotation::cur());
            let pick = meta.query_advice(pick, Rotation::cur());
            let rowidx_cur = meta.query_advice(rowidx, Rotation::cur());
            let rowidx_next = meta.query_advice(rowidx, Rotation::next());
            let sumsel_cur = meta.query_advice(sumsel, Rotation::cur());
            let sumsel_next = meta.query_advice(sumsel, Rotation::next());
            let selval_cur = meta.query_advice(selval, Rotation::cur());
            let selval_next = meta.query_advice(selval, Rotation::next());
            let selidx_cur = meta.query_advice(selidx, Rotation::cur());
            let selidx_next = meta.query_advice(selidx, Rotation::next());
            let xc_cur = meta.query_advice(xc, Rotation::cur());
            let xc_next = meta.query_advice(xc, Rotation::next());

            let one = Expression::Constant(Fp::ONE);
            // diff = xc - x, reconstructed from its range-check bits.
            let mut recomposed = Expression::Constant(Fp::ZERO);
            let mut constraints = Vec::with_capacity(DIFF_BITS + 7);
            for (k, &col) in bits.iter().enumerate() {
                let b = meta.query_advice(col, Rotation::cur());
                recomposed = recomposed + b.clone() * Expression::Constant(Fp::from(1u64 << k));
                constraints.push(b.clone() * (one.clone() - b)); // each bit is boolean
            }
            constraints.extend([
                pick.clone() * (one.clone() - pick.clone()), // pick is boolean
                rowidx_next - rowidx_cur.clone() - one,      // rowidx increments by 1
                sumsel_next - sumsel_cur - pick.clone(),     // sumsel += pick
                selval_next - selval_cur - pick.clone() * x.clone(), // selval += pick*x
                selidx_next - selidx_cur - pick * rowidx_cur, // selidx += pick*rowidx
                xc_next - xc_cur.clone(),                    // xc is constant down the column
                xc_cur - x - recomposed,                     // xc - x == Σ bit_k 2^k  (≥ 0)
            ]);
            Constraints::with_selector(s, constraints)
        });

        // Poseidon width-3 rate-2 permutation chip (shares no columns with the
        // argmax region; the two are wired only through copy-constrained cells).
        let state = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let partial_sbox = meta.advice_column();
        let rc_a = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let rc_b = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let poseidon = Pow5Chip::configure::<P128Pow5T3>(meta, state, partial_sbox, rc_a, rc_b);

        ArgmaxConfig {
            x,
            pick,
            rowidx,
            sumsel,
            selval,
            selidx,
            xc,
            bits,
            salt,
            s_arg,
            instance,
            poseidon,
        }
    }

    fn synthesize(
        &self,
        config: ArgmaxConfig,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let v = self.vocab;
        let p = self.witness.as_ref().map(Prepared::from);
        let salt_val = to_val(self.witness.as_ref().map(|w| w.salt));

        // ---- argmax region: assign logits + running selector, return the
        // logit cells and salt cell for the hash chain, plus selidx for the
        // public token binding. ----
        let (logit_cells, salt_cell, selidx_final) = layouter.assign_region(
            || "argmax",
            |mut region| {
                let g = |arr: fn(&Prepared) -> &Vec<Fp>, i: usize| {
                    to_val(p.as_ref().map(|pp| arr(pp)[i]))
                };

                // Running columns over rows 0..=v.
                let mut rowidx_cells = Vec::with_capacity(v + 1);
                let mut sumsel_cells = Vec::with_capacity(v + 1);
                let mut selval_cells = Vec::with_capacity(v + 1);
                let mut selidx_cells = Vec::with_capacity(v + 1);
                let mut xc0 = None;
                for i in 0..=v {
                    rowidx_cells.push(region.assign_advice(
                        || "rowidx",
                        config.rowidx,
                        i,
                        || g(|pp| &pp.rowidx, i),
                    )?);
                    sumsel_cells.push(region.assign_advice(
                        || "sumsel",
                        config.sumsel,
                        i,
                        || g(|pp| &pp.sumsel, i),
                    )?);
                    selval_cells.push(region.assign_advice(
                        || "selval",
                        config.selval,
                        i,
                        || g(|pp| &pp.selval, i),
                    )?);
                    selidx_cells.push(region.assign_advice(
                        || "selidx",
                        config.selidx,
                        i,
                        || g(|pp| &pp.selidx, i),
                    )?);
                    let xc_cell = region.assign_advice(
                        || "xc",
                        config.xc,
                        i,
                        || to_val(p.as_ref().map(|pp| pp.xc)),
                    )?;
                    if i == 0 {
                        xc0 = Some(xc_cell);
                    }
                }
                let xc0 = xc0.expect("v >= 0 so row 0 exists");

                // Per-logit rows 0..v.
                let mut logit_cells = Vec::with_capacity(v);
                for i in 0..v {
                    config.s_arg.enable(&mut region, i)?;
                    let xcell = region.assign_advice(|| "x", config.x, i, || g(|pp| &pp.x, i))?;
                    logit_cells.push(xcell);
                    region.assign_advice(|| "pick", config.pick, i, || g(|pp| &pp.pick, i))?;
                    for (k, &col) in config.bits.iter().enumerate() {
                        region.assign_advice(
                            || "bit",
                            col,
                            i,
                            || to_val(p.as_ref().map(|pp| pp.bits[i][k])),
                        )?;
                    }
                }

                // Boundary pins.
                region.constrain_constant(rowidx_cells[0].cell(), Fp::ZERO)?;
                region.constrain_constant(sumsel_cells[0].cell(), Fp::ZERO)?;
                region.constrain_constant(selval_cells[0].cell(), Fp::ZERO)?;
                region.constrain_constant(selidx_cells[0].cell(), Fp::ZERO)?;
                region.constrain_constant(sumsel_cells[v].cell(), Fp::ONE)?;
                // selected value equals the broadcast maximum xc (row 0 holds xc).
                region.constrain_equal(selval_cells[v].cell(), xc0.cell())?;

                // salt = acc_0 of the hash chain.
                let salt_cell = region.assign_advice(|| "salt", config.salt, 0, || salt_val)?;

                Ok((logit_cells, salt_cell, selidx_cells[v].clone()))
            },
        )?;

        // ---- Poseidon hash chain over the same logit cells ----
        let mut acc = salt_cell;
        for (i, x_cell) in logit_cells.into_iter().enumerate() {
            let chip = Pow5Chip::construct(config.poseidon.clone());
            let hasher =
                PoseidonHash::<
                    Fp,
                    Pow5Chip<Fp, WIDTH, RATE>,
                    P128Pow5T3,
                    ConstantLength<2>,
                    WIDTH,
                    RATE,
                >::init(chip, layouter.namespace(|| format!("poseidon init {i}")))?;
            acc = hasher.hash(
                layouter.namespace(|| format!("poseidon hash {i}")),
                [acc, x_cell],
            )?;
        }

        // digest == instance[0]; selected index == instance[1] (the token c).
        layouter.constrain_instance(acc.cell(), config.instance, 0)?;
        layouter.constrain_instance(selidx_final.cell(), config.instance, 1)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::chain_digest;
    use halo2_proofs::dev::MockProver;

    const K: u32 = 11;

    /// Logits with a clear maximum at `token`; returns (logits, salt, instance).
    fn setup(v: usize, token: usize) -> (Vec<i32>, Fp, Vec<Fp>) {
        let mut logits: Vec<i32> = (0..v).map(|i| ((i * 7) % 13) as i32 * 100 - 500).collect();
        let m = *logits.iter().max().unwrap();
        logits[token] = m + 250;
        let salt = Fp::from(0x00C0_FFEE);
        let logits_fp: Vec<Fp> = logits.iter().map(|&q| felt_of(q)).collect();
        let digest = chain_digest(salt, &logits_fp);
        (logits, salt, vec![digest, Fp::from(token as u64)])
    }

    #[test]
    fn mock_honest_verifies() {
        let (logits, salt, instance) = setup(16, 5);
        let circuit = ArgmaxCircuit::prover(&logits, salt, 5);
        let prover = MockProver::run(K, &circuit, vec![instance]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn mock_wrong_token_index_fails() {
        let (logits, salt, mut instance) = setup(16, 5);
        instance[1] = Fp::from(6); // claim a different index than the witnessed pick
        let circuit = ArgmaxCircuit::prover(&logits, salt, 5);
        let prover = MockProver::run(K, &circuit, vec![instance]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn mock_wrong_digest_fails() {
        let (logits, salt, mut instance) = setup(16, 5);
        instance[0] += Fp::ONE;
        let circuit = ArgmaxCircuit::prover(&logits, salt, 5);
        let prover = MockProver::run(K, &circuit, vec![instance]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn mock_non_argmax_token_fails() {
        // True max is at index 5; try to prove a non-maximal token 3. The
        // range check on xc - x[5] < 0 cannot be satisfied.
        let (logits, salt, _) = setup(16, 5);
        let logits_fp: Vec<Fp> = logits.iter().map(|&q| felt_of(q)).collect();
        let digest = chain_digest(salt, &logits_fp);
        let instance = vec![digest, Fp::from(3)];
        let circuit = ArgmaxCircuit::prover(&logits, salt, 3);
        let prover = MockProver::run(K, &circuit, vec![instance]).unwrap();
        assert!(prover.verify().is_err());
    }
}
