# Bounded-drift adversary: what sub-tolerance cheating actually buys

The v0.2 verifier re-executes challenged blocks and accepts committed
activations within a numerical tolerance τ per element (float re-execution
across backends is not bit-exact; DECISIONS.md #11). The README flags the
resulting escape hatch: a prover may perturb every committed cell by less
than τ and never fail a challenge. This report quantifies what such an
adversary can actually do to the *output tokens*, on the real model.

**Setup.** Llama-3.2-1B-Instruct Q4_K_M, Metal, greedy decoding, 48 tokens,
fixed chat prompt. Adversary model: at every commitment hook (state entering
each of the 16 blocks, plus the state entering the final norm — the exact
cells committed by `--trace`), add a perturbation δ with ‖δ‖∞ = τ; the
perturbed state feeds all downstream computation, so the committed trace is
*self-consistent up to τ per block* and passes every per-element challenge
by construction. Reproduce with:

```sh
cargo run --release -p vllm-infer --features metal --example drift_attack -- \
    model.gguf tokenizer.json
```

## Result 1: the network amplifies sub-tolerance drift ~30–80×

A single random ±τ injection at ONE hook displaces the final logits by far
more than τ (max-abs over the 128k logits, prompt forward, small-signal
τ = 0.01):

| injection point (entering block) | 0 | 1 | 2 | 4 | 8 | 12 | 15 | final norm |
|---|---|---|---|---|---|---|---|---|
| amplification \|Δlogits\|∞ / τ | 385 | 106 | 78 | 47 | 43 | 54 | 33 | 26 |

At τ = 0.5 (the default cross-backend tolerance) amplification saturates
around 40–70× — the nonlinearity clips, but the displacement is already
tens of logits.

## Result 2: random full-stack drift flips tokens even at τ = 0.01

Honest top-1 vs top-2 logit gaps on this run: median 2.14, 10th percentile
0.33, minimum 0.08. Against that, uniform random ±τ at every hook of every
position:

| τ | step-0 \|Δlogits\|∞ | first token divergence (3 seeds) |
|---|---|---|
| 0.01 | 7.3 | steps 10, 9, 6 |
| 0.05 | 30.8 | 0, 0, 0 |
| 0.10 | 31.7 | 0, 0, 0 |
| 0.25 | 34.8 | 0, 0, 0 |
| 0.50 | 34.6 | 0, 0, 0 |

Note τ = 0.01 is **5× below** the honest cross-backend deviation itself
(max ≈ 5e-2 per block) — i.e. even a tolerance tight enough to reject
honest Metal traces would not stop this adversary. And this is *random*
drift, with no objective; it flips tokens by accident within ten steps.

## Result 3: a targeted single-point attack flips half of all steps at τ = 0.01

The ∞-norm-optimal direction at the last commitment hook is
δ = τ·sign(w_b − w_a), where w are LM-head rows and (a, b) the top-2 tokens
(validated empirically through the real RMSNorm + head, not just the linear
approximation):

| τ | steps whose argmax flips (of 48) |
|---|---|
| 0.01 | 24 |
| 0.05 | 43 |
| 0.10 | 47 |
| 0.25 | 48 |
| 0.50 | 48 |

## Conclusions

1. **Per-element tolerance cannot guarantee token provenance for float
   transformers.** The Jacobian amplification (~30–80×) times the honest
   near-tie density (p10 gap ≈ 0.33) means any tolerance at or above the
   honest cross-backend noise floor admits token steering. This is a
   structural limit of tolerance-based re-execution, not a bug in the
   protocol parameters.

2. **What Layer 2 *does* guarantee** is drift-bounded computation: every
   committed cell is within τ (per element) of the true computation on the
   committed inputs, with the committed weights. Combined with Layer 3, the
   full chain guarantees: *the emitted token is the exact argmax of logits
   that are everywhere within the accumulated drift bound of the committed
   model's true logits.* That defeats lazy provers, model substitution, and
   fabricated traces — but not an adversary who runs the real model and
   nudges it within tolerance.

3. **Shipped mitigation: a distributional check.** The attacks above spend
   the *whole* budget at nearly every coordinate; honest backend drift
   concentrates two orders of magnitude below its own max (measured
   Metal→CPU per-cell **mean** deviation ≤ 8e-3 vs max 4.4e-2). `vllm
   verify` now also enforces a per-cell **mean** tolerance (default 0.05,
   `--mean-tolerance`), which caps the adversary's average per-element
   budget at the mean bound instead of τ. Per Result 2 this shrinks the
   attack's operating room by ~10× but does **not** eliminate it — stated
   here so nobody mistakes it for a fix.

4. **The actual fix is determinism, not tighter tolerances**: exact
   re-execution (tolerance zero) requires integer-only inference kernels so
   that prover and verifier compute bit-identical activations. With exact
   matching, any perturbation — however small — breaks a challenged cell.
   That is the natural v0.4: a deterministic quantized inference path,
   which would also make transcripts portable across backends. Until then,
   parties who need exact token integrity should verify on the same
   backend with `--tolerance` and `--mean-tolerance` set to the measured
   same-backend noise floor (~1e-3), which reduces — but per Result 3 does
   not fully close — the steering margin.
