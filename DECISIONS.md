# Design decisions

Non-obvious choices, with the data or reasoning behind them. Versions in
force: candle 0.11, blake3 1.8, Rust edition 2024.

## 1. BLAKE3 for all commitments

SHA-256 would work; BLAKE3 was chosen because model hashing is bulk-bound
(770 MB for a 1B Q4_K_M ⇒ ~0.5 s single-threaded) and logit hashing runs in
the decode hot loop (~1 MB/token). Both use plain domain-prefixed hashing —
no keyed/derive-key modes — so any BLAKE3 implementation can verify.

**Forward-looking caveat**: BLAKE3 is expensive inside a ZK circuit. Layer 3
will likely add a circuit-friendly commitment (e.g. Poseidon over the
fixed-point logits) *alongside* the BLAKE3 hash, both folded into the chain,
rather than proving BLAKE3 in-circuit. Decision deferred until v0.3 with a
benchmark.

## 2. Std-only GGUF parser in `vllm-core` (no candle dependency)

The commitment is over the **raw quantized bytes exactly as stored on
disk** (plus name/type/shape in the leaf preimage). No dequantization, so no
float ambiguity in the model commitment, and a verifier needs only this
crate + blake3 — it builds on any machine without candle, Metal, or a GPU.
Cost: ~250 lines of parser and a table of ggml block sizes (validated
against ggml-quants.h in unit tests). The parser also has a writer twin used
to construct tiny test models so CI never downloads weights.

## 3. Merkle construction

RFC 6962 tree shape (split at the largest power of two < n) with
domain-separated leaf (`vllm/merkle-leaf/v1`) and node
(`vllm/merkle-node/v1`) hashing to block second-preimage/ambiguity attacks.
Leaves sorted by tensor name so the root is independent of on-disk tensor
order (requantization tools may reorder). Inclusion proofs are implemented
and tested now because v0.2 challenge responses need them. Golden-vector
tests pin the construction; changing it requires bumping the domain strings.

## 4. Logit fixed-point precision: 16 fractional bits (default), i32

`q = round(x · 2^16)` as i32, hashed little-endian, precision recorded in
the transcript. i32 rather than i64 halves the bytes hashed per token
(~0.5 MB for a 128k vocab); saturation needs |logit| > 32767, unreachable
for a real model. The conversion runs in stack-sized chunks — measured
commitment overhead on Llama-3.2-1B Q4_K_M: 2.6 % of inference at
warm-cache decode speed (79 tok/s), 0.4 % cold (22 tok/s); i64 + per-step
heap buffer was 4.2 % warm. Precision choice, measured on the same setup:

- **Metal run-to-run**: bit-identical final chains across 3 runs even at 24
  fractional bits ⇒ same-machine reproducibility does not constrain the
  precision.
- **CPU vs Metal (same prompt forward)**: max |Δ| = 4.9e-1, mean |Δ| =
  8.0e-2 over the 128k-dim logit vector (different Q4_K dequant/matmul
  paths). Argmax agreed, but hash equality failed at every precision down to
  2 fractional bits ⇒ **no** quantization makes cross-backend hashes match;
  a value near a rounding boundary can always flip.

Consequences: the logit hash is a *binding commitment* to what the prover's
backend computed, not a cross-backend equality check. The v0.2 verifier
re-executes challenged layers and compares within a numerical tolerance
(calibrated from the drift above); `vllm-infer/examples/logit_diff.rs`
reproduces the measurement. 16 bits is far above observed same-backend
noise (zero) and keeps quantized logits < 2^26 in magnitude — comfortable
for a v0.3 circuit. NaN logits abort (never silently committed); ±inf
saturates.

## 5. Hash chain framing

Every variable-length field is length-prefixed and every hash input is
domain-tagged (`vllm/chain-seed/v1`, `vllm/chain-step/v1`, …). The step
index is absorbed explicitly, the sampler config is committed as raw IEEE
bits (mode tag, temperature, top_p, rng_seed) — no JSON canonicalization
ambiguity. Layer-hash folding (`vllm/chain-layers/v1`) is specified and
tested in core now so v0.2 doesn't need a chain format change.

## 6. Tiny test model: F32 weights

candle's `QMatMul::from_qtensor` auto-dequantizes F32/F16/BF16 tensors, so
the in-test 2-layer llama uses plain F32 — no hand-rolled quantization in
tests. Weights come from a fixed xorshift64 seed: the test model is
byte-identical on every run and across machines.

## 7. `--trace-layers` descoped from v0.1

Per-layer activation hashes require hooks inside the transformer forward
pass; candle-transformers' `quantized_llama` exposes none. v0.2 must vendor
that model anyway (single-layer re-execution for spot checks), so the flag
ships there. The chain/transcript formats already carry optional
`layer_hashes` per step — no format break later.

## 8. Auto-caching the model commitment

`vllm generate` re-hashes the GGUF on every run (~0.5 s for 1B) rather than
trusting a cached commitment, and writes `<model>.commitment.json` only if
absent. `--model-commitment` opts into reuse; the file's root is checked
against its own leaves before use.

## 9. Vendoring candle's quantized_llama (v0.2)

candle-transformers' `quantized_llama` exposes no per-layer hooks, so
`vllm-infer/src/model.rs` vendors it (Apache-2.0/MIT), stripped of MoE,
ggml-v1 files, and tracing spans, and extended with `forward_traced`
(activation capture), `forward_block` (single-block re-execution), `lm_head`
and `embed` (head/embedding checks). Every tensor operation was kept
byte-identical to upstream, validated by an exact final-chain parity test on
Llama-3.2-1B (Metal, greedy): vendored and upstream produce the same
transcript hash chain.

## 10. Trace design: hidden states between blocks, prefix re-execution

Cell (p, j) = hidden state entering block j at position p (plus the state
exiting the last block). A challenge (p, j) reveals the block's inputs at
positions 0..=p and its output at p; the verifier rebuilds K/V from the
revealed prefix and re-executes the whole block. Alternatives considered:
also committing per-layer KV caches (more cells, no verification gain — KV
is a function of the revealed inputs) and committing only sampled positions
(breaks prefix reveals: attention at position p needs all inputs ≤ p).
Consequence: response size grows with the challenged position (~8 KB/cell
for the 1B model); measured 15 MB for k=40 on a 60-position trace. Logit
rows are stored in the trace file for head challenges but are bound by the
chain's `logits_hash` (exact), not the Merkle tree.

## 11. Verification tolerance: default 0.5 per element

The verifier re-executes on CPU; traces are typically produced on Metal.
Measured per-block worst-case deviation (Llama-3.2-1B Q4_K_M, 112 block
challenges across three runs): 2.7e-2 / 4.9e-2 / 4.4e-2 — dominated by the
Q4_K matmul path difference, plus ±2^-17 activation quantization. Default
tolerance 0.5 keeps ~10x headroom against false accusations of honest
provers; same-backend verification can tighten to ~1e-3 (tiny-model
CPU→CPU tests pass at 0.01). The residual bounded-drift attack surface is
documented in the README; closing it exactly is what Layer 3 does for the
decode step (exact hash binding inside the circuit).

## 12. Fiat–Shamir challenges with an optional verifier nonce

Challenges derive from BLAKE3-XOF(final_chain ‖ trace_root ‖ k ‖ nonce ‖
space params), sampled without replacement (catch probability then beats
the (1-f)^k bound). Pure FS (empty nonce) admits grinding: regenerate until
the draw misses the corrupted cells, expected cost (1-f)^{-k} generations —
documented in the README rather than hidden. A verifier-chosen nonce issued
after the transcript is committed eliminates grinding at the cost of one
round of interaction; `vllm respond` refuses challenges that are not the
exact derivation for the claimed (transcript, nonce).

## 13. Proof system for Layer 3: winterfell STARK, not ezkl/halo2

Requirements: prove argmax over a committed 128k-dim logit vector in
seconds-to-a-minute on an M-series laptop, from Rust. Evaluation:

- **ezkl**: designed for ONNX graphs; a custom hash-binding statement fights
  the framework, and it drags in an enormous dependency tree plus a KZG SRS.
- **halo2 (PSE)**: the only option with *formal* zero-knowledge (blinding),
  but an in-circuit 128k-element Poseidon absorption with the standard
  width-3 gadget lands at millions of rows — minutes of proving, plus SRS
  logistics.
- **winterfell 0.13**: transparent, pure Rust, Goldilocks field; the Rescue
  AIR fits the statement naturally. Measured: **prove 0.7 s, verify 0.7 ms,
  78 KB proof** for vocab 128 256 (blowup 8, 27 queries, 16-bit grinding,
  quadratic extension ⇒ ~100-bit conjectured security).

The catch, stated loudly: winterfell has **no zero-knowledge mode** (its
"Randomized AIR" is auxiliary randomness, not ZK). Mitigations shipped: the
LDE domain is a coset, so query openings are linear projections rather than
raw trace cells, and the commitment salt (private capacity IV, free witness
column) makes the digest hiding. Residual leakage: ~27 random linear
projections of (salt ‖ logits) per proof — reconstruction-safe,
confirmation-unsafe (a fully-guessed vector can be checked). This trade
buys 50-100x faster proving than the formal-ZK alternative; revisit when
winterfell ships trace randomization (open upstream work) or by wrapping in
halo2. Argmax only; committed-seed top-p sampling in-circuit is roadmap.

## 14. The argmax AIR

51 columns x next-pow-2(vocab + 8 + 1) rows (2^17 for llama). One Rescue
round per row, absorb row every 8th (rate 8, one logit per row via seven
lane accumulators); the salt sits in the sponge capacity as an
unconstrained witness at row 0. Argmax: private constant column m, 27-bit
decomposition of m - x_i on every row (bounds the provable logit spread at
2048 logit units — measured spreads are ~40), and a boolean selector column
that must sum to 1 and may only fire at row c, where it enforces x_c = m.
The verifier needs only (digest, token, vocab): assertions pin the sponge
IV, the digest row (position derived from vocab), and the selector sum.
Rescue constants come from winterfell's Rp64_256 exports, so the native
sponge and the AIR share one source of truth; the second-to-last row
carries a full-range difference so every bit column stays non-degenerate
under winterfell's debug-mode degree checks regardless of input. The
degree-7 backward-direction round constraint (inverse S-box trick) forces
blowup 8.

## 15. Mean-deviation check (bounded-drift detector)

The bounded-drift experiment (REPORT.md) showed per-element tolerance alone
admits token steering at any usable τ: the attack must spend its budget at
nearly every coordinate, while honest backend drift concentrates ~2 orders
of magnitude below its own max (measured Metal→CPU per-cell mean ≤ 8e-3 vs
max 4.4e-2, 40 real-model challenges). `vllm verify` therefore enforces a
second, distributional bound: mean |Δ| per challenged cell ≤ 0.05 by
default (`--mean-tolerance`; ~6x headroom over honest). Tested: a uniform
+0.04 shift of one cell passes the per-element check and is rejected by the
mean check. Stated plainly in REPORT.md: this narrows the attack ~10x, it
does not close it — exact token integrity requires deterministic integer
inference (roadmap v0.4).

## 16. Deterministic backend: fixed-order float, not integer-only

REPORT.md's conclusion called for "deterministic (integer-only) inference";
building it clarified that integer arithmetic was the means, not the
requirement. Bit-exact re-execution needs exactly three things, delivered
by `vllm-infer/src/det.rs` + `det_math.rs`:

1. **Fixed evaluation order.** IEEE-754 f32/f64 basic ops are exactly
   specified; nondeterminism comes from SIMD/blocked reductions
   reassociating sums and from thread-order-dependent accumulation. The det
   kernels are sequential scalar loops; rayon parallelism only ever spans
   independent outputs. (Integer accumulation would buy order-independence
   — unnecessary once the order is pinned. Full model quality is retained;
   integer-only would have required I-BERT-style approximations of
   softmax/rmsnorm/silu.)
2. **No libm.** exp/sin/cos vary across platforms; det_math pins them with
   fixed-order polynomial implementations (~1e-13 relative accuracy).
3. **Hook requantization.** The hidden state is snapped to the trace grid
   at every commitment hook, so the committed cell IS the computation
   state, and the verifier's recomputation from committed inputs is
   bit-identical. Activation cells use 2^-8, not 2^-16: quantize∘dequantize
   must be idempotent for exactness, and beyond |x| = 128 the f32 grid is
   coarser than 2^-16 (llama outlier channels exceed 128). At 2^-8 the
   roundtrip is exact up to |x| = 32768. Logits stay at 2^-16 (|logit|<50).

Also pinned: prompt processed position-at-a-time (no separate batch path to
diverge from), weights dequantized upfront via candle's scalar CPU kernels
(per-element, no reductions — order-independent; ~5 GB resident for 1B),
transcripts tagged `det-cpu-v1` (bump on ANY computational change).

Measured (Llama-3.2-1B Q4_K_M, M-series): 10.2 tok/s decode (vs ~80 on
Metal), load 2.2 s, reruns bit-identical (equal final chains). Verification
of det transcripts is EXACT: recompute → quantize → i32 equality, zero
tolerance; 20 real-model challenges verified with 0 deviation in 4.9 s.
Tested: a fully self-consistent cheat that shifts ONE element of one cell
by ONE quantum (2^-8) — invisible at any float tolerance — is caught. This
closes REPORT.md's bounded-drift attack for deterministic transcripts and
makes them portable across machines/platforms (same binary version).

## 17. Formal-ZK Layer 3: halo2 (transparent IPA), the `vllm-zk-halo2` crate

Decision #13 shipped the argmax proof on winterfell and stated its one honest
limitation loudly: winterfell 0.13 has no trace randomization, so the STARK is
a succinct *argument*, not *formal* zero-knowledge — its query openings are
random linear projections of (salt ‖ logits), reconstruction-safe but
confirmation-unsafe. #13 named the two ways out: winterfell trace
randomization (upstream, not shipped) or "a halo2 wrapper". This is the halo2
wrapper, built as a sibling crate proving the identical statement — *the
committed token is a maximum of the committed logit vector* — with genuine
zero-knowledge.

- **Why halo2/IPA.** halo2's inner-product-argument backend over the Pasta
  curves is *transparent* (no trusted setup, like the STARK) and its prover
  **blinds every committed polynomial**, so the proof is formally ZK. That is
  exactly the property winterfell lacks; it is the whole reason this crate
  exists. Field is `pallas::Base` (Fp); commitments live on Vesta (`EqAffine`).

- **Commitment: a salted Poseidon hash chain, not a Rescue sponge.** The STARK
  hand-rolled a Rescue AIR (#14) to match a native sponge. halo2_gadgets 0.5
  already ships a Poseidon chip (`Pow5Chip`, width-3 rate-2 `P128Pow5T3`)
  whose in-circuit output the crate's *own* tests pin equal to the native
  `halo2_poseidon` primitive. So the commitment is a chain
  `accᵢ₊₁ = Poseidon2(accᵢ, xᵢ)` seeded by a private `acc₀ = salt`, using that
  vetted 2-to-1 primitive on both sides — native/in-circuit parity for free,
  no bespoke hash AIR, and variable vocab handled by looping the gadget.
  Chaining a collision-resistant compression is binding; the secret salt makes
  the digest hiding.

- **Argmax without a per-token key.** A running-selector region assigns one
  logit per row with a boolean one-hot `pick`, and four running accumulators
  (`rowidx = 0,1,2,…`, `sumsel = Σpick` pinned to 1, `selval = Σpick·x` pinned
  to a broadcast maximum `xc`, `selidx = Σpick·rowidx` pinned to the public
  index `c`). Together they force exactly one pick, at row `c`, with
  `xc = x[c]`; a per-row range check `xc − x[i] ∈ [0, 2^DIFF_BITS)` (same
  27-bit bound as #14) then gives `x[c] ≥ x[i]` for all `i`. Crucially `c` is
  read from the **instance column via a copy constraint**, not baked into the
  circuit — so a single verifying key serves every token index (a
  key-per-token circuit would be 128k keys).

- **Cost (measured, M-series, release).** vocab 32/128/256 → cold prove
  1.3 / 5.5 / 11 s, IPA proof **~5 KB** (small and ~constant). But the cold
  wall-clock is mostly *deterministic, cacheable* setup — IPA parameter
  generation (`Params::new`) plus proving/verifying-key generation, all a pure
  function of the vocab. The work a deployment that caches them pays is the
  proving proper (`create_proof`: 0.4 / 1.6 / 3.1 s) and the *proof check*
  (`verify_proof`: **6.7 / 20 / 37 ms**). IPA verification is linear in circuit
  size (not a succinct verifier like KZG), so the check grows with vocab — ≈
  seconds at 128k — but stays tiny at these sizes. The in-circuit Poseidon is
  O(vocab): a real llama vocab (128 256) lands at k≈23 (a 2²³ ≈ 8M-row
  circuit), minutes of proving and multiple GB. This is precisely the "millions
  of rows — minutes of proving" #13 predicted for in-circuit Poseidon, and the
  reason the STARK remains the default fast path (0.7 s, 78 KB) while this is
  the opt-in *formal-ZK* path.

- **Isolation and CLI wiring.** The halo2 tree is heavy, so the crate is a
  workspace member **excluded from `default-members`** and pulled into
  `vllm-cli` only behind an optional `zk-halo2` feature — the default `cargo
  build` and `vllm` binary stay free of it (same isolation principle as #13).
  It is wired as an alternate decode backend: `generate --prove-decode
  --zk-backend halo2`, `prove-decode --backend halo2`, `verify-decode` (which
  reads the backend from the proof envelope). The nice surprise: this needed
  **no transcript-format change**. The chain absorbs a salted *32-byte* digest
  per step regardless of which hash produced it, and the trace's salt slot is
  `[u64; 4]` = the same 32 bytes — so the Poseidon digest and salt reuse the
  existing Rescue slots (the CLI just converts `[u64;4]`↔`[u8;32]`). The
  published v0.4 transcript format is untouched; a halo2 transcript replays and
  binds its trace through the identical `vllm-core` machinery.

- **Range check via bit decomposition.** `xc − x[i]` is decomposed into
  DIFF_BITS booleans with a linear recomposition constraint — the simplest
  construction that is unambiguously correct (no lookup-argument corner cases).
  A limb + fixed-table lookup would cut the range-check column count ~9× and is
  the obvious width optimization; correctness-first here, since formal-ZK at
  128k is proving-bound regardless.
