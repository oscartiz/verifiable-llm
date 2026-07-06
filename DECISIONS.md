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
