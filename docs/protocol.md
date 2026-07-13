# Protocol specification

The precise security statement for each layer in one place: **what is bound**,
**what is proven**, and **the exact adversary** each layer defeats — and, just
as importantly, what it does *not*. This is the citable version of the prose
scattered through the README; where they differ, the code is authoritative.

Notation: `H` is BLAKE3 with a per-use domain tag. All integers are
little-endian and every variable-length field is length-prefixed. Logits and
activations are committed at fixed point: `q = round(x · 2^f)` as `i32`.

---

## Layer 1 — commitments (binding)

### Model commitment

- **What is bound.** A Merkle root over the GGUF tensors. Leaf `i` =
  `H_leaf(name ‖ ggml_type ‖ shape ‖ raw quantized bytes)`; leaves sorted by
  tensor name; RFC 6962 tree shape with domain-separated leaf/node hashing
  (`vllm/merkle-leaf/v1`, `vllm/merkle-node/v1`). The commitment is over the
  bytes **exactly as stored on disk** — no dequantization.
- **Statement.** Two GGUF files with the same root are byte-identical in every
  committed tensor payload, name, type, and shape (up to BLAKE3 collision
  resistance). The root is invariant to on-disk tensor *order* and to
  metadata-only edits.
- **Adversary.** A prover who swaps, fine-tunes, or re-quantizes the model after
  the fact. Any such change moves the root.

### Transcript hash chain

- **What is bound.**
  - `chain_seed = H(model_root ‖ prompt_tokens ‖ sampler_config)`, where
    `sampler_config` is the raw IEEE bits of `(mode, temperature, top_p,
    rng_seed)`.
  - `chain_i = H(chain_{i−1} ‖ i ‖ token_i ‖ H(quantize(logits_i)))`.
  - Optionally folded per step: trace-cell hashes (`--trace`) and a
    circuit-friendly logits digest (`--prove-decode`), each with its own domain.
- **Statement.** `final_chain` binds the prover to one specific ordered tuple
  `(model, prompt, sampler, per-step logits digest, tokens)`. Any retroactive
  edit to any bound field changes `final_chain` (checked by `vllm replay`).
  Same machine + backend reproduces the chain bit-for-bit.
- **Adversary.** A prover who claims a different prompt/sampler/output than what
  was committed, or who edits history. **Not** defeated: a prover who commits to
  *fabricated* logits never produced by the model — Layer 1 is binding, not
  correct. That gap is Layer 2.

Fixed-point precision `f = 16` for logits is recorded in the transcript and
absorbed into the logits digest, so it is bound transitively (lying about it
breaks the head check). `NaN` logits abort; `±inf` saturates.

---

## Layer 2 — spot checks (probabilistic soundness)

Generating with `--trace` additionally commits every activation cell
`(pos, block)` — the hidden state entering block `j` at position `p`, plus the
state exiting the last block — quantized and Merkle-rooted, with the root folded
into the chain as generation runs.

- **What is bound.** The trace root over all cells (index = `pos·(L+1) + layer`),
  bound into `final_chain`.
- **Challenge.** `k` distinct cells drawn by
  `BLAKE3-XOF(vllm/fs-challenge/v1 ‖ final_chain ‖ trace_root ‖ k ‖ nonce ‖
  n_positions ‖ n_layers ‖ first_logit_pos)`, sampled without replacement. The
  space parameters are absorbed, so the challenge space cannot be reshaped.
- **Response + check.** Per challenged block the prover reveals the block's
  committed input cells at positions `0..=p` and its output at `p`, each with a
  Merkle proof; the verifier re-executes the block with the committed weights
  over the revealed inputs and compares. Head cells additionally reveal the
  step's quantized logits, which must hash **exactly** to the chain's
  `logits_hash`. Layer-0 inputs must equal the committed tokens' embeddings.
- **Statement.** Let a *cheat* be a committed cell that does not match the
  committed model's computation on the committed inputs. With a fraction `f` of
  `N` cells inconsistent, `k` distinct challenges miss all cheats with
  probability `≤ (1 − f)^k`; equivalently `k = ⌈ln(1/δ) / f⌉` catches with
  probability `≥ 1 − δ` (the README table uses the tighter exact
  `⌈ln δ / ln(1−f)⌉`). The check has two thresholds: per-element `|Δ| ≤ τ`
  (default 0.5) and per-cell mean `|Δ| ≤ τ_mean` (default 0.05).
- **Adversary.**
  - **Defeated:** lazy provers, model substitution, fabricated traces, a
    corrupted subset of cells (caught with the probability above).
  - **Grinding:** with an empty nonce a prover can regenerate until the draw
    misses its cheats (cost ≈ `(1 − f)^{−k}`). A verifier nonce chosen *after*
    the transcript is committed removes this entirely; `respond` refuses any
    challenge that is not the exact derivation for `(transcript, nonce)`.
  - **Bounded drift (float path):** an adversary who runs the *real* model but
    perturbs every cell by `< τ` stays under the per-element check. REPORT.md
    quantifies this: the network amplifies sub-`τ` drift ~30–80×, so token
    steering is feasible at any `τ` above the honest cross-backend noise floor.
    The mean check narrows this ~10× but does **not** close it. **The closure is
    determinism, not a tighter tolerance** — see Layer 2′.

## Layer 2′ — deterministic exact verification (`--deterministic`, v0.4)

- **What changes.** A fixed-evaluation-order CPU backend (`det-cpu-v1`): scalar
  sequential reductions, no libm (fixed-polynomial exp/sin/cos), activation
  cells snapped to a `2⁻⁸` grid at every hook so a committed cell *is* the
  computation state. Transcripts are bit-identical across runs and machines.
- **Statement.** Re-execution is exact: recompute → quantize → `i32` equality,
  **zero** tolerance. A single-quantum perturbation (`2⁻⁸` on one element of one
  cell), even in a fully self-consistent cheating trace, is caught.
- **Adversary.** Closes the bounded-drift attack entirely for deterministic
  transcripts, at ~8× lower generation speed and ~5 GB extra RAM.

---

## Layer 3 — decode proof (argmax, greedy only)

Two backends prove the **same** statement about one greedy decode step. Both
commit the step's quantized logits with a salted, circuit-friendly hash whose
32-byte digest is folded into the chain, then prove:

> For a public token index `c` and chain-bound digest `d`, the prover knows a
> logit vector `x` and salt `s` such that `Commit_s(x) = d` and `x[c] ≥ x[i]`
> for all `i` — i.e. the emitted token is a maximum of the committed logits.

The argmax is enforced by a private claimed-maximum and a **27-bit** range check
on `m − x[i] ∈ [0, 2²⁷)`. Because `2²⁷ ≪` the field modulus, the field
difference equals the true integer difference with no wraparound, so a non-max
claim forces some `m − x[j] < 0` (≈ field size) and fails the range check. This
bounds the provable logit spread at `2²⁷ / 2¹⁶ = 2048` logit units (real spreads
are ~40). Sampler modes other than greedy are out of scope (roadmap).

### 3a. STARK (winterfell), the fast default — `vllm-zk`

- **Commitment.** Salted Rescue-Prime sponge (Rp64_256 over Goldilocks); the
  salt is a private capacity IV.
- **Security.** A succinct, **transparent** argument of knowledge (no trusted
  setup), ~100-bit conjectured (blowup 8, 27 queries, 16-bit grinding, quadratic
  extension). The commitment is binding.
- **Zero-knowledge — precisely, NO.** winterfell 0.13 has no trace
  randomization, so each proof reveals the trace polynomials' evaluations at the
  ~27 FRI query points — equivalently, ~27 random linear projections of the
  `(salt ‖ logits)` vector (the LDE domain is a coset, so openings are
  projections, not raw cells). Consequence: **reconstruction-safe**
  (underdetermined by ~5 orders of magnitude, and the salt blocks digest-level
  guessing) but **confirmation-unsafe** (a party already holding a candidate for
  the *entire* logit vector could confirm it from the openings).

### 3b. halo2 (transparent IPA), the formal-ZK path — `vllm-zk-halo2`

- **Commitment.** Salted Poseidon hash chain `accᵢ₊₁ = Poseidon2(accᵢ, xᵢ)`,
  `acc₀ = salt`, using the same width-3 rate-2 primitive the in-circuit gadget
  realizes (native/in-circuit parity is pinned by the crate's own tests).
- **Argmax without a per-token key.** A running one-hot selector (`sumsel = 1`,
  `selidx = c` read from the instance column via copy constraint, `selval = x[c]`)
  plus the per-row 27-bit range check. Because `c` is an instance, one verifying
  key serves every token index.
- **Security.** Transparent (IPA over Pasta, no trusted setup) **and** formally
  zero-knowledge: the prover blinds every committed witness polynomial, so the
  proof reveals nothing about the logits beyond the public claim. The trade is
  cost: the in-circuit Poseidon is `O(vocab)` — a real llama vocab lands at
  k≈23 (~8M rows), minutes of proving and multiple GB.

### Composition

Layer 2′ (or Layer 2 + REPORT's caveat) guarantees the committed logits are the
model's true logits (exactly, or within the accumulated drift bound); Layer 3
proves the emitted token is the exact argmax of those committed logits. Together:
*the emitted token is the argmax of logits that are the committed model's true
logits on the committed inputs* — with the drift qualifier on the float path and
exactly on the deterministic path.
