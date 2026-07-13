# verifiable-llm

[![ci](https://github.com/oscartiz/verifiable-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/oscartiz/verifiable-llm/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Verifiable local LLM inference on Apple Silicon: run a quantized llama GGUF
with [candle] and produce cryptographic commitments to *which model* ran and
*what it computed*, so a third party can check the claim later.

**This does NOT prove a full transformer forward pass in zero knowledge.**
That is infeasible on a laptop and is an explicit non-goal. Instead, three
composable layers of increasing cryptographic strength:

| Layer | Status | Mechanism | What it guarantees |
|---|---|---|---|
| 1 — commitments | **v0.1 (done)** | BLAKE3 Merkle tree over GGUF tensors + hash chain over decode steps | Binding: the prover is committed to one specific (model, prompt, sampler, per-step logits, tokens) tuple and cannot rewrite history afterwards |
| 2 — spot checks | **v0.2 (done)** | Fiat–Shamir challenges; verifier re-executes random single blocks on CPU | Probabilistic: cheating on a fraction *f* of the committed computation is caught with probability ≥ 1 − (1 − f)^k for k challenges |
| 3 — ZK decode | **v0.3 (done, argmax)** | STARK proof that the emitted token is the argmax of the committed logit vector | The decode step is correct without revealing the logits (succinct argument; see the precise security statement below) |
| 3′ — formal ZK decode | **v0.5 (done, argmax)** | halo2 (transparent IPA) proof of the *same* argmax statement, with polynomial blinding | Same guarantee, now **formally** zero-knowledge — the proof leaks nothing about the logits — at higher proving cost (see below) |

## Threat model (v0.1)

The adversary is a *prover* (someone claiming "model M generated text T from
prompt P") who wants to lie about it after the fact. Layer 1 makes claims
**binding**, not **correct**:

- ✅ The model commitment pins the exact quantized weight bytes on disk.
  Swapping in a different/fine-tuned/re-quantized model changes the root.
- ✅ The transcript hash chain pins prompt tokens, sampler config, RNG seed,
  every emitted token, and a fixed-precision digest of every logit vector,
  in order. Any retroactive edit breaks the chain (`vllm replay`).
- ✅ Same machine + backend re-runs reproduce the transcript bit-for-bit
  (measured: Metal is run-to-run deterministic on M-series; see
  DECISIONS.md #4).
- ✅ With `--trace` (v0.2), the prover is additionally committed to every
  intermediate activation, and random re-execution challenges verify the
  committed computation actually follows from the committed weights — see
  the protocol section below for the exact guarantee and its limits.
- ❌ Layer 1 alone does **not** prove the logits actually came from running
  model M — a prover could commit to made-up logits. That is exactly what
  Layer 2's re-execution challenges add.
- ❌ Cross-backend (CPU vs Metal) logit equality is impossible at any usable
  precision (measured max |Δ| ≈ 0.5 on a 1B Q4_K_M model); v0.2 therefore
  verifies challenged layers within a numerical tolerance, not by hash
  equality.

## Quickstart

```sh
cargo build --release            # Metal by default
cargo build --release --no-default-features   # CPU-only (verifier machines)

MODEL=path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf
TOK=path/to/tokenizer.json

# Commit to the model weights (~0.5 s for 1B): writes <model>.commitment.json
vllm commit --model $MODEL

# Check a GGUF against a commitment
vllm verify-model --model $MODEL --commitment $MODEL.commitment.json

# Generate with a committed transcript (streams tokens; ~22 tok/s on M-series)
vllm generate --model $MODEL --tokenizer $TOK \
    --prompt "In one sentence, what is a Merkle tree?" \
    --greedy --max-tokens 100 --commit run.json --bench

# Re-check the transcript's hash chain (no model or GPU needed)
vllm replay --commitment run.json

# --- v0.2: spot-check protocol (add --trace at generation time) ---
vllm generate --model $MODEL --tokenizer $TOK --prompt "..." \
    --greedy --max-tokens 100 --commit run.json --trace run.trace

vllm challenge --commitment run.json -k 60 --nonce "$(uuidgen)" --out challenge.json
vllm respond   --commitment run.json --trace run.trace \
    --challenge challenge.json --out response.json
vllm verify    --commitment run.json --model $MODEL \
    --challenge challenge.json --response response.json

# --- v0.4: deterministic mode - challenges verify with EXACT equality ---
vllm generate --model $MODEL --tokenizer $TOK --prompt "..." \
    --greedy --max-tokens 100 --deterministic --commit run.json --trace run.trace
# ... then challenge/respond as usual; verify requires zero deviation:
vllm verify --commitment run.json --model $MODEL \
    --challenge challenge.json --response response.json   # max |delta| must be 0

# --- v0.3: prove the decode step (greedy transcripts; add --prove-decode) ---
vllm generate --model $MODEL --tokenizer $TOK --prompt "..." \
    --greedy --max-tokens 100 --commit run.json --trace run.trace --prove-decode

vllm prove-decode  --commitment run.json --trace run.trace --step 5 --out step5.proof.json
vllm verify-decode --commitment run.json --proof step5.proof.json   # no model/GPU needed

# --- v0.5: the same flow, formally zero-knowledge (build --features zk-halo2) ---
# The digest is backend-agnostic in the transcript, so only the backend flag
# changes; verify-decode reads which backend from the proof file.
vllm generate --model $MODEL --tokenizer $TOK --prompt "..." \
    --greedy --max-tokens 100 --commit run.json --trace run.trace \
    --prove-decode --zk-backend halo2
vllm prove-decode  --commitment run.json --trace run.trace --step 5 \
    --backend halo2 --out step5.halo2.json
vllm verify-decode --commitment run.json --proof step5.halo2.json
```

`--bench` prints the commitment overhead. Measured on an M-series laptop,
Llama-3.2-1B Q4_K_M: **2.6 %** of inference time at warm-cache decode speed
(79 tok/s), 0.4 % on a cold run (target < 5 %).

## What is committed, exactly

- **Model**: leaf = BLAKE3 over (name, ggml type, shape, raw quantized
  bytes) per tensor, leaves sorted by name, RFC 6962-shaped Merkle tree with
  domain-separated leaf/node hashing. Roots are stable across file moves and
  metadata-only edits, and independent of tensor order in the file.
- **Generation**: `chain_0 = H(model_root ‖ prompt_tokens ‖ sampler_config)`,
  then `chain_i = H(chain_{i−1} ‖ i ‖ token_i ‖ H(quantize(logits_i)))`.
  Logits are quantized to fixed point (`round(x · 2^16)` as i32 by default;
  recorded in the transcript) before hashing. The transcript JSON carries every
  per-step value so v0.2 challenges can reference individual steps.

## The spot-check protocol (v0.2)

Generating with `--trace` additionally commits to every intermediate hidden
state: cell (p, j) = the activation vector entering block j at position p
(quantized i32, fixed point), for all positions (prompt + generated) and all
blocks, plus the state exiting the last block. Cell hashes are folded into
the hash chain *as generation runs* and Merkle-rooted; the root is in the
transcript.

- `vllm challenge` derives k pseudorandom cells from
  BLAKE3-XOF(final_chain ‖ trace_root ‖ k ‖ nonce): (position, block) pairs,
  plus LM-head cells that tie last-block activations to the committed
  logits.
- `vllm respond` reveals, per challenged block, the block's committed input
  cells at positions 0..=p and its output cell at p, each with a Merkle
  inclusion proof. Head challenges also reveal the step's quantized logits,
  which must hash **exactly** to the chain's `logits_hash`.
- `vllm verify` (CPU-only, works on a second machine) checks the chain, the
  model root, the Fiat–Shamir derivation, every Merkle proof, that layer-0
  inputs equal the committed tokens' embeddings, and finally **re-executes
  each challenged block** with the committed weights over the revealed
  inputs, requiring agreement within a numerical tolerance.

All messages are JSON files, so prover and verifier can be different
machines with only file exchange.

### Catch probability

A cheat is a committed cell that does not match the committed model's
computation on the committed inputs. With N committed cells and a fraction
f of them inconsistent, k uniformly sampled distinct challenges miss all of
them with probability ≤ (1 − f)^k, so

    k = ⌈ln(1/δ) / f⌉  challenges catch the prover with probability ≥ 1 − δ.

| cheat fraction f | k for 95 % | k for 99 % | k for 99.9 % |
|---|---|---|---|
| 20 % | 14 | 21 | 31 |
| 10 % | 29 | 44 | 66 |
| 5 %  | 59 | 90 | 135 |
| 1 %  | 300 | 459 | 688 |

`-k` is configurable on `vllm challenge`. Cost: the verifier re-executes k
single blocks over their position prefixes — roughly k/L prompt-forward
equivalents (measured: 60 challenges over a 60-position Llama-3.2-1B trace
verify in ~2 s on CPU).

**Grinding caveat.** With pure Fiat–Shamir, a prover can re-generate until
the derived challenges miss its bad cells (expected cost ≈ (1 − f)^{-k}
generations — cheap when f·k is small). For adversarial settings, the
verifier passes `--nonce <fresh randomness>` chosen *after* receiving the
transcript, which removes the prover's ability to grind entirely. The
empty-nonce mode remains useful for self-audit and archival transcripts.

**Float transcripts are tolerance-verified; deterministic transcripts are
exact.** For the default Metal backend, re-execution is tolerance-based
(float drift across backends, measured max ≈ 5e-2 per block CPU↔Metal on
the 1B model, default tolerance 0.5). An adversary may therefore inject
perturbations below the tolerance at every cell. **REPORT.md quantifies
this attack**: the network amplifies sub-tolerance drift ~30–80×, and token
steering is feasible even at τ = 0.01. Two mitigations ship: a per-cell
*mean* deviation bound (`--mean-tolerance`, default 0.05 vs measured honest
≤ 8e-3) that caps the adversary's average budget ~10×, and — the actual fix
— **`--deterministic` (v0.4)**: a fixed-evaluation-order CPU backend whose
transcripts verify with *zero tolerance* (i32 cell equality). Under exact
verification a single-quantum perturbation (2⁻⁸ on one element) is caught,
closing the bounded-drift attack entirely; the trade is speed (~10 tok/s
vs ~80 on Metal for the 1B model). See DECISIONS.md #16.

## The decode proof (v0.3)

`--prove-decode` additionally commits each step's quantized logits with a
**salted Rescue-Prime sponge** (winterfell's Rp64_256 permutation over the
64-bit Goldilocks field; the salt is a private capacity IV) and folds that
digest into the hash chain. Later, `vllm prove-decode` produces a STARK
proving, for the public token index c and chain-bound digest d:

> *"I know a logit vector x and salt s such that RescueSponge_s(x) = d,
> and x[c] ≥ x[i] for all i."*

The AIR re-computes the sponge from the witness inside the proof (one
Rescue round per row, one logit absorbed per row on average) and enforces
argmax via a private claimed-maximum column with 27-bit range-checked
differences. `vllm verify-decode` checks the proof against the transcript
— no model, no trace, no GPU.

Measured (M-series laptop, vocab 128 256): prove **0.7 s**, verify
**0.7 ms**, proof size 78 KB, commitment ~45 ms/token at generation.

**Precise security statement.** This is a succinct, *transparent*
argument (no trusted setup) with ~100-bit conjectured security, and the
commitment is binding. It is **not formal zero-knowledge**: winterfell has
no trace randomization, so each proof reveals evaluations of the trace
polynomials at ~27 out-of-domain coset points — random linear projections
of the (salt ‖ logits) vector. No individual logit can be reconstructed
(the system is underdetermined by five orders of magnitude), and the salt
prevents testing guessed vectors against the *digest*; but a party that
already knows a candidate for the entire logit vector could confirm it
from the openings. **v0.5 closes this** with a halo2 formal-ZK variant (see
below); winterfell-native trace randomization remains a possible future.
Top-p/temperature sampling proofs are also roadmap — both backends are
argmax-only (greedy).

## The formal-ZK decode proof (v0.5, halo2)

`vllm-zk-halo2` proves the *same* statement as v0.3 — the emitted token is a
maximum of the committed logit vector — but with genuine zero-knowledge
instead of a leaky succinct argument. It commits the logits with a salted
**Poseidon hash chain** (`accᵢ₊₁ = Poseidon2(accᵢ, xᵢ)`, `acc₀ = salt`) and
proves argmax in a halo2 circuit: a running one-hot selector pins the public
token index `c` (read from the instance column, so one key serves all tokens)
and range-checked differences enforce `x[c] ≥ x[i]` for every `i`. halo2's
inner-product-argument prover is **transparent** (no trusted setup, like the
STARK) and **blinds its committed witness polynomials**, so — unlike
winterfell — the proof reveals nothing about the logits beyond the public
claim (the token is the argmax).

The trade is proving cost: the commitment is an in-circuit Poseidon over the
whole vocab, O(vocab) rows. Measured (M-series, release):

| vocab | k | prove (cold) | prove (cached) | verify (cached) | proof |
|---|---|---|---|---|---|
| 32  | 12 | 1.3 s  | 0.4 s | 6.7 ms | ~5 KB |
| 128 | 14 | 5.5 s  | 1.6 s | 20 ms  | ~5 KB |
| 256 | 15 | 11 s   | 3.1 s | 37 ms  | ~5 KB |

*Cold* is the stateless API: it regenerates the transparent IPA parameters and
the proving/verifying keys on every call. That setup is deterministic (a pure
function of the vocab) and dominates the cold wall-clock. *Cached* is what a
real deployment pays once those are reused — the proving proper
(`create_proof`) and the proof check (`verify_proof`). The check is
milliseconds, but IPA verification is linear in circuit size (unlike a KZG
SNARK), so it grows with vocab — ≈ seconds at a 128k vocab — while the proof
stays ~5 KB.

A real llama vocab (128 256) needs k≈23 (a 2²³ ≈ 8M-row circuit) — minutes of
proving and multiple GB, the "millions of rows" tradeoff DECISIONS.md #13/#17
anticipates. So the STARK stays the fast default (0.7 s prove, 78 KB) and this
is the opt-in path when *formal* ZK is required.

It is wired into the `vllm` CLI behind the `zk-halo2` build feature (so the
default binary stays free of the heavy halo2 tree): `generate … --prove-decode
--zk-backend halo2`, then `prove-decode … --backend halo2` and `verify-decode`.
Because the salted 32-byte digest each backend folds into the hash chain is
backend-agnostic, this needed **no transcript-format change** — the same
`--trace`/chain machinery carries either a Rescue or a Poseidon digest.

## Workspace

- `vllm-core` — GGUF parsing (std-only), Merkle tree, hash chain,
  transcript/trace/challenge formats. Builds anywhere; a Layer-1 verifier
  and the prover's `respond` need only this.
- `vllm-infer` — candle inference with commitment hooks (`metal` feature
  gated); vendored quantized llama with per-layer hooks (`model.rs`).
  Includes `examples/logit_diff.rs` for backend-drift diagnostics.
- `vllm-verify` — the CPU re-execution verifier for v0.2 challenges.
- `vllm-cli` — the `vllm` binary.
- `vllm-zk` — the Layer-3 STARK (winterfell): salted Rescue-Prime logits
  commitment + argmax AIR. Isolated so its dependencies don't infect the
  other crates (`vllm-infer` sees only a commitment callback).
- `vllm-zk-halo2` — the Layer-3 **formal-ZK** variant (halo2 / transparent
  IPA): a salted Poseidon commitment + argmax circuit whose *blinded* proof
  leaks nothing about the logits. Same `commit / prove / verify` API as
  `vllm-zk`. Heavy proof-system deps, so it is excluded from the default build
  — `cargo test -p vllm-zk-halo2`, or `cargo run --release -p vllm-zk-halo2
  --example bench_argmax_halo2`.

Tests run without downloading weights: the integration test constructs a
tiny random-weight 2-layer llama GGUF in memory (`vllm-core` has a GGUF
writer) and runs the full commit → generate → replay → tamper cycle on CPU.

## Known limitations

- Transcripts are backend-specific: a Metal transcript will not replay
  logits-identically on CPU (see threat model). The emitted *tokens* usually
  agree, but near-ties can flip under greedy decoding.
- The llama3 chat template is hardcoded; `--raw` bypasses it.
- Tracing costs ~7 % of inference time (per-layer GPU→CPU transfers) and
  ~0.5 MB of trace per position for the 1B model; it is opt-in.
- Float (Metal) transcripts are tolerance-verified — see the protocol
  section for the bounded-drift caveat; `--deterministic` transcripts
  verify exactly at ~8x lower generation speed and ~5 GB extra RAM.
- Both decode proofs are argmax (greedy) only. v0.3 (STARK) is a succinct
  argument, not formal ZK; v0.5 (halo2) is formally ZK but O(vocab) to prove.
  See the decode-proof sections for the exact statements and the tradeoff.
- `--prove-decode` costs ~45 ms/token (native Rescue sponge over the 128k
  logit vector); opt-in.

[candle]: https://github.com/huggingface/candle
