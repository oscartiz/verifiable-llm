# Changelog

All notable changes to this project. Versions follow the workspace version;
commitment/transcript formats are versioned by domain string, so a format change
is called out explicitly.

## [0.5.0] — 2026-07-13

The formal-zero-knowledge decode proof, plus a release-hardening pass.

### Added — formal-ZK decode backend (`vllm-zk-halo2`)

A second Layer-3 backend that proves the **same** statement as the v0.3 STARK —
*the emitted greedy token is a maximum of the committed logit vector* — but with
genuine zero-knowledge.

- **Exact statement.** For a public token index `c` and a chain-bound digest `d`,
  the prover knows a logit vector `x` and salt `s` such that
  `d = PoseidonChain_s(x)` (a salted Poseidon hash chain,
  `accᵢ₊₁ = Poseidon2(accᵢ, xᵢ)`, `acc₀ = salt`) **and** `x[c] ≥ x[i]` for every
  `i`. Argmax is enforced by a running one-hot selector — the token index `c` is
  read from the instance column via a copy constraint, so a single verifying key
  serves every token — plus a 27-bit range check on `x[c] − x[i]`.
- **Why it's different from the STARK.** halo2's inner-product-argument prover is
  transparent (no trusted setup, like the STARK) **and** blinds every committed
  witness polynomial, so the proof reveals nothing about the logits beyond the
  public claim. winterfell 0.13 has no trace randomization, so the STARK's
  openings are reconstruction-safe but confirmation-unsafe (see
  [`docs/protocol.md`](docs/protocol.md)).
- **The O(vocab) tradeoff.** The commitment is an in-circuit Poseidon over the
  whole vocab. At vocab 32/128/256 the cached prove is 0.4/1.6/3.0 s and the
  proof stays ~5 KB, but a real llama vocab (128 256) needs k≈23 (~8M-row
  circuit): minutes of proving and multiple GB. So the STARK stays the fast
  default (0.7 s prove, ~78 KB) and halo2 is the opt-in path when *formal* ZK is
  required.
- **No transcript-format change.** The chain folds a salted 32-byte digest per
  step regardless of which hash produced it, so the halo2 Poseidon digest reuses
  the existing Rescue slots. Wired into the CLI behind the `zk-halo2` build
  feature: `generate … --prove-decode --zk-backend halo2`, `prove-decode …
  --backend halo2`, `verify-decode` (backend read from the proof envelope).

### Added — hardening & docs

- `Transcript::validate()`: fail-closed structural checks (non-empty prompt,
  sane trace dimensions) at every command boundary, closing an integer
  underflow on malformed transcript JSON (debug panic / release wraparound).
- `rust-toolchain.toml` pinned to stable 1.95.0 for reproducible builds.
- CI now builds and tests the `zk-halo2` feature (clippy `-D warnings`, the
  halo2 crate, and the CLI end-to-end pipeline) — previously compiled by no CI
  job — and pins both runners to the toolchain.
- Binary-level negative tests: `vllm verify-decode` rejects a forged proof body,
  a proof relabelled to another backend, and an unknown backend tag.
- `measure_costs()` and a reworked `bench_argmax_halo2` example emit the cold and
  cached prove/verify costs separately, so the documented cost table is
  reproducible from one command.
- New docs: [`docs/protocol.md`](docs/protocol.md) (per-layer formal statements),
  [`BENCHMARKS.md`](BENCHMARKS.md) (every number with command + hardware + date),
  [`SECURITY.md`](SECURITY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), a
  reviewer-facing README intro, and [`scripts/demo.sh`](scripts/demo.sh).

### Fixed

- Catch-probability table: the f=1 % / 95 % cell corrected 300 → 299 to match the
  exact `⌈ln δ / ln(1−f)⌉` bound the rest of the table uses.
- Renamed the halo2 bench example (`bench_argmax` → `bench_argmax_halo2`) to
  clear an output-filename collision with the STARK bench on workspace builds.

## [0.4.0] — deterministic exact verification

- `--deterministic`: a fixed-evaluation-order CPU backend (`det-cpu-v1`) whose
  transcripts verify with **zero** tolerance (i32 cell equality), closing the
  bounded-drift attack of [`REPORT.md`](REPORT.md). A single-quantum (2⁻⁸)
  perturbation is caught. ~10 tok/s and ~5 GB RAM for the 1B model; transcripts
  are portable across machines. See DECISIONS.md #16.

## [0.3.0] — three-layer verifiable inference (v0.1–v0.3)

- **Layer 1:** BLAKE3 Merkle model commitment + transcript hash chain.
- **Layer 2:** Fiat–Shamir spot-check challenges with CPU re-execution, plus the
  bounded-drift experiment and the mean-deviation detector.
- **Layer 3:** winterfell STARK proving the greedy decode token is the argmax of
  a salted-Rescue-committed logit vector.

[0.5.0]: https://github.com/oscartiz/verifiable-llm/releases/tag/v0.5.0
[0.4.0]: https://github.com/oscartiz/verifiable-llm/releases/tag/v0.4.0
[0.3.0]: https://github.com/oscartiz/verifiable-llm/releases/tag/v0.3.0
