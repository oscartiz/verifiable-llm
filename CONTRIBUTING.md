# Contributing

Thanks for looking. This is a research / portfolio project, so the bar is
"correct, precise, and honestly scoped" rather than "feature-complete." The most
valuable contributions are **soundness fixes**, **adversarial tests**, and
**precision fixes to the security prose** — not new features.

## Ground rules

- **Never inflate a claim.** This repo's credibility comes from understatement.
  If a security statement in the docs is stronger than what the code
  guarantees, weaken the statement (and flag it) — do not quietly strengthen the
  code to match. Prefer deleting or softening a weak claim over defending it.
- **Attacker-controlled input must fail closed.** Transcript, challenge,
  response, and proof JSON/bytes are adversarial: parsing them must never panic
  or wrap — return a clean error. Add a negative test with any such fix.
- **Every number has a command.** New performance claims go in
  [`BENCHMARKS.md`](BENCHMARKS.md) with the exact command, hardware, and date.
- **Commitment formats are versioned.** Changing a Merkle/chain/trace
  construction breaks existing commitments — bump the domain string (e.g.
  `vllm/merkle-leaf/v1` → `v2`) and the golden-vector test rather than silently
  changing the scheme.

## Developer loop

The workspace is pinned to a specific stable toolchain (`rust-toolchain.toml`),
so `rustup` will use the right compiler automatically.

```sh
# format + lint (CI runs these with -D warnings)
cargo fmt --all
cargo clippy --workspace --all-targets --no-default-features -- -D warnings

# tests, CPU-only (what a verifier machine runs — no weights downloaded)
cargo test --workspace --no-default-features

# the formal-ZK backend and its CLI wiring (heavy halo2 tree, opt-in)
cargo clippy -p vllm-cli --all-targets --no-default-features --features zk-halo2 -- -D warnings
cargo test -p vllm-cli --no-default-features --features zk-halo2
cargo test -p vllm-zk-halo2

# reproducible proof-cost benches (no weights)
cargo run --release -p vllm-zk --example bench_argmax
cargo run --release -p vllm-zk-halo2 --example bench_argmax_halo2
```

Model-dependent work (inference, drift diagnostics) needs a real
Llama-3.2-1B-Instruct Q4_K_M GGUF + tokenizer and, for the Metal diagnostics, an
Apple-Silicon GPU — see [`BENCHMARKS.md`](BENCHMARKS.md) for the commands. CI
never downloads weights; the integration tests build a tiny in-memory GGUF
instead (`vllm-infer/src/testmodel.rs`).

## Commits and PRs

- Conventional commits (`fix:`, `feat:`, `test:`, `docs:`, `ci:`, `build:`), one
  logical change per commit.
- A PR that changes behaviour should say which layer's guarantee it touches and
  include the test that would have caught the bug.
- Green CI is required: fmt, clippy `-D warnings`, and tests across the default,
  `--no-default-features`, and `zk-halo2` configurations.

## Where things live

| crate | role |
|---|---|
| `vllm-core` | GGUF parse, Merkle, hash chain, transcript/trace/challenge formats (std + blake3 only; a verifier needs only this) |
| `vllm-infer` | candle inference with commitment hooks; the deterministic backend |
| `vllm-verify` | CPU re-execution verifier for Layer-2 challenges |
| `vllm-zk` | Layer-3 STARK (winterfell): salted Rescue commitment + argmax AIR |
| `vllm-zk-halo2` | Layer-3 formal-ZK variant (halo2/IPA): salted Poseidon + argmax circuit |
| `vllm-cli` | the `vllm` binary |

The precise per-layer security statements are in
[`docs/protocol.md`](docs/protocol.md); design rationale is in
[`DECISIONS.md`](DECISIONS.md).
