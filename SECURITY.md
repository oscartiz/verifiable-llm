# Security policy

## Status: research code, not audited

`verifiable-llm` is a research / portfolio project. It implements cryptographic
commitments and zero-knowledge-flavoured arguments, but **it has not been
externally audited**, and it should not be relied on to protect anything of
value without independent review. The proof systems it builds on (winterfell,
halo2) carry their own conjectured-security caveats, which this project inherits
and documents rather than hides.

The project's design deliberately states its own limits — please read them
before trusting a guarantee:

- The **exact security statement for each layer** is in
  [`docs/protocol.md`](docs/protocol.md): what is bound, what is proven, and the
  precise adversary each layer does and does **not** defeat.
- The **STARK decode proof is not formal zero-knowledge** (winterfell 0.13 has
  no trace randomization); it is reconstruction-safe but confirmation-unsafe.
  The halo2 backend (`--features zk-halo2`) is the formally-ZK alternative.
- The **float spot-check path is tolerance-based** and admits a bounded-drift
  attack quantified in [`REPORT.md`](REPORT.md); exact token provenance requires
  `--deterministic` (DECISIONS.md #16).
- **Fiat–Shamir grinding** is possible with an empty challenge nonce; use a
  verifier-chosen nonce issued after the transcript is committed.

## Reporting a vulnerability

If you find a soundness bug — a way for a cheating prover to pass verification, a
commitment that isn't binding, a panic or non-fail-closed path reachable from
attacker-controlled input (transcript / challenge / response / proof JSON), or a
claim in the docs stronger than the code delivers — please report it.

- **Open a GitHub issue** for anything already public or low-risk, or
- **Email the maintainer** (see the repository owner's public profile) for a
  soundness break you'd rather disclose privately first.

Please include: the affected layer/crate, a minimal reproduction (a failing
test or a crafted input file is ideal), and what guarantee you believe is
violated. There is no bounty — this is a portfolio project — but soundness
reports are genuinely welcome and will be credited.

## Scope

In scope: the commitment/chain/Merkle constructions (`vllm-core`), the
spot-check protocol and verifier (`vllm-verify`), the deterministic backend
(`vllm-infer`), and both decode-proof circuits (`vllm-zk`, `vllm-zk-halo2`) —
especially **fail-closed parsing** of attacker-controlled JSON and proof bytes.

Out of scope: the underlying proof-system libraries' internals, candle/Metal
numerical behaviour beyond what the tolerance/determinism design already
addresses, and the roadmap items (top-p/temperature proofs, winterfell trace
randomization) that are documented as not-yet-built.
