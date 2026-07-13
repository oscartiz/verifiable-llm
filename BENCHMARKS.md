# Benchmarks

Every performance number quoted in the README, REPORT, and DECISIONS, with the
**exact command**, the **hardware**, and the **date** it was measured — so a
reader can reproduce it (or see why they can't without the weights).

Two classes of number:

- **Layer-3 proof costs** run on synthetic logits and need no model download, so
  they are reproduced here on the reference machine and re-run on every commit
  (the CI badge builds both benches).
- **Model-dependent numbers** (tokens/sec, commitment overhead, cross-backend
  drift, deterministic-mode speed, RAM) need the real Llama-3.2-1B-Instruct
  Q4_K_M GGUF (~770 MB) and, for the drift/logit-diff diagnostics, a Metal GPU.
  They are **not** reproduced in CI (which never downloads weights) — the command
  is given so you can run them yourself; the quoted values are the author's
  measurements on the machine noted.

## Reference machine

| | |
|---|---|
| Machine | MacBook Air, Apple M4, 16 GB |
| OS | macOS 26.5.1 (build 25F80) |
| Toolchain | rustc 1.95.0 (pinned in `rust-toolchain.toml`) |
| Build | `--release` (`lto = "thin"`) |
| Date | 2026-07-13 |

The README's headline numbers were originally taken on the author's M-series
laptop; where the chip differs from this M4, expect ±30 % on wall-clock. The
*structural* results (proof size, argmax bounds, catch probabilities) are
hardware-independent.

---

## Layer 3 — reproduced here (no weights needed)

### STARK argmax proof (winterfell), vocab 128 256

```sh
cargo run --release -p vllm-zk --example bench_argmax
```

| quantity | this M4 | README/DECISIONS #13 |
|---|---|---|
| native Rescue commit | 78–83 ms | "~45 ms/token" (author's machine) |
| prove | 0.69–0.76 s | "0.7 s" ✅ |
| verify | 0.59–0.63 ms | "0.7 ms" ✅ |
| proof size | 76–79 KiB | "78 KB" ✅ (see note) |

**Proof-size note.** The size varies ~±2 KiB run-to-run because the FRI query
positions (Fiat–Shamir–derived) change which Merkle authentication paths are
opened and how much they share prefixes. "78 KB" (decimal) ≈ 76 KiB and is
representative, not a fixed constant. The commit time is the one number that is
meaningfully hardware-bound here — this M4 measures ~80 ms/token for the native
sponge over 128 256 logits, slower than the author's "~45 ms"; both are
generation-time, opt-in, and dwarfed by the proof.

### halo2 formal-ZK argmax proof, vocab 32 / 128 / 256

```sh
cargo run --release -p vllm-zk-halo2 --example bench_argmax_halo2
```

Columns match the README/DECISIONS #17 table 1:1 (the bench times setup,
`create_proof`, and `verify_proof` separately — see `measure_costs`):

| vocab | k | prove (cold) | prove (cached) | verify (cached) | proof | README |
|---|---|---|---|---|---|---|
| 32  | 12 | 1.30 s  | 0.42 s | 17.5 ms | 5 KB | 1.3 s / 0.4 s / 6.7 ms |
| 128 | 14 | 5.46 s  | 1.50 s | 19.6 ms | 5 KB | 5.5 s / 1.6 s / 20 ms |
| 256 | 15 | 10.96 s | 2.96 s | 38.1 ms | 5 KB | 11 s / 3.1 s / 37 ms |

- **Cold** = the stateless `prove_argmax`/`verify_argmax` API, which regenerates
  the deterministic IPA parameters + keys every call (setup dominates cold
  wall-clock).
- **Cached** = the per-proof cost once those are reused: `create_proof` and
  `verify_proof` alone. IPA verification is linear in circuit size, so
  `verify (cached)` grows with vocab while the ~5 KB proof stays flat.
- The vocab-32 `verify (cached)` here (17.5 ms) runs higher than the README's
  6.7 ms — small-circuit verify is noisy and machine-sensitive; the 128/256
  rows match closely. A real llama vocab (128 256) needs k≈23 (~8M-row circuit):
  minutes of proving, multiple GB — the reason the STARK stays the fast default.

---

## Model-dependent — command given, values are the author's

Set up once:

```sh
MODEL=path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf
TOK=path/to/tokenizer.json
```

### Commitment overhead and throughput (DECISIONS #4)

```sh
vllm generate --model $MODEL --tokenizer $TOK --greedy --max-tokens 100 \
    --prompt "In one sentence, what is a Merkle tree?" --bench
```

| quantity | value | condition |
|---|---|---|
| decode throughput | ~79 tok/s | warm-cache (Metal) |
| decode throughput | ~22 tok/s | cold run |
| commitment overhead | 2.6 % | warm-cache, i32 @ 2⁻¹⁶ |
| commitment overhead | 0.4 % | cold run |

### Cross-backend logit drift (DECISIONS #4, #11)

```sh
cargo run --release -p vllm-infer --features metal --example logit_diff -- \
    $MODEL $TOK "In one sentence, what is a Merkle tree?"
```

Prints `max|Δ|`, `mean|Δ|`, and the CPU/Metal argmax. Author: max |Δ| ≈ 0.49,
mean |Δ| ≈ 0.08 over the 128 256-dim vector; argmax agrees, hash equality fails
at every precision (why Layer 2 is tolerance-based off the deterministic path).

### Bounded-drift attack (REPORT.md)

```sh
cargo run --release -p vllm-infer --features metal --example drift_attack -- \
    $MODEL $TOK
```

Reproduces all three REPORT.md tables: ~30–80× per-layer amplification of a
single ±τ injection; token divergence within ~10 steps under random full-stack
±τ even at τ = 0.01; and the targeted last-layer attack flipping 24/48 steps at
τ = 0.01. See REPORT.md for the numbers and interpretation.

### Spot-check verification cost (README)

```sh
vllm challenge --commitment run.json -k 60 --nonce "$(uuidgen)" --out c.json
vllm respond   --commitment run.json --trace run.trace --challenge c.json --out r.json
vllm verify    --commitment run.json --model $MODEL --challenge c.json --response r.json
```

Author: 60 challenges over a 60-position Llama-3.2-1B trace verify in ~2 s on
CPU (≈ k/L prompt-forward equivalents).

### Deterministic backend (DECISIONS #16)

```sh
vllm generate --model $MODEL --tokenizer $TOK --greedy --max-tokens 100 \
    --deterministic --commit run.json --trace run.trace --bench
```

Author, Llama-3.2-1B Q4_K_M: 10.2 tok/s decode (vs ~80 on Metal), model load
2.2 s, ~5 GB resident (weights dequantized to f32 up front), reruns
bit-identical, 20 real-model challenges verified with **0** deviation in 4.9 s.
