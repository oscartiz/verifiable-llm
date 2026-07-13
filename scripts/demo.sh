#!/usr/bin/env bash
# 60–90 s end-to-end demo of the full protocol, for an asciinema/GIF cast:
#
#   commit -> generate -> replay -> challenge -> respond -> verify
#          -> prove-decode -> verify-decode
#
# Needs a real Llama-3.2-1B-Instruct Q4_K_M GGUF + tokenizer (the tiny in-memory
# test model has no tokenizer, so the CLI can't drive it). To record:
#
#   asciinema rec docs/demo.cast -c "scripts/demo.sh /path/model.gguf /path/tokenizer.json"
#
# Usage: scripts/demo.sh <model.gguf> <tokenizer.json> [--deterministic]
set -euo pipefail

MODEL="${1:?usage: demo.sh <model.gguf> <tokenizer.json> [--deterministic]}"
TOK="${2:?usage: demo.sh <model.gguf> <tokenizer.json> [--deterministic]}"
DET="${3:-}"                      # pass --deterministic for exact (v0.4) verification
PROMPT="In one sentence, what is a Merkle tree?"
WORK="$(mktemp -d)"
RUN="$WORK/run.json"; TRACE="$WORK/run.trace"
CHAL="$WORK/challenge.json"; RESP="$WORK/response.json"; PROOF="$WORK/step.proof.json"

say() { printf '\n\033[1;36m$ %s\033[0m\n' "$*"; }
build() { cargo build --release --quiet "$@"; }

build            # Metal by default; add --no-default-features for CPU-only
VLLM=(./target/release/vllm)

say "vllm commit --model \$MODEL"
"${VLLM[@]}" commit --model "$MODEL"

say "vllm generate --greedy --max-tokens 60 --trace --prove-decode ${DET}"
"${VLLM[@]}" generate --model "$MODEL" --tokenizer "$TOK" --prompt "$PROMPT" \
    --greedy --max-tokens 60 --commit "$RUN" --trace "$TRACE" --prove-decode --bench $DET

say "vllm replay --commitment run.json      # chain check, no model/GPU"
"${VLLM[@]}" replay --commitment "$RUN"

say "vllm challenge -k 40 --nonce \$(uuidgen)   # verifier picks a fresh nonce"
"${VLLM[@]}" challenge --commitment "$RUN" -k 40 --nonce "$(uuidgen)" --out "$CHAL"

say "vllm respond   --trace run.trace           # prover answers"
"${VLLM[@]}" respond --commitment "$RUN" --trace "$TRACE" --challenge "$CHAL" --out "$RESP"

say "vllm verify    --model \$MODEL              # re-executes challenged blocks on CPU"
"${VLLM[@]}" verify --commitment "$RUN" --model "$MODEL" --challenge "$CHAL" --response "$RESP"

say "vllm prove-decode --step 5                 # STARK: token 5 is the argmax"
"${VLLM[@]}" prove-decode --commitment "$RUN" --trace "$TRACE" --step 5 --out "$PROOF"

say "vllm verify-decode --proof step.proof.json # no model, no trace, no GPU"
"${VLLM[@]}" verify-decode --commitment "$RUN" --proof "$PROOF"

printf '\n\033[1;32mDone — model pinned, transcript bound, challenges + decode proof verified.\033[0m\n'
rm -rf "$WORK"
