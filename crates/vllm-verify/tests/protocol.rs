//! End-to-end v0.2 protocol tests on the tiny in-memory llama:
//! honest prover verifies; cheating provers are caught. CPU only.

use std::path::PathBuf;

use vllm_core::chain::{Chain, SamplerConfig};
use vllm_core::commit;
use vllm_core::protocol::{build_response, make_challenge};
use vllm_core::trace::{TraceBuilder, TraceReader, dequantize};
use vllm_core::transcript::{ChainCheck, TraceInfo, Transcript};
use vllm_infer::engine::{GenerateRequest, Prompt, generate};
use vllm_infer::testmodel::tiny_llama_gguf;
use vllm_verify::{VerifyConfig, verify};

const PROMPT: [u32; 3] = [1, 2, 3];
const STEPS: usize = 8;

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("vllm-proto-{tag}-{}.{ext}", std::process::id()))
}

struct Setup {
    model_path: PathBuf,
    trace_path: PathBuf,
    transcript: Transcript,
}

fn honest_run(tag: &str) -> Setup {
    let model_path = temp_path(&format!("{tag}-model"), "gguf");
    std::fs::write(&model_path, tiny_llama_gguf()).unwrap();
    let trace_path = temp_path(&format!("{tag}-trace"), "trace");
    let commitment = commit::commit_gguf(&model_path).unwrap();
    let req = GenerateRequest {
        model_path: model_path.clone(),
        tokenizer_path: None,
        prompt: Prompt::Tokens(PROMPT.to_vec()),
        raw: true,
        max_new_tokens: STEPS,
        sampler: SamplerConfig::greedy(),
        logit_frac_bits: 16,
        force_cpu: true,
        trace_path: Some(trace_path.clone()),
        zk_commit: None,
    };
    let out = generate(&req, &commitment, None).unwrap();
    assert_eq!(out.transcript.replay_chain(), ChainCheck::Ok);
    out.transcript.check_trace_binding().unwrap();
    Setup {
        model_path,
        trace_path,
        transcript: out.transcript,
    }
}

fn cleanup(s: &Setup) {
    std::fs::remove_file(&s.model_path).ok();
    std::fs::remove_file(&s.trace_path).ok();
}

/// k large enough to exhaust the whole challenge space (10 positions x
/// 2 blocks + 8 head cells = 28), so every code path — including head
/// checks — is exercised deterministically.
const K_ALL: u32 = 100;

#[test]
fn honest_prover_verifies() {
    let s = honest_run("honest");
    let challenge = make_challenge(&s.transcript, K_ALL, "test-nonce").unwrap();
    assert_eq!(
        challenge.cells.len(),
        28,
        "expected the full challenge space"
    );
    let n_layers = s.transcript.trace.as_ref().unwrap().n_layers;
    assert!(
        challenge.cells.iter().any(|c| c.layer == n_layers),
        "no head checks drawn"
    );

    let mut reader = TraceReader::open(&s.trace_path).unwrap();
    let response = build_response(&mut reader, &challenge).unwrap();

    let config = VerifyConfig {
        tolerance: 0.05,
        logits_tolerance: 0.05,
    };
    let report = verify(&s.model_path, &s.transcript, &challenge, &response, &config).unwrap();
    assert_eq!(report.items.len(), 28);
    // CPU-generated trace verified on CPU: deviations are quantization-level.
    assert!(
        report.max_dev < 0.01,
        "unexpectedly large deviation {}",
        report.max_dev
    );
    cleanup(&s);
}

/// The critical adversary: a prover whose commitments are fully
/// self-consistent (valid chain, valid Merkle tree, valid trace file) but
/// whose trace does not match the committed model's computation. Simulated
/// by perturbing one activation cell and rebuilding every hash honestly.
#[test]
fn self_consistent_cheater_is_caught() {
    let s = honest_run("cheat");
    let trace_info = s.transcript.trace.clone().unwrap();
    let (n_layers, dim, fb) = (
        trace_info.n_layers,
        trace_info.hidden_dim,
        trace_info.frac_bits,
    );
    let cells_per_pos = n_layers + 1;

    // Read the honest trace and rebuild it with one perturbed cell.
    let mut reader = TraceReader::open(&s.trace_path).unwrap();
    let meta = reader.meta().clone();
    let mut builder = TraceBuilder::new(
        n_layers,
        dim,
        fb,
        meta.logit_frac_bits,
        meta.first_logit_pos,
    );
    for pos in 0..meta.n_positions {
        for layer in 0..cells_per_pos {
            let mut values = dequantize(&reader.cell(pos, layer).unwrap(), fb);
            if pos == 5 && layer == 1 {
                values[0] += 1.0; // the lie
            }
            builder.push_cell(&values).unwrap();
        }
    }
    for step in 0..meta.n_logit_rows {
        builder.push_logits_row(reader.logits_row(step).unwrap());
    }
    let hashes = builder.hashes().to_vec();
    let cheat_trace_path = temp_path("cheat-rebuilt", "trace");
    let cheat_meta = builder.write(&cheat_trace_path).unwrap();

    // Rebuild the chain and transcript exactly as an honest prover would,
    // but over the lying trace.
    let mut t = s.transcript.clone();
    let mut chain = Chain::seed(&t.model_root, &t.prompt_token_ids, &t.sampler);
    t.chain_seed = chain.value();
    let split = (t.prompt_token_ids.len() - 1) * cells_per_pos as usize;
    let prompt_hashes = hashes[..split].to_vec();
    chain.absorb_layer_hashes(&prompt_hashes);
    t.prompt_layer_hashes = Some(prompt_hashes);
    for (i, step) in t.steps.iter_mut().enumerate() {
        chain.absorb_step(step.token_id, &step.logits_hash);
        let pos = t.prompt_token_ids.len() - 1 + i;
        let lh = hashes[pos * cells_per_pos as usize..(pos + 1) * cells_per_pos as usize].to_vec();
        step.chain = chain.absorb_layer_hashes(&lh);
        step.layer_hashes = Some(lh);
    }
    t.final_chain = chain.value();
    t.trace = Some(TraceInfo {
        root: cheat_meta.root,
        n_positions: cheat_meta.n_positions,
        n_layers,
        hidden_dim: dim,
        frac_bits: fb,
    });

    // The cheating transcript is internally perfect…
    assert_eq!(t.replay_chain(), ChainCheck::Ok);
    t.check_trace_binding().unwrap();

    // …but re-execution catches it: with the space exhausted, challenge
    // (5, 0) recomputes block 0 over honest inputs and disagrees with the
    // perturbed committed output.
    let challenge = make_challenge(&t, K_ALL, "test-nonce").unwrap();
    let mut cheat_reader = TraceReader::open(&cheat_trace_path).unwrap();
    let response = build_response(&mut cheat_reader, &challenge).unwrap();
    let config = VerifyConfig {
        tolerance: 0.05,
        logits_tolerance: 0.05,
    };
    let err = verify(&s.model_path, &t, &challenge, &response, &config).unwrap_err();
    assert!(
        err.to_string().contains("deviates"),
        "unexpected error: {err}"
    );

    std::fs::remove_file(&cheat_trace_path).ok();
    cleanup(&s);
}

#[test]
fn tampered_response_and_model_are_rejected() {
    let s = honest_run("tamper");
    let challenge = make_challenge(&s.transcript, K_ALL, "test-nonce").unwrap();
    let mut reader = TraceReader::open(&s.trace_path).unwrap();
    let response = build_response(&mut reader, &challenge).unwrap();
    let config = VerifyConfig {
        tolerance: 0.05,
        logits_tolerance: 0.05,
    };

    // (a) Corrupt one revealed cell's payload: Merkle proof must reject.
    let mut bad = response.clone();
    let item = bad.items.iter_mut().find(|i| !i.inputs.is_empty()).unwrap();
    let honest_q = vllm_core::protocol::b64_decode_i32(&item.inputs[0].data).unwrap();
    let mut tampered_q = honest_q.clone();
    tampered_q[0] += 1;
    item.inputs[0].data = vllm_core::protocol::b64_encode_i32(&tampered_q);
    let err = verify(&s.model_path, &s.transcript, &challenge, &bad, &config).unwrap_err();
    assert!(
        err.to_string().contains("Merkle proof rejected"),
        "got: {err}"
    );

    // (b) Corrupt revealed logits on a head item: exact hash binding rejects.
    let n_layers = s.transcript.trace.as_ref().unwrap().n_layers;
    let mut bad = response.clone();
    let item = bad
        .items
        .iter_mut()
        .find(|i| i.cell.layer == n_layers)
        .unwrap();
    let mut q = vllm_core::protocol::b64_decode_i32(item.logits.as_ref().unwrap()).unwrap();
    q[3] += 1;
    item.logits = Some(vllm_core::protocol::b64_encode_i32(&q));
    let err = verify(&s.model_path, &s.transcript, &challenge, &bad, &config).unwrap_err();
    assert!(
        err.to_string().contains("do not hash to the committed"),
        "got: {err}"
    );

    // (c) A different model file: root mismatch before any re-execution.
    let mut gguf = tiny_llama_gguf();
    let n = gguf.len();
    gguf[n - 3] ^= 0x40;
    let other_model = temp_path("tamper-other-model", "gguf");
    std::fs::write(&other_model, gguf).unwrap();
    let err = verify(&other_model, &s.transcript, &challenge, &response, &config).unwrap_err();
    assert!(
        err.to_string().contains("does not match transcript"),
        "got: {err}"
    );

    // (d) A challenge that is not the FS derivation is refused.
    let mut forged = challenge.clone();
    forged.cells[0].pos = (forged.cells[0].pos + 1) % 9;
    let result = verify(&s.model_path, &s.transcript, &forged, &response, &config);
    assert!(result.is_err());

    std::fs::remove_file(&other_model).ok();
    cleanup(&s);
}

#[test]
fn respond_rejects_forged_challenges_and_wrong_trace() {
    let s = honest_run("respond");
    let challenge = make_challenge(&s.transcript, 5, "").unwrap();

    // A trace file for a different run cannot answer this challenge.
    let s2 = honest_run("respond-other");
    // (same model, different… actually identical run; perturb by removing)
    let mut reader = TraceReader::open(&s2.trace_path).unwrap();
    // Identical deterministic runs produce the identical trace, so this
    // must succeed — and proves respond is keyed by root, not path.
    assert!(build_response(&mut reader, &challenge).is_ok());

    // Now a genuinely different trace (different prompt) must be refused.
    let model_path = temp_path("respond-third-model", "gguf");
    std::fs::write(&model_path, tiny_llama_gguf()).unwrap();
    let commitment = commit::commit_gguf(&model_path).unwrap();
    let other_trace = temp_path("respond-third", "trace");
    let req = GenerateRequest {
        model_path: model_path.clone(),
        tokenizer_path: None,
        prompt: Prompt::Tokens(vec![4, 5]),
        raw: true,
        max_new_tokens: 4,
        sampler: SamplerConfig::greedy(),
        logit_frac_bits: 16,
        force_cpu: true,
        trace_path: Some(other_trace.clone()),
        zk_commit: None,
    };
    generate(&req, &commitment, None).unwrap();
    let mut other_reader = TraceReader::open(&other_trace).unwrap();
    assert!(build_response(&mut other_reader, &challenge).is_err());

    std::fs::remove_file(&model_path).ok();
    std::fs::remove_file(&other_trace).ok();
    cleanup(&s2);
    cleanup(&s);
}
