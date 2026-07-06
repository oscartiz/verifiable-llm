//! End-to-end integration test on a tiny random-weight 2-layer llama,
//! constructed in-test as a GGUF file — CI never downloads real weights.
//! Runs on CPU (no metal feature required).

use std::path::{Path, PathBuf};

use vllm_core::chain::{SamplerConfig, SamplerMode};
use vllm_core::commit;
use vllm_core::transcript::ChainCheck;
use vllm_infer::engine::{GenerateRequest, Prompt, generate};
use vllm_infer::testmodel::{LAYERS, VOCAB, tiny_llama_gguf};

fn temp_model(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("vllm-tiny-{tag}-{}.gguf", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn request(path: &Path, sampler: SamplerConfig) -> GenerateRequest {
    GenerateRequest {
        model_path: path.to_path_buf(),
        tokenizer_path: None,
        prompt: Prompt::Tokens(vec![1, 2, 3]),
        raw: true,
        max_new_tokens: 8,
        sampler,
        logit_frac_bits: 16,
        force_cpu: true,
        trace_path: None,
        zk_commit: None,
        deterministic: false,
    }
}

#[test]
fn tiny_model_end_to_end() {
    let bytes = tiny_llama_gguf();
    let path = temp_model("e2e", &bytes);

    let commitment = commit::commit_gguf(&path).unwrap();
    assert_eq!(commitment.tensors.len(), 3 + 9 * LAYERS as usize);

    // Greedy generation with commitments.
    let out = generate(&request(&path, SamplerConfig::greedy()), &commitment, None).unwrap();
    assert_eq!(out.transcript.steps.len(), 8);
    assert_eq!(out.transcript.vocab_size, VOCAB as u32);
    assert!(out.transcript.token_ids().all(|t| t < VOCAB as u32));
    assert_eq!(out.transcript.replay_chain(), ChainCheck::Ok);

    // Reproducible: a rerun yields the identical chain.
    let again = generate(&request(&path, SamplerConfig::greedy()), &commitment, None).unwrap();
    assert_eq!(again.transcript.final_chain, out.transcript.final_chain);

    // A different sampler config changes the chain from the seed onward.
    let topp = SamplerConfig {
        mode: SamplerMode::TopP,
        temperature: 0.8,
        top_p: 0.9,
        rng_seed: 7,
    };
    let sampled = generate(&request(&path, topp), &commitment, None).unwrap();
    assert_ne!(sampled.transcript.chain_seed, out.transcript.chain_seed);
    assert_eq!(sampled.transcript.replay_chain(), ChainCheck::Ok);

    // Tampering with one weight byte changes the model root, and a transcript
    // bound to the tampered model no longer replays against the honest root.
    let mut tampered_bytes = tiny_llama_gguf();
    let n = tampered_bytes.len();
    tampered_bytes[n - 3] ^= 0x40; // exponent bit of some final-tensor weight
    let tampered_path = temp_model("tampered", &tampered_bytes);
    let tampered_commitment = commit::commit_gguf(&tampered_path).unwrap();
    assert_ne!(tampered_commitment.root, commitment.root);

    let mut cheat = generate(
        &request(&tampered_path, SamplerConfig::greedy()),
        &tampered_commitment,
        None,
    )
    .unwrap()
    .transcript;
    cheat.model_root = commitment.root; // claim it ran on the honest model
    assert!(matches!(cheat.replay_chain(), ChainCheck::BadSeed { .. }));

    std::fs::remove_file(path).ok();
    std::fs::remove_file(tampered_path).ok();
}

/// The deterministic backend: bit-identical across runs, close to the float
/// path numerically, and traced at its own (coarser, roundtrip-exact) cell
/// precision.
#[test]
fn deterministic_backend_is_deterministic_and_close_to_float() {
    let bytes = tiny_llama_gguf();
    let path = temp_model("det", &bytes);
    let commitment = commit::commit_gguf(&path).unwrap();

    let mut det_req = request(&path, SamplerConfig::greedy());
    det_req.deterministic = true;
    det_req.trace_path = Some(path.with_extension("det.trace"));
    let a = generate(&det_req, &commitment, None).unwrap();

    let mut det_req2 = request(&path, SamplerConfig::greedy());
    det_req2.deterministic = true;
    det_req2.trace_path = Some(path.with_extension("det2.trace"));
    let b = generate(&det_req2, &commitment, None).unwrap();

    // Bit-identical: same final chain (covers every cell hash, every logit).
    assert_eq!(a.transcript.final_chain, b.transcript.final_chain);
    assert_eq!(a.transcript.env.backend, "det-cpu-v1");
    assert_eq!(a.transcript.trace.as_ref().unwrap().frac_bits, 8);
    assert_eq!(a.transcript.replay_chain(), ChainCheck::Ok);
    a.transcript.check_trace_binding().unwrap();

    // Numerically close to the float path (requantization at 2^-8 hooks
    // perturbs mildly; the tiny model's logits are O(1)).
    let float = generate(&request(&path, SamplerConfig::greedy()), &commitment, None).unwrap();
    // Compare step-0 committed logit hashes indirectly via token choice and
    // magnitude: the first greedy token should agree.
    assert_eq!(
        a.transcript.steps[0].token_id, float.transcript.steps[0].token_id,
        "det and float paths disagree on the very first greedy token"
    );

    std::fs::remove_file(path.with_extension("det.trace")).ok();
    std::fs::remove_file(path.with_extension("det2.trace")).ok();
    std::fs::remove_file(path).ok();
}
