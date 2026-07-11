//! End-to-end Layer-3 test on the tiny model: generate with salted zk
//! commitments, prove one decode step, verify it, and check the cheats.

use std::path::PathBuf;

use vllm_core::chain::SamplerConfig;
use vllm_core::commit;
use vllm_core::trace::TraceReader;
use vllm_core::transcript::ChainCheck;
use vllm_infer::engine::{GenerateRequest, Prompt, ZkCommitment, generate};
use vllm_infer::testmodel::tiny_llama_gguf;

fn temp(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("vllm-zkpipe-{tag}-{}.{ext}", std::process::id()))
}

#[test]
fn zk_commit_prove_verify_pipeline() {
    let model_path = temp("model", "gguf");
    std::fs::write(&model_path, tiny_llama_gguf()).unwrap();
    let trace_path = temp("trace", "trace");
    let commitment = commit::commit_gguf(&model_path).unwrap();

    let req = GenerateRequest {
        model_path: model_path.clone(),
        tokenizer_path: None,
        prompt: Prompt::Tokens(vec![1, 2, 3]),
        raw: true,
        max_new_tokens: 6,
        sampler: SamplerConfig::greedy(),
        logit_frac_bits: 16,
        force_cpu: true,
        trace_path: Some(trace_path.clone()),
        zk_commit: Some(Box::new(|q: &[i32]| {
            let salt = vllm_zk::random_salt();
            let digest = vllm_zk::commit_logits(q, salt).map_err(|e| e.to_string())?;
            Ok(ZkCommitment { salt, digest })
        })),
        deterministic: false,
    };
    let out = generate(&req, &commitment, None).unwrap();
    let t = out.transcript;

    // Chain replays with the zk digests folded in, and every step has one.
    assert_eq!(t.replay_chain(), ChainCheck::Ok);
    t.check_trace_binding().unwrap();
    assert!(t.steps.iter().all(|s| s.zk_digest.is_some()));

    // Tampering with a zk digest breaks the chain.
    let mut bad = t.clone();
    bad.steps[2].zk_digest.as_mut().unwrap().0[0] ^= 1;
    assert!(matches!(
        bad.replay_chain(),
        ChainCheck::BadStep { index: 2, .. }
    ));

    // Prove a step from the stored trace and verify against the transcript.
    let mut reader = TraceReader::open(&trace_path).unwrap();
    let salts = reader.meta().zk_salts.clone().expect("salts stored");
    assert_eq!(salts.len(), t.steps.len());
    let step = 3usize;
    let logits = reader.logits_row(step as u32).unwrap();
    let digest = vllm_zk::commit_logits(&logits, salts[step]).unwrap();
    assert_eq!(
        digest,
        t.steps[step].zk_digest.unwrap().0,
        "chain-bound digest matches trace"
    );

    let token = t.steps[step].token_id;
    let proof = vllm_zk::prove_argmax(&logits, salts[step], token).unwrap();
    vllm_zk::verify_argmax(&digest, token, t.vocab_size, &proof).unwrap();

    // The same proof does not verify for another step's digest.
    let other = t.steps[step + 1].zk_digest.unwrap().0;
    if other != digest {
        assert!(vllm_zk::verify_argmax(&other, token, t.vocab_size, &proof).is_err());
    }

    std::fs::remove_file(&model_path).ok();
    std::fs::remove_file(&trace_path).ok();
}

// The 32-byte zk-digest and salt slots the engine/trace expose are backend
// agnostic; the halo2 backend stores its 32-byte salt in the `[u64; 4]` slot.
#[cfg(feature = "zk-halo2")]
fn salt_u64_to_bytes(s: [u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in s.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[cfg(feature = "zk-halo2")]
fn salt_bytes_to_u64(b: [u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (i, w) in out.iter_mut().enumerate() {
        *w = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    out
}

/// The same pipeline through the formal-ZK halo2 backend (tiny vocab keeps
/// proving fast). Exercises exactly what the CLI's `--zk-backend halo2` /
/// `prove-decode --backend halo2` path does end to end.
#[cfg(feature = "zk-halo2")]
#[test]
fn zk_halo2_commit_prove_verify_pipeline() {
    let model_path = temp("halo2-model", "gguf");
    std::fs::write(&model_path, tiny_llama_gguf()).unwrap();
    let trace_path = temp("halo2-trace", "trace");
    let commitment = commit::commit_gguf(&model_path).unwrap();

    let req = GenerateRequest {
        model_path: model_path.clone(),
        tokenizer_path: None,
        prompt: Prompt::Tokens(vec![1, 2, 3]),
        raw: true,
        max_new_tokens: 6,
        sampler: SamplerConfig::greedy(),
        logit_frac_bits: 16,
        force_cpu: true,
        trace_path: Some(trace_path.clone()),
        zk_commit: Some(Box::new(|q: &[i32]| {
            let salt = vllm_zk_halo2::random_salt();
            let digest = vllm_zk_halo2::commit_logits(q, &salt).map_err(|e| e.to_string())?;
            Ok(ZkCommitment {
                salt: salt_bytes_to_u64(salt),
                digest,
            })
        })),
        deterministic: false,
    };
    let out = generate(&req, &commitment, None).unwrap();
    let t = out.transcript;

    // The Poseidon digests fold into the same chain and bind the trace.
    assert_eq!(t.replay_chain(), ChainCheck::Ok);
    t.check_trace_binding().unwrap();
    assert!(t.steps.iter().all(|s| s.zk_digest.is_some()));

    let mut reader = TraceReader::open(&trace_path).unwrap();
    let salts = reader.meta().zk_salts.clone().expect("salts stored");
    let step = 3usize;
    let logits = reader.logits_row(step as u32).unwrap();
    let salt = salt_u64_to_bytes(salts[step]);

    let digest = vllm_zk_halo2::commit_logits(&logits, &salt).unwrap();
    assert_eq!(
        digest,
        t.steps[step].zk_digest.unwrap().0,
        "chain-bound Poseidon digest matches the trace logits"
    );

    let token = t.steps[step].token_id;
    let proof = vllm_zk_halo2::prove_argmax(&logits, &salt, token).unwrap();
    vllm_zk_halo2::verify_argmax(&digest, token, t.vocab_size, &proof).unwrap();

    // A wrong committed digest is rejected.
    let mut tampered = digest;
    tampered[0] ^= 1;
    assert!(vllm_zk_halo2::verify_argmax(&tampered, token, t.vocab_size, &proof).is_err());

    std::fs::remove_file(&model_path).ok();
    std::fs::remove_file(&trace_path).ok();
}
