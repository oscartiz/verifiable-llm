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
