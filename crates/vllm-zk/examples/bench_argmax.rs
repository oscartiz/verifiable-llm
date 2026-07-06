//! Benchmark: prove/verify argmax over a llama-sized (128256) logit vector.
//! Usage: cargo run --release -p vllm-zk --example bench_argmax

use std::time::Instant;

fn main() {
    let vocab = 128_256usize;
    let mut q: Vec<i32> = (0..vocab)
        .map(|i| (((i as f64 * 0.37).sin() * 25.0 - 8.0) * 65536.0) as i32)
        .collect();
    let token = 4242u32;
    let max = *q.iter().max().unwrap();
    q[token as usize] = max + 65536; // clear argmax at `token`

    let salt = vllm_zk::random_salt();

    let t0 = Instant::now();
    let digest = vllm_zk::commit_logits(&q, salt).unwrap();
    println!(
        "commit (native sponge over {vocab} logits): {:.2?}",
        t0.elapsed()
    );

    let t0 = Instant::now();
    let proof = vllm_zk::prove_argmax(&q, salt, token).unwrap();
    println!(
        "prove:  {:.2?} ({} KB proof)",
        t0.elapsed(),
        proof.len() / 1024
    );

    let t0 = Instant::now();
    vllm_zk::verify_argmax(&digest, token, vocab as u32, &proof).unwrap();
    println!("verify: {:.2?}", t0.elapsed());
}
