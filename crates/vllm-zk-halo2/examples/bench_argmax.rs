//! Prove/verify/size benchmark for the formally-ZK halo2 argmax proof across
//! a few vocab sizes, with a projection to a real llama vocab.
//!
//!   cargo run --release -p vllm-zk-halo2 --example bench_argmax
//!
//! The halo2 proof trades speed for *formal* zero-knowledge; contrast the
//! sibling STARK bench (`cargo run --release -p vllm-zk --example bench_argmax`,
//! ~0.7 s prove at vocab 128k, but not formal ZK).

use std::time::Instant;

use vllm_zk_halo2::{commit_logits, prove_argmax, random_salt, verify_argmax};

fn synthetic_logits(vocab: usize, argmax_at: usize) -> Vec<i32> {
    let mut v: Vec<i32> = (0..vocab)
        .map(|i| ((i as f32 * 0.37).sin() * 3000.0) as i32)
        .collect();
    let max = *v.iter().max().unwrap();
    v[argmax_at] = max + 500;
    v
}

fn main() {
    println!(
        "{:>7} | {:>6} | {:>10} | {:>10} | {:>10}",
        "vocab", "k", "prove", "verify", "proof"
    );
    println!("{}", "-".repeat(56));

    for &vocab in &[32usize, 128, 256] {
        let token = vocab / 3;
        let logits = synthetic_logits(vocab, token);
        let salt = random_salt();

        let digest = commit_logits(&logits, &salt).expect("commit");

        let t0 = Instant::now();
        let proof = prove_argmax(&logits, &salt, token as u32).expect("prove");
        let prove_t = t0.elapsed();

        let t1 = Instant::now();
        verify_argmax(&digest, token as u32, vocab as u32, &proof).expect("verify");
        let verify_t = t1.elapsed();

        // k mirrors the library's own sizing (64*vocab + 256 rows).
        let k = (64u64 * vocab as u64 + 256)
            .next_power_of_two()
            .trailing_zeros();
        println!(
            "{vocab:>7} | {k:>6} | {:>10} | {:>10} | {:>7} KB",
            format!("{prove_t:.2?}"),
            format!("{verify_t:.2?}"),
            proof.len() / 1024
        );
    }

    println!(
        "\nThe prove/verify columns are cold wall-clock: both regenerate the\n\
         deterministic IPA parameters and keys for the vocab (a pure function\n\
         of it, hence cacheable). With those cached the real costs are the\n\
         proving proper and the proof *check* — the check is milliseconds here\n\
         (IPA verification is linear in circuit size, so it grows with vocab).\n\
         The ~5 KB proof stays ~flat while the circuit grows ~linearly\n\
         (Poseidon hash chain + range checks): a real llama vocab (128 256)\n\
         needs k≈23 (~8M-row circuit) — minutes of proving and multiple GB, the\n\
         tradeoff DECISIONS.md #13/#17 anticipates for *formal* ZK vs the\n\
         STARK's 0.7 s succinct argument."
    );
}
