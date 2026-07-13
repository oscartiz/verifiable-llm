//! Prove/verify/size benchmark for the formally-ZK halo2 argmax proof across
//! a few vocab sizes, with a projection to a real llama vocab.
//!
//!   cargo run --release -p vllm-zk-halo2 --example bench_argmax_halo2
//!
//! Columns match the README/DECISIONS "cold vs cached" table 1:1, so those
//! numbers are reproducible from this one command rather than asserted:
//!
//! - prove (cold): the stateless API — regenerate IPA params + keys, then
//!   prove. Setup is deterministic (a pure function of the vocab) and
//!   dominates the cold wall-clock.
//! - prove (cached): `create_proof` only, params + pk reused.
//! - verify (cached): `verify_proof` only, params + vk reused. IPA
//!   verification is linear in circuit size (unlike a KZG SNARK), so it grows
//!   with vocab while the proof stays ~flat.
//!
//! The halo2 proof trades speed for *formal* zero-knowledge; contrast the
//! sibling STARK bench (`cargo run --release -p vllm-zk --example bench_argmax`,
//! ~0.7 s prove at vocab 128k, but not formal ZK).

use vllm_zk_halo2::measure_costs;

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
        "{:>7} | {:>4} | {:>12} | {:>13} | {:>13} | {:>8}",
        "vocab", "k", "prove (cold)", "prove (cached)", "verify (cached)", "proof"
    );
    println!("{}", "-".repeat(72));

    for &vocab in &[32usize, 128, 256] {
        let token = vocab / 3;
        let logits = synthetic_logits(vocab, token);
        let c = measure_costs(&logits, token as u32).expect("measure");
        println!(
            "{:>7} | {:>4} | {:>12} | {:>13} | {:>13} | {:>5} KB",
            c.vocab,
            c.k,
            format!("{:.2?}", c.prove_cold()),
            format!("{:.2?}", c.prove_cached),
            format!("{:.2?}", c.verify_cached),
            c.proof_len / 1024
        );
    }

    println!(
        "\nCold = the stateless API (regenerates the deterministic, cacheable IPA\n\
         params + keys each call); cached = what a deployment that reuses them\n\
         pays. The ~5 KB proof stays ~flat while the circuit grows ~linearly\n\
         (Poseidon hash chain + range checks): a real llama vocab (128 256)\n\
         needs k≈23 (~8M-row circuit) — minutes of proving and multiple GB, the\n\
         tradeoff DECISIONS.md #13/#17 anticipates for *formal* ZK vs the\n\
         STARK's 0.7 s succinct argument."
    );
}
