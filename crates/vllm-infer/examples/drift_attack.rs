//! Adversarial evaluation of the v0.2 tolerance: how much can a prover who
//! perturbs every committed activation by at most ±tau (staying inside the
//! verifier's per-element tolerance at every challenged cell) actually steer
//! greedy decoding? Results and interpretation live in REPORT.md.
//!
//! Three measurements on the real model:
//!   1. per-layer amplification of a single ±tau injection (prompt forward);
//!   2. full-stack random ±tau injection at every hook of every position:
//!      first token divergence across a greedy generation;
//!   3. targeted last-layer attack: delta = tau * sign(w_b - w_a) using the
//!      dequantized LM-head rows (the linfty-optimal direction for the
//!      post-norm linear map), validated through the real head.
//!
//! Usage: cargo run --release -p vllm-infer --features metal --example
//!        drift_attack -- model.gguf tokenizer.json [--cpu]

use std::fs::File;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, IndexOp, Tensor};
use vllm_infer::model::ModelWeights;

const PROMPT: &str = "In one sentence, what is a Merkle tree?";
const MAX_TOKENS: usize = 48;
const TAUS: [f32; 5] = [0.01, 0.05, 0.1, 0.25, 0.5];
const SEEDS: [u64; 3] = [1, 2, 3];

struct Rng(u64);

impl Rng {
    fn new(key: &[u64]) -> Self {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for &k in key {
            s ^= k.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            s = s.rotate_left(31).wrapping_mul(0x94d0_49bb_1331_11eb);
        }
        Rng(s | 1)
    }

    fn next_sign(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        if self.0 & 1 == 0 { 1.0 } else { -1.0 }
    }
}

fn load_model(path: &str, device: &Device) -> Result<ModelWeights> {
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    Ok(ModelWeights::from_gguf(content, &mut file, device)?)
}

fn logits_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.squeeze(0)?.to_dtype(candle_core::DType::F32)?.to_vec1()?)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// (top-1 index, top-2 index, gap).
fn top2(v: &[f32]) -> (usize, usize, f32) {
    let a = argmax(v);
    let mut b = usize::from(a == 0);
    for (i, &x) in v.iter().enumerate() {
        if i != a && x > v[b] {
            b = i;
        }
    }
    (a, b, v[a] - v[b])
}

/// Random +-tau tensor matching `h`, deterministically keyed.
fn noise_like(h: &Tensor, tau: f32, key: &[u64]) -> Result<Tensor> {
    let dims = h.dims().to_vec();
    let n: usize = dims.iter().product();
    let mut rng = Rng::new(key);
    let data: Vec<f32> = (0..n).map(|_| tau * rng.next_sign()).collect();
    Ok(Tensor::from_vec(data, dims, h.device())?)
}

/// Greedy generation with optional injection at every hook of every forward.
/// Returns the emitted tokens and per-step logits.
fn generate(
    model: &mut ModelWeights,
    dev: &Device,
    prompt: &[u32],
    tau: Option<(f32, u64)>,
    keep_logits: bool,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    model.clear_kv_cache();
    let inject = |step: usize, j: usize, h: &Tensor| -> candle_core::Result<Option<Tensor>> {
        match tau {
            None => Ok(None),
            Some((t, seed)) => {
                let noise = noise_like(h, t, &[seed, step as u64, j as u64])
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                Ok(Some((h + noise)?))
            }
        }
    };

    let mut tokens = Vec::new();
    let mut all_logits = Vec::new();
    let input = Tensor::new(prompt, dev)?.unsqueeze(0)?;
    let mut logits = logits_vec(&model.forward_perturbed(&input, 0, &mut |j, h| inject(0, j, h))?)?;
    for step in 0..MAX_TOKENS {
        let token = argmax(&logits) as u32;
        tokens.push(token);
        if keep_logits {
            all_logits.push(std::mem::take(&mut logits));
        }
        if step + 1 == MAX_TOKENS {
            break;
        }
        let input = Tensor::new(&[token], dev)?.unsqueeze(0)?;
        let out = model.forward_perturbed(&input, prompt.len() + step, &mut |j, h| {
            inject(step + 1, j, h)
        })?;
        logits = logits_vec(&out)?;
    }
    Ok((tokens, all_logits))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (model_path, tok_path) = (
        args.get(1)
            .context("usage: drift_attack <model.gguf> <tokenizer.json>")?,
        args.get(2)
            .context("usage: drift_attack <model.gguf> <tokenizer.json>")?,
    );
    let device = if args.iter().any(|a| a == "--cpu") {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    let tokenizer = tokenizers::Tokenizer::from_file(tok_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let full = format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    let prompt = tokenizer
        .encode(full, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    let mut model = load_model(model_path, &device)?;
    let n_layers = model.n_layers();

    // --- Honest baseline: tokens, per-step gaps, last-layer hidden states.
    let (honest_tokens, honest_logits) = generate(&mut model, &device, &prompt, None, true)?;
    let mut h_last: Vec<Vec<f32>> = Vec::new(); // state entering the final norm, per step
    {
        model.clear_kv_cache();
        let input = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
        let mut capture = |j: usize, h: &Tensor| -> candle_core::Result<()> {
            if j == n_layers {
                let seq = h.dims3()?.1;
                h_last.push(
                    h.i((0, seq - 1, ..))?
                        .to_dtype(candle_core::DType::F32)?
                        .to_vec1()?,
                );
            }
            Ok(())
        };
        model.forward_traced(&input, 0, &mut capture)?;
        for (step, &token) in honest_tokens.iter().enumerate().take(MAX_TOKENS - 1) {
            let input = Tensor::new(&[token], &device)?.unsqueeze(0)?;
            model.forward_traced(&input, prompt.len() + step, &mut capture)?;
        }
    }
    let gaps: Vec<f32> = honest_logits.iter().map(|l| top2(l).2).collect();
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(f32::total_cmp);
    println!("== honest run: {} steps ==", honest_tokens.len());
    println!(
        "top1-top2 logit gap: min {:.3}  p10 {:.3}  median {:.3}  max {:.3}",
        sorted_gaps[0],
        sorted_gaps[sorted_gaps.len() / 10],
        sorted_gaps[sorted_gaps.len() / 2],
        sorted_gaps[sorted_gaps.len() - 1]
    );

    // --- 1. Per-layer amplification (single injection point, prompt forward).
    println!("\n== per-layer amplification: |dlogits|inf / tau, prompt forward ==");
    let honest_step0 = &honest_logits[0];
    for &tau in &[0.01f32, 0.5] {
        print!("tau={tau:<5} ");
        for j in 0..=n_layers {
            model.clear_kv_cache();
            let input = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
            let out = model.forward_perturbed(&input, 0, &mut |jj, h| {
                if jj == j {
                    let noise = noise_like(h, tau, &[7, j as u64])
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(Some((h + noise)?))
                } else {
                    Ok(None)
                }
            })?;
            let perturbed = logits_vec(&out)?;
            let dmax = perturbed
                .iter()
                .zip(honest_step0)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            print!("{:>6.1}", dmax / tau);
        }
        println!();
    }
    println!(
        "(columns: injection entering block 0..{}, then entering the final norm)",
        n_layers - 1
    );

    // --- 2. Full-stack random attack: first token divergence.
    println!("\n== full-stack random +-tau at every hook, greedy generation ==");
    println!("tau      step0 |dlogits|inf   first divergence step (3 seeds)");
    for &tau in &TAUS {
        let mut firsts = Vec::new();
        let mut d0 = 0f32;
        for &seed in &SEEDS {
            let (tokens, logits) = generate(&mut model, &device, &prompt, Some((tau, seed)), true)?;
            d0 = d0.max(
                logits[0]
                    .iter()
                    .zip(honest_step0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
            );
            let first = tokens
                .iter()
                .zip(&honest_tokens)
                .position(|(a, b)| a != b)
                .map(|p| p.to_string())
                .unwrap_or_else(|| "none".into());
            firsts.push(first);
        }
        println!("{tau:<8} {d0:<20.3} {}", firsts.join(", "));
    }

    // --- 3. Targeted last-layer attack: delta = tau*sign(w_b - w_a).
    println!("\n== targeted last-layer attack (per honest step) ==");
    let w = model.lm_head_weight(&device)?; // [vocab, dim]
    println!("tau      steps flipped / total");
    for &tau in &TAUS {
        let mut flipped = 0;
        for (step, logits) in honest_logits.iter().enumerate() {
            let (a, b, _gap) = top2(logits);
            let wa: Vec<f32> = w.i((a, ..))?.to_vec1()?;
            let wb: Vec<f32> = w.i((b, ..))?.to_vec1()?;
            let dim = wa.len();
            let adv: Vec<f32> = h_last[step]
                .iter()
                .zip(wa.iter().zip(&wb))
                .map(|(&h, (&ra, &rb))| h + tau * (rb - ra).signum())
                .collect();
            let adv = Tensor::from_vec(adv, (1, 1, dim), &device)?;
            let new_logits = logits_vec(&model.lm_head(&adv)?.squeeze(0)?)?;
            if argmax(&new_logits) != a {
                flipped += 1;
            }
        }
        println!("{tau:<8} {flipped} / {}", honest_logits.len());
    }

    Ok(())
}
