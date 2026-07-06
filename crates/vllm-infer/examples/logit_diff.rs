//! Diagnostic: run one prompt forward on CPU and (if compiled) Metal, and
//! report how much the logit vectors differ. Used to calibrate the logit
//! quantization default (DECISIONS.md) and the v0.2 verification tolerance.
//!
//! Usage: cargo run --release -p vllm-infer --features metal --example logit_diff -- model.gguf tokenizer.json "prompt"

use std::fs::File;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

fn forward_once(model_path: &str, device: &Device, tokens: &[u32]) -> Result<Vec<f32>> {
    let mut file = File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, device)?;
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    Ok(logits
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, model, tokenizer, prompt] = &args[..] else {
        anyhow::bail!("usage: logit_diff <model.gguf> <tokenizer.json> <prompt>");
    };
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let tokens = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
        .get_ids()
        .to_vec();
    eprintln!("{} prompt tokens", tokens.len());

    let cpu = forward_once(model, &Device::Cpu, &tokens).context("cpu forward")?;

    #[cfg(feature = "metal")]
    {
        let metal =
            forward_once(model, &Device::new_metal(0)?, &tokens).context("metal forward")?;
        assert_eq!(cpu.len(), metal.len());
        let mut max_abs = 0f32;
        let mut max_at = 0;
        let mut sum_abs = 0f64;
        for (i, (&a, &b)) in cpu.iter().zip(&metal).enumerate() {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                max_at = i;
            }
            sum_abs += d as f64;
        }
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        println!(
            "vocab={} max|Δ|={:.6e} (at index {}, cpu={:.4}, metal={:.4}) mean|Δ|={:.6e}",
            cpu.len(),
            max_abs,
            max_at,
            cpu[max_at],
            metal[max_at],
            sum_abs / cpu.len() as f64
        );
        println!("argmax: cpu={} metal={}", argmax(&cpu), argmax(&metal));
    }
    #[cfg(not(feature = "metal"))]
    println!("built without metal; cpu logits len={}", cpu.len());
    Ok(())
}
