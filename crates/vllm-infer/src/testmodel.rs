//! Test support: construct a tiny random-weight 2-layer llama GGUF entirely
//! in memory, so integration tests and CI never download real weights.
//! Weights come from a fixed xorshift64 seed — byte-identical everywhere.

use vllm_core::gguf::write::{TensorData, write_gguf};
use vllm_core::gguf::{GgmlType, MetaValue};

pub const VOCAB: u64 = 64;
pub const EMB: u64 = 32;
pub const FFN: u64 = 64;
pub const HEADS: u32 = 2;
pub const LAYERS: u32 = 2;

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Small weights keep activations tame through 2 layers.
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    }

    fn tensor(&mut self, n: u64) -> Vec<u8> {
        (0..n).flat_map(|_| self.next_f32().to_le_bytes()).collect()
    }
}

/// The GGUF bytes of the tiny model.
pub fn tiny_llama_gguf() -> Vec<u8> {
    let metadata = vec![
        (
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        ),
        (
            "llama.attention.head_count".to_string(),
            MetaValue::U32(HEADS),
        ),
        (
            "llama.attention.head_count_kv".to_string(),
            MetaValue::U32(HEADS),
        ),
        ("llama.block_count".to_string(), MetaValue::U32(LAYERS)),
        (
            "llama.embedding_length".to_string(),
            MetaValue::U32(EMB as u32),
        ),
        (
            "llama.rope.dimension_count".to_string(),
            MetaValue::U32(EMB as u32 / HEADS),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            MetaValue::F32(1e-5),
        ),
    ];

    let mut rng = Rng(0x5eed_cafe_f00d_1234);
    let ones: Vec<u8> = (0..EMB).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let mut owned: Vec<(String, Vec<u64>, Vec<u8>)> = vec![
        (
            "token_embd.weight".into(),
            vec![EMB, VOCAB],
            rng.tensor(EMB * VOCAB),
        ),
        ("output_norm.weight".into(), vec![EMB], ones.clone()),
        (
            "output.weight".into(),
            vec![EMB, VOCAB],
            rng.tensor(EMB * VOCAB),
        ),
    ];
    for i in 0..LAYERS {
        let p = format!("blk.{i}");
        owned.push((
            format!("{p}.attn_q.weight"),
            vec![EMB, EMB],
            rng.tensor(EMB * EMB),
        ));
        owned.push((
            format!("{p}.attn_k.weight"),
            vec![EMB, EMB],
            rng.tensor(EMB * EMB),
        ));
        owned.push((
            format!("{p}.attn_v.weight"),
            vec![EMB, EMB],
            rng.tensor(EMB * EMB),
        ));
        owned.push((
            format!("{p}.attn_output.weight"),
            vec![EMB, EMB],
            rng.tensor(EMB * EMB),
        ));
        owned.push((
            format!("{p}.ffn_gate.weight"),
            vec![EMB, FFN],
            rng.tensor(EMB * FFN),
        ));
        owned.push((
            format!("{p}.ffn_up.weight"),
            vec![EMB, FFN],
            rng.tensor(EMB * FFN),
        ));
        owned.push((
            format!("{p}.ffn_down.weight"),
            vec![FFN, EMB],
            rng.tensor(FFN * EMB),
        ));
        owned.push((format!("{p}.attn_norm.weight"), vec![EMB], ones.clone()));
        owned.push((format!("{p}.ffn_norm.weight"), vec![EMB], ones.clone()));
    }

    let tensors: Vec<TensorData> = owned
        .iter()
        .map(|(name, dims, data)| TensorData {
            name: name.clone(),
            ggml_type: GgmlType::F32,
            dims: dims.clone(),
            data,
        })
        .collect();

    let mut out = Vec::new();
    write_gguf(&mut out, &metadata, &tensors).expect("valid tiny model");
    out
}
