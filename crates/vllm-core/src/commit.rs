//! Model commitment: a BLAKE3 Merkle tree over the quantized weight tensors
//! of a GGUF file, leaves ordered by tensor name (byte-wise, deterministic).
//!
//! The leaf preimage covers name, ggml type, shape, and the raw quantized
//! bytes exactly as stored on disk — no dequantization, so there is nothing
//! float-ambiguous about the model commitment.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::gguf::GgufFile;
use crate::{Error, Hash32, merkle};

const TENSOR_DOMAIN: &[u8] = b"vllm/tensor/v1";
pub const MODEL_COMMITMENT_VERSION: &str = "vllm/model-commitment/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCommitment {
    pub name: String,
    pub ggml_type: u32,
    pub dims: Vec<u64>,
    /// Byte offset relative to the start of the GGUF tensor data section.
    pub offset: u64,
    pub byte_len: u64,
    pub hash: Hash32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCommitment {
    pub version: String,
    pub file_name: Option<String>,
    pub file_size: u64,
    pub alignment: u64,
    /// Merkle root over `tensors[i].hash`, in the order stored here
    /// (sorted by name).
    pub root: Hash32,
    pub tensors: Vec<TensorCommitment>,
}

impl ModelCommitment {
    /// Recompute the root from the per-tensor leaf hashes. Used to detect a
    /// commitment file whose root does not match its own leaves.
    pub fn recompute_root(&self) -> Option<Hash32> {
        let leaves: Vec<Hash32> = self.tensors.iter().map(|t| t.hash).collect();
        merkle::root(&leaves)
    }
}

/// Hash every tensor of a GGUF file and build the model commitment.
pub fn commit_gguf(path: &Path) -> Result<ModelCommitment, Error> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let gguf = GgufFile::read_header(&mut reader)?;

    let mut infos = gguf.tensors;
    infos.sort_by(|a, b| a.name.cmp(&b.name));
    if let Some(w) = infos.windows(2).find(|w| w[0].name == w[1].name) {
        return Err(Error::Gguf(format!(
            "duplicate tensor name {:?}",
            w[0].name
        )));
    }

    let mut tensors = Vec::with_capacity(infos.len());
    for info in &infos {
        let start = gguf
            .data_start
            .checked_add(info.offset)
            .filter(|s| {
                s.checked_add(info.byte_len)
                    .is_some_and(|end| end <= file_size)
            })
            .ok_or_else(|| {
                Error::Gguf(format!("tensor {:?} extends past end of file", info.name))
            })?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(TENSOR_DOMAIN);
        hasher.update(&(info.name.len() as u32).to_le_bytes());
        hasher.update(info.name.as_bytes());
        hasher.update(&info.ggml_type.0.to_le_bytes());
        hasher.update(&(info.dims.len() as u32).to_le_bytes());
        for &d in &info.dims {
            hasher.update(&d.to_le_bytes());
        }
        hasher.update(&info.byte_len.to_le_bytes());

        reader.seek(SeekFrom::Start(start))?;
        hasher.update_reader((&mut reader).take(info.byte_len))?;

        tensors.push(TensorCommitment {
            name: info.name.clone(),
            ggml_type: info.ggml_type.0,
            dims: info.dims.clone(),
            offset: info.offset,
            byte_len: info.byte_len,
            hash: hasher.finalize().into(),
        });
    }

    let leaves: Vec<Hash32> = tensors.iter().map(|t| t.hash).collect();
    let root =
        merkle::root(&leaves).ok_or_else(|| Error::Gguf("GGUF file contains no tensors".into()))?;

    Ok(ModelCommitment {
        version: MODEL_COMMITMENT_VERSION.into(),
        file_name: path.file_name().map(|n| n.to_string_lossy().into_owned()),
        file_size,
        alignment: gguf.alignment,
        root,
        tensors,
    })
}

/// Outcome of re-checking a GGUF file against a stored commitment.
#[derive(Debug, PartialEq, Eq)]
pub enum VerifyOutcome {
    Ok,
    RootMismatch {
        expected: Hash32,
        actual: Hash32,
    },
    /// First tensor whose leaf hash differs (name kept for diagnostics).
    TensorMismatch {
        name: String,
    },
    /// Tensor tables differ (renamed/added/removed tensors).
    StructureMismatch,
}

/// Re-hash `path` and compare against `expected`.
pub fn verify_gguf(path: &Path, expected: &ModelCommitment) -> Result<VerifyOutcome, Error> {
    let actual = commit_gguf(path)?;
    if actual.root == expected.root {
        return Ok(VerifyOutcome::Ok);
    }
    if actual.tensors.len() != expected.tensors.len() {
        return Ok(VerifyOutcome::StructureMismatch);
    }
    for (a, e) in actual.tensors.iter().zip(&expected.tensors) {
        if a.name != e.name {
            return Ok(VerifyOutcome::StructureMismatch);
        }
        if a.hash != e.hash {
            return Ok(VerifyOutcome::TensorMismatch {
                name: a.name.clone(),
            });
        }
    }
    Ok(VerifyOutcome::RootMismatch {
        expected: expected.root,
        actual: actual.root,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::write::{TensorData, write_gguf};
    use crate::gguf::{GgmlType, MetaValue};

    fn write_temp_gguf(name: &str, tweak: impl FnOnce(&mut Vec<u8>)) -> std::path::PathBuf {
        let t0: Vec<u8> = (0..64 * 4).map(|i| (i % 251) as u8).collect();
        let t1 = vec![9u8; 34];
        let metadata = vec![(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        )];
        let tensors = vec![
            TensorData {
                name: "z.weight".into(),
                ggml_type: GgmlType::F32,
                dims: vec![8, 8],
                data: &t0,
            },
            TensorData {
                name: "a.weight".into(),
                ggml_type: GgmlType::Q8_0,
                dims: vec![32],
                data: &t1,
            },
        ];
        let mut bytes = Vec::new();
        write_gguf(&mut bytes, &metadata, &tensors).unwrap();
        tweak(&mut bytes);
        let path =
            std::env::temp_dir().join(format!("vllm-core-test-{name}-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn commit_verify_tamper() {
        let path = write_temp_gguf("ok", |_| {});
        let commitment = commit_gguf(&path).unwrap();

        // Leaves are sorted by name regardless of file order.
        assert_eq!(commitment.tensors[0].name, "a.weight");
        assert_eq!(commitment.tensors[1].name, "z.weight");
        assert_eq!(commitment.recompute_root(), Some(commitment.root));
        assert_eq!(verify_gguf(&path, &commitment).unwrap(), VerifyOutcome::Ok);

        // Deterministic: committing twice gives the identical root.
        assert_eq!(commit_gguf(&path).unwrap().root, commitment.root);

        // Flip one byte at the end of the file: that's the payload of
        // "a.weight", written last in file order.
        let tampered = write_temp_gguf("tampered", |bytes| {
            let n = bytes.len();
            bytes[n - 1] ^= 0x01;
        });
        match verify_gguf(&tampered, &commitment).unwrap() {
            VerifyOutcome::TensorMismatch { name } => assert_eq!(name, "a.weight"),
            other => panic!("expected tensor mismatch, got {other:?}"),
        }

        std::fs::remove_file(path).ok();
        std::fs::remove_file(tampered).ok();
    }
}
