//! Activation trace: the per-position, per-layer hidden states of a
//! generation run, quantized to fixed point and committed with a Merkle tree.
//!
//! Cell (p, j) for j in 0..L is the hidden state *entering* block j at
//! position p; cell (p, L) is the state exiting the last block. Cells exist
//! for every processed position: prompt positions 0..P-1 and one position per
//! generated token that was fed back (the last generated token never is), so
//! n_positions = prompt_len + steps - 1.
//!
//! Leaf order for the Merkle tree and the file layout is
//! `index = pos * (L + 1) + layer`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Hash32, merkle};

const CELL_DOMAIN: &[u8] = b"vllm/trace-cell/v1";
pub const TRACE_VERSION: &str = "vllm/trace/v1";
const MAGIC: &[u8; 4] = b"VLTC";

/// Commit to one activation cell. `data` is the fixed-point quantized hidden
/// state (`q = round(x * 2^frac_bits)` as i32, same scheme as logits).
pub fn cell_hash(pos: u32, layer: u32, frac_bits: u8, data: &[i32]) -> Hash32 {
    let mut h = blake3::Hasher::new();
    h.update(CELL_DOMAIN);
    h.update(&pos.to_le_bytes());
    h.update(&layer.to_le_bytes());
    h.update(&[frac_bits]);
    h.update(&(data.len() as u32).to_le_bytes());
    let mut buf = [0u8; 4 * 4096];
    for chunk in data.chunks(4096) {
        for (i, &q) in chunk.iter().enumerate() {
            buf[4 * i..4 * i + 4].copy_from_slice(&q.to_le_bytes());
        }
        h.update(&buf[..4 * chunk.len()]);
    }
    h.finalize().into()
}

/// Quantize activations with the same convention as logits (NaN rejected).
pub fn quantize(values: &[f32], frac_bits: u8) -> Result<Vec<i32>, Error> {
    let scale = (1u64 << frac_bits) as f64;
    values
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            if x.is_nan() {
                return Err(Error::NonFiniteLogit { step: 0, index });
            }
            Ok((x as f64 * scale)
                .round()
                .clamp(i32::MIN as f64, i32::MAX as f64) as i32)
        })
        .collect()
}

pub fn dequantize(values: &[i32], frac_bits: u8) -> Vec<f32> {
    let scale = (1u64 << frac_bits) as f64;
    values.iter().map(|&q| (q as f64 / scale) as f32).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMeta {
    pub version: String,
    pub n_positions: u32,
    /// Number of transformer blocks L; each position has L+1 cells.
    pub n_layers: u32,
    pub hidden_dim: u32,
    pub frac_bits: u8,
    /// Merkle root over all cell hashes in index order.
    pub root: Hash32,
    /// Quantized logit rows stored after the cells (prover-local data for
    /// answering head challenges; committed via the chain, not this tree).
    pub vocab_size: u32,
    pub logit_frac_bits: u8,
    /// Position that produced logit row 0 (= prompt_len - 1); row s
    /// corresponds to position first_logit_pos + s.
    pub first_logit_pos: u32,
    pub n_logit_rows: u32,
    /// Per-step commitment salts for --prove-decode (SECRET: the hiding of
    /// the zk logits commitment rests on these staying local).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zk_salts: Option<Vec<[u64; 4]>>,
}

impl TraceMeta {
    pub fn cells_per_position(&self) -> u32 {
        self.n_layers + 1
    }

    pub fn n_cells(&self) -> u64 {
        self.n_positions as u64 * self.cells_per_position() as u64
    }

    pub fn cell_index(&self, pos: u32, layer: u32) -> u64 {
        pos as u64 * self.cells_per_position() as u64 + layer as u64
    }
}

/// In-memory trace under construction (prover side). Cells must be pushed in
/// index order: position-major, layer 0..=L within each position. Logit rows
/// are pushed independently, one per generated step, in step order.
pub struct TraceBuilder {
    n_layers: u32,
    hidden_dim: u32,
    frac_bits: u8,
    cells: Vec<Vec<i32>>,
    hashes: Vec<Hash32>,
    logit_frac_bits: u8,
    first_logit_pos: u32,
    logit_rows: Vec<Vec<i32>>,
    zk_salts: Vec<[u64; 4]>,
}

impl TraceBuilder {
    pub fn new(
        n_layers: u32,
        hidden_dim: u32,
        frac_bits: u8,
        logit_frac_bits: u8,
        first_logit_pos: u32,
    ) -> Self {
        TraceBuilder {
            n_layers,
            hidden_dim,
            frac_bits,
            cells: Vec::new(),
            hashes: Vec::new(),
            logit_frac_bits,
            first_logit_pos,
            logit_rows: Vec::new(),
            zk_salts: Vec::new(),
        }
    }

    /// Store the commitment salt of the next generated step (--prove-decode).
    pub fn push_zk_salt(&mut self, salt: [u64; 4]) {
        self.zk_salts.push(salt);
    }

    /// Store the already-quantized logits of the next generated step.
    pub fn push_logits_row(&mut self, quantized: Vec<i32>) {
        self.logit_rows.push(quantized);
    }

    /// Quantize, hash, and append the next cell. Returns its hash.
    pub fn push_cell(&mut self, values: &[f32]) -> Result<Hash32, Error> {
        if values.len() != self.hidden_dim as usize {
            return Err(Error::Gguf(format!(
                "trace cell has dim {}, expected {}",
                values.len(),
                self.hidden_dim
            )));
        }
        let cells_per_pos = self.n_layers as u64 + 1;
        let index = self.cells.len() as u64;
        let (pos, layer) = (
            (index / cells_per_pos) as u32,
            (index % cells_per_pos) as u32,
        );
        let data = quantize(values, self.frac_bits)?;
        let hash = cell_hash(pos, layer, self.frac_bits, &data);
        self.cells.push(data);
        self.hashes.push(hash);
        Ok(hash)
    }

    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn hashes(&self) -> &[Hash32] {
        &self.hashes
    }

    /// Finish: compute the root and write the trace file.
    pub fn write(self, path: &Path) -> Result<TraceMeta, Error> {
        let cells_per_pos = self.n_layers as u64 + 1;
        if self.cells.is_empty() || !(self.cells.len() as u64).is_multiple_of(cells_per_pos) {
            return Err(Error::Gguf(format!(
                "trace has {} cells, not a multiple of layers+1 = {cells_per_pos}",
                self.cells.len()
            )));
        }
        let vocab_size = self.logit_rows.first().map(|r| r.len() as u32).unwrap_or(0);
        if self.logit_rows.iter().any(|r| r.len() as u32 != vocab_size) {
            return Err(Error::Gguf("inconsistent logit row lengths".into()));
        }
        if !self.zk_salts.is_empty() && self.zk_salts.len() != self.logit_rows.len() {
            return Err(Error::Gguf(format!(
                "{} zk salts for {} logit rows",
                self.zk_salts.len(),
                self.logit_rows.len()
            )));
        }
        let root = merkle::root(&self.hashes).expect("non-empty");
        let meta = TraceMeta {
            version: TRACE_VERSION.into(),
            n_positions: (self.cells.len() as u64 / cells_per_pos) as u32,
            n_layers: self.n_layers,
            hidden_dim: self.hidden_dim,
            frac_bits: self.frac_bits,
            root,
            vocab_size,
            logit_frac_bits: self.logit_frac_bits,
            first_logit_pos: self.first_logit_pos,
            n_logit_rows: self.logit_rows.len() as u32,
            zk_salts: (!self.zk_salts.is_empty()).then_some(self.zk_salts.clone()),
        };
        let header = serde_json::to_vec(&meta).expect("serializable");
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&(header.len() as u32).to_le_bytes())?;
        w.write_all(&header)?;
        let write_row = |w: &mut BufWriter<File>, row: &[i32]| -> Result<(), Error> {
            let mut buf = Vec::with_capacity(row.len() * 4);
            for &q in row {
                buf.extend_from_slice(&q.to_le_bytes());
            }
            w.write_all(&buf)?;
            Ok(())
        };
        for cell in &self.cells {
            write_row(&mut w, cell)?;
        }
        for row in &self.logit_rows {
            write_row(&mut w, row)?;
        }
        w.flush()?;
        Ok(meta)
    }
}

/// Read-only trace file access (prover side, when answering challenges).
pub struct TraceReader {
    reader: BufReader<File>,
    meta: TraceMeta,
    data_start: u64,
    /// All cell hashes, recomputed on open (needed for Merkle paths anyway).
    hashes: Vec<Hash32>,
}

impl TraceReader {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::Gguf("not a vllm trace file".into()));
        }
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let header_len = u32::from_le_bytes(len_bytes) as usize;
        if header_len > 1 << 20 {
            return Err(Error::Gguf("implausible trace header".into()));
        }
        let mut header = vec![0u8; header_len];
        reader.read_exact(&mut header)?;
        let meta: TraceMeta = serde_json::from_slice(&header)
            .map_err(|e| Error::Gguf(format!("bad trace header: {e}")))?;
        let data_start = (4 + 4 + header_len) as u64;

        let mut tr = TraceReader {
            reader,
            meta: meta.clone(),
            data_start,
            hashes: Vec::new(),
        };
        // Recompute all cell hashes and check the root: a corrupted trace is
        // detected here rather than producing unverifiable responses.
        let mut hashes = Vec::with_capacity(meta.n_cells() as usize);
        for pos in 0..meta.n_positions {
            for layer in 0..meta.cells_per_position() {
                let data = tr.cell(pos, layer)?;
                hashes.push(cell_hash(pos, layer, meta.frac_bits, &data));
            }
        }
        if merkle::root(&hashes) != Some(meta.root) {
            return Err(Error::Gguf("trace file does not match its own root".into()));
        }
        tr.hashes = hashes;
        Ok(tr)
    }

    pub fn meta(&self) -> &TraceMeta {
        &self.meta
    }

    pub fn cell(&mut self, pos: u32, layer: u32) -> Result<Vec<i32>, Error> {
        if pos >= self.meta.n_positions || layer > self.meta.n_layers {
            return Err(Error::Gguf(format!("cell ({pos}, {layer}) out of range")));
        }
        let dim = self.meta.hidden_dim as usize;
        let offset = self.data_start + self.meta.cell_index(pos, layer) * dim as u64 * 4;
        self.read_row(offset, dim)
    }

    /// Quantized logits of generated step `s` (position first_logit_pos + s).
    pub fn logits_row(&mut self, step: u32) -> Result<Vec<i32>, Error> {
        if step >= self.meta.n_logit_rows {
            return Err(Error::Gguf(format!("logit row {step} out of range")));
        }
        let cells_bytes = self.meta.n_cells() * self.meta.hidden_dim as u64 * 4;
        let offset = self.data_start + cells_bytes + step as u64 * self.meta.vocab_size as u64 * 4;
        self.read_row(offset, self.meta.vocab_size as usize)
    }

    fn read_row(&mut self, offset: u64, len: usize) -> Result<Vec<i32>, Error> {
        self.reader.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len * 4];
        self.reader.read_exact(&mut buf)?;
        Ok(buf
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Merkle inclusion proof for a cell, against `meta.root`.
    pub fn prove_cell(&self, pos: u32, layer: u32) -> Result<Vec<Hash32>, Error> {
        let index = self.meta.cell_index(pos, layer) as usize;
        merkle::prove(&self.hashes, index)
            .ok_or_else(|| Error::Gguf(format!("cell ({pos}, {layer}) out of range")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(n_pos: u32, n_layers: u32, dim: u32) -> TraceBuilder {
        let mut b = TraceBuilder::new(n_layers, dim, 16, 16, 1);
        for pos in 0..n_pos {
            for layer in 0..=n_layers {
                let values: Vec<f32> = (0..dim)
                    .map(|i| (pos as f32 + layer as f32 * 0.1 + i as f32 * 0.01).sin())
                    .collect();
                b.push_cell(&values).unwrap();
            }
        }
        for step in 0..n_pos - 1 {
            b.push_logits_row((0..5).map(|i| (step * 100 + i) as i32).collect());
        }
        b
    }

    #[test]
    fn roundtrip_and_proofs() {
        let path = std::env::temp_dir().join(format!("vllm-trace-{}.trace", std::process::id()));
        let builder = build(3, 2, 8);
        let expected_hashes = builder.hashes().to_vec();
        let meta = builder.write(&path).unwrap();
        assert_eq!(meta.n_positions, 3);
        assert_eq!(meta.n_cells(), 9);

        let mut r = TraceReader::open(&path).unwrap();
        assert_eq!(r.meta().root, meta.root);
        assert_eq!(meta.n_logit_rows, 2);
        assert_eq!(r.logits_row(1).unwrap(), vec![100, 101, 102, 103, 104]);
        assert!(r.logits_row(2).is_err());
        for pos in 0..3 {
            for layer in 0..=2u32 {
                let data = r.cell(pos, layer).unwrap();
                let h = cell_hash(pos, layer, 16, &data);
                assert_eq!(h, expected_hashes[meta.cell_index(pos, layer) as usize]);
                let path_proof = r.prove_cell(pos, layer).unwrap();
                assert!(crate::merkle::verify(
                    &h,
                    meta.cell_index(pos, layer) as usize,
                    meta.n_cells() as usize,
                    &path_proof,
                    &meta.root
                ));
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn corrupted_trace_is_rejected_on_open() {
        let path =
            std::env::temp_dir().join(format!("vllm-trace-bad-{}.trace", std::process::id()));
        build(2, 2, 8).write(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // Corrupt a byte in the cells region (the tail of the file is logit
        // rows, which the root does not cover — they are bound by the chain's
        // per-step logits_hash instead).
        let logit_bytes = 5 * 4;
        let n = bytes.len();
        bytes[n - logit_bytes - 1] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(TraceReader::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn quantize_roundtrip_within_half_ulp() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin() * 20.0).collect();
        let q = quantize(&values, 16).unwrap();
        let back = dequantize(&q, 16);
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() <= 0.5 / 65536.0 + 1e-7);
        }
        assert!(quantize(&[f32::NAN], 16).is_err());
    }

    #[test]
    fn cell_hash_binds_coordinates() {
        let d = vec![1i32, 2, 3];
        let base = cell_hash(0, 0, 16, &d);
        assert_ne!(base, cell_hash(1, 0, 16, &d));
        assert_ne!(base, cell_hash(0, 1, 16, &d));
        assert_ne!(base, cell_hash(0, 0, 8, &d));
        assert_ne!(base, cell_hash(0, 0, 16, &[1, 2, 4]));
    }
}
