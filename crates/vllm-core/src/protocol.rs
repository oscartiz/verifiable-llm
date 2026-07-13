//! v0.2 spot-check challenge protocol: message formats and the hash-level
//! checks. The numerical re-execution of challenged layers lives in
//! `vllm-verify` (it needs candle); everything here is std + blake3, so the
//! prover side (`respond`) runs anywhere.
//!
//! Challenge derivation is Fiat–Shamir: cells are pseudorandomly drawn from
//! BLAKE3-XOF(final_chain ‖ trace_root ‖ params), so the prover cannot know
//! which cells will be probed until the transcript (and hence the trace) is
//! committed. Cell space: (pos, block) for block ∈ 0..L over every processed
//! position, plus head cells (pos, L) for positions that produced logits.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::trace::{TraceMeta, TraceReader};
use crate::transcript::Transcript;
use crate::{Error, Hash32, merkle};

pub const CHALLENGE_VERSION: &str = "vllm/challenge/v1";
pub const RESPONSE_VERSION: &str = "vllm/response/v1";
const FS_DOMAIN: &[u8] = b"vllm/fs-challenge/v1";

/// One challenged cell. `layer < n_layers`: verify block `layer` at `pos`
/// (re-execute the block over the revealed prefix). `layer == n_layers`:
/// verify the LM head at `pos` (re-execute final norm + output projection
/// and compare against the step's committed logits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChallengeCell {
    pub pos: u32,
    pub layer: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    pub version: String,
    pub final_chain: Hash32,
    pub trace_root: Hash32,
    pub k: u32,
    /// Verifier-chosen nonce mixed into the derivation. With pure
    /// Fiat-Shamir (empty nonce) a prover can regenerate until the draw
    /// misses its corrupted cells (grinding, cost ~ 1/(1-f)^k generations);
    /// a nonce picked AFTER the transcript is committed eliminates that.
    #[serde(default)]
    pub nonce: String,
    pub cells: Vec<ChallengeCell>,
}

/// Parameters that pin down the challenge space.
#[derive(Debug, Clone, Copy)]
pub struct ChallengeSpace {
    pub n_positions: u32,
    /// Number of transformer blocks L.
    pub n_layers: u32,
    /// First position that produced logits (prompt_len - 1).
    pub first_logit_pos: u32,
}

impl ChallengeSpace {
    pub fn from_transcript(t: &Transcript) -> Result<Self, Error> {
        // Fail closed on malformed input (empty prompt, degenerate trace)
        // before any position arithmetic can underflow.
        t.validate().map_err(Error::Gguf)?;
        let trace = t
            .trace
            .as_ref()
            .ok_or_else(|| Error::Gguf("transcript has no trace commitment".into()))?;
        Ok(ChallengeSpace {
            n_positions: trace.n_positions,
            n_layers: trace.n_layers,
            first_logit_pos: t.prompt_token_ids.len() as u32 - 1,
        })
    }

    fn size(&self) -> u64 {
        self.n_positions as u64 * self.n_layers as u64
            + (self.n_positions - self.first_logit_pos) as u64
    }

    fn cell_at(&self, raw: u64) -> ChallengeCell {
        let blocks = self.n_positions as u64 * self.n_layers as u64;
        if raw < blocks {
            ChallengeCell {
                pos: (raw / self.n_layers as u64) as u32,
                layer: (raw % self.n_layers as u64) as u32,
            }
        } else {
            ChallengeCell {
                pos: self.first_logit_pos + (raw - blocks) as u32,
                layer: self.n_layers,
            }
        }
    }
}

/// Derive `k` distinct challenge cells (fewer if the space is smaller than
/// `k`). Deterministic: both sides derive and cross-check.
pub fn derive_challenges(
    final_chain: &Hash32,
    trace_root: &Hash32,
    k: u32,
    nonce: &str,
    space: &ChallengeSpace,
) -> Vec<ChallengeCell> {
    let mut h = blake3::Hasher::new();
    h.update(FS_DOMAIN);
    h.update(&final_chain.0);
    h.update(&trace_root.0);
    h.update(&k.to_le_bytes());
    h.update(&(nonce.len() as u32).to_le_bytes());
    h.update(nonce.as_bytes());
    h.update(&space.n_positions.to_le_bytes());
    h.update(&space.n_layers.to_le_bytes());
    h.update(&space.first_logit_pos.to_le_bytes());
    let mut xof = h.finalize_xof();

    let size = space.size();
    let target = (k as u64).min(size) as usize;
    let mut seen = HashSet::new();
    let mut cells = Vec::with_capacity(target);
    let mut buf = [0u8; 8];
    while cells.len() < target {
        std::io::Read::read_exact(&mut xof, &mut buf).expect("XOF is infinite");
        // Modulo bias is ~size/2^64, negligible for any real trace.
        let raw = u64::from_le_bytes(buf) % size;
        if seen.insert(raw) {
            cells.push(space.cell_at(raw));
        }
    }
    cells
}

pub fn make_challenge(t: &Transcript, k: u32, nonce: &str) -> Result<Challenge, Error> {
    let space = ChallengeSpace::from_transcript(t)?;
    let trace_root = t.trace.as_ref().expect("checked").root;
    Ok(Challenge {
        version: CHALLENGE_VERSION.into(),
        final_chain: t.final_chain,
        trace_root,
        k,
        nonce: nonce.to_string(),
        cells: derive_challenges(&t.final_chain, &trace_root, k, nonce, &space),
    })
}

/// A trace cell revealed to the verifier, with its inclusion proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevealedCell {
    pub pos: u32,
    pub layer: u32,
    /// Base64 of the i32 little-endian fixed-point values.
    pub data: String,
    pub proof: Vec<Hash32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseItem {
    pub cell: ChallengeCell,
    /// Block check: cells (0..=pos, layer) — the block's inputs.
    /// Head check: the single cell (pos, n_layers).
    pub inputs: Vec<RevealedCell>,
    /// Block check: cell (pos, layer + 1) — the block's committed output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<RevealedCell>,
    /// Head check: base64 i32 quantized logits for the corresponding step;
    /// their hash must equal the step's committed logits_hash.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub version: String,
    pub trace_root: Hash32,
    pub items: Vec<ResponseItem>,
}

/// Prover: answer a challenge from the local trace file.
pub fn build_response(reader: &mut TraceReader, challenge: &Challenge) -> Result<Response, Error> {
    let meta = reader.meta().clone();
    if meta.root != challenge.trace_root {
        return Err(Error::Gguf(
            "trace file does not match challenged root".into(),
        ));
    }
    let mut items = Vec::with_capacity(challenge.cells.len());
    for &cell in &challenge.cells {
        let item = if cell.layer < meta.n_layers {
            let mut inputs = Vec::with_capacity(cell.pos as usize + 1);
            for p in 0..=cell.pos {
                inputs.push(reveal(reader, p, cell.layer)?);
            }
            ResponseItem {
                cell,
                inputs,
                output: Some(reveal(reader, cell.pos, cell.layer + 1)?),
                logits: None,
            }
        } else {
            let step = cell.pos.checked_sub(meta.first_logit_pos).ok_or_else(|| {
                Error::Gguf(format!(
                    "head challenge at pos {} < first logit pos",
                    cell.pos
                ))
            })?;
            ResponseItem {
                cell,
                inputs: vec![reveal(reader, cell.pos, meta.n_layers)?],
                output: None,
                logits: Some(b64_encode_i32(&reader.logits_row(step)?)),
            }
        };
        items.push(item);
    }
    Ok(Response {
        version: RESPONSE_VERSION.into(),
        trace_root: meta.root,
        items,
    })
}

fn reveal(reader: &mut TraceReader, pos: u32, layer: u32) -> Result<RevealedCell, Error> {
    let data = reader.cell(pos, layer)?;
    Ok(RevealedCell {
        pos,
        layer,
        data: b64_encode_i32(&data),
        proof: reader.prove_cell(pos, layer)?,
    })
}

/// Verifier, hash level: check a revealed cell's Merkle proof and decode it.
/// Returns the quantized values. Fails if the proof does not bind these
/// exact bytes to (pos, layer) under `root`.
pub fn check_revealed_cell(
    cell: &RevealedCell,
    meta: &TraceMeta,
    root: &Hash32,
) -> Result<Vec<i32>, Error> {
    let data = b64_decode_i32(&cell.data)
        .ok_or_else(|| Error::Gguf(format!("cell ({}, {}): bad base64", cell.pos, cell.layer)))?;
    if data.len() != meta.hidden_dim as usize {
        return Err(Error::Gguf(format!(
            "cell ({}, {}): dim {} != {}",
            cell.pos,
            cell.layer,
            data.len(),
            meta.hidden_dim
        )));
    }
    let hash = crate::trace::cell_hash(cell.pos, cell.layer, meta.frac_bits, &data);
    let index = meta.cell_index(cell.pos, cell.layer) as usize;
    if !merkle::verify(&hash, index, meta.n_cells() as usize, &cell.proof, root) {
        return Err(Error::Gguf(format!(
            "cell ({}, {}): Merkle proof rejected",
            cell.pos, cell.layer
        )));
    }
    Ok(data)
}

// --- minimal std-only base64 (standard alphabet, padded) ---

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn b64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        let chars = [
            B64[(n >> 18 & 63) as usize],
            B64[(n >> 12 & 63) as usize],
            B64[(n >> 6 & 63) as usize],
            B64[(n & 63) as usize],
        ];
        let keep = chunk.len() + 1;
        for (i, &c) in chars.iter().enumerate() {
            out.push(if i < keep { c as char } else { '=' });
        }
    }
    out
}

pub fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.as_bytes();
    if !s.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    for chunk in s.chunks(4) {
        let pad = chunk.iter().filter(|&&c| c == b'=').count();
        if pad > 2 || chunk[..4 - pad].contains(&b'=') {
            return None;
        }
        let mut n = 0u32;
        for &c in &chunk[..4 - pad] {
            n = n << 6 | B64.iter().position(|&b| b == c)? as u32;
        }
        n <<= 6 * pad as u32;
        let b = n.to_be_bytes();
        out.extend_from_slice(&b[1..4 - pad]);
    }
    Some(out)
}

pub fn b64_encode_i32(values: &[i32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    b64_encode(&bytes)
}

pub fn b64_decode_i32(s: &str) -> Option<Vec<i32>> {
    let bytes = b64_decode(s)?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64_roundtrip() {
        for len in 0..40 {
            let v: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(b64_decode(&b64_encode(&v)).unwrap(), v, "len {len}");
        }
        assert_eq!(b64_encode(b"hello"), "aGVsbG8=");
        assert!(b64_decode("a===").is_none());
        assert!(b64_decode("abc").is_none());
        let ints = vec![i32::MIN, -1, 0, 1, i32::MAX, 123456];
        assert_eq!(b64_decode_i32(&b64_encode_i32(&ints)).unwrap(), ints);
    }

    #[test]
    fn derivation_is_deterministic_and_distinct() {
        let a = Hash32(*blake3::hash(b"chain").as_bytes());
        let b = Hash32(*blake3::hash(b"trace").as_bytes());
        let space = ChallengeSpace {
            n_positions: 20,
            n_layers: 4,
            first_logit_pos: 9,
        };
        let c1 = derive_challenges(&a, &b, 15, "", &space);
        let c2 = derive_challenges(&a, &b, 15, "", &space);
        assert_eq!(c1, c2);
        assert_eq!(c1.len(), 15);
        let set: HashSet<_> = c1.iter().collect();
        assert_eq!(set.len(), 15, "cells must be distinct");
        for c in &c1 {
            assert!(c.pos < 20);
            assert!(c.layer <= 4);
            if c.layer == 4 {
                assert!(c.pos >= 9, "head challenge below first logit pos");
            }
        }
        // Different transcript => different challenges.
        let c3 = derive_challenges(&b, &a, 15, "", &space);
        assert_ne!(c1, c3);
        // A different verifier nonce changes the draw.
        let c4 = derive_challenges(&a, &b, 15, "beacon-42", &space);
        assert_ne!(c1, c4);
    }

    #[test]
    fn small_space_is_exhausted() {
        let a = Hash32(*blake3::hash(b"x").as_bytes());
        let space = ChallengeSpace {
            n_positions: 3,
            n_layers: 2,
            first_logit_pos: 1,
        };
        // size = 3*2 + 2 = 8 < k
        let cells = derive_challenges(&a, &a, 100, "", &space);
        assert_eq!(cells.len(), 8);
    }
}
