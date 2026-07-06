//! Minimal GGUF reader: header, metadata, tensor table, and raw tensor byte
//! ranges. We deliberately do not depend on candle here — the commitment is
//! over the quantized bytes exactly as they sit on disk, so a verifier only
//! needs this parser and blake3.
//!
//! Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::io::{self, Read};

use crate::Error;

pub const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// GGML tensor data types we know how to size. Values match ggml's enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub const F32: GgmlType = GgmlType(0);
    pub const F16: GgmlType = GgmlType(1);
    pub const Q4_0: GgmlType = GgmlType(2);
    pub const Q4_1: GgmlType = GgmlType(3);
    pub const Q5_0: GgmlType = GgmlType(6);
    pub const Q5_1: GgmlType = GgmlType(7);
    pub const Q8_0: GgmlType = GgmlType(8);
    pub const Q8_1: GgmlType = GgmlType(9);
    pub const Q2_K: GgmlType = GgmlType(10);
    pub const Q3_K: GgmlType = GgmlType(11);
    pub const Q4_K: GgmlType = GgmlType(12);
    pub const Q5_K: GgmlType = GgmlType(13);
    pub const Q6_K: GgmlType = GgmlType(14);
    pub const Q8_K: GgmlType = GgmlType(15);
    pub const I8: GgmlType = GgmlType(24);
    pub const I16: GgmlType = GgmlType(25);
    pub const I32: GgmlType = GgmlType(26);
    pub const I64: GgmlType = GgmlType(27);
    pub const F64: GgmlType = GgmlType(28);
    pub const BF16: GgmlType = GgmlType(30);

    /// (elements per block, bytes per block), or None for types we don't size.
    pub fn block_layout(self) -> Option<(u64, u64)> {
        Some(match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::Q4_0 => (32, 18),
            GgmlType::Q4_1 => (32, 20),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q5_1 => (32, 24),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q8_1 => (32, 36),
            GgmlType::Q2_K => (256, 84),
            GgmlType::Q3_K => (256, 110),
            GgmlType::Q4_K => (256, 144),
            GgmlType::Q5_K => (256, 176),
            GgmlType::Q6_K => (256, 210),
            GgmlType::Q8_K => (256, 292),
            GgmlType::I8 => (1, 1),
            GgmlType::I16 => (1, 2),
            GgmlType::I32 => (1, 4),
            GgmlType::I64 => (1, 8),
            GgmlType::F64 => (1, 8),
            _ => return None,
        })
    }
}

/// Metadata value. Arrays keep only their element type and length for scalar
/// payloads we don't need (e.g. the embedded vocab), except string/int arrays
/// which are retained because model configs live there.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            MetaValue::U8(v) => Some(v as u64),
            MetaValue::U16(v) => Some(v as u64),
            MetaValue::U32(v) => Some(v as u64),
            MetaValue::U64(v) => Some(v),
            MetaValue::I8(v) if v >= 0 => Some(v as u64),
            MetaValue::I16(v) if v >= 0 => Some(v as u64),
            MetaValue::I32(v) if v >= 0 => Some(v as u64),
            MetaValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: GgmlType,
    /// Dimensions as stored in the file (ggml order: fastest-varying first).
    pub dims: Vec<u64>,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
    /// Exact byte length of the quantized payload.
    pub byte_len: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, MetaValue>,
    /// Tensor table in file order.
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the tensor data section begins.
    pub data_start: u64,
}

impl GgufFile {
    /// Parse header, metadata, and tensor table from a reader positioned at
    /// the start of the file. Consumes exactly the non-data prefix.
    pub fn read_header<R: Read>(r: &mut R) -> Result<Self, Error> {
        let mut c = Counting { inner: r, count: 0 };

        if read_u32(&mut c)? != MAGIC {
            return Err(Error::Gguf("bad magic, not a GGUF file".into()));
        }
        let version = read_u32(&mut c)?;
        if !(2..=3).contains(&version) {
            return Err(Error::Gguf(format!("unsupported GGUF version {version}")));
        }
        let tensor_count = read_u64(&mut c)?;
        let kv_count = read_u64(&mut c)?;
        if tensor_count > 1 << 20 || kv_count > 1 << 20 {
            return Err(Error::Gguf("implausible tensor/kv count".into()));
        }

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = read_string(&mut c)?;
            let value = read_value(&mut c)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata
            .get("general.alignment")
            .and_then(MetaValue::as_u64)
        {
            Some(a) if a.is_power_of_two() => a,
            Some(a) => return Err(Error::Gguf(format!("bad alignment {a}"))),
            None => DEFAULT_ALIGNMENT,
        };

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut c)?;
            let n_dims = read_u32(&mut c)?;
            if n_dims > 4 {
                return Err(Error::Gguf(format!("tensor {name}: {n_dims} dims")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut c)?);
            }
            let ggml_type = GgmlType(read_u32(&mut c)?);
            let offset = read_u64(&mut c)?;

            let n_elems: u64 = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| Error::Gguf(format!("tensor {name}: dim overflow")))?;
            let (block, block_bytes) = ggml_type.block_layout().ok_or_else(|| {
                Error::Gguf(format!(
                    "tensor {name}: unsupported ggml type {}",
                    ggml_type.0
                ))
            })?;
            if !n_elems.is_multiple_of(block) {
                return Err(Error::Gguf(format!(
                    "tensor {name}: {n_elems} elements not divisible by block size {block}"
                )));
            }
            let byte_len = n_elems / block * block_bytes;
            tensors.push(TensorInfo {
                name,
                ggml_type,
                dims,
                offset,
                byte_len,
            });
        }

        let data_start = c.count.next_multiple_of(alignment);
        // Consume the padding so the reader is positioned at the data section.
        let pad = data_start - c.count;
        io::copy(&mut (&mut c).take(pad), &mut io::sink())?;

        Ok(GgufFile {
            version,
            alignment,
            metadata,
            tensors,
            data_start,
        })
    }
}

struct Counting<'a, R> {
    inner: &'a mut R,
    count: u64,
}

impl<R: Read> Read for Counting<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count += n as u64;
        Ok(n)
    }
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), Error> {
    r.read_exact(buf).map_err(Error::Io)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, Error> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, Error> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, Error> {
    let len = read_u64(r)?;
    if len > 1 << 24 {
        return Err(Error::Gguf(format!("implausible string length {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    read_exact(r, &mut buf)?;
    String::from_utf8(buf).map_err(|_| Error::Gguf("non-utf8 string".into()))
}

fn read_value<R: Read>(r: &mut R) -> Result<MetaValue, Error> {
    let ty = read_u32(r)?;
    read_value_of_type(r, ty)
}

/// GGUF v3 writer. Exists so tests can construct tiny models from scratch
/// (CI never downloads weights) and so tooling can re-emit modified files.
pub mod write {
    use std::io::Write;

    use super::{DEFAULT_ALIGNMENT, GgmlType, MAGIC, MetaValue};
    use crate::Error;

    pub struct TensorData<'a> {
        pub name: String,
        pub ggml_type: GgmlType,
        /// ggml order: fastest-varying dimension first.
        pub dims: Vec<u64>,
        pub data: &'a [u8],
    }

    /// Write a complete GGUF v3 file. Metadata is written in the order given
    /// (deterministic output); tensors are laid out in the order given with
    /// offsets aligned to `general.alignment` (default 32).
    pub fn write_gguf<W: Write>(
        w: &mut W,
        metadata: &[(String, MetaValue)],
        tensors: &[TensorData<'_>],
    ) -> Result<(), Error> {
        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if !alignment.is_power_of_two() {
            return Err(Error::Gguf(format!("bad alignment {alignment}")));
        }

        for t in tensors {
            let n_elems: u64 = t.dims.iter().product();
            let (block, block_bytes) = t
                .ggml_type
                .block_layout()
                .ok_or_else(|| Error::Gguf(format!("unsupported ggml type {}", t.ggml_type.0)))?;
            if !n_elems.is_multiple_of(block)
                || t.data.len() as u64 != n_elems / block * block_bytes
            {
                return Err(Error::Gguf(format!(
                    "tensor {:?}: data length {} does not match shape {:?}",
                    t.name,
                    t.data.len(),
                    t.dims
                )));
            }
        }

        // Header + metadata + tensor table, buffered so we know data_start.
        let mut head = Vec::new();
        head.extend_from_slice(&MAGIC.to_le_bytes());
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            put_string(&mut head, key);
            put_value_tagged(&mut head, value)?;
        }
        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            offset = offset.next_multiple_of(alignment);
            offsets.push(offset);
            put_string(&mut head, &t.name);
            head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for &d in &t.dims {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&t.ggml_type.0.to_le_bytes());
            head.extend_from_slice(&offset.to_le_bytes());
            offset += t.data.len() as u64;
        }

        let data_start = (head.len() as u64).next_multiple_of(alignment);
        head.resize(data_start as usize, 0);
        w.write_all(&head)?;

        let mut written = 0u64;
        for (t, &off) in tensors.iter().zip(&offsets) {
            w.write_all(&vec![0u8; (off - written) as usize])?;
            w.write_all(t.data)?;
            written = off + t.data.len() as u64;
        }
        Ok(())
    }

    fn put_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn type_tag(v: &MetaValue) -> u32 {
        match v {
            MetaValue::U8(_) => 0,
            MetaValue::I8(_) => 1,
            MetaValue::U16(_) => 2,
            MetaValue::I16(_) => 3,
            MetaValue::U32(_) => 4,
            MetaValue::I32(_) => 5,
            MetaValue::F32(_) => 6,
            MetaValue::Bool(_) => 7,
            MetaValue::String(_) => 8,
            MetaValue::Array(_) => 9,
            MetaValue::U64(_) => 10,
            MetaValue::I64(_) => 11,
            MetaValue::F64(_) => 12,
        }
    }

    fn put_value_tagged(out: &mut Vec<u8>, v: &MetaValue) -> Result<(), Error> {
        out.extend_from_slice(&type_tag(v).to_le_bytes());
        put_value(out, v)
    }

    fn put_value(out: &mut Vec<u8>, v: &MetaValue) -> Result<(), Error> {
        match v {
            MetaValue::U8(x) => out.push(*x),
            MetaValue::I8(x) => out.push(*x as u8),
            MetaValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F32(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
            MetaValue::Bool(x) => out.push(*x as u8),
            MetaValue::String(s) => put_string(out, s),
            MetaValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F64(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
            MetaValue::Array(items) => {
                let first = items
                    .first()
                    .ok_or_else(|| Error::Gguf("cannot write empty metadata array".into()))?;
                let tag = type_tag(first);
                if items.iter().any(|i| type_tag(i) != tag) {
                    return Err(Error::Gguf("heterogeneous metadata array".into()));
                }
                out.extend_from_slice(&tag.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    put_value(out, item)?;
                }
            }
        }
        Ok(())
    }
}

fn read_value_of_type<R: Read>(r: &mut R, ty: u32) -> Result<MetaValue, Error> {
    let mut b8 = [0u8; 8];
    Ok(match ty {
        0 => {
            read_exact(r, &mut b8[..1])?;
            MetaValue::U8(b8[0])
        }
        1 => {
            read_exact(r, &mut b8[..1])?;
            MetaValue::I8(b8[0] as i8)
        }
        2 => {
            read_exact(r, &mut b8[..2])?;
            MetaValue::U16(u16::from_le_bytes([b8[0], b8[1]]))
        }
        3 => {
            read_exact(r, &mut b8[..2])?;
            MetaValue::I16(i16::from_le_bytes([b8[0], b8[1]]))
        }
        4 => MetaValue::U32(read_u32(r)?),
        5 => MetaValue::I32(read_u32(r)? as i32),
        6 => MetaValue::F32(f32::from_bits(read_u32(r)?)),
        7 => {
            read_exact(r, &mut b8[..1])?;
            MetaValue::Bool(b8[0] != 0)
        }
        8 => MetaValue::String(read_string(r)?),
        9 => {
            let elem_ty = read_u32(r)?;
            let len = read_u64(r)?;
            if len > 1 << 26 {
                return Err(Error::Gguf(format!("implausible array length {len}")));
            }
            let mut items = Vec::with_capacity(len.min(1 << 16) as usize);
            for _ in 0..len {
                items.push(read_value_of_type(r, elem_ty)?);
            }
            MetaValue::Array(items)
        }
        10 => MetaValue::U64(read_u64(r)?),
        11 => MetaValue::I64(read_u64(r)? as i64),
        12 => MetaValue::F64(f64::from_bits(read_u64(r)?)),
        other => return Err(Error::Gguf(format!("unknown metadata value type {other}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::write::{TensorData, write_gguf};
    use super::*;

    fn tiny_gguf() -> Vec<u8> {
        let metadata = vec![
            (
                "general.architecture".to_string(),
                MetaValue::String("llama".into()),
            ),
            ("llama.block_count".to_string(), MetaValue::U32(2)),
            ("llama.rope.freq_base".to_string(), MetaValue::F32(10000.0)),
            (
                "tokenizer.ggml.tokens".to_string(),
                MetaValue::Array(vec![
                    MetaValue::String("a".into()),
                    MetaValue::String("b".into()),
                ]),
            ),
        ];
        let t0: Vec<u8> = (0..64 * 4).map(|i| i as u8).collect(); // 64 f32
        let t1 = vec![7u8; 34]; // one Q8_0 block of 32 elements
        let tensors = vec![
            TensorData {
                name: "b.weight".into(),
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
        let mut out = Vec::new();
        write_gguf(&mut out, &metadata, &tensors).unwrap();
        out
    }

    #[test]
    fn writer_parser_roundtrip() {
        let bytes = tiny_gguf();
        let mut cursor = std::io::Cursor::new(&bytes);
        let gguf = GgufFile::read_header(&mut cursor).unwrap();

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.alignment, DEFAULT_ALIGNMENT);
        assert_eq!(
            gguf.metadata.get("general.architecture"),
            Some(&MetaValue::String("llama".into()))
        );
        assert_eq!(
            gguf.metadata
                .get("llama.block_count")
                .and_then(MetaValue::as_u64),
            Some(2)
        );

        assert_eq!(gguf.tensors.len(), 2);
        let t0 = &gguf.tensors[0];
        assert_eq!(
            (t0.name.as_str(), t0.ggml_type, t0.byte_len),
            ("b.weight", GgmlType::F32, 256)
        );
        assert_eq!(t0.dims, vec![8, 8]);
        let t1 = &gguf.tensors[1];
        assert_eq!(
            (t1.name.as_str(), t1.ggml_type, t1.byte_len),
            ("a.weight", GgmlType::Q8_0, 34)
        );
        // Offsets are alignment-padded and relative to the data section.
        assert_eq!(t0.offset, 0);
        assert_eq!(t1.offset, 256);
        assert_eq!(gguf.data_start % gguf.alignment, 0);
        // Reader is positioned exactly at the data section after parsing.
        assert_eq!(cursor.position(), gguf.data_start);
        // And the file is exactly data_start + payload bytes.
        assert_eq!(bytes.len() as u64, gguf.data_start + 256 + 34);
    }

    #[test]
    fn rejects_garbage() {
        assert!(GgufFile::read_header(&mut std::io::Cursor::new(b"NOPE")).is_err());
        let mut truncated = tiny_gguf();
        truncated.truncate(20);
        assert!(GgufFile::read_header(&mut std::io::Cursor::new(&truncated)).is_err());
    }

    #[test]
    fn k_quant_sizes_match_ggml() {
        // One block of 256 elements for each K-quant; sizes from ggml-quants.h.
        for (ty, bytes) in [
            (GgmlType::Q2_K, 84),
            (GgmlType::Q3_K, 110),
            (GgmlType::Q4_K, 144),
            (GgmlType::Q5_K, 176),
            (GgmlType::Q6_K, 210),
            (GgmlType::Q8_K, 292),
        ] {
            assert_eq!(ty.block_layout(), Some((256, bytes)), "type {}", ty.0);
        }
    }
}
