//! Commitments, hashing, and trace formats for verifiable local LLM inference.
//!
//! This crate is dependency-light by design (blake3 + serde only) so that a
//! verifier can build it anywhere, without candle or Metal.

pub mod chain;
pub mod commit;
pub mod gguf;
pub mod hex;
pub mod merkle;
pub mod protocol;
pub mod trace;
pub mod transcript;

use std::fmt;

/// A 32-byte BLAKE3 digest, serialized as lowercase hex in JSON.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash32(pub [u8; 32]);

impl Hash32 {
    pub fn to_hex(self) -> String {
        hex::encode(&self.0)
    }

    pub fn from_hex(s: &str) -> Result<Self, Error> {
        let bytes = hex::decode(s).ok_or_else(|| Error::BadHash(s.into()))?;
        let arr: [u8; 32] = bytes.try_into().map_err(|_| Error::BadHash(s.into()))?;
        Ok(Hash32(arr))
    }
}

impl From<blake3::Hash> for Hash32 {
    fn from(h: blake3::Hash) -> Self {
        Hash32(*h.as_bytes())
    }
}

impl fmt::Debug for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash32({})", self.to_hex())
    }
}

impl fmt::Display for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl serde::Serialize for Hash32 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> serde::Deserialize<'de> for Hash32 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Hash32::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Gguf(String),
    BadHash(String),
    NonFiniteLogit { step: usize, index: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Gguf(msg) => write!(f, "gguf: {msg}"),
            Error::BadHash(s) => write!(f, "invalid hash literal: {s:?}"),
            Error::NonFiniteLogit { step, index } => {
                write!(
                    f,
                    "NaN logit at step {step}, index {index}; refusing to commit"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
