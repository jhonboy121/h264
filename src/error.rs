//! Error types for the decode and encode paths.

use core::fmt;

/// Errors produced while decoding an H.264 bitstream.
///
/// Bitstream-level problems return `Err` rather than panicking; panics are
/// reserved for internal invariant violations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Reader ran past the end of the available bits.
    UnexpectedEof,
    /// An Exp-Golomb / syntax element was out of its legal range.
    InvalidSyntax(&'static str),
    /// A referenced parameter set (SPS/PPS) was not present.
    MissingParameterSet,
    /// A feature in the bitstream is not supported by this port.
    Unsupported(&'static str),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnexpectedEof => write!(f, "unexpected end of bitstream"),
            DecodeError::InvalidSyntax(s) => write!(f, "invalid syntax: {s}"),
            DecodeError::MissingParameterSet => write!(f, "missing parameter set"),
            DecodeError::Unsupported(s) => write!(f, "unsupported feature: {s}"),
        }
    }
}

/// Errors produced while encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// Input configuration was invalid.
    InvalidConfig(&'static str),
    /// Output buffer too small.
    OutputTooSmall,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            EncodeError::OutputTooSmall => write!(f, "output buffer too small"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}
#[cfg(feature = "std")]
impl std::error::Error for EncodeError {}
