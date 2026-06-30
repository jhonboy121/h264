//! Pure-Rust port of Cisco OpenH264 (internal use).
//!
//! See `docs/PORTING_PLAN.md` for scope and `docs/TRACKER.md` for status.
//!
//! The crate is `no_std + alloc` by default; the `std` feature (on by default)
//! enables file helpers and the test/oracle harness. Codec state owns its scratch
//! buffers (allocated once, reused per frame) — there is no global mutable state.
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::too_many_arguments)] // faithful ports of C kernels carry wide signatures

extern crate alloc;

pub mod bits;
#[cfg(feature = "decoder")]
pub mod decoder;
pub mod dsp;
pub mod error;

#[cfg(feature = "decoder")]
pub mod api;
#[cfg(feature = "decoder")]
pub mod formats;

pub use error::{DecodeError, EncodeError};

#[cfg(feature = "decoder")]
pub use api::{nal_units, Decoder};
#[cfg(feature = "decoder")]
pub use formats::yuv::{DecodedYuv, Frame, VisibleRegion, YUVSource};

/// Crate version of the upstream OpenH264 this port tracks.
pub const UPSTREAM_OPENH264_VERSION: &str = "2.x (master @ e3f5b10)";
