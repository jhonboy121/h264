//! Pure-Rust port of Cisco OpenH264 (internal use).
//!
//! See `docs/PORTING_PLAN.md` for scope and `docs/TRACKER.md` for status.
//!
//! The crate is `no_std + alloc` by default; the `std` feature (on by default)
//! enables file helpers and the test/oracle harness. Codec state owns its scratch
//! buffers (allocated once, reused per frame) — there is no global mutable state.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bits;
#[cfg(feature = "decoder")]
pub mod decoder;
pub mod dsp;
#[cfg(feature = "encoder")]
pub mod encoder;
pub mod error;
pub mod image;
pub mod nal;
pub mod processing;

#[cfg(feature = "decoder")]
pub mod api;
#[cfg(feature = "decoder")]
pub mod formats;

pub use error::{DecodeError, EncodeError};
pub use image::{I420, YuvRef};
pub use processing::downsample::scale_i420;

#[cfg(feature = "decoder")]
pub use api::{Decoder, nal_units, sps_dimensions};
#[cfg(feature = "decoder")]
pub use formats::yuv::{
    ColorInfo, ColorMatrix, ColorRange, DecodedYuv, Frame, VisibleRegion, YUVSource,
};

/// Crate version of the upstream OpenH264 this port tracks.
pub const UPSTREAM_OPENH264_VERSION: &str = "2.x (master @ e3f5b10)";
