//! Output pixel formats: YUV views and YUV→RGB conversion.
//!
//! [`yuv`] defines the [`YUVSource`](yuv::YUVSource) trait plus the
//! [`DecodedYuv`](yuv::DecodedYuv) / [`Frame`](yuv::Frame) decoder-output types;
//! [`rgb`] holds the scalar BT.601 conversion kernel used by their default
//! `write_rgb8` / `write_rgba8` methods.

pub mod rgb;
pub mod yuv;
