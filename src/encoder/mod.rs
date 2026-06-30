//! Encoder DSP and core (feature `encoder`).
//!
//! Pure-Rust ports of the OpenH264 encoder kernels. Shared transform / quant /
//! SATD kernels live in [`crate::dsp`]; this module holds the encoder-only
//! pieces (intra prediction for mode decision, and — in later phases — mode
//! decision, motion estimation, CAVLC/CABAC writing and rate control).

pub mod cavlc_writer;
pub mod intra_pred;
