//! DSP kernels shared by the decoder and encoder.
//!
//! Each kernel has an always-compiled scalar reference (the conformance
//! baseline). SIMD variants land in P10 behind the `simd` feature with runtime
//! dispatch. Ports `reference/codec/common/src/*` and the transform/intra/MC
//! kernels from the decoder/encoder cores. See `docs/REFERENCE_MAP.md`.

pub mod transform;

pub use transform::clip1;

/// Generic clamp of `x` into `[lo, hi]` (`WELS_CLIP3`).
#[inline(always)]
pub fn clip3(x: i32, lo: i32, hi: i32) -> i32 {
    x.clamp(lo, hi)
}

// Submodules are added per P1 item:
pub mod copy;
pub mod intra_pred;
pub mod sad;
// pub mod mc;
// pub mod deblock;
// pub mod expand;
// pub mod tables;
