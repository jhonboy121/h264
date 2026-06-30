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

/// Block geometry: width/height of a rectangular sample block.
#[derive(Clone, Copy)]
pub struct Dim {
    pub w: usize,
    pub h: usize,
}

/// A motion vector (quarter-pel for luma, eighth-pel for chroma).
#[derive(Clone, Copy)]
pub struct Mv {
    pub x: i16,
    pub y: i16,
}

/// Read-only rectangular view into a plane: element `(y, x)` is
/// `data[off + y * stride + x]`.
#[derive(Clone, Copy)]
pub struct Blk<'a> {
    pub data: &'a [u8],
    pub off: usize,
    pub stride: usize,
}

/// Mutable rectangular view into a plane (see [`Blk`]).
pub struct BlkMut<'a> {
    pub data: &'a mut [u8],
    pub off: usize,
    pub stride: usize,
}

// Submodules are added per P1 item:
pub mod copy;
pub mod deblock;
pub mod expand;
pub mod intra_pred;
pub mod mc;
pub mod sad;
#[cfg(feature = "simd")]
pub mod simd;
pub mod satd;
pub mod tables;
// Added in later phases:
// pub mod expand;   // P3 (coupled to padded Plane buffer)
