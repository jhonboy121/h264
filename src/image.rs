//! Planar I420 (4:2:0, 8-bit) picture buffers shared by the encoder, the
//! downsampler and the decoder's output views.

use alloc::vec;
use alloc::vec::Vec;

/// Borrowed I420 planes. `y[0]` / `u[0]` / `v[0]` are the top-left samples;
/// chroma planes are `width.div_ceil(2)` x `height.div_ceil(2)`.
#[derive(Debug, Clone, Copy)]
pub struct YuvRef<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub y_stride: usize,
    pub c_stride: usize,
    pub width: u32,
    pub height: u32,
}

impl YuvRef<'_> {
    /// Chroma plane width.
    pub const fn chroma_width(&self) -> u32 {
        self.width.div_ceil(2)
    }

    /// Chroma plane height.
    pub const fn chroma_height(&self) -> u32 {
        self.height.div_ceil(2)
    }

    /// True when every plane holds `width`x`height` samples at its stride.
    pub fn is_valid(&self) -> bool {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (self.chroma_width() as usize, self.chroma_height() as usize);
        w > 0
            && h > 0
            && self.y_stride >= w
            && self.c_stride >= cw
            && plane_fits(self.y.len(), self.y_stride, w, h)
            && plane_fits(self.u.len(), self.c_stride, cw, ch)
            && plane_fits(self.v.len(), self.c_stride, cw, ch)
    }
}

/// `len` covers `h` rows of `w` samples at `stride` (the last row may be short).
const fn plane_fits(len: usize, stride: usize, w: usize, h: usize) -> bool {
    h == 0 || len >= (h - 1) * stride + w
}

/// Owned I420 picture, tightly packed (`y_stride = width`,
/// `c_stride = width.div_ceil(2)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct I420 {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl I420 {
    /// A black (Y = 0, U = V = 128) picture of the given size.
    pub fn new(width: u32, height: u32) -> Self {
        const CHROMA_ZERO: u8 = 128;
        let luma = width as usize * height as usize;
        let chroma = width.div_ceil(2) as usize * height.div_ceil(2) as usize;
        I420 {
            width,
            height,
            y: vec![0; luma],
            u: vec![CHROMA_ZERO; chroma],
            v: vec![CHROMA_ZERO; chroma],
        }
    }

    /// Borrow the planes.
    pub fn as_ref(&self) -> YuvRef<'_> {
        YuvRef {
            y: &self.y,
            u: &self.u,
            v: &self.v,
            y_stride: self.width as usize,
            c_stride: self.width.div_ceil(2) as usize,
            width: self.width,
            height: self.height,
        }
    }
}
