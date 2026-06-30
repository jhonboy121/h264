//! Decoded-picture sample buffers (Y/Cb/Cr planes with a padding border).
//!
//! Mirrors the role of `PPicture`/`pData[]` in
//! `reference/codec/decoder/core/inc/picture.h`. Each plane is allocated once
//! (`Picture::new`) with a `PADDING` border on every side so that future motion
//! compensation and intra prediction can read a few samples outside the coded
//! area without bounds checks. 4:2:0 8-bit only.

use alloc::vec;
use alloc::vec::Vec;

/// Border, in samples, around every plane. OpenH264 uses 32 (`PADDING_LENGTH`).
pub const PADDING: usize = 32;

/// A single decoded frame: planar 4:2:0 8-bit, each plane padded by [`PADDING`].
#[derive(Clone, Debug)]
pub struct Picture {
    /// Luma plane (length `luma_stride * (mb_height*16 + 2*PADDING)`).
    pub y: Vec<u8>,
    /// Cb plane.
    pub u: Vec<u8>,
    /// Cr plane.
    pub v: Vec<u8>,
    /// Row stride of the luma plane (including both borders).
    pub luma_stride: usize,
    /// Row stride of each chroma plane (including both borders).
    pub chroma_stride: usize,
    /// Coded luma width in pixels (`mb_width * 16`).
    pub width: usize,
    /// Coded luma height in pixels (`mb_height * 16`).
    pub height: usize,
    pub mb_width: usize,
    pub mb_height: usize,
}

impl Picture {
    /// Allocate all three planes for an `mb_width` x `mb_height` macroblock grid.
    pub fn new(mb_width: usize, mb_height: usize) -> Self {
        let width = mb_width * 16;
        let height = mb_height * 16;
        let cwidth = mb_width * 8;
        let cheight = mb_height * 8;

        let luma_stride = width + 2 * PADDING;
        let chroma_stride = cwidth + 2 * PADDING;
        let luma_len = luma_stride * (height + 2 * PADDING);
        let chroma_len = chroma_stride * (cheight + 2 * PADDING);

        Picture {
            y: vec![0u8; luma_len],
            u: vec![0u8; chroma_len],
            v: vec![0u8; chroma_len],
            luma_stride,
            chroma_stride,
            width,
            height,
            mb_width,
            mb_height,
        }
    }

    /// Byte offset of luma pixel (0, 0) — the top-left of the coded area.
    #[inline]
    pub fn luma_origin(&self) -> usize {
        PADDING * self.luma_stride + PADDING
    }

    /// Byte offset of chroma pixel (0, 0).
    #[inline]
    pub fn chroma_origin(&self) -> usize {
        PADDING * self.chroma_stride + PADDING
    }

    /// Luma top-left sample offset of macroblock `(mb_x, mb_y)`.
    #[inline]
    pub fn luma_mb_offset(&self, mb_x: usize, mb_y: usize) -> usize {
        self.luma_origin() + mb_y * 16 * self.luma_stride + mb_x * 16
    }

    /// Chroma top-left sample offset of macroblock `(mb_x, mb_y)`.
    #[inline]
    pub fn chroma_mb_offset(&self, mb_x: usize, mb_y: usize) -> usize {
        self.chroma_origin() + mb_y * 8 * self.chroma_stride + mb_x * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_and_offsets() {
        // QCIF: 11 x 9 macroblocks -> 176 x 144 luma, 88 x 72 chroma.
        let p = Picture::new(11, 9);
        assert_eq!(p.width, 176);
        assert_eq!(p.height, 144);
        assert_eq!(p.luma_stride, 176 + 64);
        assert_eq!(p.chroma_stride, 88 + 64);
        assert_eq!(p.y.len(), p.luma_stride * (144 + 64));
        assert_eq!(p.u.len(), p.chroma_stride * (72 + 64));

        // The top-left neighbor of MB(0,0) (origin - stride - 1) is in bounds.
        let o = p.luma_mb_offset(0, 0);
        assert_eq!(o, p.luma_origin());
        assert!(o >= p.luma_stride + 1);

        // Adjacent macroblocks are 16 luma / 8 chroma samples apart.
        assert_eq!(p.luma_mb_offset(1, 0) - p.luma_mb_offset(0, 0), 16);
        assert_eq!(p.luma_mb_offset(0, 1) - p.luma_mb_offset(0, 0), 16 * p.luma_stride);
        assert_eq!(p.chroma_mb_offset(1, 0) - p.chroma_mb_offset(0, 0), 8);
    }
}
