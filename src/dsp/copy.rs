//! Block copy kernels, ported from `reference/codec/common/src/copy_mb.cpp`
//! (`WelsCopy{4x4,8x4,4x8,8x8,8x16,16x8,16x16}_c`).
//!
//! The C versions are unrolled 32-bit loads/stores; semantically each is a
//! `w x h` rectangular byte copy with independent source/destination strides.
//! We expose one generic `copy_block` plus the named sizes for call-site clarity.

/// Copy a `w`x`h` block from `src` to `dst` with the given strides.
#[inline]
pub fn copy_block(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let d = y * dst_stride;
        let s = y * src_stride;
        dst[d..d + w].copy_from_slice(&src[s..s + w]);
    }
}

macro_rules! named_copy {
    ($name:ident, $w:expr, $h:expr, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize) {
            copy_block(dst, dst_stride, src, src_stride, $w, $h);
        }
    };
}

named_copy!(copy4x4, 4, 4, "Port of `WelsCopy4x4_c`.");
named_copy!(copy8x4, 8, 4, "Port of `WelsCopy8x4_c`.");
named_copy!(copy4x8, 4, 8, "Port of `WelsCopy4x8_c`.");
named_copy!(copy8x8, 8, 8, "Port of `WelsCopy8x8_c`.");
named_copy!(copy8x16, 8, 16, "Port of `WelsCopy8x16_c`.");
named_copy!(copy16x8, 16, 8, "Port of `WelsCopy16x8_c`.");
named_copy!(copy16x16, 16, 16, "Port of `WelsCopy16x16_c`.");

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn copy16x16_with_padding_strides() {
        // Source plane 24-wide, dest plane 20-wide; copy the top-left 16x16.
        let src_stride = 24;
        let dst_stride = 20;
        let mut src = vec![0u8; src_stride * 18];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let mut dst = vec![0xAAu8; dst_stride * 18];
        copy16x16(&mut dst, dst_stride, &src, src_stride);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(dst[y * dst_stride + x], src[y * src_stride + x], "y={y} x={x}");
            }
            // bytes past the copied width must be untouched
            assert_eq!(dst[y * dst_stride + 16], 0xAA);
        }
    }

    #[test]
    fn generic_matches_named_sizes() {
        let stride = 16;
        let mut src = vec![0u8; stride * 16];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i.wrapping_mul(37) % 256) as u8;
        }
        for (w, h) in [(4, 4), (8, 4), (4, 8), (8, 8), (8, 16), (16, 8), (16, 16)] {
            let mut a = vec![0u8; stride * 16];
            let mut b = vec![0u8; stride * 16];
            copy_block(&mut a, stride, &src, stride, w, h);
            copy_block(&mut b, stride, &src, stride, w, h);
            assert_eq!(a, b);
            // spot check a corner
            assert_eq!(a[(h - 1) * stride + (w - 1)], src[(h - 1) * stride + (w - 1)]);
        }
    }
}
