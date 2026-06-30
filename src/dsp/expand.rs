//! Reference-picture border extension (edge replication).
//!
//! Port of `ExpandPictureLuma_c` / `ExpandPictureChroma_c`
//! (`reference/codec/common/src/expand_pic.cpp`): each plane's coded area is
//! replicated outward into its padding border so the 6-tap luma / bilinear
//! chroma motion-compensation halo can read a few samples past the picture
//! edge. Corners take the corner sample. The motion-vector clamp in the inter
//! path keeps every read within this border, so replicating the full padding
//! (matching or exceeding OpenH264's `PADDING_LENGTH`) is bit-exact.

#[cfg(feature = "decoder")]
use crate::decoder::picture::Picture;

/// Replicate the edges of one plane into its `border`-wide padding on all four
/// sides. `origin` is the linear index of coded pixel (0, 0); the coded area is
/// `width` x `height` with row stride `stride`.
pub fn expand_plane(
    plane: &mut [u8],
    stride: usize,
    border: usize,
    width: usize,
    height: usize,
    origin: usize,
) {
    // Left / right borders on each coded row.
    for y in 0..height {
        let row = origin + y * stride;
        let left = plane[row];
        let right = plane[row + width - 1];
        for x in 1..=border {
            plane[row - x] = left;
            plane[row + width - 1 + x] = right;
        }
    }
    // Top / bottom borders, copying whole (already left/right-extended) rows so
    // the corners are filled with the corner sample.
    let row_lo = origin - border; // start of the extended top coded row
    let span = width + 2 * border;
    for y in 1..=border {
        let top_src = origin - border;
        let bot_src = origin + (height - 1) * stride - border;
        let top_dst = top_src - y * stride;
        let bot_dst = bot_src + y * stride;
        plane.copy_within(top_src..top_src + span, top_dst);
        plane.copy_within(bot_src..bot_src + span, bot_dst);
    }
    let _ = row_lo;
}

/// Expand all three planes of a decoded [`Picture`] into their borders.
#[cfg(feature = "decoder")]
pub fn expand_picture(pic: &mut Picture) {
    let lstride = pic.luma_stride;
    let cstride = pic.chroma_stride;
    let border = crate::decoder::picture::PADDING;
    let lo = pic.luma_origin();
    let co = pic.chroma_origin();
    let (lw, lh) = (pic.width, pic.height);
    let (cw, ch) = (pic.width / 2, pic.height / 2);
    expand_plane(&mut pic.y, lstride, border, lw, lh, lo);
    expand_plane(&mut pic.u, cstride, border, cw, ch, co);
    expand_plane(&mut pic.v, cstride, border, cw, ch, co);
}

#[cfg(all(test, feature = "decoder"))]
mod tests {
    use super::*;
    use crate::decoder::picture::{PADDING, Picture};

    #[test]
    fn replicates_edges_and_corners() {
        let mut pic = Picture::new(2, 2); // 32x32 luma
        let stride = pic.luma_stride;
        let o = pic.luma_origin();
        // Fill the coded area with a ramp so edges are distinguishable.
        for y in 0..pic.height {
            for x in 0..pic.width {
                pic.y[o + y * stride + x] = (x as u8).wrapping_mul(3).wrapping_add(y as u8);
            }
        }
        expand_picture(&mut pic);
        let w = pic.width;
        let h = pic.height;
        // Left border equals column 0; right equals last column.
        for y in 0..h {
            let row = o + y * stride;
            assert_eq!(pic.y[row - PADDING], pic.y[row]);
            assert_eq!(pic.y[row - 1], pic.y[row]);
            assert_eq!(pic.y[row + w - 1 + PADDING], pic.y[row + w - 1]);
        }
        // Top border equals row 0 (including its extended borders).
        for x in 0..w {
            assert_eq!(pic.y[o - stride + x], pic.y[o + x]);
            assert_eq!(pic.y[o - PADDING * stride + x], pic.y[o + x]);
        }
        // Top-left corner = pixel(0,0).
        assert_eq!(pic.y[o - stride - 1], pic.y[o]);
        assert_eq!(pic.y[o - PADDING * stride - PADDING], pic.y[o]);
        // Bottom-right corner = pixel(w-1,h-1).
        let br = o + (h - 1) * stride + w - 1;
        assert_eq!(pic.y[br + stride + 1], pic.y[br]);
    }
}
