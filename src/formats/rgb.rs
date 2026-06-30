//! Scalar YUV→RGB conversion (BT.601, full-range / JFIF integer form).
//!
//! Mirrors the integer color-conversion helpers shipped with OpenH264's
//! utilities: a fixed-point BT.601 transform with rounding and `[0, 255]`
//! clamping. Full-range is used so that neutral chroma reproduces luma exactly
//! (`Y, 128, 128 -> (Y, Y, Y)`).

use super::yuv::YUVSource;

// 16.16 fixed-point BT.601 coefficients (full range).
//   R = Y                 + 1.402  * (V-128)
//   G = Y - 0.344136*(U-128) - 0.714136 * (V-128)
//   B = Y + 1.772  * (U-128)
const CR_R: i32 = 91_881; // 1.402   << 16
const CB_G: i32 = 22_554; // 0.344136 << 16
const CR_G: i32 = 46_802; // 0.714136 << 16
const CB_B: i32 = 116_130; // 1.772  << 16

#[inline]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Convert one YUV sample to `(r, g, b)`.
#[inline]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let yi = y as i32;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    let r = yi + ((CR_R * e + 0x8000) >> 16);
    let g = yi - ((CB_G * d + CR_G * e + 0x8000) >> 16);
    let b = yi + ((CB_B * d + 0x8000) >> 16);
    (clamp8(r), clamp8(g), clamp8(b))
}

/// Pack `src`'s visible region into `out` as RGB8 (`alpha = false`) or RGBA8
/// (`alpha = true`). Panics if `out` is shorter than the required length.
pub fn write<S: YUVSource + ?Sized>(src: &S, out: &mut [u8], alpha: bool) {
    let (w, h) = src.dimensions();
    let (ys, us, vs) = src.strides();
    let bpp = if alpha { 4 } else { 3 };
    let need = w * h * bpp;
    assert!(out.len() >= need, "output buffer too small: {} < {}", out.len(), need);

    let yp = src.y();
    let up = src.u();
    let vp = src.v();

    for row in 0..h {
        let yrow = row * ys;
        let crow = (row / 2) * us;
        let vrow = (row / 2) * vs;
        for col in 0..w {
            let (r, g, b) = yuv_to_rgb(yp[yrow + col], up[crow + col / 2], vp[vrow + col / 2]);
            let o = (row * w + col) * bpp;
            out[o] = r;
            out[o + 1] = g;
            out[o + 2] = b;
            if alpha {
                out[o + 3] = 255;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::yuv::YUVSource;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A 2x2 solid-color YUV source (single chroma sample) for unit testing.
    struct Solid {
        yp: Vec<u8>,
        up: Vec<u8>,
        vp: Vec<u8>,
    }
    impl Solid {
        fn new(y: u8, u: u8, v: u8) -> Self {
            Solid { yp: vec![y; 4], up: vec![u; 1], vp: vec![v; 1] }
        }
    }
    impl YUVSource for Solid {
        fn dimensions(&self) -> (usize, usize) {
            (2, 2)
        }
        fn strides(&self) -> (usize, usize, usize) {
            (2, 1, 1)
        }
        fn y(&self) -> &[u8] {
            &self.yp
        }
        fn u(&self) -> &[u8] {
            &self.up
        }
        fn v(&self) -> &[u8] {
            &self.vp
        }
    }

    fn near(a: u8, b: u8, tol: i32) -> bool {
        (a as i32 - b as i32).abs() <= tol
    }

    #[test]
    fn neutral_gray_is_exact() {
        // Y=128, U=V=128 -> (128,128,128) exactly (full-range BT.601).
        assert_eq!(yuv_to_rgb(128, 128, 128), (128, 128, 128));
        assert_eq!(yuv_to_rgb(0, 128, 128), (0, 0, 0));
        assert_eq!(yuv_to_rgb(255, 128, 128), (255, 255, 255));
    }

    #[test]
    fn primaries_round_trip() {
        // JFIF/full-range encodings of the RGB primaries.
        let red = yuv_to_rgb(76, 84, 255);
        assert!(near(red.0, 255, 3) && near(red.1, 0, 3) && near(red.2, 0, 3), "{red:?}");
        let green = yuv_to_rgb(150, 44, 21);
        assert!(near(green.0, 0, 4) && near(green.1, 255, 4) && near(green.2, 0, 4), "{green:?}");
        let blue = yuv_to_rgb(29, 255, 107);
        assert!(near(blue.0, 0, 3) && near(blue.1, 0, 3) && near(blue.2, 255, 3), "{blue:?}");
    }

    #[test]
    fn write_rgb8_fills_buffer() {
        let src = Solid::new(128, 128, 128);
        let mut out = vec![0u8; src.rgb8_len()];
        src.write_rgb8(&mut out);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|&b| b == 128));

        let mut rgba = vec![0u8; src.rgba8_len()];
        src.write_rgba8(&mut rgba);
        assert_eq!(rgba, vec![128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255]);
    }
}
