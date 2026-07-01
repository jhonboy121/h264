//! YUV→RGB conversion.
//!
//! Two paths share the [`write`] entry point, selected at compile time:
//!
//! * **Default (scalar).** A fixed-point BT.601 **full-range** transform with
//!   rounding and `[0, 255]` clamping, mirroring OpenH264's integer color
//!   helpers. `no_std`, zero-dependency, and BT.601-full-*only* — it ignores the
//!   picture's [`ColorInfo`](super::yuv::ColorInfo). Full range is used so that
//!   neutral chroma reproduces luma exactly (`Y, 128, 128 -> (Y, Y, Y)`).
//! * **`yuv-convert` feature.** Routes through the SIMD `yuv` crate, honoring the
//!   picture's matrix (BT.601/709/2020) and range (full/limited) from the SPS
//!   VUI. Requires `std`.

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
///
/// With the `yuv-convert` feature this routes through the SIMD `yuv` crate and
/// honors the source's [`ColorInfo`](super::yuv::ColorInfo); otherwise it uses
/// the scalar BT.601-full-range kernel below.
pub fn write<S: YUVSource + ?Sized>(src: &S, out: &mut [u8], alpha: bool) {
    #[cfg(feature = "yuv-convert")]
    {
        write_yuv_crate(src, out, alpha);
    }
    #[cfg(not(feature = "yuv-convert"))]
    {
        write_scalar(src, out, alpha);
    }
}

/// Scalar BT.601 full-range conversion. Always used on the default (no_std,
/// zero-dep) build; ignores the source's [`ColorInfo`](super::yuv::ColorInfo).
#[cfg(not(feature = "yuv-convert"))]
fn write_scalar<S: YUVSource + ?Sized>(src: &S, out: &mut [u8], alpha: bool) {
    let (w, h) = src.dimensions();
    let (ys, us, vs) = src.strides();
    let bpp = if alpha { 4 } else { 3 };
    let need = w * h * bpp;
    assert!(
        out.len() >= need,
        "output buffer too small: {} < {}",
        out.len(),
        need
    );

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

/// SIMD conversion via the `yuv` crate, honoring the source's matrix + range.
#[cfg(feature = "yuv-convert")]
fn write_yuv_crate<S: YUVSource + ?Sized>(src: &S, out: &mut [u8], alpha: bool) {
    use super::yuv::{ColorMatrix, ColorRange};
    use yuv::{YuvPlanarImage, YuvRange, YuvStandardMatrix};

    let (w, h) = src.dimensions();
    let (ys, us, vs) = src.strides();
    let bpp = if alpha { 4 } else { 3 };
    let need = w * h * bpp;
    assert!(
        out.len() >= need,
        "output buffer too small: {} < {}",
        out.len(),
        need
    );

    let info = src.color_info();
    let range = match info.range {
        ColorRange::Limited => YuvRange::Limited,
        ColorRange::Full => YuvRange::Full,
    };
    let matrix = match info.matrix {
        ColorMatrix::Bt601 => YuvStandardMatrix::Bt601,
        ColorMatrix::Bt709 => YuvStandardMatrix::Bt709,
        ColorMatrix::Bt2020 => YuvStandardMatrix::Bt2020,
    };

    // 4:2:0 chroma dimensions round up, so odd visible sizes are covered.
    let image = YuvPlanarImage {
        y_plane: src.y(),
        y_stride: ys as u32,
        u_plane: src.u(),
        u_stride: us as u32,
        v_plane: src.v(),
        v_stride: vs as u32,
        width: w as u32,
        height: h as u32,
    };
    let dst_stride = (w * bpp) as u32;
    let dst = &mut out[..need];
    let result = if alpha {
        yuv::yuv420_to_rgba(&image, dst, dst_stride, range, matrix)
    } else {
        yuv::yuv420_to_rgb(&image, dst, dst_stride, range, matrix)
    };
    result.expect("yuv420_to_rgb(a) conversion failed");
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
            Solid {
                yp: vec![y; 4],
                up: vec![u; 1],
                vp: vec![v; 1],
            }
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
        assert!(
            near(red.0, 255, 3) && near(red.1, 0, 3) && near(red.2, 0, 3),
            "{red:?}"
        );
        let green = yuv_to_rgb(150, 44, 21);
        assert!(
            near(green.0, 0, 4) && near(green.1, 255, 4) && near(green.2, 0, 4),
            "{green:?}"
        );
        let blue = yuv_to_rgb(29, 255, 107);
        assert!(
            near(blue.0, 0, 3) && near(blue.1, 0, 3) && near(blue.2, 255, 3),
            "{blue:?}"
        );
    }

    // Scalar path: neutral gray reproduces luma exactly. Feature-off only, so
    // the byte-exact expectation is preserved unchanged.
    #[cfg(not(feature = "yuv-convert"))]
    #[test]
    fn write_rgb8_fills_buffer() {
        let src = Solid::new(128, 128, 128);
        let mut out = vec![0u8; src.rgb8_len()];
        src.write_rgb8(&mut out);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|&b| b == 128));

        let mut rgba = vec![0u8; src.rgba8_len()];
        src.write_rgba8(&mut rgba);
        assert_eq!(
            rgba,
            vec![
                128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255
            ]
        );
    }

    // `yuv-convert` path: the SIMD crate rounds slightly differently, so the same
    // conversions are checked within a small tolerance. Buffer lengths and the
    // full-range neutral-gray / primary reproductions still hold.
    #[cfg(feature = "yuv-convert")]
    #[test]
    fn write_rgb8_via_crate() {
        // rgb8_len / rgba8_len are the packed sizes.
        let src = Solid::new(128, 128, 128);
        assert_eq!(src.rgb8_len(), 12);
        assert_eq!(src.rgba8_len(), 16);

        // Neutral gray (full range) -> ~(128,128,128).
        let mut out = vec![0u8; src.rgb8_len()];
        src.write_rgb8(&mut out);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|&b| near(b, 128, 2)), "{out:?}");

        // RGBA alpha is opaque; color channels track the RGB result.
        let mut rgba = vec![0u8; src.rgba8_len()];
        src.write_rgba8(&mut rgba);
        assert_eq!(rgba.len(), 16);
        for px in rgba.chunks_exact(4) {
            assert!(near(px[0], 128, 2) && near(px[1], 128, 2) && near(px[2], 128, 2));
            assert_eq!(px[3], 255);
        }

        // Full-range primaries reproduce within tolerance.
        let expect = [
            (76u8, 84u8, 255u8, (255u8, 0u8, 0u8)), // red
            (150, 44, 21, (0, 255, 0)),             // green
            (29, 255, 107, (0, 0, 255)),            // blue
        ];
        for (y, u, v, (er, eg, eb)) in expect {
            let s = Solid::new(y, u, v);
            let mut o = vec![0u8; s.rgb8_len()];
            s.write_rgb8(&mut o);
            assert!(
                near(o[0], er, 6) && near(o[1], eg, 6) && near(o[2], eb, 6),
                "yuv({y},{u},{v}) -> {:?}, want ~({er},{eg},{eb})",
                &o[0..3]
            );
        }
    }
}
