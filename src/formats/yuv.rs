//! YUV output views and the [`YUVSource`] abstraction.
//!
//! [`YUVSource`] mirrors the trait of the `openh264` crate: a planar 4:2:0
//! 8-bit picture exposing visible dimensions, per-plane strides, and plane
//! slices whose first sample is the top-left of the *visible* (cropped) region.
//! Any implementor gains BT.601 YUV→RGB conversion for free via the default
//! methods, which write into caller-provided buffers (no allocation).

use crate::decoder::params::Sps;
use crate::decoder::picture::Picture;
use crate::image::YuvRef;

use super::rgb;

/// YUV→RGB matrix coefficients, as carried by the SPS VUI `matrix_coeffs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMatrix {
    /// BT.601 / SMPTE-170M (the default; also JFIF when paired with full range).
    #[default]
    Bt601,
    /// BT.709 (HD).
    Bt709,
    /// BT.2020 non-constant-luminance (UHD).
    Bt2020,
}

/// Quantization range of the YUV samples (VUI `video_full_range_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorRange {
    /// Studio-swing / limited range (Y ∈ [16, 235]).
    Limited,
    /// Full range (Y ∈ [0, 255]); the default, matching the scalar converter.
    #[default]
    Full,
}

/// Colorimetry a converter should honor: matrix coefficients plus range.
///
/// Derived from the SPS VUI ([`from_sps`](ColorInfo::from_sps)). When the VUI is
/// absent or leaves a field unspecified, the default is **BT.601 full-range** —
/// exactly what the scalar fallback produces. The scalar path always uses
/// BT.601-full regardless of this value; only the `yuv-convert` crate path
/// honors BT.709/BT.2020 and limited range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColorInfo {
    pub matrix: ColorMatrix,
    pub range: ColorRange,
}

impl ColorInfo {
    /// Derive colorimetry from an SPS. Unspecified fields fall back to the
    /// default (BT.601 full-range), matching the scalar converter's output.
    pub(crate) fn from_sps(sps: &Sps) -> Self {
        if !sps.vui_parameters_present_flag {
            return Self::default();
        }
        let vui = &sps.vui;
        let range = match (
            vui.video_signal_type_present_flag,
            vui.video_full_range_flag,
        ) {
            (true, false) => ColorRange::Limited,
            _ => ColorRange::Full,
        };
        let matrix = if vui.colour_description_present_flag {
            match vui.matrix_coeffs {
                // Table E-5: 1 = BT.709, 9 = BT.2020 NCL, 5/6 = BT.601;
                // everything else (incl. unspecified/reserved) → BT.601.
                1 => ColorMatrix::Bt709,
                9 => ColorMatrix::Bt2020,
                _ => ColorMatrix::Bt601,
            }
        } else {
            ColorMatrix::Bt601
        };
        ColorInfo { matrix, range }
    }
}

/// The visible (post-crop) rectangle of a decoded picture, expressed as sample
/// offsets into the coded planes plus visible dimensions. 4:2:0 only.
#[derive(Debug, Clone, Copy)]
pub struct VisibleRegion {
    /// Visible luma width in pixels.
    pub width: usize,
    /// Visible luma height in pixels.
    pub height: usize,
    /// Luma column offset of the visible top-left, from the coded origin.
    pub luma_x: usize,
    /// Luma row offset of the visible top-left, from the coded origin.
    pub luma_y: usize,
    /// Chroma column offset of the visible top-left, from the coded origin.
    pub chroma_x: usize,
    /// Chroma row offset of the visible top-left, from the coded origin.
    pub chroma_y: usize,
}

/// A planar 4:2:0 8-bit YUV picture: visible dimensions, plane strides, and
/// plane accessors. Conversion helpers ([`write_rgb8`](YUVSource::write_rgb8) /
/// [`write_rgba8`](YUVSource::write_rgba8)) come for free as default methods.
pub trait YUVSource {
    /// Visible luma `(width, height)` in pixels.
    fn dimensions(&self) -> (usize, usize);

    /// Row strides in samples of the `(y, u, v)` planes.
    fn strides(&self) -> (usize, usize, usize);

    /// Luma plane, indexed `y()[row * stride.0 + col]`; `[0]` is the visible
    /// top-left sample.
    fn y(&self) -> &[u8];

    /// Cb (U) plane; `[0]` is the visible top-left chroma sample.
    fn u(&self) -> &[u8];

    /// Cr (V) plane; `[0]` is the visible top-left chroma sample.
    fn v(&self) -> &[u8];

    /// Colorimetry to convert with. Defaults to BT.601 full-range (what the
    /// scalar converter always produces); decoder outputs override this from the
    /// SPS VUI so the `yuv-convert` path can honor BT.709/BT.2020 + limited range.
    fn color_info(&self) -> ColorInfo {
        ColorInfo::default()
    }

    /// Bytes required by [`write_rgb8`](Self::write_rgb8): `width * height * 3`.
    fn rgb8_len(&self) -> usize {
        let (w, h) = self.dimensions();
        w * h * 3
    }

    /// Bytes required by [`write_rgba8`](Self::write_rgba8): `width * height * 4`.
    fn rgba8_len(&self) -> usize {
        let (w, h) = self.dimensions();
        w * h * 4
    }

    /// Convert to packed 24-bit RGB (BT.601) into `out`, which must be at least
    /// [`rgb8_len`](Self::rgb8_len) bytes.
    fn write_rgb8(&self, out: &mut [u8]) {
        rgb::write(self, out, false);
    }

    /// Convert to packed 32-bit RGBA (BT.601, alpha = 255) into `out`, which
    /// must be at least [`rgba8_len`](Self::rgba8_len) bytes.
    fn write_rgba8(&self, out: &mut [u8]) {
        rgb::write(self, out, true);
    }
}

/// A borrowed view over a decoded [`Picture`]'s visible region.
///
/// Yielded by [`crate::Decoder::decode`]; valid until the next decode call.
#[derive(Debug, Clone, Copy)]
pub struct DecodedYuv<'a> {
    pic: &'a Picture,
    region: VisibleRegion,
}

impl<'a> DecodedYuv<'a> {
    pub(crate) fn new(pic: &'a Picture, region: VisibleRegion) -> Self {
        DecodedYuv { pic, region }
    }
}

/// An owned decoded frame: a [`Picture`] plus its visible region.
///
/// Yielded by [`crate::Decoder::decode_all`]. Implements [`YUVSource`] directly
/// and can also lend a borrowed [`DecodedYuv`] view via [`view`](Frame::view).
#[derive(Debug, Clone)]
pub struct Frame {
    pic: Picture,
    region: VisibleRegion,
}

impl Frame {
    pub(crate) fn new(pic: Picture, region: VisibleRegion) -> Self {
        Frame { pic, region }
    }

    /// Borrow this frame as a [`DecodedYuv`] view.
    pub fn view(&self) -> DecodedYuv<'_> {
        DecodedYuv::new(&self.pic, self.region)
    }

    /// Consume the frame, returning the underlying [`Picture`] with its visible
    /// (post-crop) region stamped onto it so consumers of the bare `Picture`
    /// (e.g. [`crate::decoder::decode_stream`]) can emit the cropped rectangle.
    pub(crate) fn into_picture(mut self) -> Picture {
        self.pic.visible_x = self.region.luma_x;
        self.pic.visible_y = self.region.luma_y;
        self.pic.visible_width = self.region.width;
        self.pic.visible_height = self.region.height;
        self.pic
    }
}

fn luma_slice<'a>(pic: &'a Picture, r: &VisibleRegion) -> &'a [u8] {
    let off = pic.luma_origin() + r.luma_y * pic.luma_stride + r.luma_x;
    &pic.y[off..]
}

fn chroma_slice<'a>(plane: &'a [u8], pic: &Picture, r: &VisibleRegion) -> &'a [u8] {
    let off = pic.chroma_origin() + r.chroma_y * pic.chroma_stride + r.chroma_x;
    &plane[off..]
}

impl YUVSource for DecodedYuv<'_> {
    fn dimensions(&self) -> (usize, usize) {
        (self.region.width, self.region.height)
    }
    fn strides(&self) -> (usize, usize, usize) {
        (
            self.pic.luma_stride,
            self.pic.chroma_stride,
            self.pic.chroma_stride,
        )
    }
    fn y(&self) -> &[u8] {
        luma_slice(self.pic, &self.region)
    }
    fn u(&self) -> &[u8] {
        chroma_slice(&self.pic.u, self.pic, &self.region)
    }
    fn v(&self) -> &[u8] {
        chroma_slice(&self.pic.v, self.pic, &self.region)
    }
    fn color_info(&self) -> ColorInfo {
        self.pic.color
    }
}

impl YUVSource for Frame {
    fn dimensions(&self) -> (usize, usize) {
        (self.region.width, self.region.height)
    }
    fn strides(&self) -> (usize, usize, usize) {
        (
            self.pic.luma_stride,
            self.pic.chroma_stride,
            self.pic.chroma_stride,
        )
    }
    fn y(&self) -> &[u8] {
        luma_slice(&self.pic, &self.region)
    }
    fn u(&self) -> &[u8] {
        chroma_slice(&self.pic.u, &self.pic, &self.region)
    }
    fn v(&self) -> &[u8] {
        chroma_slice(&self.pic.v, &self.pic, &self.region)
    }
    fn color_info(&self) -> ColorInfo {
        self.pic.color
    }
}

fn yuv_ref<'a>(pic: &'a Picture, r: &VisibleRegion) -> YuvRef<'a> {
    YuvRef {
        y: luma_slice(pic, r),
        u: chroma_slice(&pic.u, pic, r),
        v: chroma_slice(&pic.v, pic, r),
        y_stride: pic.luma_stride,
        c_stride: pic.chroma_stride,
        width: r.width as u32,
        height: r.height as u32,
    }
}

impl DecodedYuv<'_> {
    /// The visible (cropped) picture as borrowed I420 planes.
    pub fn yuv(&self) -> YuvRef<'_> {
        yuv_ref(self.pic, &self.region)
    }
}

impl Frame {
    /// The visible (cropped) picture as borrowed I420 planes.
    pub fn yuv(&self) -> YuvRef<'_> {
        yuv_ref(&self.pic, &self.region)
    }
}
