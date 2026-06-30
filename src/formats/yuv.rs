//! YUV output views and the [`YUVSource`] abstraction.
//!
//! [`YUVSource`] mirrors the trait of the `openh264` crate: a planar 4:2:0
//! 8-bit picture exposing visible dimensions, per-plane strides, and plane
//! slices whose first sample is the top-left of the *visible* (cropped) region.
//! Any implementor gains BT.601 YUV→RGB conversion for free via the default
//! methods, which write into caller-provided buffers (no allocation).

use crate::decoder::picture::Picture;

use super::rgb;

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
        (self.pic.luma_stride, self.pic.chroma_stride, self.pic.chroma_stride)
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
}

impl YUVSource for Frame {
    fn dimensions(&self) -> (usize, usize) {
        (self.region.width, self.region.height)
    }
    fn strides(&self) -> (usize, usize, usize) {
        (self.pic.luma_stride, self.pic.chroma_stride, self.pic.chroma_stride)
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
}
