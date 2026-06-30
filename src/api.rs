//! Ergonomic public decode API.
//!
//! [`Decoder`] is a self-contained H.264 decoder (no global state): construct
//! it, feed Annex-B packets to [`decode`](Decoder::decode), and read back
//! [`DecodedYuv`] pictures. The shape mirrors the `openh264` crate's `Decoder`
//! while being a pure-Rust implementation over this crate's own decode path.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::decoder::frame::StreamDecoder;
use crate::decoder::nal::annexb_nal_units;
use crate::error::DecodeError;
use crate::formats::yuv::{DecodedYuv, Frame};

/// A stateful H.264 decoder owning its parameter sets and decoded-picture
/// buffer. Supports baseline + Main profile (I/P slices, CAVLC or CABAC,
/// 4:2:0 8-bit).
///
/// ```
/// use h264::{Decoder, YUVSource};
///
/// # const STREAM: &[u8] = include_bytes!(concat!(
/// #     env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/BANM_MW_D.264"));
/// let mut decoder = Decoder::new();
/// // Decode a whole Annex-B stream into owned frames...
/// let frames = decoder.decode_all(STREAM).unwrap();
/// let first = &frames[0];
/// assert_eq!(first.dimensions(), (176, 144));
///
/// // ...and convert one to RGB into a caller-owned buffer.
/// let mut rgb = vec![0u8; first.rgb8_len()];
/// first.write_rgb8(&mut rgb);
/// ```
pub struct Decoder {
    inner: StreamDecoder,
    ready: VecDeque<Frame>,
    front: Option<Frame>,
}

impl Decoder {
    /// Create a new decoder with empty state.
    pub fn new() -> Self {
        Decoder {
            inner: StreamDecoder::new(),
            ready: VecDeque::new(),
            front: None,
        }
    }

    /// Feed Annex-B NAL data and return the next completed picture, if any.
    ///
    /// A picture is emitted once the *next* picture's first slice has been seen,
    /// so the final frame of a stream is retrieved with [`flush`](Self::flush).
    /// The returned [`DecodedYuv`] borrows the decoder and is invalidated by the
    /// next `decode`/`flush` call.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<DecodedYuv<'_>>, DecodeError> {
        let mut produced = Vec::new();
        self.inner.feed(packet, &mut produced)?;
        self.ready.extend(produced);
        self.front = self.ready.pop_front();
        Ok(self.front.as_ref().map(Frame::view))
    }

    /// Flush the trailing in-flight picture (and any queued frames), one per
    /// call. Returns `None` once the decoder is fully drained.
    pub fn flush(&mut self) -> Option<DecodedYuv<'_>> {
        let mut produced = Vec::new();
        self.inner.finish(&mut produced);
        self.ready.extend(produced);
        self.front = self.ready.pop_front();
        self.front.as_ref().map(Frame::view)
    }

    /// Decode an entire Annex-B stream into owned [`Frame`]s in decode order.
    ///
    /// Convenience over repeated [`decode`](Self::decode) + [`flush`](Self::flush)
    /// when the whole stream is available up front.
    pub fn decode_all(&mut self, annexb: &[u8]) -> Result<Vec<Frame>, DecodeError> {
        let mut out = Vec::new();
        self.inner.feed(annexb, &mut out)?;
        self.inner.finish(&mut out);
        Ok(out)
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterate the Annex-B NAL units in `data`, yielding each NAL's bytes (header
/// byte first, start code excluded). Mirrors `openh264`'s `nal_units` helper.
pub fn nal_units(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    annexb_nal_units(data)
}
