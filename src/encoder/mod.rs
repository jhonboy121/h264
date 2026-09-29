//! Encoder DSP and core (feature `encoder`).
//!
//! Pure-Rust ports of the OpenH264 encoder kernels. Shared transform / quant /
//! SATD kernels live in [`crate::dsp`]; this module holds the encoder-only
//! pieces (intra prediction, CAVLC writing, parameter-set + NAL generation,
//! rate control, and the frame encode loop).
//!
//! [`Encoder`] is the public facade. [`Encoder::with_config`] +
//! [`Encoder::encode`] is the rate-controlled path (target bitrate, IDR
//! interval, slices); [`Encoder::new`] + [`Encoder::encode_frame`] codes every
//! frame at one fixed QP. Either way the first frame is an IDR access unit
//! (SPS + PPS + IDR slices) and the stream decodes back through
//! [`crate::decoder`] bit-exactly.

mod encode_mb;

pub mod cavlc_writer;
pub mod intra_pred;
pub mod motion_est;
pub mod nal_encap;
pub mod paraset;
pub mod ratectl;

use alloc::vec;
use alloc::vec::Vec;

use crate::bits::BitWriter;
use crate::dsp::sad::sad;
use crate::error::EncodeError;
use crate::image::YuvRef;
use encode_mb::{BORDER, FrameEnc, MbDims, PlaneRefs};
use nal_encap::append_annexb_nal;
use paraset::ParamConfig;
use ratectl::RateControl;

/// NAL unit types we emit.
const NAL_NON_IDR_SLICE: u8 = 1;
const NAL_IDR_SLICE: u8 = crate::nal::IDR;
const NAL_SPS: u8 = crate::nal::SPS;
const NAL_PPS: u8 = crate::nal::PPS;
/// `nal_ref_idc` of parameter sets / IDR slices and of P slices.
const REF_IDC_HIGHEST: u8 = 3;
const REF_IDC_P: u8 = 2;
/// `pic_init_qp` of rate-controlled streams; each slice header carries the
/// frame QP as a delta from it.
const PIC_INIT_QP: u8 = 26;
/// `idr_pic_id` alternates between back-to-back IDRs (spec 7.4.3).
const IDR_PIC_ID_CYCLE: u32 = 2;
/// Largest QP the syntax allows.
const MAX_QP: u8 = 51;
const BITS_PER_BYTE: i64 = 8;
const KBIT: u32 = 1000;

/// The encoder's own reconstructed, border-extended reference planes (the most
/// recent decoded frame). Single short-term reference is sufficient for IPPP.
struct RefPlanes {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

/// Rate-controlled encoder configuration, as a phone's MediaCodec encoder
/// would take it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Any even size; a size that is not a multiple of 16 is signalled with
    /// SPS frame cropping.
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_kbps: u32,
    /// Frames between IDRs; `None` = only the first and on request.
    pub keyframe_interval: Option<u32>,
    /// Slices per frame for slice-parallel encoding (1 = serial). The `threads`
    /// feature encodes them in parallel.
    pub slices: u32,
}

/// One encoded access unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    /// One Annex-B access unit; an IDR is led by SPS + PPS.
    pub data: Vec<u8>,
    pub keyframe: bool,
    /// The QP the frame was coded at (every MB uses it).
    pub qp: u8,
}

/// Baseline H.264 encoder emitting IPPP access units (IDR, then P slices).
pub struct Encoder {
    width: u32,
    height: u32,
    cfg: ParamConfig,
    /// Reconstructed reference for the next P frame (`None` => emit IDR).
    reference: Option<RefPlanes>,
    /// Display/decode-order frame counter (0 == the leading IDR).
    frame_index: u32,
    /// Slices per frame (>= 1). Each frame is partitioned into this many
    /// contiguous MB-row bands, every band emitted as its own slice NAL with an
    /// independent CAVLC bitstream and prediction state reset at the boundary.
    /// Capped at the MB height (one slice can never be fewer than one MB row).
    slices: u32,
    /// `Some` for [`with_config`](Self::with_config) encoders.
    rc: Option<RateControl>,
    keyframe_interval: Option<u32>,
    /// Frames coded since the last IDR, that IDR included.
    frames_since_idr: u32,
    /// `idr_pic_id` of the next IDR.
    idr_pic_id: u32,
    /// Whether the previous access unit was an IDR.
    last_was_idr: bool,
    /// The previous frame's padded source luma (P-frame complexity).
    prev_src_y: Vec<u8>,
}

/// A frame's source planes, padded to whole macroblocks.
struct Source {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    c_stride: usize,
}

impl Encoder {
    /// Create an encoder for `width`x`height` frames at a fixed quantiser `qp`
    /// (0..=51). Returns an error for empty dimensions or an out-of-range QP.
    pub fn new(width: u32, height: u32, qp: u8) -> Result<Self, EncodeError> {
        Self::new_with_slices(width, height, qp, 1)
    }

    /// Like [`new`](Self::new) but partitions every frame into `slices`
    /// contiguous slices (MB-row bands). `slices` is clamped to `[1, mb_height]`.
    /// Multi-slice streams stay bit-exact through this crate's decoder and are
    /// the unit of parallelism for the `threads`-gated encode paths.
    pub fn new_with_slices(
        width: u32,
        height: u32,
        qp: u8,
        slices: u32,
    ) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 {
            return Err(EncodeError::InvalidConfig("zero dimension"));
        }
        if qp > MAX_QP {
            return Err(EncodeError::InvalidConfig("qp out of range"));
        }
        let cfg = ParamConfig::new(width, height, qp);
        let slices = slices.clamp(1, cfg.mb_height);
        Ok(Encoder {
            width,
            height,
            cfg,
            reference: None,
            frame_index: 0,
            slices,
            rc: None,
            keyframe_interval: None,
            frames_since_idr: 0,
            idr_pic_id: 0,
            last_was_idr: false,
            prev_src_y: Vec::new(),
        })
    }

    /// A rate-controlled encoder (OpenH264's bitrate mode, see [`ratectl`]).
    /// Frames go in through [`encode`](Self::encode).
    pub fn with_config(config: EncoderConfig) -> Result<Self, EncodeError> {
        let EncoderConfig {
            width,
            height,
            fps,
            bitrate_kbps,
            keyframe_interval,
            slices,
        } = config;
        if width % 2 != 0 || height % 2 != 0 {
            return Err(EncodeError::InvalidConfig("odd dimension"));
        }
        if fps == 0 {
            return Err(EncodeError::InvalidConfig("zero frame rate"));
        }
        if bitrate_kbps == 0 {
            return Err(EncodeError::InvalidConfig("zero bitrate"));
        }
        let mut enc = Self::new_with_slices(width, height, PIC_INIT_QP, slices)?;
        let bitrate = bitrate_kbps as u64 * KBIT as u64;
        enc.cfg.fps = Some(fps);
        enc.cfg.level_idc = paraset::level_idc(enc.cfg.mb_width, enc.cfg.mb_height, fps, bitrate);
        enc.rc = Some(RateControl::new(width, height, bitrate as i64, fps));
        enc.keyframe_interval = keyframe_interval;
        Ok(enc)
    }

    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Slices emitted per frame (after clamping to the MB height).
    pub const fn slices(&self) -> u32 {
        self.slices
    }

    /// Set the number of slices per frame (clamped to `[1, mb_height]`).
    pub fn set_slices(&mut self, slices: u32) {
        self.slices = slices.clamp(1, self.cfg.mb_height);
    }

    /// Frame rate: advertised in the SPS VUI timing info (from the next IDR;
    /// `0` disables VUI) and, for a rate-controlled encoder, the per-frame bit
    /// budget from the next frame.
    pub fn set_frame_rate(&mut self, fps: u32) {
        self.cfg.fps = if fps > 0 { Some(fps) } else { None };
        if let Some(rc) = &mut self.rc
            && fps > 0
        {
            rc.set_fps(fps);
        }
    }

    /// Target bitrate of a rate-controlled encoder, from the next frame. No IDR
    /// is forced (like MediaCodec's `video-bitrate` parameter). A no-op for a
    /// fixed-QP encoder or `0`.
    pub fn set_bitrate(&mut self, kbps: u32) {
        if let Some(rc) = &mut self.rc
            && kbps > 0
        {
            rc.set_bitrate(kbps as i64 * KBIT as i64);
        }
    }

    /// Force the next frame to be an IDR (clears the reference + resets the
    /// frame counter), e.g. for a clean random-access point.
    pub fn force_idr(&mut self) {
        self.reference = None;
        self.frame_index = 0;
    }

    /// The last encoded frame's reconstruction (what a decoder outputs for
    /// it), cropped to the visible size. `None` before the first frame.
    pub fn reconstruction(&self) -> Option<YuvRef<'_>> {
        let r = self.reference.as_ref()?;
        let (y_stride, c_stride, _, _) =
            encode_mb::rec_dims(self.cfg.mb_width as usize, self.cfg.mb_height as usize);
        Some(YuvRef {
            y: &r.y[BORDER * y_stride + BORDER..],
            u: &r.u[BORDER * c_stride + BORDER..],
            v: &r.v[BORDER * c_stride + BORDER..],
            y_stride,
            c_stride,
            width: self.width,
            height: self.height,
        })
    }

    /// Encode one I420 frame. A rate-controlled encoder picks the frame QP and
    /// the IDR cadence; a fixed-QP one codes at its QP. The slices of a
    /// multi-slice frame are encoded in parallel with the `threads` feature.
    pub fn encode(&mut self, frame: &YuvRef<'_>) -> Result<EncodedFrame, EncodeError> {
        if frame.width != self.width || frame.height != self.height {
            return Err(EncodeError::InvalidFrame("frame size differs from the encoder's"));
        }
        if !frame.is_valid() {
            return Err(EncodeError::InvalidFrame("plane shorter than its size and stride"));
        }
        if let Some(interval) = self.keyframe_interval
            && self.frames_since_idr >= interval.max(1)
        {
            self.force_idr();
        }
        let src = self.pad_source(frame.y, frame.y_stride, frame.u, frame.v, frame.c_stride);
        let idr = self.reference.is_none();
        let mb_w = self.cfg.mb_width as usize;
        let mb_h = self.cfg.mb_height as usize;
        let (qp, complexity) = match &mut self.rc {
            Some(rc) => {
                let complexity = if idr {
                    intra_complexity(&src.y, src.y_stride, mb_w, mb_h)
                } else {
                    inter_complexity(&src.y, &self.prev_src_y, src.y_stride, mb_w, mb_h)
                };
                (rc.picture_init(idr, complexity), complexity)
            }
            None => (self.cfg.qp, 0),
        };
        #[cfg(feature = "threads")]
        let parallel = self.slices > 1;
        #[cfg(not(feature = "threads"))]
        let parallel = false;
        let data = self.encode_source(&src, qp, parallel);
        if let Some(rc) = &mut self.rc {
            rc.picture_update(idr, qp, data.len() as i64 * BITS_PER_BYTE, complexity);
        }
        self.prev_src_y = src.y;
        self.frames_since_idr = if idr { 1 } else { self.frames_since_idr.saturating_add(1) };
        Ok(EncodedFrame {
            data,
            keyframe: idr,
            qp,
        })
    }

    /// Encode one I420 frame at the fixed QP. The first frame (or any frame
    /// after [`force_idr`](Self::force_idr)) is a self-contained IDR access unit
    /// (SPS + PPS + IDR slice); subsequent frames are P (non-IDR) slices coded
    /// against the previous reconstructed frame. `y`/`u`/`v` are planar 8-bit
    /// samples with the given strides (`u`/`v` are quarter-resolution, 4:2:0).
    pub fn encode_frame(
        &mut self,
        y: &[u8],
        y_stride: usize,
        u: &[u8],
        v: &[u8],
        c_stride: usize,
    ) -> Vec<u8> {
        let src = self.pad_source(y, y_stride, u, v, c_stride);
        self.encode_source(&src, self.cfg.qp, false)
    }

    /// Pad the source to MB-aligned planes with edge replication.
    fn pad_source(&self, y: &[u8], y_stride: usize, u: &[u8], v: &[u8], c_stride: usize) -> Source {
        let mb_width = self.cfg.mb_width as usize;
        let mb_height = self.cfg.mb_height as usize;
        let (w, h) = (self.width as usize, self.height as usize);
        let (y, sy_stride) = pad_plane(y, y_stride, w, h, mb_width * 16, mb_height * 16);
        let (cw, ch) = (w >> 1, h >> 1);
        let (u, sc_stride) = pad_plane(u, c_stride, cw, ch, mb_width * 8, mb_height * 8);
        let (v, _) = pad_plane(v, c_stride, cw, ch, mb_width * 8, mb_height * 8);
        Source {
            y,
            u,
            v,
            y_stride: sy_stride,
            c_stride: sc_stride,
        }
    }

    /// Encode a padded frame at `qp` into one access unit, serially or (with
    /// `threads`) slice-parallel, and keep its reconstruction as the reference.
    fn encode_source(&mut self, src: &Source, qp: u8, parallel: bool) -> Vec<u8> {
        let mb_width = self.cfg.mb_width as usize;
        let mb_height = self.cfg.mb_height as usize;
        let reference = self.reference.take();
        let is_p = reference.is_some();
        let (ref_y, ref_u, ref_v): (&[u8], &[u8], &[u8]) = match &reference {
            Some(r) => (&r.y, &r.u, &r.v),
            None => (&[], &[], &[]),
        };
        let idr_pic_id = if self.rc.is_some() && self.last_was_idr && !is_p {
            (self.idr_pic_id + 1) % IDR_PIC_ID_CYCLE
        } else {
            0
        };
        let job = SliceJob {
            dims: MbDims {
                width: mb_width,
                height: mb_height,
            },
            qp,
            src: PlaneRefs {
                y: &src.y,
                u: &src.u,
                v: &src.v,
            },
            sy_stride: src.y_stride,
            sc_stride: src.c_stride,
            refs: PlaneRefs {
                y: ref_y,
                u: ref_u,
                v: ref_v,
            },
            cfg: self.cfg,
            frame_index: self.frame_index,
            idr_pic_id,
            is_p,
        };
        let bands = slice_bands(mb_height, self.slices);

        // IDR access unit (re)transmits the parameter sets ahead of its slices.
        let mut out = Vec::new();
        if !is_p {
            let sps = paraset::write_sps(&self.cfg);
            append_annexb_nal(&mut out, REF_IDC_HIGHEST, NAL_SPS, &sps);
            let pps = paraset::write_pps(&self.cfg);
            append_annexb_nal(&mut out, REF_IDC_HIGHEST, NAL_PPS, &pps);
        }
        let (ref_idc, nal_type) = if is_p {
            (REF_IDC_P, NAL_NON_IDR_SLICE)
        } else {
            (REF_IDC_HIGHEST, NAL_IDR_SLICE)
        };

        let planes = if parallel {
            encode_bands_parallel(job, &bands, &mut out, ref_idc, nal_type)
        } else {
            encode_bands_serial(job, &bands, &mut out, ref_idc, nal_type)
        };

        let (y, u, v) = planes;
        self.reference = Some(RefPlanes { y, u, v });
        self.frame_index = self.frame_index.wrapping_add(1);
        if !is_p {
            self.idr_pic_id = idr_pic_id;
        }
        self.last_was_idr = !is_p;
        out
    }
}

/// Reconstructed (border-extended) Y, U, V planes of a coded frame.
type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Encode the bands on one frame encoder, each band its own slice NAL.
fn encode_bands_serial(
    job: SliceJob<'_>,
    bands: &[(usize, usize)],
    out: &mut Vec<u8>,
    ref_idc: u8,
    nal_type: u8,
) -> Planes {
    let mut frame = job.frame_enc();
    for &band in bands {
        let rbsp = job.encode_band(&mut frame, band);
        append_annexb_nal(out, ref_idc, nal_type, &rbsp);
    }
    frame.into_reference()
}

#[cfg(not(feature = "threads"))]
fn encode_bands_parallel(
    job: SliceJob<'_>,
    bands: &[(usize, usize)],
    out: &mut Vec<u8>,
    ref_idc: u8,
    nal_type: u8,
) -> Planes {
    encode_bands_serial(job, bands, out, ref_idc, nal_type)
}

/// Encode each band on its own [`FrameEnc`] across worker threads (contiguous
/// static chunking, so results come back in band order), append the slice NALs
/// in order, and stitch the per-slice reconstructions into one border-extended
/// reference. Byte-identical to the serial path: each slice is independent
/// (deblocking off, neighbours gated at the boundary).
#[cfg(feature = "threads")]
fn encode_bands_parallel(
    job: SliceJob<'_>,
    bands: &[(usize, usize)],
    out: &mut Vec<u8>,
    ref_idc: u8,
    nal_type: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let nbands = bands.len();
    let par = std::thread::available_parallelism().map_or(1, |n| n.get());
    let nthreads = nbands.min(par).max(1);
    let results: Vec<(Vec<u8>, FrameEnc)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .map(|w| {
                let lo = w * nbands / nthreads;
                let hi = (w + 1) * nbands / nthreads;
                scope.spawn(move || {
                    bands[lo..hi]
                        .iter()
                        .map(|&band| {
                            let mut fe = job.frame_enc();
                            let rbsp = job.encode_band(&mut fe, band);
                            (rbsp, fe)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });

    for (rbsp, _) in &results {
        append_annexb_nal(out, ref_idc, nal_type, rbsp);
    }
    let mut unified = encode_mb::RecPlanes::new(job.dims);
    for ((_, fe), &(fy, ly)) in results.iter().zip(bands) {
        fe.copy_band_into(&mut unified, fy, ly);
    }
    encode_mb::expand_reference(
        &mut unified.y,
        &mut unified.u,
        &mut unified.v,
        job.dims,
        unified.ystride,
        unified.cstride,
    );
    (unified.y, unified.u, unified.v)
}

/// Inter complexity (VAA `iFrameSad`): the frame's 16x16 SADs against the
/// previous source frame.
fn inter_complexity(cur: &[u8], prev: &[u8], stride: usize, mb_w: usize, mb_h: usize) -> i64 {
    if prev.len() != cur.len() {
        return 0;
    }
    let mut total = 0i64;
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let o = mb_y * 16 * stride + mb_x * 16;
            total += sad(&cur[o..], stride, &prev[o..], stride, 16, 16) as i64;
        }
    }
    total
}

/// Intra complexity, after `CComplexityAnalysis::AnalyzeGomComplexityViaSad`'s
/// I-frame path: per MB the smaller SAD of a vertical (row above) or
/// horizontal (column to the left) prediction from the source itself.
fn intra_complexity(src: &[u8], stride: usize, mb_w: usize, mb_h: usize) -> i64 {
    let mut total = 0i64;
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let o = mb_y * 16 * stride + mb_x * 16;
            let mut best = u32::MAX;
            if mb_y > 0 {
                let above = &src[o - stride..o - stride + 16];
                let mut s = 0u32;
                for r in 0..16 {
                    let row = &src[o + r * stride..o + r * stride + 16];
                    s += row.iter().zip(above).map(|(&a, &b)| a.abs_diff(b) as u32).sum::<u32>();
                }
                best = s;
            }
            if mb_x > 0 {
                let mut s = 0u32;
                for r in 0..16 {
                    let row = &src[o + r * stride - 1..o + r * stride + 16];
                    let left = row[0];
                    s += row[1..].iter().map(|&a| a.abs_diff(left) as u32).sum::<u32>();
                }
                best = best.min(s);
            }
            if best != u32::MAX {
                total += best as i64;
            }
        }
    }
    total
}

/// Partition a frame's `mb_height` MB-rows into `slices` contiguous bands as
/// evenly as possible (band sizes differ by at most one MB row). Returns
/// `[(first_mb_y, last_mb_y), ...]` half-open row ranges.
fn slice_bands(mb_height: usize, slices: u32) -> Vec<(usize, usize)> {
    let s = (slices as usize).clamp(1, mb_height.max(1));
    (0..s)
        .map(|b| (b * mb_height / s, (b + 1) * mb_height / s))
        .collect()
}

/// Write an IDR I-slice header (CAVLC, frame-only) starting at `first_mb`.
fn write_idr_slice_header(bw: &mut BitWriter, job: &SliceJob<'_>, first_mb: u32) {
    let cfg = &job.cfg;
    bw.write_ue(first_mb); // first_mb_in_slice
    bw.write_ue(7); // slice_type = I (7 -> "all I" form)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_bits(0, cfg.log2_max_frame_num); // frame_num = 0
    bw.write_ue(job.idr_pic_id);
    bw.write_bits(0, cfg.log2_max_poc_lsb); // pic_order_cnt_lsb = 0
    // dec_ref_pic_marking (IDR): no_output_of_prior_pics + long_term_reference.
    bw.write_flag(false);
    bw.write_flag(false);
    bw.write_se(job.qp as i32 - cfg.qp as i32); // slice_qp_delta
    bw.write_ue(1); // disable_deblocking_filter_idc = 1 (off; output == recon)
}

/// Write a P-slice header (CAVLC, single short-term reference) starting at
/// `first_mb`. All slices of a picture share `frame_index` (frame_num / POC).
fn write_p_slice_header(bw: &mut BitWriter, job: &SliceJob<'_>, first_mb: u32) {
    let cfg = &job.cfg;
    let max_fn = 1u32 << cfg.log2_max_frame_num;
    let max_poc = 1u32 << cfg.log2_max_poc_lsb;
    bw.write_ue(first_mb); // first_mb_in_slice
    bw.write_ue(5); // slice_type = P (5 -> "all P" form)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_bits(job.frame_index % max_fn, cfg.log2_max_frame_num); // frame_num
    bw.write_bits(
        (job.frame_index.wrapping_mul(2)) % max_poc,
        cfg.log2_max_poc_lsb,
    ); // poc_lsb
    bw.write_flag(false); // num_ref_idx_active_override_flag (use PPS default = 1)
    bw.write_flag(false); // ref_pic_list_modification_flag_l0
    // dec_ref_pic_marking (non-IDR, nal_ref_idc != 0): adaptive flag off.
    bw.write_flag(false);
    bw.write_se(job.qp as i32 - cfg.qp as i32); // slice_qp_delta
    bw.write_ue(1); // disable_deblocking_filter_idc = 1 (off; output == recon)
}

/// One input frame for the all-intra frame-parallel encode path: borrowed
/// planar I420 planes with their strides (`u`/`v` quarter-resolution, 4:2:0).
#[cfg(feature = "threads")]
pub struct FrameInput<'a> {
    pub y: &'a [u8],
    pub y_stride: usize,
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub c_stride: usize,
}

#[cfg(feature = "threads")]
impl Encoder {
    /// Slice-parallel sibling of [`encode_frame`](Self::encode_frame): encode
    /// the frame's `slices` independent slices concurrently with
    /// [`std::thread::scope`] (up to `available_parallelism` worker threads),
    /// then concatenate the slice NALs in order. Frames stay serial (a P frame
    /// references the previous reconstruction); slices within a frame run in
    /// parallel. The output is **byte-identical** to `encode_frame` — each slice
    /// is fully independent (deblocking off, neighbours gated at the boundary),
    /// so the order of execution cannot change a single bit.
    pub fn encode_frame_parallel(
        &mut self,
        y: &[u8],
        y_stride: usize,
        u: &[u8],
        v: &[u8],
        c_stride: usize,
    ) -> Vec<u8> {
        let src = self.pad_source(y, y_stride, u, v, c_stride);
        self.encode_source(&src, self.cfg.qp, true)
    }

    /// Frame-parallel all-intra encode: encode `frames` as independent IDR
    /// access units concurrently (each its own encoder state) and return the
    /// AUs in input order. Each AU is byte-identical to encoding that frame on a
    /// fresh single encoder, so the result matches the serial all-intra path
    /// exactly. Uses up to `available_parallelism` worker threads.
    pub fn encode_frames_parallel(&self, frames: &[FrameInput<'_>]) -> Vec<Vec<u8>> {
        let n = frames.len();
        if n == 0 {
            return Vec::new();
        }
        let par = std::thread::available_parallelism().map_or(1, |p| p.get());
        let nthreads = n.min(par).max(1);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..nthreads)
                .map(|t| {
                    let lo = t * n / nthreads;
                    let hi = (t + 1) * n / nthreads;
                    scope.spawn(move || {
                        frames[lo..hi]
                            .iter()
                            .map(|f| {
                                let mut enc = self.fresh_fixed_qp();
                                enc.encode_frame(f.y, f.y_stride, f.u, f.v, f.c_stride)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        })
    }

    /// A new fixed-QP encoder with this one's size, QP, slices and frame rate.
    fn fresh_fixed_qp(&self) -> Encoder {
        Encoder {
            width: self.width,
            height: self.height,
            cfg: self.cfg,
            reference: None,
            frame_index: 0,
            slices: self.slices,
            rc: None,
            keyframe_interval: None,
            frames_since_idr: 0,
            idr_pic_id: 0,
            last_was_idr: false,
            prev_src_y: Vec::new(),
        }
    }
}

/// The constant inputs for encoding one frame's slice bands, bundled so the
/// threaded slice-parallel path can hand a single `Copy` value to each worker
/// (the only per-band variable is the MB-row range). All fields are `Copy`: the
/// planes are borrowed (`PlaneRefs`) and `ParamConfig` is small + `Copy`.
#[derive(Clone, Copy)]
struct SliceJob<'a> {
    dims: MbDims,
    qp: u8,
    src: PlaneRefs<'a>,
    sy_stride: usize,
    sc_stride: usize,
    refs: PlaneRefs<'a>,
    cfg: ParamConfig,
    frame_index: u32,
    idr_pic_id: u32,
    is_p: bool,
}

impl<'a> SliceJob<'a> {
    fn frame_enc(&self) -> FrameEnc<'a> {
        FrameEnc::new(
            self.dims,
            self.qp as i32,
            self.src,
            self.sy_stride,
            self.sc_stride,
            self.refs,
        )
    }

    /// Encode MB rows `band` of `fe` into one slice RBSP (header included).
    fn encode_band(&self, fe: &mut FrameEnc<'a>, band: (usize, usize)) -> Vec<u8> {
        let (first_mb_y, last_mb_y) = band;
        let mut bw = BitWriter::new();
        let first_mb = (first_mb_y * self.dims.width) as u32;
        if self.is_p {
            write_p_slice_header(&mut bw, self, first_mb);
        } else {
            write_idr_slice_header(&mut bw, self, first_mb);
        }
        fe.encode_band(&mut bw, first_mb_y, last_mb_y, self.is_p);
        bw.write_trailing_bits();
        bw.finish()
    }
}

/// Copy `src` (`w`x`h`, row stride `src_stride`) into a tightly-strided
/// `dst_w`x`dst_h` buffer, replicating the right/bottom edges into the padding.
fn pad_plane(
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    dst_w: usize,
    dst_h: usize,
) -> (Vec<u8>, usize) {
    let mut dst = vec![0u8; dst_w * dst_h];
    for y in 0..dst_h {
        let sy = y.min(h - 1);
        let row = &src[sy * src_stride..sy * src_stride + w];
        let drow = &mut dst[y * dst_w..y * dst_w + dst_w];
        drow[..w].copy_from_slice(row);
        let edge = row[w - 1];
        for d in &mut drow[w..] {
            *d = edge;
        }
    }
    (dst, dst_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_produces_three_nals() {
        // A flat grey frame should encode into SPS + PPS + IDR slice.
        let w = 32usize;
        let h = 32usize;
        let y = vec![128u8; w * h];
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let mut enc = Encoder::new(w as u32, h as u32, 26).unwrap();
        let au = enc.encode_frame(&y, w, &u, &v, w / 2);

        let count = crate::decoder::nal::annexb_nal_units(&au).count();
        assert_eq!(count, 3, "expected SPS + PPS + IDR slice");
        assert!(au.starts_with(&[0, 0, 0, 1]));
    }

    #[test]
    fn slice_headers_carry_frame_qp_and_alternate_idr_pic_id() {
        use crate::decoder::nal::parse_nal;
        use crate::decoder::params::{parse_pps, parse_sps};
        use crate::decoder::slice_header::parse_slice_header;

        let (w, h) = (64u32, 48u32);
        let config = EncoderConfig {
            width: w,
            height: h,
            fps: 30,
            bitrate_kbps: 200,
            keyframe_interval: Some(1),
            slices: 2,
        };
        let mut enc = Encoder::with_config(config).unwrap();
        let frame = crate::image::I420::new(w, h);
        let mut ids = Vec::new();
        for _ in 0..3 {
            let au = enc.encode(&frame.as_ref()).unwrap();
            assert!(au.keyframe);
            let nals: Vec<_> = crate::decoder::nal::annexb_nal_units(&au.data)
                .map(|n| parse_nal(n).unwrap())
                .collect();
            let sps = parse_sps(&nals[0].rbsp).unwrap();
            let pps = parse_pps(&nals[1].rbsp, Some(&sps)).unwrap();
            assert_eq!(pps.pic_init_qp, PIC_INIT_QP as i32);
            for slice in &nals[2..] {
                let sh = parse_slice_header(&slice.rbsp, slice.ref_idc, true, &sps, &pps).unwrap();
                assert_eq!(sh.slice_qp, au.qp as i32);
                ids.push(sh.idr_pic_id);
            }
        }
        assert_eq!(ids, [0, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn level_follows_size_and_rate() {
        let level = |w: u32, h: u32, fps: u32, kbps: u64| {
            paraset::level_idc(w.div_ceil(16), h.div_ceil(16), fps, kbps * 1000)
        };
        assert_eq!(level(640, 360, 30, 900), 30);
        assert_eq!(level(1280, 720, 30, 2000), 31);
        assert_eq!(level(1920, 1080, 30, 4000), 40);
        assert_eq!(level(1920, 1080, 60, 6000), 42);
    }
}
