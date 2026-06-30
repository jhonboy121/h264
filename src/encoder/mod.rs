//! Encoder DSP and core (feature `encoder`).
//!
//! Pure-Rust ports of the OpenH264 encoder kernels. Shared transform / quant /
//! SATD kernels live in [`crate::dsp`]; this module holds the encoder-only
//! pieces (intra prediction, CAVLC writing, parameter-set + NAL generation, and
//! the intra-frame encode loop).
//!
//! [`Encoder`] is the public facade: construct it for a (width, height, qp) and
//! call [`Encoder::encode_frame`] to produce a complete Annex-B IDR access unit
//! (SPS + PPS + IDR slice) that decodes back through [`crate::decoder`].

mod encode_mb;

pub mod cavlc_writer;
pub mod intra_pred;
pub mod motion_est;
pub mod nal_encap;
pub mod paraset;

use alloc::vec;
use alloc::vec::Vec;

use crate::bits::BitWriter;
use crate::error::EncodeError;
use encode_mb::{FrameEnc, MbDims, PlaneRefs};
use nal_encap::append_annexb_nal;
use paraset::ParamConfig;

/// NAL unit types we emit.
const NAL_NON_IDR_SLICE: u8 = 1;
const NAL_IDR_SLICE: u8 = 5;
const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;

/// The encoder's own reconstructed, border-extended reference planes (the most
/// recent decoded frame). Single short-term reference is sufficient for IPPP.
struct RefPlanes {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

/// Baseline H.264 encoder emitting IPPP access units (one IDR, then P slices).
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
        if qp > 51 {
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
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Slices emitted per frame (after clamping to the MB height).
    pub fn slices(&self) -> u32 {
        self.slices
    }

    /// Set the number of slices per frame (clamped to `[1, mb_height]`).
    pub fn set_slices(&mut self, slices: u32) {
        self.slices = slices.clamp(1, self.cfg.mb_height);
    }

    /// Advertise `fps` in the SPS VUI timing info so raw Annex-B players
    /// (ffplay/QuickTime) and the decoder know the playback rate. Set it before
    /// the first encoded frame (the SPS is emitted with each IDR). `0` disables
    /// VUI. Note: this is a declaration — feed frames at this rate for it to be
    /// accurate.
    pub fn set_frame_rate(&mut self, fps: u32) {
        self.cfg.fps = if fps > 0 { Some(fps) } else { None };
    }

    /// Force the next [`encode_frame`](Self::encode_frame) to emit an IDR
    /// (clears the reference + resets the frame counter), e.g. for a clean
    /// random-access point.
    pub fn force_idr(&mut self) {
        self.reference = None;
        self.frame_index = 0;
    }

    /// Encode one I420 frame. The first frame (or any frame after
    /// [`force_idr`](Self::force_idr)) is a self-contained IDR access unit
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
        let mb_width = self.cfg.mb_width as usize;
        let mb_height = self.cfg.mb_height as usize;
        let qp = self.cfg.qp as i32;

        // Pad source to MB-aligned planes with edge replication.
        let (src_y, sy_stride) = pad_plane(
            y,
            y_stride,
            self.width as usize,
            self.height as usize,
            mb_width * 16,
            mb_height * 16,
        );
        let cw = self.width as usize >> 1;
        let ch = self.height as usize >> 1;
        let (src_u, sc_stride) = pad_plane(u, c_stride, cw, ch, mb_width * 8, mb_height * 8);
        let (src_v, _) = pad_plane(v, c_stride, cw, ch, mb_width * 8, mb_height * 8);

        let reference = self.reference.take();
        let is_p = reference.is_some();
        let (ref_y, ref_u, ref_v): (&[u8], &[u8], &[u8]) = match &reference {
            Some(r) => (&r.y, &r.u, &r.v),
            None => (&[], &[], &[]),
        };

        let dims = MbDims {
            width: mb_width,
            height: mb_height,
        };
        let mut frame = FrameEnc::new(
            dims,
            qp,
            PlaneRefs {
                y: &src_y,
                u: &src_u,
                v: &src_v,
            },
            sy_stride,
            sc_stride,
            PlaneRefs {
                y: ref_y,
                u: ref_u,
                v: ref_v,
            },
        );

        // IDR access unit (re)transmits the parameter sets ahead of its slices.
        let mut out = Vec::new();
        if !is_p {
            append_annexb_nal(&mut out, 3, NAL_SPS, &paraset::write_sps(&self.cfg));
            append_annexb_nal(&mut out, 3, NAL_PPS, &paraset::write_pps(&self.cfg));
        }

        // Encode each slice (MB-row band) into its own NAL, serially.
        for &(first_mb_y, last_mb_y) in &slice_bands(mb_height, self.slices) {
            let mut bw = BitWriter::new();
            let first_mb = (first_mb_y * mb_width) as u32;
            if is_p {
                write_p_slice_header(&mut bw, &self.cfg, self.frame_index, first_mb);
            } else {
                write_idr_slice_header(&mut bw, &self.cfg, first_mb);
            }
            frame.encode_band(&mut bw, first_mb_y, last_mb_y, is_p);
            bw.write_trailing_bits();
            let (ref_idc, nal_type) = if is_p {
                (2, NAL_NON_IDR_SLICE)
            } else {
                (3, NAL_IDR_SLICE)
            };
            append_annexb_nal(&mut out, ref_idc, nal_type, &bw.finish());
        }

        // Keep this frame's (border-extended) reconstruction as the next ref.
        let (ry, ru, rv) = frame.into_reference();
        self.reference = Some(RefPlanes {
            y: ry,
            u: ru,
            v: rv,
        });
        self.frame_index = self.frame_index.wrapping_add(1);

        out
    }
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
fn write_idr_slice_header(bw: &mut BitWriter, cfg: &ParamConfig, first_mb: u32) {
    bw.write_ue(first_mb); // first_mb_in_slice
    bw.write_ue(7); // slice_type = I (7 -> "all I" form)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_bits(0, cfg.log2_max_frame_num); // frame_num = 0
    bw.write_ue(0); // idr_pic_id
    bw.write_bits(0, cfg.log2_max_poc_lsb); // pic_order_cnt_lsb = 0
    // dec_ref_pic_marking (IDR): no_output_of_prior_pics + long_term_reference.
    bw.write_flag(false);
    bw.write_flag(false);
    bw.write_se(0); // slice_qp_delta (slice qp == pic_init_qp)
    bw.write_ue(1); // disable_deblocking_filter_idc = 1 (off; output == recon)
}

/// Write a P-slice header (CAVLC, single short-term reference) starting at
/// `first_mb`. All slices of a picture share `frame_index` (frame_num / POC).
fn write_p_slice_header(bw: &mut BitWriter, cfg: &ParamConfig, frame_index: u32, first_mb: u32) {
    let max_fn = 1u32 << cfg.log2_max_frame_num;
    let max_poc = 1u32 << cfg.log2_max_poc_lsb;
    bw.write_ue(first_mb); // first_mb_in_slice
    bw.write_ue(5); // slice_type = P (5 -> "all P" form)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_bits(frame_index % max_fn, cfg.log2_max_frame_num); // frame_num
    bw.write_bits(
        (frame_index.wrapping_mul(2)) % max_poc,
        cfg.log2_max_poc_lsb,
    ); // poc_lsb
    bw.write_flag(false); // num_ref_idx_active_override_flag (use PPS default = 1)
    bw.write_flag(false); // ref_pic_list_modification_flag_l0
    // dec_ref_pic_marking (non-IDR, nal_ref_idc != 0): adaptive flag off.
    bw.write_flag(false);
    bw.write_se(0); // slice_qp_delta
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
        let mb_width = self.cfg.mb_width as usize;
        let mb_height = self.cfg.mb_height as usize;
        let qp = self.cfg.qp as i32;

        let (src_y, sy_stride) = pad_plane(
            y,
            y_stride,
            self.width as usize,
            self.height as usize,
            mb_width * 16,
            mb_height * 16,
        );
        let cw = self.width as usize >> 1;
        let ch = self.height as usize >> 1;
        let (src_u, sc_stride) = pad_plane(u, c_stride, cw, ch, mb_width * 8, mb_height * 8);
        let (src_v, _) = pad_plane(v, c_stride, cw, ch, mb_width * 8, mb_height * 8);

        let reference = self.reference.take();
        let is_p = reference.is_some();
        let (ref_y, ref_u, ref_v): (&[u8], &[u8], &[u8]) = match &reference {
            Some(r) => (&r.y, &r.u, &r.v),
            None => (&[], &[], &[]),
        };

        let dims = MbDims {
            width: mb_width,
            height: mb_height,
        };
        let job = SliceJob {
            dims,
            qp,
            src: PlaneRefs {
                y: &src_y,
                u: &src_u,
                v: &src_v,
            },
            sy_stride,
            sc_stride,
            refs: PlaneRefs {
                y: ref_y,
                u: ref_u,
                v: ref_v,
            },
            cfg: self.cfg,
            frame_index: self.frame_index,
            is_p,
        };
        let bands = slice_bands(mb_height, self.slices);
        let nbands = bands.len();
        let par = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let nthreads = nbands.min(par).max(1);

        // Encode each band on a worker (contiguous static chunking, so results
        // come back in band order regardless of thread count).
        let results: Vec<(Vec<u8>, FrameEnc)> = std::thread::scope(|scope| {
            let bands = &bands;
            let handles: Vec<_> = (0..nthreads)
                .map(|w| {
                    let lo = w * nbands / nthreads;
                    let hi = (w + 1) * nbands / nthreads;
                    scope.spawn(move || {
                        (lo..hi)
                            .map(|bi| encode_one_band(job, bands[bi]))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });

        // Assemble the access unit in slice order.
        let mut out = Vec::new();
        if !is_p {
            append_annexb_nal(&mut out, 3, NAL_SPS, &paraset::write_sps(&self.cfg));
            append_annexb_nal(&mut out, 3, NAL_PPS, &paraset::write_pps(&self.cfg));
        }
        let (ref_idc, nal_type) = if is_p {
            (2, NAL_NON_IDR_SLICE)
        } else {
            (3, NAL_IDR_SLICE)
        };
        for (rbsp, _) in &results {
            append_annexb_nal(&mut out, ref_idc, nal_type, rbsp);
        }

        // Stitch the per-slice reconstructions into one frame, border-extend,
        // and keep it as the next reference (the serial deblock/expand step).
        let mut unified = encode_mb::RecPlanes::new(dims);
        for ((_, fe), &(fy, ly)) in results.iter().zip(bands.iter()) {
            fe.copy_band_into(&mut unified, fy, ly);
        }
        encode_mb::expand_reference(
            &mut unified.y,
            &mut unified.u,
            &mut unified.v,
            dims,
            unified.ystride,
            unified.cstride,
        );
        self.reference = Some(RefPlanes {
            y: unified.y,
            u: unified.u,
            v: unified.v,
        });
        self.frame_index = self.frame_index.wrapping_add(1);
        out
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
        let par = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let nthreads = n.min(par).max(1);
        let (w, h, qp, slices, fps) = (
            self.width,
            self.height,
            self.cfg.qp,
            self.slices,
            self.cfg.fps,
        );
        std::thread::scope(|scope| {
            let frames = &frames;
            let handles: Vec<_> = (0..nthreads)
                .map(|t| {
                    let lo = t * n / nthreads;
                    let hi = (t + 1) * n / nthreads;
                    scope.spawn(move || {
                        (lo..hi)
                            .map(|i| {
                                let mut enc = Encoder::new_with_slices(w, h, qp, slices).unwrap();
                                if let Some(fps) = fps {
                                    enc.set_frame_rate(fps);
                                }
                                let f = &frames[i];
                                enc.encode_frame(f.y, f.y_stride, f.u, f.v, f.c_stride)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }
}

/// The constant inputs for encoding one slice band, bundled so the threaded
/// slice-parallel path can hand a single `Copy` value to each worker (the only
/// per-band variable is the MB-row range). All fields are `Copy`: the planes
/// are borrowed (`PlaneRefs`) and `ParamConfig` is small + `Copy`.
#[cfg(feature = "threads")]
#[derive(Clone, Copy)]
struct SliceJob<'a> {
    dims: MbDims,
    qp: i32,
    src: PlaneRefs<'a>,
    sy_stride: usize,
    sc_stride: usize,
    refs: PlaneRefs<'a>,
    cfg: ParamConfig,
    frame_index: u32,
    is_p: bool,
}

/// Encode a single slice band into its own [`FrameEnc`] (independent
/// reconstruction + bitstream), returning the slice RBSP and the frame encoder
/// holding the reconstructed band. Shared by the threaded slice-parallel path.
#[cfg(feature = "threads")]
fn encode_one_band<'a>(job: SliceJob<'a>, band: (usize, usize)) -> (Vec<u8>, FrameEnc<'a>) {
    let (first_mb_y, last_mb_y) = band;
    let mut fe = FrameEnc::new(
        job.dims,
        job.qp,
        job.src,
        job.sy_stride,
        job.sc_stride,
        job.refs,
    );
    let mut bw = BitWriter::new();
    let first_mb = (first_mb_y * job.dims.width) as u32;
    if job.is_p {
        write_p_slice_header(&mut bw, &job.cfg, job.frame_index, first_mb);
    } else {
        write_idr_slice_header(&mut bw, &job.cfg, first_mb);
    }
    fe.encode_band(&mut bw, first_mb_y, last_mb_y, job.is_p);
    bw.write_trailing_bits();
    (bw.finish(), fe)
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
}
