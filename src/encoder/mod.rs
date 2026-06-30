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
use encode_mb::FrameEnc;
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
}

impl Encoder {
    /// Create an encoder for `width`x`height` frames at a fixed quantiser `qp`
    /// (0..=51). Returns an error for empty dimensions or an out-of-range QP.
    pub fn new(width: u32, height: u32, qp: u8) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 {
            return Err(EncodeError::InvalidConfig("zero dimension"));
        }
        if qp > 51 {
            return Err(EncodeError::InvalidConfig("qp out of range"));
        }
        Ok(Encoder {
            width,
            height,
            cfg: ParamConfig::new(width, height, qp),
            reference: None,
            frame_index: 0,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
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
        let (src_y, sy_stride) = pad_plane(y, y_stride, self.width as usize, self.height as usize, mb_width * 16, mb_height * 16);
        let cw = self.width as usize >> 1;
        let ch = self.height as usize >> 1;
        let (src_u, sc_stride) = pad_plane(u, c_stride, cw, ch, mb_width * 8, mb_height * 8);
        let (src_v, _) = pad_plane(v, c_stride, cw, ch, mb_width * 8, mb_height * 8);

        let is_p = self.reference.is_some();
        let (ref_y, ref_u, ref_v) = match self.reference.take() {
            Some(r) => (r.y, r.u, r.v),
            None => (Vec::new(), Vec::new(), Vec::new()),
        };

        let mut frame =
            FrameEnc::new(mb_width, mb_height, qp, src_y, src_u, src_v, sy_stride, sc_stride, ref_y, ref_u, ref_v);

        let mut out = Vec::new();
        let mut bw = BitWriter::new();

        if is_p {
            self.write_p_slice_header(&mut bw);
            let mut pending = 0u32;
            for mb_y in 0..mb_height {
                for mb_x in 0..mb_width {
                    frame.encode_mb_p(&mut bw, mb_x, mb_y, &mut pending);
                }
            }
            // Flush any trailing skipped macroblocks.
            if pending > 0 {
                bw.write_ue(pending);
            }
            bw.write_trailing_bits();
            append_annexb_nal(&mut out, 2, NAL_NON_IDR_SLICE, &bw.finish());
        } else {
            // IDR access unit: (re)transmit the parameter sets, then the I slice.
            append_annexb_nal(&mut out, 3, NAL_SPS, &paraset::write_sps(&self.cfg));
            append_annexb_nal(&mut out, 3, NAL_PPS, &paraset::write_pps(&self.cfg));
            self.write_idr_slice_header(&mut bw);
            for mb_y in 0..mb_height {
                for mb_x in 0..mb_width {
                    frame.encode_mb(&mut bw, mb_x, mb_y);
                }
            }
            bw.write_trailing_bits();
            append_annexb_nal(&mut out, 3, NAL_IDR_SLICE, &bw.finish());
        }

        // Keep this frame's (border-extended) reconstruction as the next ref.
        let (ry, ru, rv) = frame.into_reference();
        self.reference = Some(RefPlanes { y: ry, u: ru, v: rv });
        self.frame_index = self.frame_index.wrapping_add(1);

        out
    }

    /// Write the IDR I-slice header (CAVLC, frame-only, single slice).
    fn write_idr_slice_header(&self, bw: &mut BitWriter) {
        bw.write_ue(0); // first_mb_in_slice
        bw.write_ue(7); // slice_type = I (7 -> "all I" form)
        bw.write_ue(0); // pic_parameter_set_id
        bw.write_bits(0, self.cfg.log2_max_frame_num); // frame_num = 0
        bw.write_ue(0); // idr_pic_id
        bw.write_bits(0, self.cfg.log2_max_poc_lsb); // pic_order_cnt_lsb = 0
        // dec_ref_pic_marking (IDR): no_output_of_prior_pics + long_term_reference.
        bw.write_flag(false);
        bw.write_flag(false);
        bw.write_se(0); // slice_qp_delta (slice qp == pic_init_qp)
        bw.write_ue(1); // disable_deblocking_filter_idc = 1 (off; output == recon)
    }

    /// Write a P-slice header (CAVLC, single slice, single short-term reference).
    fn write_p_slice_header(&self, bw: &mut BitWriter) {
        let max_fn = 1u32 << self.cfg.log2_max_frame_num;
        let max_poc = 1u32 << self.cfg.log2_max_poc_lsb;
        bw.write_ue(0); // first_mb_in_slice
        bw.write_ue(5); // slice_type = P (5 -> "all P" form)
        bw.write_ue(0); // pic_parameter_set_id
        bw.write_bits(self.frame_index % max_fn, self.cfg.log2_max_frame_num); // frame_num
        bw.write_bits((self.frame_index.wrapping_mul(2)) % max_poc, self.cfg.log2_max_poc_lsb); // poc_lsb
        bw.write_flag(false); // num_ref_idx_active_override_flag (use PPS default = 1)
        bw.write_flag(false); // ref_pic_list_modification_flag_l0
        // dec_ref_pic_marking (non-IDR, nal_ref_idc != 0): adaptive flag off.
        bw.write_flag(false);
        bw.write_se(0); // slice_qp_delta
        bw.write_ue(1); // disable_deblocking_filter_idc = 1 (off; output == recon)
    }
}

/// Copy `src` (`w`x`h`, row stride `src_stride`) into a tightly-strided
/// `dst_w`x`dst_h` buffer, replicating the right/bottom edges into the padding.
fn pad_plane(src: &[u8], src_stride: usize, w: usize, h: usize, dst_w: usize, dst_h: usize) -> (Vec<u8>, usize) {
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
