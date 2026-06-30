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
const NAL_IDR_SLICE: u8 = 5;
const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;

/// Intra-only (IDR) baseline H.264 encoder.
pub struct Encoder {
    width: u32,
    height: u32,
    cfg: ParamConfig,
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
        Ok(Encoder { width, height, cfg: ParamConfig::new(width, height, qp) })
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Encode one I420 frame into a self-contained Annex-B IDR access unit
    /// (SPS + PPS + IDR slice). `y`/`u`/`v` are the planar 8-bit samples with
    /// the given row strides; `u`/`v` are quarter-resolution (4:2:0).
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

        let mut frame = FrameEnc::new(mb_width, mb_height, qp, src_y, src_u, src_v, sy_stride, sc_stride);

        // --- Parameter sets ---
        let mut out = Vec::new();
        append_annexb_nal(&mut out, 3, NAL_SPS, &paraset::write_sps(&self.cfg));
        append_annexb_nal(&mut out, 3, NAL_PPS, &paraset::write_pps(&self.cfg));

        // --- IDR slice ---
        let mut bw = BitWriter::new();
        self.write_slice_header(&mut bw);
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                frame.encode_mb(&mut bw, mb_x, mb_y);
            }
        }
        bw.write_trailing_bits();
        let rbsp = bw.finish();
        append_annexb_nal(&mut out, 3, NAL_IDR_SLICE, &rbsp);

        out
    }

    /// Write the IDR I-slice header (CAVLC, frame-only, single slice).
    fn write_slice_header(&self, bw: &mut BitWriter) {
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
        for x in w..dst_w {
            drow[x] = edge;
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
