//! Baseline (profile 66) SPS/PPS generation, ported from the syntax in
//! `reference/codec/encoder/core/src/au_set.cpp`
//! (`WelsWriteSpsSyntax` / `WelsWritePpsSyntax`).
//!
//! These produce the RBSP payloads (with rbsp_trailing_bits) for a CAVLC,
//! frame-only, single-slice baseline stream. They are the exact inverse of the
//! decoder's [`crate::decoder::params::parse_sps`] / `parse_pps`.

use crate::bits::BitWriter;
use alloc::vec::Vec;

/// Baseline profile.
pub const PROFILE_BASELINE: u8 = 66;

/// `constraint_set0_flag` + `constraint_set1_flag` (then 4 zero bits): the
/// stream obeys Baseline and Main, i.e. Constrained Baseline. OpenH264 sets both
/// for baseline (`WelsInitSps`).
const CONSTRAINED_BASELINE_FLAGS: u32 = 0b1100_0000;

/// Level 3.0, the fallback before [`level_idc`] existed.
const DEFAULT_LEVEL_IDC: u8 = 30;

/// One row of `g_ksLevelLimits` (Table A-1): the limits this encoder can hit.
struct LevelLimits {
    level_idc: u8,
    max_mbps: u64,
    max_fs: u64,
    max_dpb_mbs: u64,
    /// kbit/s, in units of 1200 bit/s for Baseline (`cpbBrVclFactor`).
    max_br: u64,
}

const fn lvl(level_idc: u8, max_mbps: u64, max_fs: u64, max_dpb_mbs: u64, max_br: u64) -> LevelLimits {
    LevelLimits {
        level_idc,
        max_mbps,
        max_fs,
        max_dpb_mbs,
        max_br,
    }
}

/// `g_ksLevelLimits` without level 1b (a Baseline 1b stream is signalled as 1.1).
const LEVEL_LIMITS: [LevelLimits; 16] = [
    lvl(10, 1485, 99, 396, 64),
    lvl(11, 3000, 396, 900, 192),
    lvl(12, 6000, 396, 2376, 384),
    lvl(13, 11880, 396, 2376, 768),
    lvl(20, 11880, 396, 2376, 2000),
    lvl(21, 19800, 792, 4752, 4000),
    lvl(22, 20250, 1620, 8100, 4000),
    lvl(30, 40500, 1620, 8100, 10000),
    lvl(31, 108000, 3600, 18000, 14000),
    lvl(32, 216000, 5120, 20480, 20000),
    lvl(40, 245760, 8192, 32768, 20000),
    lvl(41, 245760, 8192, 32768, 50000),
    lvl(42, 522240, 8704, 34816, 50000),
    lvl(50, 589824, 22080, 110400, 135000),
    lvl(51, 983040, 36864, 184320, 240000),
    lvl(52, 2073600, 36864, 184320, 240000),
];
/// `cpbBrVclFactor` for Baseline.
const BR_FACTOR: u64 = 1200;
/// `WelsGetLevelIdc`'s fallback.
const MAX_LEVEL_IDC: u8 = 51;

/// Smallest level whose limits admit the stream (`WelsGetLevelIdc` /
/// `WelsCheckLevelLimitation`, one reference frame).
pub fn level_idc(mb_width: u32, mb_height: u32, fps: u32, bitrate: u64) -> u8 {
    let (w, h) = (mb_width as u64, mb_height as u64);
    let mbs = w * h;
    LEVEL_LIMITS
        .iter()
        .find(|l| {
            l.max_mbps >= mbs * fps as u64
                && l.max_fs >= mbs
                && l.max_fs * 8 >= w * w
                && l.max_fs * 8 >= h * h
                && l.max_dpb_mbs >= mbs
                && l.max_br * BR_FACTOR >= bitrate
        })
        .map_or(MAX_LEVEL_IDC, |l| l.level_idc)
}

/// Encoder-side parameter-set configuration derived from the public knobs.
#[derive(Debug, Clone, Copy)]
pub struct ParamConfig {
    pub mb_width: u32,
    pub mb_height: u32,
    /// Chroma-cropping offsets (in crop units) to recover the visible size.
    pub crop_right: u32,
    pub crop_bottom: u32,
    /// pic_init_qp (0..=51).
    pub qp: u8,
    pub level_idc: u8,
    /// log2_max_frame_num (>= 4).
    pub log2_max_frame_num: u32,
    /// log2_max_pic_order_cnt_lsb (>= 4).
    pub log2_max_poc_lsb: u32,
    /// Frame rate (fps) to advertise in VUI timing info. `None` omits VUI
    /// entirely (the historical behaviour); `Some(fps)` makes the raw Annex-B
    /// stream self-describing so players honour the playback rate.
    pub fps: Option<u32>,
}

impl ParamConfig {
    /// Build a config for the given visible `width`x`height` and `qp`.
    pub fn new(width: u32, height: u32, qp: u8) -> Self {
        let mb_width = (width + 15) >> 4;
        let mb_height = (height + 15) >> 4;
        // 4:2:0 crop units are 2 luma samples; offsets recover the visible size.
        let crop_right = (mb_width * 16 - width) / 2;
        let crop_bottom = (mb_height * 16 - height) / 2;
        ParamConfig {
            mb_width,
            mb_height,
            crop_right,
            crop_bottom,
            qp,
            level_idc: DEFAULT_LEVEL_IDC,
            log2_max_frame_num: 4,
            log2_max_poc_lsb: 4,
            fps: None,
        }
    }

    pub fn frame_cropping(&self) -> bool {
        self.crop_right != 0 || self.crop_bottom != 0
    }
}

/// Generate the baseline SPS RBSP (sps_id = 0).
pub fn write_sps(cfg: &ParamConfig) -> Vec<u8> {
    let mut bw = BitWriter::new();

    bw.write_bits(PROFILE_BASELINE as u32, 8); // profile_idc
    // constraint_set0..5 flags + 2 reserved zero bits (8 bits total).
    bw.write_bits(CONSTRAINED_BASELINE_FLAGS, 8);
    bw.write_bits(cfg.level_idc as u32, 8); // level_idc
    bw.write_ue(0); // seq_parameter_set_id

    // Baseline (not high profile): no chroma_format_idc / bit-depth block.

    bw.write_ue(cfg.log2_max_frame_num - 4); // log2_max_frame_num_minus4
    bw.write_ue(0); // pic_order_cnt_type = 0
    bw.write_ue(cfg.log2_max_poc_lsb - 4); // log2_max_pic_order_cnt_lsb_minus4

    bw.write_ue(1); // max_num_ref_frames
    bw.write_flag(false); // gaps_in_frame_num_value_allowed_flag
    bw.write_ue(cfg.mb_width - 1); // pic_width_in_mbs_minus1
    bw.write_ue(cfg.mb_height - 1); // pic_height_in_map_units_minus1
    bw.write_flag(true); // frame_mbs_only_flag
    bw.write_flag(true); // direct_8x8_inference_flag

    let crop = cfg.frame_cropping();
    bw.write_flag(crop); // frame_cropping_flag
    if crop {
        bw.write_ue(0); // crop_left
        bw.write_ue(cfg.crop_right);
        bw.write_ue(0); // crop_top
        bw.write_ue(cfg.crop_bottom);
    }
    match cfg.fps {
        // VUI carrying only timing_info, so raw Annex-B players know the frame
        // rate. fps = time_scale / (2 * num_units_in_tick) => num_units_in_tick=1,
        // time_scale=2*fps. Mirrors the decoder's `parse_vui`.
        Some(fps) if fps > 0 => {
            bw.write_flag(true); // vui_parameters_present_flag
            bw.write_flag(false); // aspect_ratio_info_present_flag
            bw.write_flag(false); // overscan_info_present_flag
            bw.write_flag(false); // video_signal_type_present_flag
            bw.write_flag(false); // chroma_loc_info_present_flag
            bw.write_flag(true); // timing_info_present_flag
            bw.write_bits(1, 32); // num_units_in_tick
            bw.write_bits(2 * fps, 32); // time_scale
            bw.write_flag(true); // fixed_frame_rate_flag
            bw.write_flag(false); // nal_hrd_parameters_present_flag
            bw.write_flag(false); // vcl_hrd_parameters_present_flag
            bw.write_flag(false); // pic_struct_present_flag
            bw.write_flag(false); // bitstream_restriction_flag
        }
        _ => bw.write_flag(false), // vui_parameters_present_flag
    }

    bw.write_trailing_bits();
    bw.finish()
}

/// Generate the baseline PPS RBSP (pps_id = 0, sps_id = 0, CAVLC).
pub fn write_pps(cfg: &ParamConfig) -> Vec<u8> {
    let mut bw = BitWriter::new();

    bw.write_ue(0); // pic_parameter_set_id
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_flag(false); // entropy_coding_mode_flag (CAVLC)
    bw.write_flag(false); // bottom_field_pic_order_in_frame_present_flag
    bw.write_ue(0); // num_slice_groups_minus1
    bw.write_ue(0); // num_ref_idx_l0_default_active_minus1
    bw.write_ue(0); // num_ref_idx_l1_default_active_minus1
    bw.write_flag(false); // weighted_pred_flag
    bw.write_bits(0, 2); // weighted_bipred_idc
    bw.write_se(cfg.qp as i32 - 26); // pic_init_qp_minus26
    bw.write_se(0); // pic_init_qs_minus26
    bw.write_se(0); // chroma_qp_index_offset
    bw.write_flag(true); // deblocking_filter_control_present_flag
    bw.write_flag(false); // constrained_intra_pred_flag
    bw.write_flag(false); // redundant_pic_cnt_present_flag

    bw.write_trailing_bits();
    bw.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::params::{parse_pps, parse_sps};

    #[test]
    fn sps_roundtrip_dimensions() {
        for &(w, h) in &[
            (176u32, 144u32),
            (640, 480),
            (1920, 1080),
            (160, 120),
            (100, 64),
        ] {
            for &qp in &[26u8, 32, 40] {
                let cfg = ParamConfig::new(w, h, qp);
                let sps = parse_sps(&write_sps(&cfg)).expect("parse_sps");
                assert_eq!(sps.profile_idc, PROFILE_BASELINE as u32, "{w}x{h}");
                assert_eq!(sps.width, w, "width {w}x{h}");
                assert_eq!(sps.height, h, "height {w}x{h}");
                assert_eq!(sps.mb_width, (w + 15) >> 4);
                assert_eq!(sps.mb_height, (h + 15) >> 4);
                assert_eq!(sps.sps_id, 0);
                assert!(sps.frame_mbs_only_flag);
            }
        }
    }

    #[test]
    fn sps_roundtrip_vui_timing() {
        // No fps -> no VUI.
        let sps = parse_sps(&write_sps(&ParamConfig::new(640, 480, 26))).unwrap();
        assert!(!sps.vui_parameters_present_flag);

        // fps set -> VUI timing such that time_scale / (2 * num_units_in_tick) == fps.
        for &fps in &[24u32, 25, 30, 60] {
            let mut cfg = ParamConfig::new(1920, 1080, 26);
            cfg.fps = Some(fps);
            let sps = parse_sps(&write_sps(&cfg)).expect("parse_sps");
            assert!(sps.vui_parameters_present_flag, "{fps}");
            assert!(sps.vui.timing_info_present_flag, "{fps}");
            assert_eq!(sps.vui.num_units_in_tick, 1, "{fps}");
            assert_eq!(sps.vui.time_scale, 2 * fps, "{fps}");
            assert!(sps.vui.fixed_frame_rate_flag, "{fps}");
            // Dimensions still decode correctly with VUI present.
            assert_eq!(sps.width, 1920, "{fps}");
            assert_eq!(sps.height, 1080, "{fps}");
        }
    }

    #[test]
    fn pps_roundtrip_qp() {
        for &qp in &[10u8, 26, 28, 32, 40, 51] {
            let cfg = ParamConfig::new(176, 144, qp);
            let sps = parse_sps(&write_sps(&cfg)).unwrap();
            let pps = parse_pps(&write_pps(&cfg), Some(&sps)).expect("parse_pps");
            assert_eq!(pps.pps_id, 0);
            assert_eq!(pps.sps_id, 0);
            assert!(!pps.entropy_coding_mode_flag);
            assert_eq!(pps.pic_init_qp, qp as i32, "qp {qp}");
        }
    }
}
