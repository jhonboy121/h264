//! Sequence- and Picture-Parameter-Set parsing.
//!
//! Faithful port of `ParseSps`/`WelsParseSps`, `ParsePps`, `ParseVui` and
//! `ParseScalingList`/`SetScalingListValue` from
//! `reference/codec/decoder/core/src/au_parser.cpp`, with the struct shapes
//! mirroring `TagSps`/`TagPps`/`TagVui` in
//! `reference/codec/decoder/core/inc/parameter_sets.h`.
//!
//! Where OpenH264 deliberately rejects valid bitstreams (chroma_format_idc > 1,
//! bit depth != 8, `frame_mbs_only_flag == 0`, slice_group_map_type > 1) this
//! port instead parses the syntax per the H.264 spec so the parser is a complete
//! reader; the values are still stored for downstream code to validate.

use crate::bits::BitReader;
use crate::error::DecodeError;

type Result<T> = core::result::Result<T, DecodeError>;

// ---- Constants ported from au_parser.cpp / wels_const*.h / dec_golomb.h ----

const MAX_SPS_COUNT: u32 = 32;
const MAX_PPS_COUNT: u32 = 256;
const MAX_SLICEGROUP_IDS: usize = 8;
const MAX_MB_SIZE: u32 = 36864;

const SPS_LOG2_MAX_FRAME_NUM_MINUS4_MAX: u32 = 12;
const SPS_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4_MAX: u32 = 12;
const SPS_NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE_MAX: u32 = 255;
const SPS_MAX_NUM_REF_FRAMES_MAX: u32 = 16;

const PPS_PIC_INIT_QP_QS_MIN: i32 = 0;
const PPS_PIC_INIT_QP_QS_MAX: i32 = 51;
const PPS_CHROMA_QP_INDEX_OFFSET_MIN: i32 = -12;
const PPS_CHROMA_QP_INDEX_OFFSET_MAX: i32 = 12;

const SCALING_LIST_DELTA_SCALE_MIN: i32 = -128;
const SCALING_LIST_DELTA_SCALE_MAX: i32 = 127;

const EXTENDED_SAR: u32 = 255;

// Sample-aspect-ratio table for aspect_ratio_idc in 1..=16 (Table E-1).
// Index 0 is the "Unspecified" placeholder (idc 0); idc N maps to entry N.
const VUI_SAR: [(u32, u32); 17] = [
    (0, 0),
    (1, 1),
    (12, 11),
    (10, 11),
    (16, 11),
    (40, 33),
    (24, 11),
    (20, 11),
    (32, 11),
    (80, 33),
    (18, 11),
    (15, 11),
    (64, 33),
    (160, 99),
    (4, 3),
    (3, 2),
    (2, 1),
];

// 4x4 residual zig-zag scan order (`g_kuiZigzagScan`).
const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

// 8x8 residual zig-zag scan order (`g_kuiZigzagScan8x8`).
#[rustfmt::skip]
const ZIGZAG_8X8: [usize; 64] = [
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// Default 4x4 scaling lists (`g_kuiDequantScaling4x4Default`).
#[rustfmt::skip]
const DEQUANT_4X4_DEFAULT: [[u8; 16]; 2] = [
    [6, 13, 20, 28, 13, 20, 28, 32, 20, 28, 32, 37, 28, 32, 37, 42],
    [10, 14, 20, 24, 14, 20, 24, 27, 20, 24, 27, 30, 24, 27, 30, 34],
];

// Default 8x8 scaling lists (`g_kuiDequantScaling8x8Default`).
#[rustfmt::skip]
const DEQUANT_8X8_DEFAULT: [[u8; 64]; 2] = [
    [
        6, 10, 13, 16, 18, 23, 25, 27, 10, 11, 16, 18, 23, 25, 27, 29,
        13, 16, 18, 23, 25, 27, 29, 31, 16, 18, 23, 25, 27, 29, 31, 33,
        18, 23, 25, 27, 29, 31, 33, 36, 23, 25, 27, 29, 31, 33, 36, 38,
        25, 27, 29, 31, 33, 36, 38, 40, 27, 29, 31, 33, 36, 38, 40, 42,
    ],
    [
        9, 13, 15, 17, 19, 21, 22, 24, 13, 13, 17, 19, 21, 22, 24, 25,
        15, 17, 19, 21, 22, 24, 25, 27, 17, 19, 21, 22, 24, 25, 27, 28,
        19, 21, 22, 24, 25, 27, 28, 30, 21, 22, 24, 25, 27, 28, 30, 32,
        22, 24, 25, 27, 28, 30, 32, 33, 24, 25, 27, 28, 30, 32, 33, 35,
    ],
];

/// Video Usability Information (subset of Annex E parsed by OpenH264's
/// `ParseVui`). HRD parameter bodies are intentionally not stored.
#[derive(Debug, Clone, Default)]
pub struct Vui {
    pub aspect_ratio_info_present_flag: bool,
    pub aspect_ratio_idc: u32,
    pub sar_width: u32,
    pub sar_height: u32,
    pub overscan_info_present_flag: bool,
    pub overscan_appropriate_flag: bool,
    pub video_signal_type_present_flag: bool,
    pub video_format: u8,
    pub video_full_range_flag: bool,
    pub colour_description_present_flag: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coeffs: u8,
    pub chroma_loc_info_present_flag: bool,
    pub chroma_sample_loc_type_top_field: u32,
    pub chroma_sample_loc_type_bottom_field: u32,
    pub timing_info_present_flag: bool,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate_flag: bool,
    pub nal_hrd_parameters_present_flag: bool,
    pub vcl_hrd_parameters_present_flag: bool,
    pub pic_struct_present_flag: bool,
    pub bitstream_restriction_flag: bool,
    pub motion_vectors_over_pic_boundaries_flag: bool,
    pub max_bytes_per_pic_denom: u32,
    pub max_bits_per_mb_denom: u32,
    pub log2_max_mv_length_horizontal: u32,
    pub log2_max_mv_length_vertical: u32,
    pub max_num_reorder_frames: u32,
    pub max_dec_frame_buffering: u32,
}

/// Sequence Parameter Set (mirrors `TagSps`).
#[derive(Debug, Clone)]
pub struct Sps {
    pub sps_id: u32,
    pub profile_idc: u32,
    pub level_idc: u8,
    pub constraint_set_flags: [bool; 6],

    pub chroma_format_idc: u32,
    pub chroma_array_type: u32,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub qpprime_y_zero_transform_bypass_flag: bool,
    pub seq_scaling_matrix_present_flag: bool,
    pub seq_scaling_list_present_flag: [bool; 12],
    pub scaling_list_4x4: [[u8; 16]; 6],
    pub scaling_list_8x8: [[u8; 64]; 6],

    pub log2_max_frame_num: u32,
    pub pic_order_cnt_type: u32,
    /// POC type 0: log2_max_pic_order_cnt_lsb.
    pub log2_max_poc_lsb: i32,
    /// POC type 1.
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub num_ref_frames_in_poc_cycle: u32,
    pub offset_for_ref_frame: [i32; 256],

    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,

    pub mb_width: u32,
    pub mb_height: u32,
    pub total_mb_count: u32,

    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,

    pub frame_cropping_flag: bool,
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,

    pub vui_parameters_present_flag: bool,
    pub vui: Vui,

    /// Cropped luma picture width in pixels.
    pub width: u32,
    /// Cropped luma picture height in pixels.
    pub height: u32,
}

impl Default for Sps {
    fn default() -> Self {
        Sps {
            sps_id: 0,
            profile_idc: 0,
            level_idc: 0,
            constraint_set_flags: [false; 6],
            chroma_format_idc: 1,
            chroma_array_type: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_list_present_flag: [false; 12],
            scaling_list_4x4: [[0; 16]; 6],
            scaling_list_8x8: [[0; 64]; 6],
            log2_max_frame_num: 0,
            pic_order_cnt_type: 0,
            log2_max_poc_lsb: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_poc_cycle: 0,
            offset_for_ref_frame: [0; 256],
            max_num_ref_frames: 0,
            gaps_in_frame_num_value_allowed_flag: false,
            mb_width: 0,
            mb_height: 0,
            total_mb_count: 0,
            frame_mbs_only_flag: false,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: false,
            frame_cropping_flag: false,
            crop_left: 0,
            crop_right: 0,
            crop_top: 0,
            crop_bottom: 0,
            vui_parameters_present_flag: false,
            vui: Vui::default(),
            width: 0,
            height: 0,
        }
    }
}

/// Picture Parameter Set (mirrors `TagPps`).
#[derive(Debug, Clone)]
pub struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,

    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,

    pub num_slice_groups: u32,
    pub slice_group_map_type: u32,
    /// map_type 0.
    pub run_length: [u32; MAX_SLICEGROUP_IDS],
    /// map_type 2.
    pub top_left: [u32; MAX_SLICEGROUP_IDS],
    pub bottom_right: [u32; MAX_SLICEGROUP_IDS],
    /// map_type 3/4/5.
    pub slice_group_change_direction_flag: bool,
    pub slice_group_change_rate: u32,
    /// map_type 6.
    pub pic_size_in_map_units: u32,
    pub slice_group_id: alloc::vec::Vec<u32>,

    pub num_ref_idx_l0_active: u32,
    pub num_ref_idx_l1_active: u32,

    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,

    pub pic_init_qp: i32,
    pub pic_init_qs: i32,
    /// [cb, cr] chroma QP index offsets.
    pub chroma_qp_index_offset: [i32; 2],

    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,

    pub transform_8x8_mode_flag: bool,
    pub pic_scaling_matrix_present_flag: bool,
    pub pic_scaling_list_present_flag: [bool; 12],
    pub scaling_list_4x4: [[u8; 16]; 6],
    pub scaling_list_8x8: [[u8; 64]; 6],
    pub second_chroma_qp_index_offset: i32,
}

impl Default for Pps {
    fn default() -> Self {
        Pps {
            pps_id: 0,
            sps_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups: 1,
            slice_group_map_type: 0,
            run_length: [0; MAX_SLICEGROUP_IDS],
            top_left: [0; MAX_SLICEGROUP_IDS],
            bottom_right: [0; MAX_SLICEGROUP_IDS],
            slice_group_change_direction_flag: false,
            slice_group_change_rate: 0,
            pic_size_in_map_units: 0,
            slice_group_id: alloc::vec::Vec::new(),
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 1,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp: 26,
            pic_init_qs: 26,
            chroma_qp_index_offset: [0; 2],
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            transform_8x8_mode_flag: false,
            pic_scaling_matrix_present_flag: false,
            pic_scaling_list_present_flag: [false; 12],
            scaling_list_4x4: [[0; 16]; 6],
            scaling_list_8x8: [[0; 64]; 6],
            second_chroma_qp_index_offset: 0,
        }
    }
}

#[inline]
fn is_high_profile(profile_idc: u32) -> bool {
    // PRO_SCALABLE_BASELINE/HIGH(83/86), PRO_HIGH(100), PRO_HIGH10(110),
    // PRO_HIGH422(122), PRO_HIGH444(144), PRO_CAVLC444(244), and 44.
    matches!(profile_idc, 83 | 86 | 100 | 110 | 122 | 144 | 244 | 44)
}

/// `Ceil(Log2(n))` — the bit width used for `slice_group_id` (map type 6).
#[inline]
fn ceil_log2(n: u32) -> u32 {
    let mut k = 0u32;
    while (1u32 << k) < n {
        k += 1;
    }
    k
}

/// Parse a Sequence Parameter Set from RBSP (header byte already stripped).
pub fn parse_sps(rbsp: &[u8]) -> Result<Sps> {
    let mut bs = BitReader::new(rbsp);
    let mut sps = Sps::default();

    let profile_idc = bs.read_bits(8)?;
    sps.profile_idc = profile_idc;
    for i in 0..6 {
        sps.constraint_set_flags[i] = bs.read_flag()?;
    }
    let _reserved_zero_2bits = bs.read_bits(2)?;
    sps.level_idc = bs.read_bits(8)? as u8;

    let sps_id = bs.read_ue()?;
    if sps_id >= MAX_SPS_COUNT {
        return Err(DecodeError::InvalidSyntax("sps_id out of range"));
    }
    sps.sps_id = sps_id;

    // Defaults (set before the optional high-profile block, matching the C).
    sps.chroma_format_idc = 1;
    sps.chroma_array_type = 1;

    if is_high_profile(profile_idc) {
        sps.chroma_format_idc = bs.read_ue()?;
        if sps.chroma_format_idc > 3 {
            return Err(DecodeError::InvalidSyntax("chroma_format_idc"));
        }
        if sps.chroma_format_idc == 3 {
            sps.separate_colour_plane_flag = bs.read_flag()?;
        }
        sps.chroma_array_type = if sps.separate_colour_plane_flag {
            0
        } else {
            sps.chroma_format_idc
        };
        sps.bit_depth_luma = 8 + bs.read_ue()?;
        sps.bit_depth_chroma = 8 + bs.read_ue()?;
        sps.qpprime_y_zero_transform_bypass_flag = bs.read_flag()?;
        sps.seq_scaling_matrix_present_flag = bs.read_flag()?;
        if sps.seq_scaling_matrix_present_flag {
            let num = if sps.chroma_format_idc != 3 { 8 } else { 12 };
            // For SPS scaling lists there is no prior SPS matrix to fall back on.
            let (mut l4, mut l8) = (sps.scaling_list_4x4, sps.scaling_list_8x8);
            parse_scaling_list(
                &mut bs,
                num,
                &mut sps.seq_scaling_list_present_flag,
                &mut l4,
                &mut l8,
            )?;
            sps.scaling_list_4x4 = l4;
            sps.scaling_list_8x8 = l8;
        }
    }

    let log2_max_frame_num_minus4 = bs.read_ue_max(
        SPS_LOG2_MAX_FRAME_NUM_MINUS4_MAX,
        "log2_max_frame_num_minus4",
    )?;
    sps.log2_max_frame_num = 4 + log2_max_frame_num_minus4;

    sps.pic_order_cnt_type = bs.read_ue()?;
    if sps.pic_order_cnt_type == 0 {
        let v = bs.read_ue_max(
            SPS_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4_MAX,
            "log2_max_pic_order_cnt_lsb_minus4",
        )?;
        sps.log2_max_poc_lsb = 4 + v as i32;
    } else if sps.pic_order_cnt_type == 1 {
        sps.delta_pic_order_always_zero_flag = bs.read_flag()?;
        sps.offset_for_non_ref_pic = bs.read_se()?;
        sps.offset_for_top_to_bottom_field = bs.read_se()?;
        let n = bs.read_ue_max(
            SPS_NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE_MAX,
            "num_ref_frames_in_pic_order_cnt_cycle",
        )?;
        sps.num_ref_frames_in_poc_cycle = n;
        for i in 0..n as usize {
            sps.offset_for_ref_frame[i] = bs.read_se()?;
        }
    } else if sps.pic_order_cnt_type > 2 {
        return Err(DecodeError::InvalidSyntax("pic_order_cnt_type"));
    }

    sps.max_num_ref_frames = bs.read_ue()?;
    sps.gaps_in_frame_num_value_allowed_flag = bs.read_flag()?;

    let mb_width = 1 + bs.read_ue()?;
    if mb_width == 0 || mb_width > MAX_MB_SIZE {
        return Err(DecodeError::InvalidSyntax("pic_width_in_mbs"));
    }
    sps.mb_width = mb_width;

    let mb_height = 1 + bs.read_ue()?;
    if mb_height == 0 || mb_height > MAX_MB_SIZE {
        return Err(DecodeError::InvalidSyntax("pic_height_in_map_units"));
    }
    sps.mb_height = mb_height;
    sps.total_mb_count = mb_width * mb_height;

    if sps.max_num_ref_frames > SPS_MAX_NUM_REF_FRAMES_MAX {
        return Err(DecodeError::InvalidSyntax("max_num_ref_frames"));
    }

    sps.frame_mbs_only_flag = bs.read_flag()?;
    if !sps.frame_mbs_only_flag {
        // Spec: interlaced coding. OpenH264 rejects this; we parse it.
        sps.mb_adaptive_frame_field_flag = bs.read_flag()?;
    }
    sps.direct_8x8_inference_flag = bs.read_flag()?;

    sps.frame_cropping_flag = bs.read_flag()?;
    if sps.frame_cropping_flag {
        sps.crop_left = bs.read_ue()?;
        sps.crop_right = bs.read_ue()?;
        if (sps.crop_left + sps.crop_right) > (mb_width * 16 / 2) {
            return Err(DecodeError::InvalidSyntax("frame_crop horizontal"));
        }
        sps.crop_top = bs.read_ue()?;
        sps.crop_bottom = bs.read_ue()?;
        if (sps.crop_top + sps.crop_bottom) > (mb_height * 16 / 2) {
            return Err(DecodeError::InvalidSyntax("frame_crop vertical"));
        }
    }

    sps.vui_parameters_present_flag = bs.read_flag()?;
    if sps.vui_parameters_present_flag {
        parse_vui(&mut bs, &mut sps.vui)?;
    }

    // Derived pixel dimensions, accounting for cropping (spec 7.4.2.1.1).
    let frame_mbs = if sps.frame_mbs_only_flag { 1 } else { 2 };
    let width_pixels = mb_width * 16;
    let height_pixels = mb_height * 16 * frame_mbs;
    let (sub_w, sub_h) = match sps.chroma_array_type {
        1 => (2u32, 2u32), // 4:2:0
        2 => (2, 1),       // 4:2:2
        3 => (1, 1),       // 4:4:4
        _ => (1, 1),       // 4:0:0 / separate colour plane -> ChromaArrayType 0
    };
    let (crop_unit_x, crop_unit_y) = if sps.chroma_array_type == 0 {
        (1, frame_mbs)
    } else {
        (sub_w, sub_h * frame_mbs)
    };
    sps.width = width_pixels.saturating_sub(crop_unit_x * (sps.crop_left + sps.crop_right));
    sps.height = height_pixels.saturating_sub(crop_unit_y * (sps.crop_top + sps.crop_bottom));

    Ok(sps)
}

/// Parse a Picture Parameter Set from RBSP. `sps` is the referenced SPS (needed
/// for chroma_format_idc when a pic_scaling_matrix is present); pass `None` if
/// unavailable — parsing then errors only if a scaling matrix is actually
/// signalled, mirroring `ParsePps`.
pub fn parse_pps(rbsp: &[u8], sps: Option<&Sps>) -> Result<Pps> {
    let mut bs = BitReader::new(rbsp);
    let mut pps = Pps::default();

    let pps_id = bs.read_ue()?;
    if pps_id >= MAX_PPS_COUNT {
        return Err(DecodeError::InvalidSyntax("pps_id out of range"));
    }
    pps.pps_id = pps_id;

    let sps_id = bs.read_ue()?;
    if sps_id >= MAX_SPS_COUNT {
        return Err(DecodeError::InvalidSyntax("sps_id out of range"));
    }
    pps.sps_id = sps_id;

    pps.entropy_coding_mode_flag = bs.read_flag()?;
    pps.bottom_field_pic_order_in_frame_present_flag = bs.read_flag()?;

    pps.num_slice_groups = 1 + bs.read_ue()?;
    if pps.num_slice_groups > 1 {
        pps.slice_group_map_type = bs.read_ue()?;
        match pps.slice_group_map_type {
            0 => {
                let count = pps.num_slice_groups as usize;
                if count > MAX_SLICEGROUP_IDS {
                    return Err(DecodeError::InvalidSyntax("num_slice_groups"));
                }
                for i in 0..count {
                    pps.run_length[i] = 1 + bs.read_ue()?;
                }
            }
            2 => {
                let count = (pps.num_slice_groups - 1) as usize;
                if count > MAX_SLICEGROUP_IDS {
                    return Err(DecodeError::InvalidSyntax("num_slice_groups"));
                }
                for i in 0..count {
                    pps.top_left[i] = bs.read_ue()?;
                    pps.bottom_right[i] = bs.read_ue()?;
                }
            }
            3..=5 => {
                pps.slice_group_change_direction_flag = bs.read_flag()?;
                pps.slice_group_change_rate = 1 + bs.read_ue()?;
            }
            6 => {
                pps.pic_size_in_map_units = 1 + bs.read_ue()?;
                let bits = ceil_log2(pps.num_slice_groups);
                pps.slice_group_id
                    .reserve(pps.pic_size_in_map_units as usize);
                for _ in 0..pps.pic_size_in_map_units {
                    pps.slice_group_id.push(bs.read_bits(bits)?);
                }
            }
            _ => return Err(DecodeError::InvalidSyntax("slice_group_map_type")),
        }
    }

    pps.num_ref_idx_l0_active = 1 + bs.read_ue()?;
    pps.num_ref_idx_l1_active = 1 + bs.read_ue()?;
    if pps.num_ref_idx_l0_active > 32 || pps.num_ref_idx_l1_active > 32 {
        return Err(DecodeError::InvalidSyntax("num_ref_idx_default_active"));
    }

    pps.weighted_pred_flag = bs.read_flag()?;
    pps.weighted_bipred_idc = bs.read_bits(2)? as u8;

    pps.pic_init_qp = 26 + bs.read_se()?;
    if pps.pic_init_qp < PPS_PIC_INIT_QP_QS_MIN || pps.pic_init_qp > PPS_PIC_INIT_QP_QS_MAX {
        return Err(DecodeError::InvalidSyntax("pic_init_qp"));
    }
    pps.pic_init_qs = 26 + bs.read_se()?;
    if pps.pic_init_qs < PPS_PIC_INIT_QP_QS_MIN || pps.pic_init_qs > PPS_PIC_INIT_QP_QS_MAX {
        return Err(DecodeError::InvalidSyntax("pic_init_qs"));
    }

    let cb = bs.read_se()?;
    if !(PPS_CHROMA_QP_INDEX_OFFSET_MIN..=PPS_CHROMA_QP_INDEX_OFFSET_MAX).contains(&cb) {
        return Err(DecodeError::InvalidSyntax("chroma_qp_index_offset"));
    }
    pps.chroma_qp_index_offset[0] = cb;
    pps.chroma_qp_index_offset[1] = cb; // default cr = cb
    pps.second_chroma_qp_index_offset = cb;

    pps.deblocking_filter_control_present_flag = bs.read_flag()?;
    pps.constrained_intra_pred_flag = bs.read_flag()?;
    pps.redundant_pic_cnt_present_flag = bs.read_flag()?;

    if bs.more_rbsp_data() {
        pps.transform_8x8_mode_flag = bs.read_flag()?;
        pps.pic_scaling_matrix_present_flag = bs.read_flag()?;
        if pps.pic_scaling_matrix_present_flag {
            let sps = sps.ok_or(DecodeError::MissingParameterSet)?;
            let num = 6 + if pps.transform_8x8_mode_flag {
                if sps.chroma_format_idc != 3 { 2 } else { 6 }
            } else {
                0
            };
            let (mut l4, mut l8) = (pps.scaling_list_4x4, pps.scaling_list_8x8);
            parse_scaling_list_pps(
                &mut bs,
                sps,
                num,
                &mut pps.pic_scaling_list_present_flag,
                &mut l4,
                &mut l8,
            )?;
            pps.scaling_list_4x4 = l4;
            pps.scaling_list_8x8 = l8;
        }
        let cr = bs.read_se()?;
        if !(PPS_CHROMA_QP_INDEX_OFFSET_MIN..=PPS_CHROMA_QP_INDEX_OFFSET_MAX).contains(&cr) {
            return Err(DecodeError::InvalidSyntax("second_chroma_qp_index_offset"));
        }
        pps.chroma_qp_index_offset[1] = cr;
        pps.second_chroma_qp_index_offset = cr;
    }

    Ok(pps)
}

/// `SetScalingListValue`: read one scaling list of `count` entries (16 or 64),
/// writing them in raster order via the matching zig-zag scan. Returns whether
/// `use_default_scaling_matrix_flag` was set.
fn set_scaling_list_value(bs: &mut BitReader<'_>, list: &mut [u8], count: usize) -> Result<bool> {
    let mut last_scale: i32 = 8;
    let mut next_scale: i32 = 8;
    let mut use_default = false;
    for j in 0..count {
        if next_scale != 0 {
            let delta = bs.read_se()?;
            if !(SCALING_LIST_DELTA_SCALE_MIN..=SCALING_LIST_DELTA_SCALE_MAX).contains(&delta) {
                return Err(DecodeError::InvalidSyntax("scaling_list delta_scale"));
            }
            next_scale = (last_scale + delta + 256) % 256;
            use_default = j == 0 && next_scale == 0;
            if use_default {
                break;
            }
        }
        let idx = if count == 16 {
            ZIGZAG_4X4[j]
        } else {
            ZIGZAG_8X8[j]
        };
        list[idx] = if next_scale == 0 {
            last_scale as u8
        } else {
            next_scale as u8
        };
        last_scale = list[idx] as i32;
    }
    Ok(use_default)
}

/// `ParseScalingList` for the SPS case (`bPPS == false`, no fall-back matrices).
fn parse_scaling_list(
    bs: &mut BitReader<'_>,
    count: usize,
    present: &mut [bool; 12],
    l4: &mut [[u8; 16]; 6],
    l8: &mut [[u8; 64]; 6],
) -> Result<()> {
    // For the SPS path the "default" fall-backs are the spec flat defaults.
    let default_4x4: [[u8; 16]; 2] = [DEQUANT_4X4_DEFAULT[0], DEQUANT_4X4_DEFAULT[1]];
    let default_8x8: [[u8; 64]; 2] = [DEQUANT_8X8_DEFAULT[0], DEQUANT_8X8_DEFAULT[1]];
    scaling_list_loop(bs, count, present, l4, l8, &default_4x4, &default_8x8)
}

/// `ParseScalingList` for the PPS case (`bPPS == true`): the fall-back matrices
/// come from the SPS when `seq_scaling_matrix_present_flag` was set.
fn parse_scaling_list_pps(
    bs: &mut BitReader<'_>,
    sps: &Sps,
    count: usize,
    present: &mut [bool; 12],
    l4: &mut [[u8; 16]; 6],
    l8: &mut [[u8; 64]; 6],
) -> Result<()> {
    let init = sps.seq_scaling_matrix_present_flag;
    let default_4x4: [[u8; 16]; 2] = if init {
        [sps.scaling_list_4x4[0], sps.scaling_list_4x4[3]]
    } else {
        [DEQUANT_4X4_DEFAULT[0], DEQUANT_4X4_DEFAULT[1]]
    };
    let default_8x8: [[u8; 64]; 2] = if init {
        [sps.scaling_list_8x8[0], sps.scaling_list_8x8[1]]
    } else {
        [DEQUANT_8X8_DEFAULT[0], DEQUANT_8X8_DEFAULT[1]]
    };
    scaling_list_loop(bs, count, present, l4, l8, &default_4x4, &default_8x8)
}

/// Shared body of `ParseScalingList`, parameterised by the fall-back matrices.
fn scaling_list_loop(
    bs: &mut BitReader<'_>,
    count: usize,
    present: &mut [bool; 12],
    l4: &mut [[u8; 16]; 6],
    l8: &mut [[u8; 64]; 6],
    default_4x4: &[[u8; 16]; 2],
    default_8x8: &[[u8; 64]; 2],
) -> Result<()> {
    for i in 0..count {
        let flag = bs.read_flag()?;
        present[i] = flag;
        if flag {
            if i < 6 {
                let use_default = set_scaling_list_value(bs, &mut l4[i], 16)?;
                if use_default {
                    l4[i] = DEQUANT_4X4_DEFAULT[i / 3];
                }
            } else {
                let use_default = set_scaling_list_value(bs, &mut l8[i - 6], 64)?;
                if use_default {
                    l8[i - 6] = DEQUANT_8X8_DEFAULT[(i - 6) & 1];
                }
            }
        } else if i < 6 {
            if i != 0 && i != 3 {
                l4[i] = l4[i - 1];
            } else {
                l4[i] = default_4x4[i / 3];
            }
        } else if i == 6 || i == 7 {
            l8[i - 6] = default_8x8[i & 1];
        } else {
            l8[i - 6] = l8[i - 8];
        }
    }
    Ok(())
}

/// `ParseVui` — Annex E. HRD parameter bodies are not stored: OpenH264's port
/// (`_PARSE_NALHRD_VCLHRD_PARAMS_` undefined) does not consume them, so the
/// flag is recorded but the body is left for the caller. We match that: if an
/// HRD is signalled we record the flag and stop reading further VUI fields,
/// exactly as the reference decoder does on the no-HRD build.
fn parse_vui(bs: &mut BitReader<'_>, vui: &mut Vui) -> Result<()> {
    vui.aspect_ratio_info_present_flag = bs.read_flag()?;
    if vui.aspect_ratio_info_present_flag {
        vui.aspect_ratio_idc = bs.read_bits(8)?;
        if vui.aspect_ratio_idc < 17 {
            let (w, h) = VUI_SAR[vui.aspect_ratio_idc as usize];
            vui.sar_width = w;
            vui.sar_height = h;
        } else if vui.aspect_ratio_idc == EXTENDED_SAR {
            vui.sar_width = bs.read_bits(16)?;
            vui.sar_height = bs.read_bits(16)?;
        }
    }
    vui.overscan_info_present_flag = bs.read_flag()?;
    if vui.overscan_info_present_flag {
        vui.overscan_appropriate_flag = bs.read_flag()?;
    }
    vui.video_signal_type_present_flag = bs.read_flag()?;
    if vui.video_signal_type_present_flag {
        vui.video_format = bs.read_bits(3)? as u8;
        vui.video_full_range_flag = bs.read_flag()?;
        vui.colour_description_present_flag = bs.read_flag()?;
        if vui.colour_description_present_flag {
            vui.colour_primaries = bs.read_bits(8)? as u8;
            vui.transfer_characteristics = bs.read_bits(8)? as u8;
            vui.matrix_coeffs = bs.read_bits(8)? as u8;
        }
    }
    vui.chroma_loc_info_present_flag = bs.read_flag()?;
    if vui.chroma_loc_info_present_flag {
        vui.chroma_sample_loc_type_top_field = bs.read_ue()?;
        vui.chroma_sample_loc_type_bottom_field = bs.read_ue()?;
    }
    vui.timing_info_present_flag = bs.read_flag()?;
    if vui.timing_info_present_flag {
        vui.num_units_in_tick = bs.read_bits(32)?;
        vui.time_scale = bs.read_bits(32)?;
        vui.fixed_frame_rate_flag = bs.read_flag()?;
    }
    vui.nal_hrd_parameters_present_flag = bs.read_flag()?;
    if vui.nal_hrd_parameters_present_flag {
        // HRD body not parsed (matches OpenH264's default build). Returning
        // Unsupported keeps the bit position honest rather than misreading on.
        return Err(DecodeError::Unsupported("vui nal_hrd_parameters"));
    }
    vui.vcl_hrd_parameters_present_flag = bs.read_flag()?;
    if vui.vcl_hrd_parameters_present_flag {
        return Err(DecodeError::Unsupported("vui vcl_hrd_parameters"));
    }
    vui.pic_struct_present_flag = bs.read_flag()?;
    vui.bitstream_restriction_flag = bs.read_flag()?;
    if vui.bitstream_restriction_flag {
        vui.motion_vectors_over_pic_boundaries_flag = bs.read_flag()?;
        vui.max_bytes_per_pic_denom = bs.read_ue()?;
        vui.max_bits_per_mb_denom = bs.read_ue()?;
        vui.log2_max_mv_length_horizontal = bs.read_ue()?;
        vui.log2_max_mv_length_vertical = bs.read_ue()?;
        vui.max_num_reorder_frames = bs.read_ue()?;
        vui.max_dec_frame_buffering = bs.read_ue()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::nal::{NalUnitType, annexb_nal_units, parse_nal};
    use alloc::vec::Vec;

    fn collect_nals(stream: &[u8]) -> Vec<crate::decoder::nal::NalUnit> {
        annexb_nal_units(stream).filter_map(parse_nal).collect()
    }

    fn find_sps(stream: &[u8]) -> Sps {
        for nal in collect_nals(stream) {
            if nal.unit_type == NalUnitType::Sps {
                return parse_sps(&nal.rbsp).expect("SPS parse");
            }
        }
        panic!("no SPS in stream");
    }

    #[test]
    fn banm_mw_d_sps() {
        let stream = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/BANM_MW_D.264"
        ));
        let sps = find_sps(stream);
        // Baseline/Main-class profile, sane level.
        assert!(sps.profile_idc >= 66 && sps.profile_idc <= 244);
        assert!(sps.level_idc > 0);
        assert!(sps.frame_mbs_only_flag);
        // Positive, macroblock-aligned dimensions for a CIF/QCIF-class clip.
        assert!(sps.width > 0 && sps.width <= 1024);
        assert!(sps.height > 0 && sps.height <= 1024);
        assert!(sps.width <= sps.mb_width * 16);
        assert!(sps.height <= sps.mb_height * 16);
        assert_eq!(sps.total_mb_count, sps.mb_width * sps.mb_height);
    }

    #[test]
    fn banm_mw_d_pps() {
        let stream = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/BANM_MW_D.264"
        ));
        let sps = find_sps(stream);
        let mut found = false;
        for nal in collect_nals(stream) {
            if nal.unit_type == NalUnitType::Pps {
                let pps = parse_pps(&nal.rbsp, Some(&sps)).expect("PPS parse");
                assert!(pps.pic_init_qp >= 0 && pps.pic_init_qp <= 51);
                assert!(pps.num_slice_groups >= 1);
                assert_eq!(pps.sps_id, sps.sps_id);
                found = true;
            }
        }
        assert!(found, "no PPS in stream");
    }

    #[test]
    fn ba1_ft_c_sps() {
        let stream = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/BA1_FT_C.264"
        ));
        let sps = find_sps(stream);
        assert!(sps.width > 0 && sps.height > 0);
        assert!(sps.profile_idc > 0);
    }

    #[test]
    fn sva_nl1_b_sps() {
        let stream = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/SVA_NL1_B.264"
        ));
        let sps = find_sps(stream);
        assert!(sps.width > 0 && sps.height > 0);
        assert!(sps.profile_idc > 0);
    }

    /// Hand-built minimal baseline SPS: profile 66, level 1.0, sps_id 0,
    /// log2_max_frame_num_minus4=0, poc_type 0, log2_max_poc_lsb_minus4=0,
    /// max_num_ref_frames=1, no gaps, 11x9 MBs (QCIF 176x144), frame_mbs_only,
    /// no direct8x8, no crop, no VUI.
    #[test]
    fn handbuilt_minimal_sps() {
        // Build the RBSP bit-by-bit.
        let mut bits: Vec<u8> = Vec::new();
        let push_bits = |val: u32, n: u32, bits: &mut Vec<u8>| {
            for i in (0..n).rev() {
                bits.push(((val >> i) & 1) as u8);
            }
        };
        // ue(v) encoder.
        let push_ue = |val: u32, bits: &mut Vec<u8>| {
            let code = val + 1;
            let len = 32 - code.leading_zeros();
            for _ in 0..(len - 1) {
                bits.push(0);
            }
            for i in (0..len).rev() {
                bits.push(((code >> i) & 1) as u8);
            }
        };

        push_bits(66, 8, &mut bits); // profile_idc
        bits.extend(core::iter::repeat_n(0, 6)); // constraint flags
        push_bits(0, 2, &mut bits); // reserved_zero_2bits
        push_bits(10, 8, &mut bits); // level_idc = 1.0 (10)
        push_ue(0, &mut bits); // sps_id
        push_ue(0, &mut bits); // log2_max_frame_num_minus4
        push_ue(0, &mut bits); // pic_order_cnt_type
        push_ue(0, &mut bits); // log2_max_pic_order_cnt_lsb_minus4
        push_ue(1, &mut bits); // max_num_ref_frames
        bits.push(0); // gaps_in_frame_num_value_allowed_flag
        push_ue(10, &mut bits); // pic_width_in_mbs_minus1 = 10 -> 11 MBs
        push_ue(8, &mut bits); // pic_height_in_map_units_minus1 = 8 -> 9 MBs
        bits.push(1); // frame_mbs_only_flag
        bits.push(0); // direct_8x8_inference_flag
        bits.push(0); // frame_cropping_flag
        bits.push(0); // vui_parameters_present_flag
        bits.push(1); // rbsp_stop_one_bit

        // Pack bits MSB-first into bytes.
        let mut bytes = Vec::new();
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            bytes.push(b);
        }

        let sps = parse_sps(&bytes).expect("hand-built SPS");
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.level_idc, 10);
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.mb_width, 11);
        assert_eq!(sps.mb_height, 9);
        assert_eq!(sps.width, 176);
        assert_eq!(sps.height, 144);
        assert!(sps.frame_mbs_only_flag);
        assert_eq!(sps.max_num_ref_frames, 1);
    }
}
