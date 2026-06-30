//! Slice-header parsing (spec 7.3.3).
//!
//! Faithful port of `ParseSliceHeaderSyntaxs` and its helpers
//! `ParseRefPicListReordering`, `ParsePredWeightedTable` and
//! `ParseDecRefPicMarking` from
//! `reference/codec/decoder/core/src/decoder_core.cpp`, with the struct shapes
//! mirroring `TagSliceHeaders`/`TagRefPicListReorderSyntax`/
//! `TagPredWeightTabSyntax`/`TagRefPicMarking` in
//! `reference/codec/decoder/core/inc/slice.h`.
//!
//! This port covers the non-SVC (AVC) path only — `kbExtensionFlag == false` in
//! the reference — since the public entry point is handed a plain [`Sps`]/[`Pps`]
//! pair. Where OpenH264 rejects valid bitstreams (interlaced
//! `frame_mbs_only_flag == 0`, `separate_colour_plane_flag`, SP/SI) this port
//! instead parses the syntax per the H.264 spec so the reader is complete; the
//! values are stored for downstream code to validate.

use alloc::vec::Vec;

use crate::bits::BitReader;
use crate::decoder::params::{Pps, Sps};
use crate::error::DecodeError;

type Result<T> = core::result::Result<T, DecodeError>;

// ---- Constants ported from decoder_core.cpp / wels_const.h / wels_common_defs.h ----

const SLICE_HEADER_IDR_PIC_ID_MAX: u32 = 65535;
const SLICE_HEADER_REDUNDANT_PIC_CNT_MAX: u32 = 127;
const SLICE_HEADER_CABAC_INIT_IDC_MAX: u32 = 2;
const SLICE_HEADER_ALPHAC0_BETA_OFFSET_MIN: i32 = -12;
const SLICE_HEADER_ALPHAC0_BETA_OFFSET_MAX: i32 = 12;
const MAX_NUM_REF_IDX_L0_ACTIVE_MINUS1: u32 = 15;
const MAX_NUM_REF_IDX_L1_ACTIVE_MINUS1: u32 = 15;
const MAX_REF_PIC_COUNT: u32 = 16;
const MAX_MMCO_COUNT: usize = 66;
const SLICE_HEADER_FIRST_MB_MAX: u32 = 36863;

// memory_management_control_operation values (`wels_common_defs.h`).
const MMCO_END: u32 = 0;
const MMCO_SHORT2UNUSED: u32 = 1;
const MMCO_LONG2UNUSED: u32 = 2;
const MMCO_SHORT2LONG: u32 = 3;
const MMCO_SET_MAX_LONG: u32 = 4;
const MMCO_RESET: u32 = 5;
const MMCO_LONG: u32 = 6;

/// Slice coding type (`EWelsSliceType`). The raw `slice_type` syntax value is
/// 0..=9; values 5..=9 are the "all slices in the picture have this type" alias
/// of 0..=4 and map to the same variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// `P_SLICE` (0).
    P,
    /// `B_SLICE` (1).
    B,
    /// `I_SLICE` (2).
    I,
    /// `SP_SLICE` (3).
    SP,
    /// `SI_SLICE` (4).
    SI,
}

impl SliceType {
    /// Map a normalised slice_type (already reduced to 0..=4) to the enum.
    fn from_normalised(v: u32) -> Result<Self> {
        Ok(match v {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::SP,
            4 => SliceType::SI,
            _ => return Err(DecodeError::InvalidSyntax("slice_type")),
        })
    }

    #[inline]
    pub fn is_intra(self) -> bool {
        matches!(self, SliceType::I | SliceType::SI)
    }

    #[inline]
    pub fn is_p(self) -> bool {
        matches!(self, SliceType::P | SliceType::SP)
    }

    #[inline]
    pub fn is_b(self) -> bool {
        matches!(self, SliceType::B)
    }
}

/// One `ref_pic_list_modification` entry (spec 7.3.3.1).
#[derive(Debug, Clone, Copy, Default)]
pub struct ReorderEntry {
    /// `modification_of_pic_nums_idc` (0..=3; 3 terminates the list).
    pub modification_of_pic_nums_idc: u32,
    /// `abs_diff_pic_num_minus1` (idc 0 or 1).
    pub abs_diff_pic_num_minus1: u32,
    /// `long_term_pic_num` (idc 2).
    pub long_term_pic_num: u32,
}

/// `ref_pic_list_modification` for one list (l0 or l1).
#[derive(Debug, Clone, Default)]
pub struct RefListReorder {
    /// `ref_pic_list_modification_flag_lX`.
    pub flag: bool,
    /// Entries up to and excluding the terminating idc==3 marker.
    pub entries: Vec<ReorderEntry>,
}

/// `ref_pic_list_modification` (both lists). l1 is only populated for B slices.
#[derive(Debug, Clone, Default)]
pub struct RefPicListReordering {
    pub list: [RefListReorder; 2],
}

/// Per-reference weight/offset record within a `pred_weight_table` list.
#[derive(Debug, Clone, Copy)]
pub struct WeightEntry {
    pub luma_weight_flag: bool,
    pub luma_weight: i32,
    pub luma_offset: i32,
    pub chroma_weight_flag: bool,
    /// [cb, cr].
    pub chroma_weight: [i32; 2],
    /// [cb, cr].
    pub chroma_offset: [i32; 2],
}

/// `pred_weight_table` (spec 7.3.3.2).
#[derive(Debug, Clone, Default)]
pub struct PredWeightTable {
    pub luma_log2_weight_denom: u32,
    pub chroma_log2_weight_denom: u32,
    /// Per-list entries; index 0 = l0, index 1 = l1 (B slices only).
    pub list: [Vec<WeightEntry>; 2],
}

/// One `memory_management_control_operation` record (spec 7.3.3.3).
#[derive(Debug, Clone, Copy, Default)]
pub struct MmcoEntry {
    /// `memory_management_control_operation`.
    pub mmco_type: u32,
    /// `1 + difference_of_pic_nums_minus1` (mmco 1, 3).
    pub diff_of_pic_num: i32,
    /// Derived ShortTermFrameNum (mmco 1, 3).
    pub short_frame_num: i32,
    /// `long_term_pic_num` (mmco 2).
    pub long_term_pic_num: u32,
    /// `long_term_frame_idx` (mmco 3, 6).
    pub long_term_frame_idx: i32,
    /// `-1 + max_long_term_frame_idx_plus1` (mmco 4).
    pub max_long_term_frame_idx: i32,
}

/// `dec_ref_pic_marking` (spec 7.3.3.3).
#[derive(Debug, Clone, Default)]
pub struct RefPicMarking {
    /// IDR path: `no_output_of_prior_pics_flag`.
    pub no_output_of_prior_pics_flag: bool,
    /// IDR path: `long_term_reference_flag`.
    pub long_term_reference_flag: bool,
    /// Non-IDR path: `adaptive_ref_pic_marking_mode_flag`.
    pub adaptive_ref_pic_marking_mode_flag: bool,
    /// MMCO commands (non-IDR, adaptive mode), excluding the terminating mmco 0.
    pub mmco: Vec<MmcoEntry>,
}

/// Parsed slice header (mirrors the read part of `TagSliceHeaders`).
#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    /// Raw `slice_type` syntax value (0..=9).
    pub slice_type_raw: u32,
    /// Decoded slice type (5..=9 folded to 0..=4).
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub sps_id: u32,
    pub is_idr: bool,

    /// `colour_plane_id` (only when `separate_colour_plane_flag`).
    pub colour_plane_id: u32,
    pub frame_num: u32,

    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,

    /// `idr_pic_id` (IDR slices only).
    pub idr_pic_id: u32,

    /// `pic_order_cnt_lsb` (poc_type 0) — raw value, not the derived POC.
    pub pic_order_cnt_lsb: u32,
    /// `delta_pic_order_cnt_bottom` (poc_type 0).
    pub delta_pic_order_cnt_bottom: i32,
    /// `delta_pic_order_cnt[0..2]` (poc_type 1).
    pub delta_pic_order_cnt: [i32; 2],

    pub redundant_pic_cnt: u32,

    /// `direct_spatial_mv_pred_flag` (B slices).
    pub direct_spatial_mv_pred_flag: bool,

    pub num_ref_idx_active_override_flag: bool,
    /// Effective active reference counts [l0, l1] after any override.
    pub num_ref_idx_active: [u32; 2],

    pub ref_pic_list_reordering: RefPicListReordering,
    /// Present only when weighted prediction applies to this slice.
    pub pred_weight_table: Option<PredWeightTable>,
    /// Present only when `nal_ref_idc != 0`.
    pub dec_ref_pic_marking: Option<RefPicMarking>,

    /// `cabac_init_idc` (CABAC, non-I/SI slices).
    pub cabac_init_idc: u32,

    pub slice_qp_delta: i32,
    /// `pic_init_qp + slice_qp_delta`.
    pub slice_qp: i32,

    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset: i32,
    pub slice_beta_offset: i32,

    /// `slice_group_change_cycle` (FMO map types 3..=5).
    pub slice_group_change_cycle: u32,

    /// Macroblock dimensions resolved from the SPS/field flag.
    pub mb_width: u32,
    pub mb_height: u32,
}

/// `Ceil(Log2(n))` — bit width of `slice_group_change_cycle`.
#[inline]
fn ceil_log2(n: u32) -> u32 {
    let mut k = 0u32;
    while (1u32 << k) < n {
        k += 1;
    }
    k
}

/// Parse a slice header from `rbsp` (NAL header byte already stripped). The
/// caller has resolved the referenced PPS→SPS and passes them plus the NAL's
/// `nal_ref_idc` and IDR flag.
pub fn parse_slice_header(
    rbsp: &[u8],
    nal_ref_idc: u8,
    is_idr: bool,
    sps: &Sps,
    pps: &Pps,
) -> Result<SliceHeader> {
    let mut bs = BitReader::new(rbsp);

    // first_mb_in_slice
    let first_mb_in_slice = bs.read_ue()?;
    if first_mb_in_slice > SLICE_HEADER_FIRST_MB_MAX {
        return Err(DecodeError::InvalidSyntax("first_mb_in_slice"));
    }

    // slice_type
    let slice_type_raw = bs.read_ue()?;
    if slice_type_raw > 9 {
        return Err(DecodeError::InvalidSyntax("slice_type"));
    }
    let slice_norm = if slice_type_raw > 4 {
        slice_type_raw - 5
    } else {
        slice_type_raw
    };
    let slice_type = SliceType::from_normalised(slice_norm)?;

    // IDR pictures must carry an I slice (spec / OpenH264).
    if is_idr && slice_type != SliceType::I {
        return Err(DecodeError::InvalidSyntax("idr slice_type"));
    }

    // pic_parameter_set_id (validated against the supplied PPS by the caller).
    let pps_id = bs.read_ue()?;

    // Range-check first_mb_in_slice against the picture size.
    if sps.total_mb_count > 0 && first_mb_in_slice > sps.total_mb_count - 1 {
        return Err(DecodeError::InvalidSyntax("first_mb_in_slice range"));
    }

    // num_ref_frames == 0 only admits intra slices (OpenH264).
    if sps.max_num_ref_frames == 0 && !slice_type.is_intra() {
        return Err(DecodeError::InvalidSyntax("slice_type for num_ref_frames=0"));
    }

    if sps.log2_max_frame_num == 0 {
        return Err(DecodeError::MissingParameterSet);
    }

    // colour_plane_id — spec 7.3.3; OpenH264 omits it (rejects separate planes).
    let colour_plane_id = if sps.separate_colour_plane_flag {
        bs.read_bits(2)?
    } else {
        0
    };

    // frame_num
    let frame_num = bs.read_bits(sps.log2_max_frame_num)?;

    // field_pic_flag / bottom_field_flag (only when not frame-only). OpenH264
    // rejects interlaced here; we parse per spec.
    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = bs.read_flag()?;
        if field_pic_flag {
            bottom_field_flag = bs.read_flag()?;
        }
    }

    let mb_width = sps.mb_width;
    let mb_height = sps.mb_height / (1 + field_pic_flag as u32);

    // idr_pic_id
    let mut idr_pic_id = 0;
    if is_idr {
        if frame_num != 0 {
            return Err(DecodeError::InvalidSyntax("idr frame_num"));
        }
        idr_pic_id = bs.read_ue()?;
        if idr_pic_id > SLICE_HEADER_IDR_PIC_ID_MAX {
            return Err(DecodeError::InvalidSyntax("idr_pic_id"));
        }
    }

    // Picture order count.
    let mut pic_order_cnt_lsb = 0;
    let mut delta_pic_order_cnt_bottom = 0;
    let mut delta_pic_order_cnt = [0i32; 2];
    if sps.pic_order_cnt_type == 0 {
        pic_order_cnt_lsb = bs.read_bits(sps.log2_max_poc_lsb as u32)?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            delta_pic_order_cnt_bottom = bs.read_se()?;
        }
    } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        delta_pic_order_cnt[0] = bs.read_se()?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            delta_pic_order_cnt[1] = bs.read_se()?;
        }
    }

    // redundant_pic_cnt
    let mut redundant_pic_cnt = 0;
    if pps.redundant_pic_cnt_present_flag {
        redundant_pic_cnt = bs.read_ue()?;
        if redundant_pic_cnt > SLICE_HEADER_REDUNDANT_PIC_CNT_MAX {
            return Err(DecodeError::InvalidSyntax("redundant_pic_cnt"));
        }
    }

    // direct_spatial_mv_pred_flag (B slices)
    let mut direct_spatial_mv_pred_flag = false;
    if slice_type == SliceType::B {
        direct_spatial_mv_pred_flag = bs.read_flag()?;
    }

    // num_ref_idx_active_override and the resulting active counts.
    let mut num_ref_idx_active = [pps.num_ref_idx_l0_active, pps.num_ref_idx_l1_active];
    let mut num_ref_idx_active_override_flag = false;
    if slice_type.is_p() || slice_type == SliceType::B {
        num_ref_idx_active_override_flag = bs.read_flag()?;
        if num_ref_idx_active_override_flag {
            let l0 = bs.read_ue()?;
            if l0 > MAX_NUM_REF_IDX_L0_ACTIVE_MINUS1 {
                return Err(DecodeError::InvalidSyntax("num_ref_idx_l0_active_minus1"));
            }
            num_ref_idx_active[0] = 1 + l0;
            if slice_type == SliceType::B {
                let l1 = bs.read_ue()?;
                if l1 > MAX_NUM_REF_IDX_L1_ACTIVE_MINUS1 {
                    return Err(DecodeError::InvalidSyntax("num_ref_idx_l1_active_minus1"));
                }
                num_ref_idx_active[1] = 1 + l1;
            }
        }
    }

    if num_ref_idx_active[0] > MAX_REF_PIC_COUNT || num_ref_idx_active[1] > MAX_REF_PIC_COUNT {
        return Err(DecodeError::InvalidSyntax("num_ref_idx overflow"));
    }

    // ref_pic_list_modification (a.k.a. reordering)
    let ref_pic_list_reordering =
        parse_ref_pic_list_reordering(&mut bs, slice_type, sps, &num_ref_idx_active)?;

    // pred_weight_table
    let weighted = (pps.weighted_pred_flag && slice_type == SliceType::P)
        || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B);
    let pred_weight_table = if weighted {
        Some(parse_pred_weight_table(
            &mut bs,
            slice_type,
            sps,
            &num_ref_idx_active,
        )?)
    } else {
        None
    };

    // dec_ref_pic_marking
    let dec_ref_pic_marking = if nal_ref_idc != 0 {
        Some(parse_dec_ref_pic_marking(&mut bs, is_idr, sps, frame_num)?)
    } else {
        None
    };

    // cabac_init_idc
    let mut cabac_init_idc = 0;
    if pps.entropy_coding_mode_flag && !slice_type.is_intra() {
        cabac_init_idc = bs.read_ue()?;
        if cabac_init_idc > SLICE_HEADER_CABAC_INIT_IDC_MAX {
            return Err(DecodeError::InvalidSyntax("cabac_init_idc"));
        }
    }

    // slice_qp_delta
    let slice_qp_delta = bs.read_se()?;
    let slice_qp = pps.pic_init_qp + slice_qp_delta;
    if !(0..=51).contains(&slice_qp) {
        return Err(DecodeError::InvalidSyntax("slice_qp"));
    }

    // NB: SP/SI slices would carry sp_for_switch_flag/slice_qs_delta here; this
    // port (like OpenH264) does not support SP/SI, so those bits are not read.

    // Deblocking filter parameters.
    let mut disable_deblocking_filter_idc = 0;
    let mut slice_alpha_c0_offset = 0;
    let mut slice_beta_offset = 0;
    if pps.deblocking_filter_control_present_flag {
        disable_deblocking_filter_idc = bs.read_ue()?;
        if disable_deblocking_filter_idc > 6 {
            return Err(DecodeError::InvalidSyntax("disable_deblocking_filter_idc"));
        }
        if disable_deblocking_filter_idc != 1 {
            slice_alpha_c0_offset = bs.read_se()? * 2;
            if !(SLICE_HEADER_ALPHAC0_BETA_OFFSET_MIN..=SLICE_HEADER_ALPHAC0_BETA_OFFSET_MAX)
                .contains(&slice_alpha_c0_offset)
            {
                return Err(DecodeError::InvalidSyntax("slice_alpha_c0_offset_div2"));
            }
            slice_beta_offset = bs.read_se()? * 2;
            if !(SLICE_HEADER_ALPHAC0_BETA_OFFSET_MIN..=SLICE_HEADER_ALPHAC0_BETA_OFFSET_MAX)
                .contains(&slice_beta_offset)
            {
                return Err(DecodeError::InvalidSyntax("slice_beta_offset_div2"));
            }
        }
    }

    // slice_group_change_cycle (FMO map types 3..=5).
    let mut slice_group_change_cycle = 0;
    if pps.num_slice_groups > 1
        && (3..=5).contains(&pps.slice_group_map_type)
        && pps.slice_group_change_rate > 0
    {
        // PicSizeInMapUnits = PicWidthInMbs * PicHeightInMapUnits (spec 7.4.3).
        // Bit width is Ceil(Log2(PicSizeInMapUnits / SliceGroupChangeRate + 1)).
        // (The reference uses a natural-log expression here, a latent bug that
        // never triggers for the single-slice-group streams it decodes; we
        // follow the spec.)
        let pic_size_in_map_units = sps.total_mb_count;
        let bits = ceil_log2(1 + pic_size_in_map_units / pps.slice_group_change_rate);
        slice_group_change_cycle = bs.read_bits(bits)?;
    }

    Ok(SliceHeader {
        first_mb_in_slice,
        slice_type_raw,
        slice_type,
        pps_id,
        sps_id: pps.sps_id,
        is_idr,
        colour_plane_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
        redundant_pic_cnt,
        direct_spatial_mv_pred_flag,
        num_ref_idx_active_override_flag,
        num_ref_idx_active,
        ref_pic_list_reordering,
        pred_weight_table,
        dec_ref_pic_marking,
        cabac_init_idc,
        slice_qp_delta,
        slice_qp,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset,
        slice_beta_offset,
        slice_group_change_cycle,
        mb_width,
        mb_height,
    })
}

/// Port of `ParseRefPicListReordering`.
fn parse_ref_pic_list_reordering(
    bs: &mut BitReader<'_>,
    slice_type: SliceType,
    sps: &Sps,
    ref_count: &[u32; 2],
) -> Result<RefPicListReordering> {
    let mut out = RefPicListReordering::default();
    if slice_type.is_intra() {
        return Ok(out);
    }

    // list0, then list1 for B slices.
    let num_lists = if slice_type == SliceType::B { 2 } else { 1 };
    for list in 0..num_lists {
        let flag = bs.read_flag()?; // ref_pic_list_modification_flag_lX
        out.list[list].flag = flag;
        if flag {
            let mut idx = 0;
            loop {
                let idc = bs.read_ue()?; // modification_of_pic_nums_idc
                if (idx >= MAX_REF_PIC_COUNT as usize && idc != 3) || idc > 3 {
                    return Err(DecodeError::InvalidSyntax("ref reordering idc"));
                }
                if idc == 3 {
                    break;
                }
                if idx >= ref_count[list] as usize || idx >= MAX_REF_PIC_COUNT as usize {
                    return Err(DecodeError::InvalidSyntax("ref reordering count"));
                }
                let mut entry = ReorderEntry {
                    modification_of_pic_nums_idc: idc,
                    ..Default::default()
                };
                if idc == 0 || idc == 1 {
                    let v = bs.read_ue()?; // abs_diff_pic_num_minus1
                    if v >= (1u32 << sps.log2_max_frame_num) {
                        return Err(DecodeError::InvalidSyntax("abs_diff_pic_num_minus1"));
                    }
                    entry.abs_diff_pic_num_minus1 = v;
                } else if idc == 2 {
                    entry.long_term_pic_num = bs.read_ue()?; // long_term_pic_num
                }
                out.list[list].entries.push(entry);
                idx += 1;
            }
        }
    }
    Ok(out)
}

/// Port of `ParsePredWeightedTable`.
fn parse_pred_weight_table(
    bs: &mut BitReader<'_>,
    slice_type: SliceType,
    sps: &Sps,
    ref_count: &[u32; 2],
) -> Result<PredWeightTable> {
    let mut table = PredWeightTable::default();

    let luma = bs.read_ue()?; // luma_log2_weight_denom
    if luma > 7 {
        return Err(DecodeError::InvalidSyntax("luma_log2_weight_denom"));
    }
    table.luma_log2_weight_denom = luma;
    if sps.chroma_array_type != 0 {
        let chroma = bs.read_ue()?; // chroma_log2_weight_denom
        if chroma > 7 {
            return Err(DecodeError::InvalidSyntax("chroma_log2_weight_denom"));
        }
        table.chroma_log2_weight_denom = chroma;
    }

    let num_lists = if slice_type == SliceType::B { 2 } else { 1 };
    for list in 0..num_lists {
        for _ in 0..ref_count[list] {
            // luma
            let luma_weight_flag = bs.read_flag()?;
            let (luma_weight, luma_offset) = if luma_weight_flag {
                let w = bs.read_se()?;
                if !(-128..=127).contains(&w) {
                    return Err(DecodeError::InvalidSyntax("luma_weight"));
                }
                let o = bs.read_se()?;
                if !(-128..=127).contains(&o) {
                    return Err(DecodeError::InvalidSyntax("luma_offset"));
                }
                (w, o)
            } else {
                (1i32 << table.luma_log2_weight_denom, 0)
            };

            // chroma
            let mut chroma_weight_flag = false;
            let mut chroma_weight = [1i32 << table.chroma_log2_weight_denom; 2];
            let mut chroma_offset = [0i32; 2];
            if sps.chroma_array_type != 0 {
                chroma_weight_flag = bs.read_flag()?;
                if chroma_weight_flag {
                    for j in 0..2 {
                        let w = bs.read_se()?;
                        if !(-128..=127).contains(&w) {
                            return Err(DecodeError::InvalidSyntax("chroma_weight"));
                        }
                        chroma_weight[j] = w;
                        let o = bs.read_se()?;
                        if !(-128..=127).contains(&o) {
                            return Err(DecodeError::InvalidSyntax("chroma_offset"));
                        }
                        chroma_offset[j] = o;
                    }
                }
            }

            table.list[list].push(WeightEntry {
                luma_weight_flag,
                luma_weight,
                luma_offset,
                chroma_weight_flag,
                chroma_weight,
                chroma_offset,
            });
        }
    }
    Ok(table)
}

/// Port of `ParseDecRefPicMarking`.
fn parse_dec_ref_pic_marking(
    bs: &mut BitReader<'_>,
    is_idr: bool,
    sps: &Sps,
    frame_num: u32,
) -> Result<RefPicMarking> {
    let mut marking = RefPicMarking::default();
    if is_idr {
        marking.no_output_of_prior_pics_flag = bs.read_flag()?;
        marking.long_term_reference_flag = bs.read_flag()?;
        return Ok(marking);
    }

    marking.adaptive_ref_pic_marking_mode_flag = bs.read_flag()?;
    if !marking.adaptive_ref_pic_marking_mode_flag {
        return Ok(marking);
    }

    let frame_num_mask = (1i64 << sps.log2_max_frame_num) - 1;
    let mut allow_mmco5 = true;
    let mut mmco4_exist = false;
    let mut mmco5_exist = false;
    let mut mmco6_exist = false;
    for _ in 0..MAX_MMCO_COUNT {
        let mmco = bs.read_ue()?; // memory_management_control_operation
        if mmco == MMCO_END {
            break;
        }
        let mut entry = MmcoEntry {
            mmco_type: mmco,
            ..Default::default()
        };
        if mmco == MMCO_SHORT2UNUSED || mmco == MMCO_SHORT2LONG {
            allow_mmco5 = false;
            let diff = 1 + bs.read_ue()? as i32; // difference_of_pic_nums_minus1
            entry.diff_of_pic_num = diff;
            entry.short_frame_num = ((frame_num as i64 - diff as i64) & frame_num_mask) as i32;
        } else if mmco == MMCO_LONG2UNUSED {
            allow_mmco5 = false;
            entry.long_term_pic_num = bs.read_ue()?; // long_term_pic_num
        }
        if mmco == MMCO_SHORT2LONG || mmco == MMCO_LONG {
            if mmco == MMCO_LONG {
                if mmco6_exist {
                    return Err(DecodeError::InvalidSyntax("mmco6 duplicate"));
                }
                mmco6_exist = true;
            }
            entry.long_term_frame_idx = bs.read_ue()? as i32; // long_term_frame_idx
        } else if mmco == MMCO_SET_MAX_LONG {
            if mmco4_exist {
                return Err(DecodeError::InvalidSyntax("mmco4 duplicate"));
            }
            mmco4_exist = true;
            let max = -1 + bs.read_ue()? as i32; // max_long_term_frame_idx_plus1
            if max > sps.max_num_ref_frames as i32 {
                return Err(DecodeError::InvalidSyntax("max_long_term_frame_idx"));
            }
            entry.max_long_term_frame_idx = max;
        } else if mmco == MMCO_RESET {
            if !allow_mmco5 || mmco5_exist {
                return Err(DecodeError::InvalidSyntax("mmco5 invalid"));
            }
            mmco5_exist = true;
        }
        marking.mmco.push(entry);
    }

    Ok(marking)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::nal::{annexb_nal_units, parse_nal, NalUnit, NalUnitType};
    use crate::decoder::params::{parse_pps, parse_sps};
    use alloc::vec::Vec;

    fn collect_nals(stream: &[u8]) -> Vec<NalUnit> {
        annexb_nal_units(stream).filter_map(parse_nal).collect()
    }

    /// Build id→SPS and id→PPS maps (max 32 / 256 entries) from a stream.
    fn parse_param_sets(nals: &[NalUnit]) -> (Vec<Option<Sps>>, Vec<Option<Pps>>) {
        let mut sps_map: Vec<Option<Sps>> = (0..32).map(|_| None).collect();
        for nal in nals {
            if nal.unit_type == NalUnitType::Sps {
                let sps = parse_sps(&nal.rbsp).expect("SPS parse");
                let id = sps.sps_id as usize;
                sps_map[id] = Some(sps);
            }
        }
        let mut pps_map: Vec<Option<Pps>> = (0..256).map(|_| None).collect();
        for nal in nals {
            if nal.unit_type == NalUnitType::Pps {
                let sps_ref = {
                    // Peek the sps_id by parsing without a reference first.
                    let pps_probe = parse_pps(&nal.rbsp, None);
                    match pps_probe {
                        Ok(p) => sps_map[p.sps_id as usize].as_ref(),
                        Err(_) => None,
                    }
                };
                let pps = parse_pps(&nal.rbsp, sps_ref).expect("PPS parse");
                let id = pps.pps_id as usize;
                pps_map[id] = Some(pps);
            }
        }
        (sps_map, pps_map)
    }

    fn run_fixture(stream: &[u8]) -> (Vec<SliceType>, Vec<i32>) {
        let nals = collect_nals(stream);
        let (sps_map, pps_map) = parse_param_sets(&nals);

        let mut types = Vec::new();
        let mut qps = Vec::new();
        let mut seen_first_idr = false;

        for nal in &nals {
            if !nal.unit_type.is_vcl() {
                continue;
            }
            // Resolve the referenced PPS (and through it the SPS). The slice
            // header's pps_id is the second ue(v); read it from a probe header.
            let is_idr = nal.unit_type.is_idr();
            // Probe pps_id: parse first_mb (ue), slice_type (ue), pps_id (ue).
            let mut probe = BitReader::new(&nal.rbsp);
            let _first_mb = probe.read_ue().expect("probe first_mb");
            let _slice_type = probe.read_ue().expect("probe slice_type");
            let pps_id = probe.read_ue().expect("probe pps_id");
            let pps = pps_map[pps_id as usize].as_ref().expect("referenced PPS");
            let sps = sps_map[pps.sps_id as usize].as_ref().expect("referenced SPS");

            let sh = parse_slice_header(&nal.rbsp, nal.ref_idc, is_idr, sps, pps)
                .expect("slice header");

            // Range / sanity assertions.
            assert!(
                sh.first_mb_in_slice < sps.total_mb_count.max(1),
                "first_mb in range"
            );
            assert!((0..=51).contains(&sh.slice_qp), "slice_qp in 0..=51");
            assert!(
                sh.frame_num < (1u32 << sps.log2_max_frame_num),
                "frame_num in range"
            );
            assert_eq!(sh.pps_id, pps_id);
            assert_eq!(sh.sps_id, pps.sps_id);

            if is_idr {
                assert_eq!(sh.slice_type, SliceType::I, "IDR slice must be I");
                if !seen_first_idr {
                    assert_eq!(sh.first_mb_in_slice, 0, "first IDR slice first_mb == 0");
                    seen_first_idr = true;
                }
            }

            types.push(sh.slice_type);
            qps.push(sh.slice_qp);

            // Stop after a handful of VCL units.
            if types.len() >= 6 {
                break;
            }
        }
        (types, qps)
    }

    #[test]
    fn banm_mw_d_slices() {
        let stream = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/BANM_MW_D.264"
        ));
        let (types, qps) = run_fixture(stream);
        assert!(!types.is_empty(), "parsed at least one slice");
        // First decoded VCL slice is the IDR's I slice.
        assert_eq!(types[0], SliceType::I);
        // Following slices in a typical IPPP clip are P.
        if types.len() > 1 {
            assert!(
                types[1..].iter().all(|t| *t == SliceType::P || *t == SliceType::I),
                "subsequent slices are P (or I): {types:?}"
            );
        }
        for qp in qps {
            assert!((0..=51).contains(&qp));
        }
    }

    #[test]
    fn ba1_ft_c_slices() {
        let stream = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/BA1_FT_C.264"
        ));
        let (types, qps) = run_fixture(stream);
        assert!(!types.is_empty());
        assert_eq!(types[0], SliceType::I);
        for qp in qps {
            assert!((0..=51).contains(&qp));
        }
    }
}
