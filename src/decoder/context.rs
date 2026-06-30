//! Per-frame decoder state: parameter sets, the current [`Picture`], and the
//! per-macroblock arrays needed for neighbour-based prediction (nC derivation,
//! intra-mode prediction, deblocking later).
//!
//! This collapses the role of `PWelsDecoderContext` + `PDqLayer` from
//! `reference/codec/decoder/core/inc/decoder_context.h` down to what the
//! baseline intra (CAVLC, 4:2:0) path actually reads. State is kept in flat
//! per-MB `Vec`s sized to `total_mb`, indexed by raster MB address
//! `mb_xy = mb_y * mb_width + mb_x`.

use alloc::vec;
use alloc::vec::Vec;

use super::params::{Pps, Sps};
use super::picture::Picture;

/// Macroblock prediction class. Covers the baseline intra path plus the P-slice
/// inter partition kinds (mirrors the relevant `MB_TYPE_*` from
/// `wels_common_defs.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbType {
    /// `MB_TYPE_INTRA4x4`.
    Intra4x4,
    /// `MB_TYPE_INTRA16x16`.
    Intra16x16,
    /// `MB_TYPE_16x16` (P_L0_16x16).
    Inter16x16,
    /// `MB_TYPE_16x8` (P_L0_L0_16x8).
    Inter16x8,
    /// `MB_TYPE_8x16` (P_L0_L0_8x16).
    Inter8x16,
    /// `MB_TYPE_8x8` (P_8x8).
    Inter8x8,
    /// `MB_TYPE_8x8_REF0` (P_8x8ref0).
    Inter8x8Ref0,
    /// `MB_TYPE_SKIP` (P_Skip).
    PSkip,
}

impl MbType {
    /// True for `IS_INTRANxN` (only I_4x4 here; I_8x8 is High-profile, deferred).
    #[inline]
    pub fn is_intra_nxn(self) -> bool {
        matches!(self, MbType::Intra4x4)
    }

    /// `IS_INTRA`.
    #[inline]
    pub fn is_intra(self) -> bool {
        matches!(self, MbType::Intra4x4 | MbType::Intra16x16)
    }

    /// `IS_INTER` (skip counts as inter).
    #[inline]
    pub fn is_inter(self) -> bool {
        !self.is_intra()
    }

    /// `IS_SKIP`.
    #[inline]
    pub fn is_skip(self) -> bool {
        matches!(self, MbType::PSkip)
    }

    /// `IS_INTER_16x16` (16x16 or skip).
    #[inline]
    pub fn is_inter_16x16(self) -> bool {
        matches!(self, MbType::Inter16x16 | MbType::PSkip)
    }
}

/// Sub-macroblock partition kind for a P_8x8 macroblock partition
/// (`g_ksInterPSubMbTypeInfo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubMbType {
    /// `SUB_MB_TYPE_8x8` (1 part, width 2).
    P8x8,
    /// `SUB_MB_TYPE_8x4` (2 parts, width 2).
    P8x4,
    /// `SUB_MB_TYPE_4x8` (2 parts, width 1).
    P4x8,
    /// `SUB_MB_TYPE_4x4` (4 parts, width 1).
    P4x4,
}

impl SubMbType {
    /// (partition count, partition width in 4x4 units).
    #[inline]
    pub fn part_info(self) -> (usize, usize) {
        match self {
            SubMbType::P8x8 => (1, 2),
            SubMbType::P8x4 => (2, 2),
            SubMbType::P4x8 => (2, 1),
            SubMbType::P4x4 => (4, 1),
        }
    }
}

/// Availability of the four neighbouring macroblocks within the current slice.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeighborAvail {
    pub left: bool,
    pub top: bool,
    pub top_left: bool,
    pub top_right: bool,
    /// `mb_xy` of the left / top neighbours (only valid when the flag is set).
    pub left_xy: usize,
    pub top_xy: usize,
}

/// Frame-level decoder state for the baseline intra path.
pub struct DecoderContext {
    /// SPS by id (`sps_id` indexes; `None` if not yet seen).
    pub sps: Vec<Option<Sps>>,
    /// PPS by id.
    pub pps: Vec<Option<Pps>>,

    pub mb_width: usize,
    pub mb_height: usize,
    pub total_mb: usize,

    /// Slice id per MB (`-1` until the MB is assigned). Two MBs are mutual
    /// neighbours only when their slice ids match (spec availability).
    pub slice_idc: Vec<i32>,

    pub mb_type: Vec<MbType>,
    /// Adjusted intra16x16 luma prediction mode (`I16_PRED_*`, 0..=6).
    pub i16_mode: Vec<i8>,
    /// Adjusted chroma prediction mode (`C_PRED_*`, 0..=6).
    pub chroma_mode: Vec<i8>,
    /// Coded block pattern (`cbp_luma | cbp_chroma << 4`).
    pub cbp: Vec<u8>,
    /// Per-MB CABAC `coded_block_flag` DC state (`pCbfDc`): bit `iResProperty`
    /// set when that DC block had a non-zero `coded_block_flag`. Bits used:
    /// 1 = luma DC, 7 = Cb DC, 8 = Cr DC. Read by the CABAC cbf neighbour
    /// derivation; unused by CAVLC.
    pub cbf_dc: Vec<u16>,
    pub luma_qp: Vec<i8>,
    /// Per-MB chroma QP, 2 per MB (Cb then Cr), derived from luma QP via the
    /// chroma QP mapping + `chroma_qp_index_offset`. Used by the deblocker.
    pub chroma_qp: Vec<i8>,

    /// Per-MB `disable_deblocking_filter_idc` from the owning slice header
    /// (0 = filter all, 1 = off, 2 = off across slice boundaries).
    pub deblock_idc: Vec<u8>,
    /// Per-MB `slice_alpha_c0_offset` (already doubled, spec range -12..=12).
    pub deblock_alpha_off: Vec<i8>,
    /// Per-MB `slice_beta_offset`.
    pub deblock_beta_off: Vec<i8>,

    /// Per-block `iBestMode` (the pre-adjustment intra4x4 mode used by the
    /// neighbour-mode predictor), 16 entries per MB in raster order.
    pub i4_best_mode: Vec<i8>,
    /// Per-block final intra4x4 mode (`I4_PRED_*` incl. DC_L/DDL_TOP variants).
    pub i4_final_mode: Vec<i8>,
    /// Per-block luma non-zero-coefficient count, 16/MB raster order.
    pub nzc_luma: Vec<i8>,
    /// Per-block chroma non-zero-coefficient count, 8/MB: Cb 0..3 then Cr 0..3,
    /// each component in raster order.
    pub nzc_chroma: Vec<i8>,

    /// Per-4x4-block list-0 motion vectors, 16 per MB in raster (`g_kuiScan4`)
    /// order, `[x, y]` in quarter-pel. Stored flat: `mv[(mb*16 + raster)*2 + c]`.
    pub mv: Vec<i16>,
    /// Per-4x4-block list-0 reference index (slice-local, 0-based), 16 per MB in
    /// raster order; `REF_NOT_IN_LIST` (-1) for intra blocks.
    pub ref_idx: Vec<i8>,
    /// Per-4x4-block resolved reference-picture identity used by the inter
    /// deblock bS rule, 16 per MB raster order; `-1` for intra / not-in-list.
    pub ref_pic_id: Vec<i32>,
    /// Per-MB sub-mb partition kinds (P_8x8 only), 4 per MB.
    pub sub_mb_type: Vec<SubMbType>,

    pub picture: Picture,
}

impl DecoderContext {
    /// Allocate all per-MB state and the picture for an `mb_width`x`mb_height`
    /// frame. `sps`/`pps` carry the already-parsed parameter sets.
    pub fn new(
        sps: Vec<Option<Sps>>,
        pps: Vec<Option<Pps>>,
        mb_width: usize,
        mb_height: usize,
    ) -> Self {
        let total_mb = mb_width * mb_height;
        DecoderContext {
            sps,
            pps,
            mb_width,
            mb_height,
            total_mb,
            slice_idc: vec![-1; total_mb],
            mb_type: vec![MbType::Intra4x4; total_mb],
            i16_mode: vec![0; total_mb],
            chroma_mode: vec![0; total_mb],
            cbp: vec![0; total_mb],
            cbf_dc: vec![0; total_mb],
            luma_qp: vec![0; total_mb],
            chroma_qp: vec![0; total_mb * 2],
            deblock_idc: vec![0; total_mb],
            deblock_alpha_off: vec![0; total_mb],
            deblock_beta_off: vec![0; total_mb],
            i4_best_mode: vec![-1; total_mb * 16],
            i4_final_mode: vec![2; total_mb * 16],
            nzc_luma: vec![0; total_mb * 16],
            nzc_chroma: vec![0; total_mb * 8],
            mv: vec![0; total_mb * 16 * 2],
            ref_idx: vec![-1; total_mb * 16],
            ref_pic_id: vec![-1; total_mb * 16],
            sub_mb_type: vec![SubMbType::P8x8; total_mb * 4],
            picture: Picture::new(mb_width, mb_height),
        }
    }

    /// Neighbour availability for MB `mb_xy`, gated on matching slice id
    /// (`GetNeighborAvailMbType`).
    pub fn neighbors(&self, mb_xy: usize) -> NeighborAvail {
        let mb_x = mb_xy % self.mb_width;
        let mb_y = mb_xy / self.mb_width;
        let cur = self.slice_idc[mb_xy];
        let mut n = NeighborAvail::default();

        if mb_x != 0 {
            let xy = mb_xy - 1;
            if self.slice_idc[xy] == cur {
                n.left = true;
                n.left_xy = xy;
            }
        }
        if mb_y != 0 {
            let xy = mb_xy - self.mb_width;
            if self.slice_idc[xy] == cur {
                n.top = true;
                n.top_xy = xy;
            }
            if mb_x != 0 {
                let xy = mb_xy - self.mb_width - 1;
                if self.slice_idc[xy] == cur {
                    n.top_left = true;
                }
            }
            if mb_x != self.mb_width - 1 {
                let xy = mb_xy - self.mb_width + 1;
                if self.slice_idc[xy] == cur {
                    n.top_right = true;
                }
            }
        }
        n
    }

    /// Slice of this MB's 16 luma non-zero counts (raster order).
    #[inline]
    pub fn nzc_luma_mb(&self, mb_xy: usize) -> &[i8] {
        &self.nzc_luma[mb_xy * 16..mb_xy * 16 + 16]
    }

    /// Slice of this MB's 8 chroma non-zero counts.
    #[inline]
    pub fn nzc_chroma_mb(&self, mb_xy: usize) -> &[i8] {
        &self.nzc_chroma[mb_xy * 8..mb_xy * 8 + 8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_availability_within_slice() {
        let mut ctx = DecoderContext::new(Vec::new(), Vec::new(), 11, 9);
        // Mark the whole frame as one slice.
        for s in ctx.slice_idc.iter_mut() {
            *s = 0;
        }

        // MB(0,0): no neighbours.
        let n = ctx.neighbors(0);
        assert!(!n.left && !n.top && !n.top_left && !n.top_right);

        // MB(5,5) = xy 5*11+5 = 60: all four neighbours present.
        let xy = 5 * 11 + 5;
        let n = ctx.neighbors(xy);
        assert!(n.left && n.top && n.top_left && n.top_right);
        assert_eq!(n.left_xy, xy - 1);
        assert_eq!(n.top_xy, xy - 11);

        // Right-column MB has no top-right.
        let xy = 1 * 11 + 10;
        let n = ctx.neighbors(xy);
        assert!(n.left && n.top && n.top_left && !n.top_right);
    }

    #[test]
    fn slice_boundary_breaks_availability() {
        let mut ctx = DecoderContext::new(Vec::new(), Vec::new(), 11, 9);
        // MB 0 in slice 0, MB 1 in slice 1: not neighbours.
        ctx.slice_idc[0] = 0;
        ctx.slice_idc[1] = 1;
        let n = ctx.neighbors(1);
        assert!(!n.left);
    }
}
