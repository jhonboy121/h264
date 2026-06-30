//! B-slice direct (spatial and temporal) motion prediction.
//!
//! Faithful port of `GetColocatedMb`, `PredMvBDirectSpatial`,
//! `PredBDirectTemporal`, `FillSpatialDirect8x8Mv` and `FillTemporalDirect8x8Mv`
//! from `reference/codec/decoder/core/src/mv_pred.cpp`. Entropy-independent:
//! shared by the CAVLC and CABAC B-slice parse paths.

use super::context::DecoderContext;
use super::dpb::ColMotion;
use super::mv_pred::{median, REF_NOT_AVAIL, REF_NOT_IN_LIST, SCAN4};

/// `WELS_MIN_POSITIVE`: prefer the non-negative value; if both are non-negative
/// take the minimum.
#[inline]
fn min_positive(a: i8, b: i8) -> i8 {
    if a < 0 {
        b
    } else if b < 0 {
        a
    } else {
        a.min(b)
    }
}

/// raster index of the colocated corner block used under
/// `direct_8x8_inference_flag` (each 8x8 maps to its outer corner).
const COL_CORNER: [usize; 16] = [0, 0, 3, 3, 0, 0, 3, 3, 12, 12, 15, 15, 12, 12, 15, 15];

/// Colocated reference picture (`list1[0]`) info for B direct prediction.
pub struct ColRef<'a> {
    pub col: &'a ColMotion,
    pub is_long: bool,
    pub inference: bool,
    /// Per-ref list-0 MV scale factors for temporal direct (`iMvScale`).
    pub mv_scale: &'a [i32],
    pub ref0_count: usize,
}

impl ColRef<'_> {
    /// Colocated `(intra, ref0, ref1, mv0, mv1)` for current-MB 4x4 raster block
    /// `r`, honouring `direct_8x8_inference_flag` corner replication.
    fn at(&self, mb: usize, r: usize) -> (bool, i8, i8, [i16; 2], [i16; 2]) {
        let intra = self.col.intra.get(mb).copied().unwrap_or(true);
        let cr = if self.inference { COL_CORNER[r] } else { r };
        let base = mb * 16 + cr;
        let r0 = self.col.ref_idx[0].get(base).copied().unwrap_or(-1);
        let mv0 = [
            self.col.mv[0].get(base * 2).copied().unwrap_or(0),
            self.col.mv[0].get(base * 2 + 1).copied().unwrap_or(0),
        ];
        let uses_l1 = self.col.uses_l1.get(mb).copied().unwrap_or(false);
        let (r1, mv1) = if uses_l1 {
            (
                self.col.ref_idx[1].get(base).copied().unwrap_or(-1),
                [
                    self.col.mv[1].get(base * 2).copied().unwrap_or(0),
                    self.col.mv[1].get(base * 2 + 1).copied().unwrap_or(0),
                ],
            )
        } else {
            (REF_NOT_IN_LIST, [0, 0])
        };
        (intra, r0, r1, mv0, mv1)
    }
}

/// Result of the MB-level direct prediction (before per-partition fill).
pub struct DirectInfo {
    /// `[list0, list1]` predicted MV (spatial) — base MV for every block.
    pub mvp: [[i16; 2]; 2],
    /// `[list0, list1]` reference index (negative = list unused).
    pub iref: [i8; 2],
    /// The direct MB resolved to a single 16x16 partition.
    pub mb16x16: bool,
    /// Sub-partition is 4x4 (only when colocated is Inter_8x8 without inference).
    pub sub_4x4: bool,
}

#[inline]
fn nb(ctx: &DecoderContext, list: usize, xy: usize, blk: usize) -> ([i16; 2], i8) {
    let (mv, rf) = if list == 0 {
        (
            [ctx.mv[(xy * 16 + blk) * 2], ctx.mv[(xy * 16 + blk) * 2 + 1]],
            ctx.ref_idx[xy * 16 + blk],
        )
    } else {
        (
            [ctx.mv_l1[(xy * 16 + blk) * 2], ctx.mv_l1[(xy * 16 + blk) * 2 + 1]],
            ctx.ref_idx_l1[xy * 16 + blk],
        )
    };
    (mv, rf)
}

/// Temporal direct prediction (`PredBDirectTemporal` / `FillTemporalDirect8x8Mv`,
/// spec 8.4.1.2.3). Scales the colocated list-0 MV by the POC-distance ratio in
/// `cr.mv_scale` and derives the backward MV as `mvL0 - mvCol`. Fills `ctx`.
///
/// NOTE: `cr.mv_scale[0]` is used for the reference index (this is exact for the
/// single-reference case `num_ref_idx_l0_active == 1`); the general
/// `MapColToList0` mapping requires the colocated picture's own reference list,
/// which the DPB does not retain. No corpus stream exercises this path (the
/// temporal-direct streams require transform_8x8, out of scope), so it is
/// unvalidated.
pub fn b_direct_temporal(ctx: &mut DecoderContext, mb_xy: usize, cur_is_8x8: bool, cr: &ColRef) {
    let scale0 = cr.mv_scale.first().copied().unwrap_or(256);
    let mb16x16 = !cur_is_8x8;
    let scale_mv = |mv: [i16; 2]| -> [i16; 2] {
        [
            ((scale0 * mv[0] as i32 + 128) >> 8) as i16,
            ((scale0 * mv[1] as i32 + 128) >> 8) as i16,
        ]
    };
    let parts: &[usize] = if mb16x16 { &[0] } else { &[0, 4, 8, 12] };
    for &part_idx in parts {
        let scan4 = SCAN4[part_idx];
        let blocks: &[usize] = if mb16x16 {
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        } else {
            &[scan4, scan4 + 1, scan4 + 4, scan4 + 5]
        };
        let (intra, cref0, _cref1, cmv0, cmv1) = cr.at(mb_xy, scan4);
        let (mv0, mv1) = if intra {
            ([0, 0], [0, 0])
        } else {
            let mvcol = if cref0 >= 0 { cmv0 } else { cmv1 };
            let l0 = scale_mv(mvcol);
            ([l0[0], l0[1]], [l0[0] - mvcol[0], l0[1] - mvcol[1]])
        };
        for &r in blocks {
            let b = (mb_xy * 16 + r) * 2;
            ctx.mv[b] = mv0[0];
            ctx.mv[b + 1] = mv0[1];
            ctx.mv_l1[b] = mv1[0];
            ctx.mv_l1[b + 1] = mv1[1];
            ctx.ref_idx[mb_xy * 16 + r] = 0;
            ctx.ref_idx_l1[mb_xy * 16 + r] = 0;
            ctx.direct[mb_xy * 16 + r] = 1;
        }
        if !mb16x16 {
            ctx.sub_mb_type[mb_xy * 4 + (part_idx >> 2)] = super::context::SubMbType::P8x8;
        }
    }
}

/// `PredMvBDirectSpatial` neighbour derivation: compute the per-list reference
/// index and predicted MV from the spatial neighbours (left/top/top-right/
/// top-left). Does not fill `ctx`.
pub fn b_direct_spatial(ctx: &DecoderContext, mb_xy: usize, cur_is_8x8: bool) -> DirectInfo {
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;
    let cur_slice = ctx.slice_idc[mb_xy];
    let avail = |xy: usize| ctx.slice_idc[xy] == cur_slice;

    let left = mb_x != 0 && avail(mb_xy - 1);
    let top = mb_y != 0 && avail(mb_xy - mb_width);
    let left_top = mb_x != 0 && mb_y != 0 && avail(mb_xy - mb_width - 1);
    let right_top = mb_x != mb_width - 1 && mb_y != 0 && avail(mb_xy - mb_width + 1);

    let left_xy = mb_xy.wrapping_sub(1);
    let top_xy = mb_xy.wrapping_sub(mb_width);
    let left_top_xy = mb_xy.wrapping_sub(mb_width + 1);
    let right_top_xy = (mb_xy + 1).wrapping_sub(mb_width);

    let left_inter = left && ctx.mb_type[left_xy].is_inter();
    let top_inter = top && ctx.mb_type[top_xy].is_inter();
    let lt_inter = left_top && ctx.mb_type[left_top_xy].is_inter();
    let rt_inter = right_top && ctx.mb_type[right_top_xy].is_inter();

    let mut mvp = [[0i16; 2]; 2];
    let mut iref = [REF_NOT_IN_LIST; 2];

    for list in 0..2 {
        let (mv_a, ref_a) = if left_inter {
            nb(ctx, list, left_xy, 3)
        } else {
            ([0, 0], if left { REF_NOT_IN_LIST } else { REF_NOT_AVAIL })
        };
        let (mv_b, ref_b) = if top_inter {
            nb(ctx, list, top_xy, 12)
        } else {
            ([0, 0], if top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL })
        };
        let (mv_c0, ref_c0) = if rt_inter {
            nb(ctx, list, right_top_xy, 12)
        } else {
            ([0, 0], if right_top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL })
        };
        let (mv_d, ref_d) = if lt_inter {
            nb(ctx, list, left_top_xy, 15)
        } else {
            ([0, 0], if left_top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL })
        };

        let mut diag = ref_c0;
        let mut mv_c = mv_c0;
        if diag == REF_NOT_AVAIL {
            diag = ref_d;
            mv_c = mv_d;
        }

        let ref_temp = min_positive(ref_b, diag);
        let r = min_positive(ref_a, ref_temp);
        if r >= 0 {
            let match_count =
                (ref_a == r) as i32 + (ref_b == r) as i32 + (diag == r) as i32;
            if match_count == 1 {
                mvp[list] = if ref_a == r {
                    mv_a
                } else if ref_b == r {
                    mv_b
                } else {
                    mv_c
                };
            } else {
                mvp[list] = [
                    median(mv_a[0] as i32, mv_b[0] as i32, mv_c[0] as i32) as i16,
                    median(mv_a[1] as i32, mv_b[1] as i32, mv_c[1] as i32) as i16,
                ];
            }
            iref[list] = r;
        } else {
            mvp[list] = [0, 0];
            iref[list] = REF_NOT_IN_LIST;
        }
    }

    // Cross-list reconciliation (spec 8.4.1.2.2 / lines 556-564).
    if iref[0] <= REF_NOT_IN_LIST && iref[1] <= REF_NOT_IN_LIST {
        iref[0] = 0;
        iref[1] = 0;
    }
    // (When only one list is unused, the resolved MbType drops that list; the
    // negative ref index propagates through fill so the block uses one list.)

    // For an intra-or-16x16 colocated MB the direct MB resolves to 16x16; a B_8x8
    // current MB keeps 8x8 sub-partitions.
    DirectInfo {
        mvp,
        iref,
        mb16x16: !cur_is_8x8,
        sub_4x4: false,
    }
}

/// Apply the spatial-direct `colZeroFlag` test for one colocated 4x4 block and,
/// when triggered, zero that list's MV (spec 8.4.1.2.2 / `FillSpatialDirect8x8Mv`
/// lines 1010-1071). Returns the (possibly zeroed) `[list0, list1]` MV.
fn col_zero_mv(info: &DirectInfo, cr: &ColRef, mb: usize, r: usize) -> [[i16; 2]; 2] {
    let mut mv = info.mvp;
    let nonzero = (mv[0] != [0, 0]) || (mv[1] != [0, 0]);
    if !nonzero {
        return mv;
    }
    let (intra, cref0, cref1, cmv0, cmv1) = cr.at(mb, r);
    let coloc_zero = !intra
        && !cr.is_long
        && (cref0 == 0 || (cref0 < 0 && cref1 == 0));
    if !coloc_zero {
        return mv;
    }
    let cmv = if cref0 == 0 { cmv0 } else { cmv1 };
    let small = (cmv[0] as i32 + 1) as u32 <= 2 && (cmv[1] as i32 + 1) as u32 <= 2;
    if small {
        if info.iref[0] == 0 {
            mv[0] = [0, 0];
        }
        if info.iref[1] == 0 {
            mv[1] = [0, 0];
        }
    }
    mv
}

/// Store a B-direct MV/ref for one 4x4 raster block into `ctx` (both lists).
#[inline]
fn store_block(ctx: &mut DecoderContext, mb_xy: usize, r: usize, mv: [[i16; 2]; 2], iref: [i8; 2], ref_pic: [i32; 2]) {
    let b = (mb_xy * 16 + r) * 2;
    ctx.mv[b] = mv[0][0];
    ctx.mv[b + 1] = mv[0][1];
    ctx.mv_l1[b] = mv[1][0];
    ctx.mv_l1[b + 1] = mv[1][1];
    ctx.ref_idx[mb_xy * 16 + r] = iref[0];
    ctx.ref_idx_l1[mb_xy * 16 + r] = iref[1];
    ctx.ref_pic_id[mb_xy * 16 + r] = if iref[0] >= 0 { ref_pic[0] } else { -1 };
    ctx.ref_pic_id_l1[mb_xy * 16 + r] = if iref[1] >= 0 { ref_pic[1] } else { -1 };
    ctx.direct[mb_xy * 16 + r] = 1;
}

/// Fill all 16 blocks of a 16x16 spatial-direct MB (`UpdateP16x16MotionInfo`
/// then colZero, lines 571-587). `ref_pic` is the resolved reference-picture id
/// for `[list0, list1]` (index 0 of each list).
pub fn fill_direct_16x16(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    info: &DirectInfo,
    cr: &ColRef,
    ref_pic: [i32; 2],
) {
    // colZero uses colocated block 0 for the whole-MB case.
    let mv = col_zero_mv(info, cr, mb_xy, 0);
    for r in 0..16 {
        store_block(ctx, mb_xy, r, mv, info.iref, ref_pic);
    }
}

/// Fill one 8x8 partition of a spatial-direct B_8x8 sub-block. `idx8` is the
/// 8x8 index (0..3); `part_w`/`part_count` come from the sub_mb_type.
#[allow(clippy::too_many_arguments)]
pub fn fill_direct_8x8(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    idx8: usize,
    part_count: usize,
    part_w: usize,
    info: &DirectInfo,
    cr: &ColRef,
    ref_pic: [i32; 2],
) {
    let base_part = idx8 << 2;
    for j in 0..part_count {
        let part_idx = base_part + j * part_w;
        let scan4 = SCAN4[part_idx];
        let mv = col_zero_mv(info, cr, mb_xy, scan4);
        if info.sub_4x4 {
            store_block(ctx, mb_xy, scan4, mv, info.iref, ref_pic);
        } else {
            // 8x8 sub-partition: 2x2 raster block.
            for &r in &[scan4, scan4 + 1, scan4 + 4, scan4 + 5] {
                store_block(ctx, mb_xy, r, mv, info.iref, ref_pic);
            }
        }
    }
}

