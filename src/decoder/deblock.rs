//! In-loop deblocking filter driver, ported from `WelsDeblockingFilterSlice` /
//! `WelsDeblockingMb` / `DeblockingIntraMb` (`FilteringEdgeLumaHV` +
//! `FilteringEdgeChromaHV`) in
//! `reference/codec/decoder/core/src/deblocking.cpp`.
//!
//! This is the picture-level pass: after the whole frame is reconstructed it
//! walks macroblocks in raster order and filters each MB's vertical edges
//! (left boundary + internal x = 4, 8, 12) then horizontal edges
//! (top boundary + internal y = 4, 8, 12), luma then chroma, reading
//! already-filtered left/top neighbours.
//!
//! Intra MBs use the fixed rule (bS = 4 on MB boundaries, bS = 3 on internal
//! 4x4 edges). Inter MBs derive a per-4x4-edge boundary strength
//! (`DeblockingBsMarginalMBAvcbase` / `DeblockingBSInsideMBNormal` /
//! `DeblockingBSInsideMBAvsbase`): bS = 4 on an intra-neighbour MB boundary,
//! else bS = 2 if either side has coefficients, else bS = 1 if the references
//! differ or the MVs differ by >= 4 quarter-pel, else 0.

use crate::dsp::deblock::{
    ChromaPair, deblock_chroma_eq42, deblock_chroma_eq4_h, deblock_chroma_eq4_v,
    deblock_chroma_lt42, deblock_chroma_lt4_h, deblock_chroma_lt4_v, deblock_luma_eq4_h,
    deblock_luma_eq4_v, deblock_luma_lt4_h, deblock_luma_lt4_v,
};

use super::context::DecoderContext;

/// One luma plane plus the MB sample offset and stride, bundled (with `&mut`) to
/// keep the deblock entry points within the argument-count budget.
struct LumaPlane<'a> {
    y: &'a mut [u8],
    y_off: usize,
    stride: usize,
}

/// The two chroma planes plus the MB sample offset and stride.
struct ChromaPlanes<'a> {
    cb: &'a mut [u8],
    cr: &'a mut [u8],
    off: usize,
    stride: usize,
}

/// Neighbour-edge availability (left/top MB boundaries).
#[derive(Clone, Copy)]
struct AvailEdges {
    left: bool,
    top: bool,
}

/// The slice's alpha/beta deblock offsets.
#[derive(Clone, Copy)]
struct EdgeOffsets {
    aoff: i32,
    boff: i32,
}

/// Luma QPs of the current MB and its left/top neighbours.
#[derive(Clone, Copy)]
struct LumaQp {
    cur: i32,
    left: i32,
    top: i32,
}

/// Chroma (Cb/Cr) QPs of the current MB and its left/top neighbours.
#[derive(Clone, Copy)]
struct ChromaQp {
    cur: [i32; 2],
    left: [i32; 2],
    top: [i32; 2],
}

// --- Threshold tables (Tables 8-16 / 8-17), copied verbatim from
// deblocking.cpp. Indexed by `(qp + offset) + 12`. ---

/// `g_kuiAlphaTable[52 + 24]`.
static ALPHA: [u8; 76] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 4, 4, 5, 6, //
    7, 8, 9, 10, 12, 13, 15, 17, 20, 22, //
    25, 28, 32, 36, 40, 45, 50, 56, 63, 71, //
    80, 90, 101, 113, 127, 144, 162, 182, 203, 226, //
    255, 255, //
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

/// `g_kiBetaTable[52 + 24]`.
static BETA: [u8; 76] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 2, 2, 2, 3, //
    3, 3, 3, 4, 4, 4, 6, 6, 7, 7, //
    8, 8, 9, 9, 10, 10, 11, 11, 12, 12, //
    13, 13, 14, 14, 15, 15, 16, 16, 17, 17, //
    18, 18, //
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
];

/// `g_kiTc0Table[52 + 24][4]`.
static TC0: [[i8; 4]; 76] = [
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 1],
    [-1, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 1, 1], [-1, 0, 1, 1], [-1, 1, 1, 1],
    [-1, 1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, 2], [-1, 1, 1, 2], [-1, 1, 1, 2],
    [-1, 1, 1, 2], [-1, 1, 2, 3], [-1, 1, 2, 3], [-1, 2, 2, 3], [-1, 2, 2, 4], [-1, 2, 3, 4],
    [-1, 2, 3, 4], [-1, 3, 3, 5], [-1, 3, 4, 6], [-1, 3, 4, 6], [-1, 4, 5, 7], [-1, 4, 5, 8],
    [-1, 4, 6, 9], [-1, 5, 7, 10], [-1, 6, 8, 11], [-1, 6, 8, 13], [-1, 7, 10, 14], [-1, 8, 11, 16],
    [-1, 9, 12, 18], [-1, 10, 13, 20], [-1, 11, 15, 23], [-1, 13, 17, 25],
    [-1, 13, 17, 25], [-1, 13, 17, 25], [-1, 13, 17, 25], [-1, 13, 17, 25], [-1, 13, 17, 25],
    [-1, 13, 17, 25], [-1, 13, 17, 25], [-1, 13, 17, 25], [-1, 13, 17, 25], [-1, 13, 17, 25],
    [-1, 13, 17, 25], [-1, 13, 17, 25],
];

/// `GET_ALPHA_BETA_FROM_QP`: alpha from `qp + alpha_off`, beta from
/// `qp + beta_off` (the `+12` table bias is folded in here).
#[inline]
fn alpha_beta(qp: i32, alpha_off: i32, beta_off: i32) -> (i32, i32) {
    let a = ALPHA[(qp + alpha_off + 12) as usize] as i32;
    let b = BETA[(qp + beta_off + 12) as usize] as i32;
    (a, b)
}

/// `TC0_TBL_LOOKUP` for a uniform boundary strength `bs` across the edge: every
/// `tc[k] = g_kiTc0Table(indexA)[bs] + bChroma`.
#[inline]
fn tc_uniform(qp: i32, alpha_off: i32, bs: usize, bchroma: i32) -> [i8; 4] {
    let t = (TC0[(qp + alpha_off + 12) as usize][bs] as i32 + bchroma) as i8;
    [t, t, t, t]
}

/// Internal-edge bS for the all-intra path (§8.7.2.1: bS = 3 for 4x4 edges
/// inside an intra MB).
const BS_INTERNAL: usize = 3;

/// Filter the whole reconstructed picture in place (intra boundary-strength
/// rule). Mirrors `WelsDeblockingFilterSlice` for a single full-frame slice:
/// MBs in raster order, each filtered against its already-filtered neighbours.
pub fn deblock_frame(ctx: &mut DecoderContext) {
    let mb_width = ctx.mb_width;
    let total_mb = ctx.total_mb;
    for mb_xy in 0..total_mb {
        deblock_mb(ctx, mb_xy, mb_width);
    }
}

fn deblock_mb(ctx: &mut DecoderContext, mb_xy: usize, mb_width: usize) {
    let idc = ctx.deblock_idc[mb_xy];
    if idc == 1 {
        return; // deblocking disabled for this MB
    }
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;

    // Boundary availability (DeblockingAvailableNoInterlayer).
    let (left, top) = if idc == 2 {
        let cur = ctx.slice_idc[mb_xy];
        let l = mb_x > 0 && ctx.slice_idc[mb_xy - 1] == cur;
        let t = mb_y > 0 && ctx.slice_idc[mb_xy - mb_width] == cur;
        (l, t)
    } else {
        (mb_x > 0, mb_y > 0)
    };

    if ctx.mb_type[mb_xy].is_intra() {
        deblock_intra_mb(ctx, mb_xy, mb_width, mb_x, mb_y, left, top);
    } else {
        deblock_inter_mb(ctx, mb_xy, mb_width, mb_x, mb_y, left, top);
    }
}

/// All-intra macroblock: bS = 4 on MB boundaries, bS = 3 on internal 4x4 edges.
fn deblock_intra_mb(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    mb_width: usize,
    mb_x: usize,
    mb_y: usize,
    left: bool,
    top: bool,
) {
    let aoff = ctx.deblock_alpha_off[mb_xy] as i32;
    let boff = ctx.deblock_beta_off[mb_xy] as i32;

    // Snapshot QPs (copy out before borrowing the picture mutably).
    let cur_lqp = ctx.luma_qp[mb_xy] as i32;
    let left_lqp = if left { ctx.luma_qp[mb_xy - 1] as i32 } else { 0 };
    let top_lqp = if top { ctx.luma_qp[mb_xy - mb_width] as i32 } else { 0 };
    let cur_cqp = [ctx.chroma_qp[mb_xy * 2] as i32, ctx.chroma_qp[mb_xy * 2 + 1] as i32];
    let left_cqp = if left {
        [ctx.chroma_qp[(mb_xy - 1) * 2] as i32, ctx.chroma_qp[(mb_xy - 1) * 2 + 1] as i32]
    } else {
        [0, 0]
    };
    let top_cqp = if top {
        [ctx.chroma_qp[(mb_xy - mb_width) * 2] as i32, ctx.chroma_qp[(mb_xy - mb_width) * 2 + 1] as i32]
    } else {
        [0, 0]
    };

    let ystride = ctx.picture.luma_stride;
    let cstride = ctx.picture.chroma_stride;
    let y_off = ctx.picture.luma_mb_offset(mb_x, mb_y);
    let c_off = ctx.picture.chroma_mb_offset(mb_x, mb_y);

    deblock_luma(
        LumaPlane { y: &mut ctx.picture.y, y_off, stride: ystride },
        AvailEdges { left, top },
        EdgeOffsets { aoff, boff },
        LumaQp { cur: cur_lqp, left: left_lqp, top: top_lqp },
        ctx.transform_8x8[mb_xy],
    );
    deblock_chroma(
        ChromaPlanes { cb: &mut ctx.picture.u, cr: &mut ctx.picture.v, off: c_off, stride: cstride },
        AvailEdges { left, top },
        EdgeOffsets { aoff, boff },
        ChromaQp { cur: cur_cqp, left: left_cqp, top: top_cqp },
    );
}

fn deblock_luma(
    plane: LumaPlane,
    avail: AvailEdges,
    edge: EdgeOffsets,
    qp: LumaQp,
    transform_8x8: bool,
) {
    let LumaPlane { y, y_off, stride } = plane;
    let AvailEdges { left, top } = avail;
    let EdgeOffsets { aoff, boff } = edge;
    let LumaQp { cur: cur_qp, left: left_qp, top: top_qp } = qp;
    // --- Vertical edges (filtered with the H kernels) ---
    if left {
        let qp = (cur_qp + left_qp + 1) >> 1;
        let (a, b) = alpha_beta(qp, aoff, boff);
        if (a | b) != 0 {
            deblock_luma_eq4_h(y, y_off, stride, a, b);
        }
    }
    let (a, b) = alpha_beta(cur_qp, aoff, boff);
    let tc = tc_uniform(cur_qp, aoff, BS_INTERNAL, 0);
    if (a | b) != 0 {
        // 8x8 transform: only the x = 8 internal edge is a transform-block
        // boundary; the x = 4 / 12 edges are skipped.
        if !transform_8x8 {
            deblock_luma_lt4_h(y, y_off + 4, stride, a, b, &tc);
        }
        deblock_luma_lt4_h(y, y_off + 8, stride, a, b, &tc);
        if !transform_8x8 {
            deblock_luma_lt4_h(y, y_off + 12, stride, a, b, &tc);
        }
    }

    // --- Horizontal edges (filtered with the V kernels) ---
    if top {
        let qp = (cur_qp + top_qp + 1) >> 1;
        let (a2, b2) = alpha_beta(qp, aoff, boff);
        if (a2 | b2) != 0 {
            deblock_luma_eq4_v(y, y_off, stride, a2, b2);
        }
    }
    if (a | b) != 0 {
        if !transform_8x8 {
            deblock_luma_lt4_v(y, y_off + 4 * stride, stride, a, b, &tc);
        }
        deblock_luma_lt4_v(y, y_off + 8 * stride, stride, a, b, &tc);
        if !transform_8x8 {
            deblock_luma_lt4_v(y, y_off + 12 * stride, stride, a, b, &tc);
        }
    }
}

fn deblock_chroma(planes: ChromaPlanes, avail: AvailEdges, edge: EdgeOffsets, qp_set: ChromaQp) {
    let ChromaPlanes { cb, cr, off: c_off, stride } = planes;
    let AvailEdges { left, top } = avail;
    let EdgeOffsets { aoff, boff } = edge;
    let ChromaQp { cur: cur_qp, left: left_qp, top: top_qp } = qp_set;
    // --- Vertical edges (H kernels): left boundary x=0, internal x=4 ---
    if left {
        let qp = [(cur_qp[0] + left_qp[0] + 1) >> 1, (cur_qp[1] + left_qp[1] + 1) >> 1];
        chroma_edge_eq4(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off, stride }, EdgeOffsets { aoff, boff }, qp, true);
    }
    chroma_edge_lt4(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off + 4, stride }, EdgeOffsets { aoff, boff }, cur_qp, true);

    // --- Horizontal edges (V kernels): top boundary y=0, internal y=4 ---
    if top {
        let qp = [(cur_qp[0] + top_qp[0] + 1) >> 1, (cur_qp[1] + top_qp[1] + 1) >> 1];
        chroma_edge_eq4(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off, stride }, EdgeOffsets { aoff, boff }, qp, false);
    }
    chroma_edge_lt4(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off + 4 * stride, stride }, EdgeOffsets { aoff, boff }, cur_qp, false);
}

/// Strong (bS = 4) chroma edge. `vertical` selects the H (vertical-edge) vs V
/// (horizontal-edge) kernel. Uses the two-plane kernel when Cb/Cr share a QP
/// (always true for baseline), else the single-plane kernels per component.
fn chroma_edge_eq4(planes: ChromaPlanes, edge: EdgeOffsets, qp: [i32; 2], vertical: bool) {
    let ChromaPlanes { cb, cr, off, stride } = planes;
    let EdgeOffsets { aoff, boff } = edge;
    if qp[0] == qp[1] {
        let (a, b) = alpha_beta(qp[0], aoff, boff);
        if (a | b) != 0 {
            if vertical {
                deblock_chroma_eq4_h(cb, off, cr, off, stride, a, b);
            } else {
                deblock_chroma_eq4_v(cb, off, cr, off, stride, a, b);
            }
        }
    } else {
        let (sx, sy) = if vertical { (1isize, stride as isize) } else { (stride as isize, 1isize) };
        for (i, plane) in [&mut *cb, &mut *cr].into_iter().enumerate() {
            let (a, b) = alpha_beta(qp[i], aoff, boff);
            if (a | b) != 0 {
                deblock_chroma_eq42(plane, off, sx, sy, a, b);
            }
        }
    }
}

/// Normal (bS = 3 internal) chroma edge.
fn chroma_edge_lt4(planes: ChromaPlanes, edge: EdgeOffsets, qp: [i32; 2], vertical: bool) {
    let ChromaPlanes { cb, cr, off, stride } = planes;
    let EdgeOffsets { aoff, boff } = edge;
    if qp[0] == qp[1] {
        let (a, b) = alpha_beta(qp[0], aoff, boff);
        if (a | b) != 0 {
            let tc = tc_uniform(qp[0], aoff, BS_INTERNAL, 1);
            if vertical {
                deblock_chroma_lt4_h(ChromaPair { cb: &mut *cb, cb_off: off, cr: &mut *cr, cr_off: off }, stride, a, b, &tc);
            } else {
                deblock_chroma_lt4_v(ChromaPair { cb: &mut *cb, cb_off: off, cr: &mut *cr, cr_off: off }, stride, a, b, &tc);
            }
        }
    } else {
        let (sx, sy) = if vertical { (1isize, stride as isize) } else { (stride as isize, 1isize) };
        for (i, plane) in [&mut *cb, &mut *cr].into_iter().enumerate() {
            let (a, b) = alpha_beta(qp[i], aoff, boff);
            if (a | b) != 0 {
                let tc = tc_uniform(qp[i], aoff, BS_INTERNAL, 1);
                deblock_chroma_lt42(plane, off, sx, sy, a, b, &tc);
            }
        }
    }
}

// ===================== Inter macroblock deblocking =====================

// g_kuiTableBIdx: per direction (0=vertical edge, 1=horizontal edge), the four
// current-MB 4x4 raster indices [0..4] and the four neighbour indices [4..8].
const TABLE_B_IDX: [[usize; 8]; 2] = [
    [0, 4, 8, 12, 3, 7, 11, 15],
    [0, 1, 2, 3, 12, 13, 14, 15],
];

/// `g_kiTc0Table(indexA)[bs] + bchroma` for each of four segments, with `bs`
/// taken as `nbs[k] & 3` exactly as `TC0_TBL_LOOKUP`.
#[inline]
fn tc_from_bs(qp: i32, alpha_off: i32, nbs: &[u8; 4], bchroma: i32) -> [i8; 4] {
    let row = &TC0[(qp + alpha_off + 12) as usize];
    [
        (row[(nbs[0] & 3) as usize] as i32 + bchroma) as i8,
        (row[(nbs[1] & 3) as usize] as i32 + bchroma) as i8,
        (row[(nbs[2] & 3) as usize] as i32 + bchroma) as i8,
        (row[(nbs[3] & 3) as usize] as i32 + bchroma) as i8,
    ]
}

/// `MB_BS_MV`: bS = 1 if the two blocks use a different reference picture or
/// their MVs differ by >= 4 quarter-pel in either component, else 0.
#[inline]
fn mb_bs_mv(ref_a: i32, ref_b: i32, mv_a: [i16; 2], mv_b: [i16; 2]) -> u8 {
    let diff = ref_a != ref_b
        || (mv_a[0] as i32 - mv_b[0] as i32).abs() >= 4
        || (mv_a[1] as i32 - mv_b[1] as i32).abs() >= 4;
    diff as u8
}

#[inline]
fn mv_diff(a: [i16; 2], b: [i16; 2]) -> bool {
    (a[0] as i32 - b[0] as i32).abs() >= 4 || (a[1] as i32 - b[1] as i32).abs() >= 4
}

/// Snapshot of one 4x4 block's bi-predictive reference identities + MVs.
#[derive(Clone, Copy)]
struct BBlk {
    r0: i32,
    r1: i32,
    m0: [i16; 2],
    m1: [i16; 2],
}

/// B-slice inter boundary strength (0 or 1) for two blocks `p`/`q`, allowing
/// cross-list reference matching (`ON_MB_BS` / `IN_SMB_EDGE_MV`, spec 8.7.2.1).
#[inline]
fn b_bs_inter(p: BBlk, q: BBlk) -> u8 {
    let matched = (p.r0 == q.r0 && p.r1 == q.r1) || (p.r0 == q.r1 && p.r1 == q.r0);
    if !matched {
        return 1;
    }
    let v = if p.r0 != p.r1 {
        if p.r0 == q.r0 {
            mv_diff(p.m0, q.m0) || mv_diff(p.m1, q.m1)
        } else {
            mv_diff(p.m0, q.m1) || mv_diff(p.m1, q.m0)
        }
    } else {
        (mv_diff(p.m0, q.m0) || mv_diff(p.m1, q.m1)) && (mv_diff(p.m0, q.m1) || mv_diff(p.m1, q.m0))
    };
    v as u8
}

/// Effective per-block luma nonzero count for deblocking bS: for an MB coded
/// with the 8x8 transform, a 4x4 sub-block's "transform block has coefficients"
/// test uses the whole 8x8 block (OR of its four 4x4 sub-block counts), per
/// `DeblockingBSInsideMB` / `DeblockingBsMarginalMB` (`i8x8NnzTab`).
#[inline]
fn eff_nzc(ctx: &DecoderContext, xy: usize, r: usize) -> i32 {
    if ctx.transform_8x8[xy] {
        let bx8 = (r % 4) / 2;
        let by8 = (r / 4) / 2;
        let base = by8 * 8 + bx8 * 2;
        ctx.nzc_luma[xy * 16 + base] as i32
            | ctx.nzc_luma[xy * 16 + base + 1] as i32
            | ctx.nzc_luma[xy * 16 + base + 4] as i32
            | ctx.nzc_luma[xy * 16 + base + 5] as i32
    } else {
        ctx.nzc_luma[xy * 16 + r] as i32
    }
}

/// Compute the boundary-strength array for a B-slice inter MB, using both
/// reference lists (`DeblockingBSliceBsMarginalMBAvcbase` +
/// `DeblockingBSliceBSInsideMBNormal`).
fn inter_bs_b(
    ctx: &DecoderContext,
    mb_xy: usize,
    mb_width: usize,
    left: bool,
    top: bool,
) -> [[[u8; 4]; 4]; 2] {
    let mut nbs = [[[0u8; 4]; 4]; 2];
    let nzc = |xy: usize, r: usize| eff_nzc(ctx, xy, r);
    let blk = |xy: usize, r: usize| BBlk {
        r0: ctx.ref_pic_id[xy * 16 + r],
        r1: ctx.ref_pic_id_l1[xy * 16 + r],
        m0: [ctx.mv[(xy * 16 + r) * 2], ctx.mv[(xy * 16 + r) * 2 + 1]],
        m1: [ctx.mv_l1[(xy * 16 + r) * 2], ctx.mv_l1[(xy * 16 + r) * 2 + 1]],
    };

    // MB boundaries (edge 0).
    for dir in 0..2 {
        let avail = if dir == 0 { left } else { top };
        if !avail {
            continue;
        }
        let nb_xy = if dir == 0 { mb_xy - 1 } else { mb_xy - mb_width };
        if ctx.mb_type[nb_xy].is_intra() {
            nbs[dir][0] = [4, 4, 4, 4];
            continue;
        }
        for i in 0..4 {
            let cur = TABLE_B_IDX[dir][i];
            let neigh = TABLE_B_IDX[dir][4 + i];
            nbs[dir][0][i] = if nzc(mb_xy, cur) != 0 || nzc(nb_xy, neigh) != 0 {
                2
            } else {
                b_bs_inter(blk(mb_xy, cur), blk(nb_xy, neigh))
            };
        }
    }

    // Internal edges (1,2,3) — vertical then horizontal.
    for (e, edge) in nbs[0].iter_mut().enumerate().skip(1) {
        for (seg, cell) in edge.iter_mut().enumerate() {
            let idx = seg * 4 + e;
            let nidx = seg * 4 + e - 1;
            *cell = if nzc(mb_xy, idx) != 0 || nzc(mb_xy, nidx) != 0 {
                2
            } else {
                b_bs_inter(blk(mb_xy, idx), blk(mb_xy, nidx))
            };
        }
    }
    for (e, edge) in nbs[1].iter_mut().enumerate().skip(1) {
        for (s, cell) in edge.iter_mut().enumerate() {
            let idx = e * 4 + s;
            let nidx = (e - 1) * 4 + s;
            *cell = if nzc(mb_xy, idx) != 0 || nzc(mb_xy, nidx) != 0 {
                2
            } else {
                b_bs_inter(blk(mb_xy, idx), blk(mb_xy, nidx))
            };
        }
    }
    nbs
}

/// Compute the boundary-strength array `nbs[dir][edge][seg]` for an inter MB.
/// `dir` 0 = vertical edges (left boundary + x=4,8,12), 1 = horizontal.
fn inter_bs(
    ctx: &DecoderContext,
    mb_xy: usize,
    mb_width: usize,
    left: bool,
    top: bool,
) -> [[[u8; 4]; 4]; 2] {
    let mut nbs = [[[0u8; 4]; 4]; 2];
    let nzc = |xy: usize, r: usize| eff_nzc(ctx, xy, r);
    let refp = |r: usize| ctx.ref_pic_id[mb_xy * 16 + r];
    let mv = |r: usize| {
        let b = (mb_xy * 16 + r) * 2;
        [ctx.mv[b], ctx.mv[b + 1]]
    };
    let mb_type = ctx.mb_type[mb_xy];

    // --- MB boundaries (edge 0) ---
    for dir in 0..2 {
        let avail = if dir == 0 { left } else { top };
        if !avail {
            continue;
        }
        let nb_xy = if dir == 0 { mb_xy - 1 } else { mb_xy - mb_width };
        if ctx.mb_type[nb_xy].is_intra() {
            nbs[dir][0] = [4, 4, 4, 4];
            continue;
        }
        let nb_mv = |r: usize| {
            let b = (nb_xy * 16 + r) * 2;
            [ctx.mv[b], ctx.mv[b + 1]]
        };
        let nb_ref = |r: usize| ctx.ref_pic_id[nb_xy * 16 + r];
        for i in 0..4 {
            let cur = TABLE_B_IDX[dir][i];
            let neigh = TABLE_B_IDX[dir][4 + i];
            nbs[dir][0][i] = if nzc(mb_xy, cur) != 0 || nzc(nb_xy, neigh) != 0 {
                2
            } else {
                mb_bs_mv(refp(cur), nb_ref(neigh), mv(cur), nb_mv(neigh))
            };
        }
    }

    // --- Internal edges (edges 1,2,3) ---
    if mb_type.is_skip() {
        return nbs; // skip: internal bS all 0
    }
    if mb_type.is_inter_16x16() {
        // DeblockingBSInsideMBAvsbase: bS = 2 if either 4x4 block carries
        // coefficients, else 0 (the MV is uniform across a 16x16 partition).
        for (e, edge) in nbs[0].iter_mut().enumerate().skip(1) {
            for (seg, cell) in edge.iter_mut().enumerate() {
                let a = nzc(mb_xy, seg * 4 + e - 1);
                let b = nzc(mb_xy, seg * 4 + e);
                *cell = if (a | b) != 0 { 2 } else { 0 };
            }
        }
        for (e, edge) in nbs[1].iter_mut().enumerate().skip(1) {
            for (s, cell) in edge.iter_mut().enumerate() {
                let a = nzc(mb_xy, (e - 1) * 4 + s);
                let b = nzc(mb_xy, e * 4 + s);
                *cell = if (a | b) != 0 { 2 } else { 0 };
            }
        }
    } else {
        // DeblockingBSInsideMBNormal: BS_EDGE — compares the two 4x4 blocks'
        // reference pictures (different ref_idx across 8x8/16x8/8x16 partitions)
        // as well as their MVs.
        let bs_edge = |bsx1: i32, idx: usize, nidx: usize| -> u8 {
            let smb = mb_bs_mv(refp(idx), refp(nidx), mv(idx), mv(nidx));
            if bsx1 != 0 { 2 } else { smb }
        };
        for (e, edge) in nbs[0].iter_mut().enumerate().skip(1) {
            for (seg, cell) in edge.iter_mut().enumerate() {
                let idx = seg * 4 + e;
                let nidx = seg * 4 + e - 1;
                *cell = bs_edge(nzc(mb_xy, idx) | nzc(mb_xy, nidx), idx, nidx);
            }
        }
        for (e, edge) in nbs[1].iter_mut().enumerate().skip(1) {
            for (s, cell) in edge.iter_mut().enumerate() {
                let idx = e * 4 + s;
                let nidx = (e - 1) * 4 + s;
                *cell = bs_edge(nzc(mb_xy, idx) | nzc(mb_xy, nidx), idx, nidx);
            }
        }
    }
    nbs
}

fn deblock_inter_mb(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    mb_width: usize,
    mb_x: usize,
    mb_y: usize,
    left: bool,
    top: bool,
) {
    let nbs = if ctx.is_b_slice {
        inter_bs_b(ctx, mb_xy, mb_width, left, top)
    } else {
        inter_bs(ctx, mb_xy, mb_width, left, top)
    };

    let aoff = ctx.deblock_alpha_off[mb_xy] as i32;
    let boff = ctx.deblock_beta_off[mb_xy] as i32;
    let cur_lqp = ctx.luma_qp[mb_xy] as i32;
    let cur_cqp = [ctx.chroma_qp[mb_xy * 2] as i32, ctx.chroma_qp[mb_xy * 2 + 1] as i32];
    let left_lqp = if left { ctx.luma_qp[mb_xy - 1] as i32 } else { 0 };
    let top_lqp = if top { ctx.luma_qp[mb_xy - mb_width] as i32 } else { 0 };
    let left_cqp = if left {
        [ctx.chroma_qp[(mb_xy - 1) * 2] as i32, ctx.chroma_qp[(mb_xy - 1) * 2 + 1] as i32]
    } else {
        [0, 0]
    };
    let top_cqp = if top {
        [ctx.chroma_qp[(mb_xy - mb_width) * 2] as i32, ctx.chroma_qp[(mb_xy - mb_width) * 2 + 1] as i32]
    } else {
        [0, 0]
    };

    let ystride = ctx.picture.luma_stride;
    let cstride = ctx.picture.chroma_stride;
    let y_off = ctx.picture.luma_mb_offset(mb_x, mb_y);
    let c_off = ctx.picture.chroma_mb_offset(mb_x, mb_y);

    inter_luma(
        LumaPlane { y: &mut ctx.picture.y, y_off, stride: ystride },
        &nbs,
        AvailEdges { left, top },
        EdgeOffsets { aoff, boff },
        LumaQp { cur: cur_lqp, left: left_lqp, top: top_lqp },
        ctx.transform_8x8[mb_xy],
    );
    inter_chroma(
        ChromaPlanes { cb: &mut ctx.picture.u, cr: &mut ctx.picture.v, off: c_off, stride: cstride },
        &nbs,
        AvailEdges { left, top },
        EdgeOffsets { aoff, boff },
        ChromaQp { cur: cur_cqp, left: left_cqp, top: top_cqp },
    );
}

fn inter_luma(
    plane: LumaPlane,
    nbs: &[[[u8; 4]; 4]; 2],
    avail: AvailEdges,
    edge: EdgeOffsets,
    qp: LumaQp,
    transform_8x8: bool,
) {
    let LumaPlane { y, y_off, stride } = plane;
    let AvailEdges { left, top } = avail;
    let EdgeOffsets { aoff, boff } = edge;
    let LumaQp { cur: cur_qp, left: left_qp, top: top_qp } = qp;
    // For an 8x8-transform MB only the central internal edge (e == 2) is a
    // transform-block boundary; the e == 1 / e == 3 edges are not filtered.
    let skip_internal = |e: usize| transform_8x8 && e != 2;
    // Vertical edges (H kernels).
    if left {
        if nbs[0][0][0] == 4 {
            let qp = (cur_qp + left_qp + 1) >> 1;
            let (a, b) = alpha_beta(qp, aoff, boff);
            if (a | b) != 0 {
                deblock_luma_eq4_h(y, y_off, stride, a, b);
            }
        } else if nbs[0][0].iter().any(|&v| v != 0) {
            let qp = (cur_qp + left_qp + 1) >> 1;
            let (a, b) = alpha_beta(qp, aoff, boff);
            if (a | b) != 0 {
                let tc = tc_from_bs(qp, aoff, &nbs[0][0], 0);
                deblock_luma_lt4_h(y, y_off, stride, a, b, &tc);
            }
        }
    }
    let (a, b) = alpha_beta(cur_qp, aoff, boff);
    if (a | b) != 0 {
        for (e, edge) in nbs[0].iter().enumerate().skip(1) {
            if !skip_internal(e) && edge.iter().any(|&v| v != 0) {
                let tc = tc_from_bs(cur_qp, aoff, edge, 0);
                deblock_luma_lt4_h(y, y_off + e * 4, stride, a, b, &tc);
            }
        }
    }

    // Horizontal edges (V kernels).
    if top {
        if nbs[1][0][0] == 4 {
            let qp = (cur_qp + top_qp + 1) >> 1;
            let (a2, b2) = alpha_beta(qp, aoff, boff);
            if (a2 | b2) != 0 {
                deblock_luma_eq4_v(y, y_off, stride, a2, b2);
            }
        } else if nbs[1][0].iter().any(|&v| v != 0) {
            let qp = (cur_qp + top_qp + 1) >> 1;
            let (a2, b2) = alpha_beta(qp, aoff, boff);
            if (a2 | b2) != 0 {
                let tc = tc_from_bs(qp, aoff, &nbs[1][0], 0);
                deblock_luma_lt4_v(y, y_off, stride, a2, b2, &tc);
            }
        }
    }
    if (a | b) != 0 {
        for (e, edge) in nbs[1].iter().enumerate().skip(1) {
            if !skip_internal(e) && edge.iter().any(|&v| v != 0) {
                let tc = tc_from_bs(cur_qp, aoff, edge, 0);
                deblock_luma_lt4_v(y, y_off + e * 4 * stride, stride, a, b, &tc);
            }
        }
    }
}

fn inter_chroma(planes: ChromaPlanes, nbs: &[[[u8; 4]; 4]; 2], avail: AvailEdges, edge: EdgeOffsets, qp_set: ChromaQp) {
    let ChromaPlanes { cb, cr, off: c_off, stride } = planes;
    let AvailEdges { left, top } = avail;
    let EdgeOffsets { aoff, boff } = edge;
    let ChromaQp { cur: cur_qp, left: left_qp, top: top_qp } = qp_set;
    // Vertical: boundary (bS from nbs[0][0]) then internal x=4 (nbs[0][2]).
    if left {
        let qp = [(cur_qp[0] + left_qp[0] + 1) >> 1, (cur_qp[1] + left_qp[1] + 1) >> 1];
        inter_chroma_edge(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off, stride }, EdgeOffsets { aoff, boff }, qp, &nbs[0][0], true);
    }
    inter_chroma_edge(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off + 4, stride }, EdgeOffsets { aoff, boff }, cur_qp, &nbs[0][2], true);

    // Horizontal: boundary then internal y=4.
    if top {
        let qp = [(cur_qp[0] + top_qp[0] + 1) >> 1, (cur_qp[1] + top_qp[1] + 1) >> 1];
        inter_chroma_edge(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off, stride }, EdgeOffsets { aoff, boff }, qp, &nbs[1][0], false);
    }
    inter_chroma_edge(ChromaPlanes { cb: &mut *cb, cr: &mut *cr, off: c_off + 4 * stride, stride }, EdgeOffsets { aoff, boff }, cur_qp, &nbs[1][2], false);
}

/// Filter one chroma edge with per-segment bS. `bS == 4` selects the strong
/// (eq4) filter; otherwise the `Lt4` filter with `tc` from the bS array.
fn inter_chroma_edge(planes: ChromaPlanes, edge: EdgeOffsets, qp: [i32; 2], nbs: &[u8; 4], vertical: bool) {
    let ChromaPlanes { cb, cr, off, stride } = planes;
    let EdgeOffsets { aoff, boff } = edge;
    if nbs[0] == 4 {
        // Strong filter (intra-neighbour boundary).
        if qp[0] == qp[1] {
            let (a, b) = alpha_beta(qp[0], aoff, boff);
            if (a | b) != 0 {
                if vertical {
                    deblock_chroma_eq4_h(cb, off, cr, off, stride, a, b);
                } else {
                    deblock_chroma_eq4_v(cb, off, cr, off, stride, a, b);
                }
            }
        } else {
            let (sx, sy) = if vertical { (1isize, stride as isize) } else { (stride as isize, 1isize) };
            for (i, plane) in [&mut *cb, &mut *cr].into_iter().enumerate() {
                let (a, b) = alpha_beta(qp[i], aoff, boff);
                if (a | b) != 0 {
                    deblock_chroma_eq42(plane, off, sx, sy, a, b);
                }
            }
        }
        return;
    }
    if !nbs.iter().any(|&v| v != 0) {
        return;
    }
    if qp[0] == qp[1] {
        let (a, b) = alpha_beta(qp[0], aoff, boff);
        if (a | b) != 0 {
            let tc = tc_from_bs(qp[0], aoff, nbs, 1);
            if vertical {
                deblock_chroma_lt4_h(ChromaPair { cb: &mut *cb, cb_off: off, cr: &mut *cr, cr_off: off }, stride, a, b, &tc);
            } else {
                deblock_chroma_lt4_v(ChromaPair { cb: &mut *cb, cb_off: off, cr: &mut *cr, cr_off: off }, stride, a, b, &tc);
            }
        }
    } else {
        let (sx, sy) = if vertical { (1isize, stride as isize) } else { (stride as isize, 1isize) };
        for (i, plane) in [&mut *cb, &mut *cr].into_iter().enumerate() {
            let (a, b) = alpha_beta(qp[i], aoff, boff);
            if (a | b) != 0 {
                let tc = tc_from_bs(qp[i], aoff, nbs, 1);
                deblock_chroma_lt42(plane, off, sx, sy, a, b, &tc);
            }
        }
    }
}
