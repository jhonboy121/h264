//! In-loop deblocking filter driver, ported from `WelsDeblockingFilterSlice` /
//! `WelsDeblockingMb` / `DeblockingIntraMb` (`FilteringEdgeLumaHV` +
//! `FilteringEdgeChromaHV`) in
//! `reference/codec/decoder/core/src/deblocking.cpp`.
//!
//! This is the picture-level pass: after the whole frame is reconstructed it
//! walks macroblocks in raster order and filters each MB's vertical edges
//! (left boundary + internal x = 4, 8, 12) then horizontal edges (top boundary
//! + internal y = 4, 8, 12), luma then chroma, reading already-filtered
//! left/top neighbours.
//!
//! Intra MBs use the fixed rule (bS = 4 on MB boundaries, bS = 3 on internal
//! 4x4 edges). Inter MBs derive a per-4x4-edge boundary strength
//! (`DeblockingBsMarginalMBAvcbase` / `DeblockingBSInsideMBNormal` /
//! `DeblockingBSInsideMBAvsbase`): bS = 4 on an intra-neighbour MB boundary,
//! else bS = 2 if either side has coefficients, else bS = 1 if the references
//! differ or the MVs differ by >= 4 quarter-pel, else 0.

use crate::dsp::deblock::{
    deblock_chroma_eq42, deblock_chroma_eq4_h, deblock_chroma_eq4_v, deblock_chroma_lt42,
    deblock_chroma_lt4_h, deblock_chroma_lt4_v, deblock_luma_eq4_h, deblock_luma_eq4_v,
    deblock_luma_lt4_h, deblock_luma_lt4_v,
};

use super::context::DecoderContext;

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
#[allow(clippy::too_many_arguments)]
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
        &mut ctx.picture.y, y_off, ystride, left, top, aoff, boff, cur_lqp, left_lqp, top_lqp,
    );
    deblock_chroma(
        &mut ctx.picture.u, &mut ctx.picture.v, c_off, cstride, left, top, aoff, boff, cur_cqp,
        left_cqp, top_cqp,
    );
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma(
    y: &mut [u8],
    y_off: usize,
    stride: usize,
    left: bool,
    top: bool,
    aoff: i32,
    boff: i32,
    cur_qp: i32,
    left_qp: i32,
    top_qp: i32,
) {
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
        deblock_luma_lt4_h(y, y_off + 4, stride, a, b, &tc);
        deblock_luma_lt4_h(y, y_off + 8, stride, a, b, &tc);
        deblock_luma_lt4_h(y, y_off + 12, stride, a, b, &tc);
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
        deblock_luma_lt4_v(y, y_off + 4 * stride, stride, a, b, &tc);
        deblock_luma_lt4_v(y, y_off + 8 * stride, stride, a, b, &tc);
        deblock_luma_lt4_v(y, y_off + 12 * stride, stride, a, b, &tc);
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_chroma(
    cb: &mut [u8],
    cr: &mut [u8],
    c_off: usize,
    stride: usize,
    left: bool,
    top: bool,
    aoff: i32,
    boff: i32,
    cur_qp: [i32; 2],
    left_qp: [i32; 2],
    top_qp: [i32; 2],
) {
    // --- Vertical edges (H kernels): left boundary x=0, internal x=4 ---
    if left {
        let qp = [(cur_qp[0] + left_qp[0] + 1) >> 1, (cur_qp[1] + left_qp[1] + 1) >> 1];
        chroma_edge_eq4(cb, cr, c_off, stride, aoff, boff, qp, true);
    }
    chroma_edge_lt4(cb, cr, c_off + 4, stride, aoff, boff, cur_qp, true);

    // --- Horizontal edges (V kernels): top boundary y=0, internal y=4 ---
    if top {
        let qp = [(cur_qp[0] + top_qp[0] + 1) >> 1, (cur_qp[1] + top_qp[1] + 1) >> 1];
        chroma_edge_eq4(cb, cr, c_off, stride, aoff, boff, qp, false);
    }
    chroma_edge_lt4(cb, cr, c_off + 4 * stride, stride, aoff, boff, cur_qp, false);
}

/// Strong (bS = 4) chroma edge. `vertical` selects the H (vertical-edge) vs V
/// (horizontal-edge) kernel. Uses the two-plane kernel when Cb/Cr share a QP
/// (always true for baseline), else the single-plane kernels per component.
#[allow(clippy::too_many_arguments)]
fn chroma_edge_eq4(
    cb: &mut [u8],
    cr: &mut [u8],
    off: usize,
    stride: usize,
    aoff: i32,
    boff: i32,
    qp: [i32; 2],
    vertical: bool,
) {
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
#[allow(clippy::too_many_arguments)]
fn chroma_edge_lt4(
    cb: &mut [u8],
    cr: &mut [u8],
    off: usize,
    stride: usize,
    aoff: i32,
    boff: i32,
    qp: [i32; 2],
    vertical: bool,
) {
    if qp[0] == qp[1] {
        let (a, b) = alpha_beta(qp[0], aoff, boff);
        if (a | b) != 0 {
            let tc = tc_uniform(qp[0], aoff, BS_INTERNAL, 1);
            if vertical {
                deblock_chroma_lt4_h(cb, off, cr, off, stride, a, b, &tc);
            } else {
                deblock_chroma_lt4_v(cb, off, cr, off, stride, a, b, &tc);
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
    let nzc = |xy: usize, r: usize| ctx.nzc_luma[xy * 16 + r] as i32;
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
        for seg in 0..4 {
            for e in 1..4 {
                let a = nzc(mb_xy, seg * 4 + e - 1);
                let b = nzc(mb_xy, seg * 4 + e);
                nbs[0][e][seg] = if (a | b) != 0 { 2 } else { 0 };
            }
        }
        for s in 0..4 {
            for e in 1..4 {
                let a = nzc(mb_xy, (e - 1) * 4 + s);
                let b = nzc(mb_xy, e * 4 + s);
                nbs[1][e][s] = if (a | b) != 0 { 2 } else { 0 };
            }
        }
    } else {
        // DeblockingBSInsideMBNormal: BS_EDGE with within-MB MV-difference check.
        let bs_edge = |bsx1: i32, idx: usize, nidx: usize| -> u8 {
            let smb = mb_bs_mv(0, 0, mv(idx), mv(nidx)); // ref ignored within MB
            if bsx1 != 0 { 2 } else { smb }
        };
        for seg in 0..4 {
            for e in 1..4 {
                let idx = seg * 4 + e;
                let nidx = seg * 4 + e - 1;
                nbs[0][e][seg] = bs_edge(nzc(mb_xy, idx) | nzc(mb_xy, nidx), idx, nidx);
            }
        }
        for s in 0..4 {
            for e in 1..4 {
                let idx = e * 4 + s;
                let nidx = (e - 1) * 4 + s;
                nbs[1][e][s] = bs_edge(nzc(mb_xy, idx) | nzc(mb_xy, nidx), idx, nidx);
            }
        }
    }
    nbs
}

#[allow(clippy::too_many_arguments)]
fn deblock_inter_mb(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    mb_width: usize,
    mb_x: usize,
    mb_y: usize,
    left: bool,
    top: bool,
) {
    let nbs = inter_bs(ctx, mb_xy, mb_width, left, top);

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
        &mut ctx.picture.y, y_off, ystride, &nbs, left, top, aoff, boff, cur_lqp, left_lqp, top_lqp,
    );
    inter_chroma(
        &mut ctx.picture.u, &mut ctx.picture.v, c_off, cstride, &nbs, left, top, aoff, boff, cur_cqp,
        left_cqp, top_cqp,
    );
}

#[allow(clippy::too_many_arguments)]
fn inter_luma(
    y: &mut [u8],
    y_off: usize,
    stride: usize,
    nbs: &[[[u8; 4]; 4]; 2],
    left: bool,
    top: bool,
    aoff: i32,
    boff: i32,
    cur_qp: i32,
    left_qp: i32,
    top_qp: i32,
) {
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
        for e in 1..4 {
            if nbs[0][e].iter().any(|&v| v != 0) {
                let tc = tc_from_bs(cur_qp, aoff, &nbs[0][e], 0);
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
        for e in 1..4 {
            if nbs[1][e].iter().any(|&v| v != 0) {
                let tc = tc_from_bs(cur_qp, aoff, &nbs[1][e], 0);
                deblock_luma_lt4_v(y, y_off + e * 4 * stride, stride, a, b, &tc);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn inter_chroma(
    cb: &mut [u8],
    cr: &mut [u8],
    c_off: usize,
    stride: usize,
    nbs: &[[[u8; 4]; 4]; 2],
    left: bool,
    top: bool,
    aoff: i32,
    boff: i32,
    cur_qp: [i32; 2],
    left_qp: [i32; 2],
    top_qp: [i32; 2],
) {
    // Vertical: boundary (bS from nbs[0][0]) then internal x=4 (nbs[0][2]).
    if left {
        let qp = [(cur_qp[0] + left_qp[0] + 1) >> 1, (cur_qp[1] + left_qp[1] + 1) >> 1];
        inter_chroma_edge(cb, cr, c_off, stride, aoff, boff, qp, &nbs[0][0], true);
    }
    inter_chroma_edge(cb, cr, c_off + 4, stride, aoff, boff, cur_qp, &nbs[0][2], true);

    // Horizontal: boundary then internal y=4.
    if top {
        let qp = [(cur_qp[0] + top_qp[0] + 1) >> 1, (cur_qp[1] + top_qp[1] + 1) >> 1];
        inter_chroma_edge(cb, cr, c_off, stride, aoff, boff, qp, &nbs[1][0], false);
    }
    inter_chroma_edge(cb, cr, c_off + 4 * stride, stride, aoff, boff, cur_qp, &nbs[1][2], false);
}

/// Filter one chroma edge with per-segment bS. `bS == 4` selects the strong
/// (eq4) filter; otherwise the `Lt4` filter with `tc` from the bS array.
#[allow(clippy::too_many_arguments)]
fn inter_chroma_edge(
    cb: &mut [u8],
    cr: &mut [u8],
    off: usize,
    stride: usize,
    aoff: i32,
    boff: i32,
    qp: [i32; 2],
    nbs: &[u8; 4],
    vertical: bool,
) {
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
                deblock_chroma_lt4_h(cb, off, cr, off, stride, a, b, &tc);
            } else {
                deblock_chroma_lt4_v(cb, off, cr, off, stride, a, b, &tc);
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
