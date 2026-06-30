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
//! Only the all-intra boundary-strength rule is implemented here (bS = 4 on MB
//! boundaries, bS = 3 on internal 4x4 edges), which is exact for baseline intra
//! frames. The inter bS derivation (`DeblockingBsMarginalMBAvcbase` /
//! `DeblockingBSInsideMBNormal`) is a later phase; the structure leaves room for
//! it (a per-edge bS would replace the hard-coded `3`/`4`).

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
