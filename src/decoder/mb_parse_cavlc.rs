//! I-slice macroblock syntax + residual parsing (CAVLC, 4:2:0 8-bit).
//!
//! Faithful port of `WelsActualDecodeMbCavlcISlice`
//! (`reference/codec/decoder/core/src/decode_slice.cpp`) together with the
//! helpers it calls in `parse_mb_syn_cavlc.cpp`: `GetNeighborAvailMbType`,
//! `WelsFillCacheNonZeroCount`, `ParseIntra4x4Mode`/`ParseIntra16x16Mode`,
//! `PredIntra4x4Mode`, `CheckIntraNxNPredMode`/`CheckIntra16x16PredMode`/
//! `CheckIntraChromaPredMode`, and `WelsResidualBlockCavlc` (residual store +
//! dequant). The non-zero-count cache and intra-mode cache are expressed
//! directly as spatial neighbour look-ups (the 48/30-entry C caches implement
//! exactly these), which is bit-identical for the baseline path.
//!
//! Deferred (guarded with [`DecodeError::Unsupported`]): I_PCM, transform_8x8 /
//! I_8x8 (High profile), and scaling lists (`bUseScalingList`). None occur in
//! the baseline CAVLC fixture.

use crate::bits::BitReader;
use crate::dsp::tables::{
    G_KUI_CHROMA_DC_SCAN, G_KUI_DEQUANT_COEFF, G_KUI_LUMA_DC_ZIGZAG_SCAN, G_KUI_ZIGZAG_SCAN,
};
use crate::error::DecodeError;

use super::cavlc::residual_block_cavlc;
use super::context::{DecoderContext, MbType, NeighborAvail};
use super::params::Pps;

type Result<T> = core::result::Result<T, DecodeError>;

// --- Per-MB 4x4 block geometry (luma) -------------------------------------
// Block scan index i (the order residuals are coded, == g_kuiScan8 order) maps
// to a raster position (bx,by) within the macroblock.
const BLOCK_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
const BLOCK_BX: [usize; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
const BLOCK_BY: [usize; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];

// g_kuiCache30ScanIdx: block i -> position in the 30-entry (6-wide) sample
// availability grid used by CheckIntraNxNPredMode.
const CACHE30_SCAN_IDX: [usize; 16] = [7, 8, 13, 14, 9, 10, 15, 16, 19, 20, 25, 26, 21, 22, 27, 28];

// g_kuiI16CbpTable.
const I16_CBP_TABLE: [u8; 6] = [0, 16, 32, 15, 31, 47];

// g_kuiIntra4x4CbpTable (chroma_format_idc != 0).
#[rustfmt::skip]
const INTRA4X4_CBP_TABLE: [u8; 48] = [
    47, 31, 15,  0, 23, 27, 29, 30,  7, 11, 13, 14, 39, 43, 45, 46,
    16,  3,  5, 10, 12, 19, 21, 26, 28, 35, 37, 42, 44,  1,  2,  4,
     8, 17, 18, 20, 24,  6,  9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

// g_kuiChromaQpTable.
#[rustfmt::skip]
const CHROMA_QP_TABLE: [u8; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37,
    37, 38, 38, 38, 39, 39, 39, 39,
];

// g_ksI16PredInfo / g_ksChromaPredInfo / g_ksI4PredInfo: (pred_mode, need_left,
// need_top, need_left_top). Availability flags are compared with `>=` exactly as
// the C CHECK_* macros (which pass masked ints).
const I16_PRED_INFO: [(i8, i32, i32, i32); 4] = [
    (0, 0, 1, 0), // V
    (1, 1, 0, 0), // H
    (0, 0, 0, 0), // (DC handled separately)
    (3, 1, 1, 1), // P
];
const CHROMA_PRED_INFO: [(i8, i32, i32, i32); 4] = [
    (0, 0, 0, 0), // (DC handled separately)
    (1, 1, 0, 0), // H
    (2, 0, 1, 0), // V
    (3, 1, 1, 1), // P
];
const I4_PRED_INFO: [(i8, i32, i32, i32); 9] = [
    (0, 0, 1, 0), // V
    (1, 1, 0, 0), // H
    (0, 0, 0, 0), // (DC handled separately)
    (3, 0, 1, 0), // DDL
    (4, 1, 1, 1), // DDR
    (5, 1, 1, 1), // VR
    (6, 1, 1, 1), // HD
    (7, 0, 1, 0), // VL
    (8, 1, 0, 0), // HU
];

#[inline]
fn clip3(x: i32, lo: i32, hi: i32) -> i32 {
    x.clamp(lo, hi)
}

/// `WELS_NON_ZERO_COUNT_AVERAGE`: nC from the left/top neighbour counts
/// (`-1` == unavailable).
#[inline]
fn nc_average(na: i32, nb: i32) -> i32 {
    let mut nc = na + nb + 1;
    if na != -1 && nb != -1 {
        nc >>= 1;
    }
    if na == -1 && nb == -1 {
        nc += 1;
    }
    nc
}

/// Snapshot of a neighbour MB's per-block state, used so the parse can borrow
/// `ctx` immutably up front and then mutate it freely.
#[derive(Clone, Copy)]
struct NeighborSnap {
    avail: bool,
    is_nxn: bool,
    nzc_luma: [i8; 16],
    nzc_chroma: [i8; 8],
    best_mode: [i8; 16],
}

impl NeighborSnap {
    fn unavailable() -> Self {
        NeighborSnap {
            avail: false,
            is_nxn: false,
            nzc_luma: [-1; 16],
            nzc_chroma: [-1; 8],
            best_mode: [-1; 16],
        }
    }

    fn from_ctx(ctx: &DecoderContext, avail: bool, xy: usize) -> Self {
        if !avail {
            return Self::unavailable();
        }
        let mut s = NeighborSnap {
            avail: true,
            is_nxn: ctx.mb_type[xy].is_intra_nxn(),
            nzc_luma: [0; 16],
            nzc_chroma: [0; 8],
            best_mode: [0; 16],
        };
        s.nzc_luma.copy_from_slice(ctx.nzc_luma_mb(xy));
        s.nzc_chroma.copy_from_slice(ctx.nzc_chroma_mb(xy));
        s.best_mode.copy_from_slice(&ctx.i4_best_mode[xy * 16..xy * 16 + 16]);
        s
    }
}

/// Parse one intra macroblock of an I slice into `ctx` (modes, cbp, qp, per-MB
/// non-zero counts) and into `coeffs` (the 384-entry dequantised residual store,
/// laid out exactly like `pScaledTCoeff`: 16 luma 4x4 blocks of 16, then Cb and
/// Cr each 4 blocks of 16). `last_mb_qp` carries `pSlice->iLastMbQp` across MBs.
pub fn parse_intra_mb_cavlc(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    mb_xy: usize,
    pps: &Pps,
    last_mb_qp: &mut i32,
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let ui_mb_type = bs.read_ue()?;
    parse_intra_mb_core(bs, ctx, mb_xy, pps, last_mb_qp, coeffs, ui_mb_type)
}

/// Body of intra-MB parse with `ui_mb_type` already read (so the P-slice path
/// can hand in the value after its `-5` adjustment).
pub(super) fn parse_intra_mb_core(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    mb_xy: usize,
    pps: &Pps,
    last_mb_qp: &mut i32,
    coeffs: &mut [i16; 384],
    ui_mb_type: u32,
) -> Result<()> {
    let neigh = ctx.neighbors(mb_xy);
    let left = NeighborSnap::from_ctx(ctx, neigh.left, neigh.left_xy);
    let top = NeighborSnap::from_ctx(ctx, neigh.top, neigh.top_xy);

    // Current-MB block state, accumulated locally then written back to ctx.
    let mut cur_nzc_luma = [0i8; 16];
    let mut cur_nzc_chroma = [0i8; 8];
    let mut best_mode = [-1i8; 16];
    let mut final_mode = [2i8; 16];

    let mb_type;
    let mut i16_mode = 0i8;
    let mut chroma_mode;
    let cbp: u8;

    if ui_mb_type > 25 {
        return Err(DecodeError::InvalidSyntax("intra mb_type"));
    }
    if ui_mb_type == 25 {
        return Err(DecodeError::Unsupported("I_PCM"));
    }

    if ui_mb_type == 0 {
        // I_NxN. transform_size_8x8 (I_8x8) is High-profile only.
        if pps.transform_8x8_mode_flag {
            let t8 = bs.read_flag()?;
            if t8 {
                return Err(DecodeError::Unsupported("transform_size_8x8 / I_8x8"));
            }
        }
        mb_type = MbType::Intra4x4;
        chroma_mode =
            parse_intra4x4(bs, &neigh, &left, &top, &mut best_mode, &mut final_mode)?;

        let ui_cbp = bs.read_ue()?;
        if ui_cbp > 47 {
            return Err(DecodeError::InvalidSyntax("intra4x4 cbp"));
        }
        cbp = INTRA4X4_CBP_TABLE[ui_cbp as usize];
    } else {
        // I_16x16.
        mb_type = MbType::Intra16x16;
        i16_mode = ((ui_mb_type - 1) & 3) as i8;
        cbp = I16_CBP_TABLE[((ui_mb_type - 1) >> 2) as usize];

        let neigh_avail = ((neigh.left as i32) << 2)
            | ((neigh.top_left as i32) << 1)
            | (neigh.top as i32);
        check_intra16x16_mode(neigh_avail, &mut i16_mode)?;
        // intra_chroma_pred_mode
        let cm = bs.read_ue()?;
        if cm > 3 {
            return Err(DecodeError::InvalidSyntax("intra_chroma_pred_mode"));
        }
        chroma_mode = cm as i8;
        check_intra_chroma_mode(neigh_avail, &mut chroma_mode)?;
    }

    let cbp_l = cbp & 0x0f;
    let cbp_c = cbp >> 4;

    // mb_qp_delta + residuals.
    let luma_qp: i32;
    if cbp != 0 || mb_type == MbType::Intra16x16 {
        coeffs.iter_mut().for_each(|c| *c = 0);
        let qp_delta = bs.read_se()?;
        if !(-26..=25).contains(&qp_delta) {
            return Err(DecodeError::InvalidSyntax("mb_qp_delta"));
        }
        luma_qp = (*last_mb_qp + qp_delta + 52) % 52;
        *last_mb_qp = luma_qp;
    } else {
        luma_qp = *last_mb_qp;
    }
    let chroma_qp = [
        CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[0], 0, 51) as usize] as i32,
        CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[1], 0, 51) as usize] as i32,
    ];

    if cbp != 0 || mb_type == MbType::Intra16x16 {
        parse_residuals(
            bs,
            mb_type,
            cbp_l,
            cbp_c,
            luma_qp,
            &chroma_qp,
            &left,
            &top,
            &mut cur_nzc_luma,
            &mut cur_nzc_chroma,
            coeffs,
        )?;
    }

    // Commit MB state to the context.
    ctx.mb_type[mb_xy] = mb_type;
    ctx.i16_mode[mb_xy] = i16_mode;
    ctx.chroma_mode[mb_xy] = chroma_mode;
    ctx.cbp[mb_xy] = cbp;
    ctx.luma_qp[mb_xy] = luma_qp as i8;
    ctx.chroma_qp[mb_xy * 2] = chroma_qp[0] as i8;
    ctx.chroma_qp[mb_xy * 2 + 1] = chroma_qp[1] as i8;
    ctx.nzc_luma[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&cur_nzc_luma);
    ctx.nzc_chroma[mb_xy * 8..mb_xy * 8 + 8].copy_from_slice(&cur_nzc_chroma);
    ctx.i4_best_mode[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&best_mode);
    ctx.i4_final_mode[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&final_mode);

    // Intra MBs carry no list-0 motion; mark blocks as not-in-list so that
    // neighbouring inter MBs see `REF_NOT_IN_LIST` and zero MVs.
    for b in 0..16 {
        ctx.ref_idx[mb_xy * 16 + b] = -1;
        ctx.ref_pic_id[mb_xy * 16 + b] = -1;
        ctx.mv[(mb_xy * 16 + b) * 2] = 0;
        ctx.mv[(mb_xy * 16 + b) * 2 + 1] = 0;
    }

    Ok(())
}

/// `ParseIntra4x4Mode`: 16 luma modes + chroma mode. Returns the (checked)
/// chroma prediction mode.
fn parse_intra4x4(
    bs: &mut BitReader<'_>,
    neigh: &NeighborAvail,
    left: &NeighborSnap,
    top: &NeighborSnap,
    best_mode: &mut [i8; 16],
    final_mode: &mut [i8; 16],
) -> Result<i8> {
    // Neighbour mode cache rows (WelsFillCacheConstrain0IntraNxN).
    let top_modes: [i8; 4] = if top.avail && top.is_nxn {
        [top.best_mode[12], top.best_mode[13], top.best_mode[14], top.best_mode[15]]
    } else if top.avail {
        [2; 4]
    } else {
        [-1; 4]
    };
    let left_modes: [i8; 4] = if left.avail && left.is_nxn {
        [left.best_mode[3], left.best_mode[7], left.best_mode[11], left.best_mode[15]]
    } else if left.avail {
        [2; 4]
    } else {
        [-1; 4]
    };

    // 30-entry (6-wide) sample-availability grid (WelsMapNxNNeighToSampleNormal).
    let mut sample_avail = [0i32; 30];
    if neigh.left {
        sample_avail[6] = 1;
        sample_avail[12] = 1;
        sample_avail[18] = 1;
        sample_avail[24] = 1;
    }
    if neigh.top_left {
        sample_avail[0] = 1;
    }
    if neigh.top {
        sample_avail[1] = 1;
        sample_avail[2] = 1;
        sample_avail[3] = 1;
        sample_avail[4] = 1;
    }
    if neigh.top_right {
        sample_avail[5] = 1;
    }

    let chroma_neigh_avail =
        (sample_avail[6] << 2) | (sample_avail[0] << 1) | sample_avail[1];

    for i in 0..16 {
        let raster = BLOCK_RASTER[i];
        let bx = BLOCK_BX[i];
        let by = BLOCK_BY[i];

        let prev_flag = bs.read_flag()?;
        let top_mode = if by > 0 { best_mode[(by - 1) * 4 + bx] } else { top_modes[bx] };
        let left_mode = if bx > 0 { best_mode[by * 4 + bx - 1] } else { left_modes[by] };
        let pred_mode = if left_mode == -1 || top_mode == -1 {
            2
        } else {
            left_mode.min(top_mode)
        };

        let cur_best = if prev_flag {
            pred_mode
        } else {
            let rem = bs.read_bits(3)? as i8;
            rem + if rem >= pred_mode { 1 } else { 0 }
        };

        let cur_final = check_intra_nxn_mode(&sample_avail, cur_best, i)?;

        best_mode[raster] = cur_best;
        final_mode[raster] = cur_final;
        sample_avail[CACHE30_SCAN_IDX[i]] = 1;
    }

    // intra_chroma_pred_mode
    let cm = bs.read_ue()?;
    if cm > 3 {
        return Err(DecodeError::InvalidSyntax("intra_chroma_pred_mode"));
    }
    let mut chroma_mode = cm as i8;
    check_intra_chroma_mode(chroma_neigh_avail, &mut chroma_mode)?;
    Ok(chroma_mode)
}

/// `CheckIntraNxNPredMode` (4x4): validate `mode` against availability, return
/// the final mode (with DC / DDL_TOP / VL_TOP variants resolved).
fn check_intra_nxn_mode(sample_avail: &[i32; 30], mode: i8, i: usize) -> Result<i8> {
    if !(0..=8).contains(&mode) {
        return Err(DecodeError::InvalidSyntax("intra4x4 pred mode"));
    }
    let idx = CACHE30_SCAN_IDX[i];
    let left_avail = sample_avail[idx - 1];
    let top_avail = sample_avail[idx - 6];
    let left_top_avail = sample_avail[idx - 7];
    let right_top_avail = sample_avail[idx - 5];

    if mode == 2 {
        // DC
        return Ok(if left_avail != 0 && top_avail != 0 {
            2
        } else if left_avail != 0 {
            9 // DC_L
        } else if top_avail != 0 {
            10 // DC_T
        } else {
            11 // DC_128
        });
    }

    let (pm, nl, nt, nlt) = I4_PRED_INFO[mode as usize];
    if mode != pm || left_avail < nl || top_avail < nt || left_top_avail < nlt {
        return Err(DecodeError::InvalidSyntax("intra4x4 pred mode unavail"));
    }
    let mut final_mode = mode;
    if mode == 3 && right_top_avail == 0 {
        final_mode = 12; // DDL_TOP
    } else if mode == 7 && right_top_avail == 0 {
        final_mode = 13; // VL_TOP
    }
    Ok(final_mode)
}

/// `CheckIntra16x16PredMode`. `neigh_avail` = (left<<2)|(left_top<<1)|top.
fn check_intra16x16_mode(neigh_avail: i32, mode: &mut i8) -> Result<()> {
    let left = neigh_avail & 0x04;
    let left_top = neigh_avail & 0x02;
    let top = neigh_avail & 0x01;
    if !(0..=3).contains(mode) {
        return Err(DecodeError::InvalidSyntax("i16x16 pred mode"));
    }
    if *mode == 2 {
        *mode = if left != 0 && top != 0 {
            2
        } else if left != 0 {
            4 // DC_L
        } else if top != 0 {
            5 // DC_T
        } else {
            6 // DC_128
        };
        return Ok(());
    }
    let (pm, nl, nt, nlt) = I16_PRED_INFO[*mode as usize];
    if *mode != pm || left < nl || top < nt || left_top < nlt {
        return Err(DecodeError::InvalidSyntax("i16x16 pred mode unavail"));
    }
    Ok(())
}

/// `CheckIntraChromaPredMode`. `neigh_avail` = (left<<2)|(left_top<<1)|top.
fn check_intra_chroma_mode(neigh_avail: i32, mode: &mut i8) -> Result<()> {
    let left = neigh_avail & 0x04;
    let left_top = neigh_avail & 0x02;
    let top = neigh_avail & 0x01;
    if *mode == 0 {
        *mode = if left != 0 && top != 0 {
            0
        } else if left != 0 {
            4 // DC_L
        } else if top != 0 {
            5 // DC_T
        } else {
            6 // DC_128
        };
        return Ok(());
    }
    let (pm, nl, nt, nlt) = CHROMA_PRED_INFO[*mode as usize];
    if *mode != pm || left < nl || top < nt || left_top < nlt {
        return Err(DecodeError::InvalidSyntax("chroma pred mode unavail"));
    }
    Ok(())
}

/// nC for a luma 4x4 block (raster bx,by) from the already-decoded current MB
/// counts and the left/top neighbour snapshots.
fn nc_luma(cur: &[i8; 16], left: &NeighborSnap, top: &NeighborSnap, bx: usize, by: usize) -> i32 {
    let na = if bx > 0 {
        cur[by * 4 + bx - 1] as i32
    } else if left.avail {
        left.nzc_luma[by * 4 + 3] as i32
    } else {
        -1
    };
    let nb = if by > 0 {
        cur[(by - 1) * 4 + bx] as i32
    } else if top.avail {
        top.nzc_luma[12 + bx] as i32
    } else {
        -1
    };
    nc_average(na, nb)
}

/// nC for a chroma 4x4 block (component c, raster bx,by within the 2x2 grid).
fn nc_chroma(
    cur: &[i8; 8],
    left: &NeighborSnap,
    top: &NeighborSnap,
    c: usize,
    bx: usize,
    by: usize,
) -> i32 {
    let base = c * 4;
    let na = if bx > 0 {
        cur[base + by * 2 + bx - 1] as i32
    } else if left.avail {
        left.nzc_chroma[base + by * 2 + 1] as i32
    } else {
        -1
    };
    let nb = if by > 0 {
        cur[base + (by - 1) * 2 + bx] as i32
    } else if top.avail {
        top.nzc_chroma[base + 2 + bx] as i32
    } else {
        -1
    };
    nc_average(na, nb)
}

#[allow(clippy::too_many_arguments)]
fn parse_residuals(
    bs: &mut BitReader<'_>,
    mb_type: MbType,
    cbp_l: u8,
    cbp_c: u8,
    luma_qp: i32,
    chroma_qp: &[i32; 2],
    left: &NeighborSnap,
    top: &NeighborSnap,
    cur_nzc_luma: &mut [i8; 16],
    cur_nzc_chroma: &mut [i8; 8],
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let deq_l = &G_KUI_DEQUANT_COEFF[luma_qp as usize];
    let mut out = [0i32; 16];

    if mb_type == MbType::Intra16x16 {
        // Luma DC (16 coeffs, luma-DC zig-zag, then Hadamard dequant-IDCT).
        let nc0 = nc_luma(cur_nzc_luma, left, top, 0, 0);
        out.fill(0);
        residual_block_cavlc(bs, nc0, 16, &mut out)?;
        for (s, &lvl) in out.iter().enumerate() {
            coeffs[G_KUI_LUMA_DC_ZIGZAG_SCAN[s] as usize] = lvl as i16;
        }
        luma_dc_dequant_idct(coeffs, luma_qp);

        // Luma AC (cbp_l is 0 or 15 for I16x16; if set, all 16 blocks).
        if cbp_l != 0 {
            for i in 0..16 {
                let raster = BLOCK_RASTER[i];
                let nc = nc_luma(cur_nzc_luma, left, top, BLOCK_BX[i], BLOCK_BY[i]);
                out.fill(0);
                let total = residual_block_cavlc(bs, nc, 15, &mut out)?;
                let base = i * 16;
                for s in 0..15 {
                    if out[s] != 0 {
                        let j = G_KUI_ZIGZAG_SCAN[s + 1] as usize;
                        coeffs[base + j] = (out[s] * deq_l[j & 7] as i32) as i16;
                    }
                }
                cur_nzc_luma[raster] = total as i8;
            }
        }
    } else {
        // I_4x4: per 8x8, DC+AC together (full 16-coeff zig-zag).
        for id8 in 0..4 {
            if cbp_l & (1 << id8) == 0 {
                continue;
            }
            for id4 in 0..4 {
                let i = id8 * 4 + id4;
                let raster = BLOCK_RASTER[i];
                let nc = nc_luma(cur_nzc_luma, left, top, BLOCK_BX[i], BLOCK_BY[i]);
                out.fill(0);
                let total = residual_block_cavlc(bs, nc, 16, &mut out)?;
                let base = i * 16;
                for s in 0..16 {
                    if out[s] != 0 {
                        let j = G_KUI_ZIGZAG_SCAN[s] as usize;
                        coeffs[base + j] = (out[s] * deq_l[j & 7] as i32) as i16;
                    }
                }
                cur_nzc_luma[raster] = total as i8;
            }
        }
    }

    // Chroma DC (Cb, Cr) when cbp_c is 1 or 2.
    if cbp_c == 1 || cbp_c == 2 {
        for c in 0..2 {
            let cbase = 256 + c * 64;
            out.fill(0);
            residual_block_cavlc(bs, -1, 4, &mut out)?;
            for k in 0..4 {
                coeffs[cbase + G_KUI_CHROMA_DC_SCAN[k] as usize] = out[k] as i16;
            }
            chroma_dc_idct(&mut coeffs[cbase..cbase + 64]);
            let qmul = G_KUI_DEQUANT_COEFF[chroma_qp[c] as usize][0] as i32;
            for k in 0..4 {
                let j = cbase + G_KUI_CHROMA_DC_SCAN[k] as usize;
                coeffs[j] = ((coeffs[j] as i32 * qmul) >> 1) as i16;
            }
        }
    }

    // Chroma AC when cbp_c == 2.
    if cbp_c == 2 {
        for c in 0..2 {
            let deq_c = &G_KUI_DEQUANT_COEFF[chroma_qp[c] as usize];
            for b in 0..4 {
                let bx = b % 2;
                let by = b / 2;
                let nc = nc_chroma(cur_nzc_chroma, left, top, c, bx, by);
                out.fill(0);
                let total = residual_block_cavlc(bs, nc, 15, &mut out)?;
                let base = 256 + c * 64 + b * 16;
                for s in 0..15 {
                    if out[s] != 0 {
                        let j = G_KUI_ZIGZAG_SCAN[s + 1] as usize;
                        coeffs[base + j] = (out[s] * deq_c[j & 7] as i32) as i16;
                    }
                }
                cur_nzc_chroma[c * 4 + b] = total as i8;
            }
        }
    }

    Ok(())
}

/// `WelsLumaDcDequantIdct`: inverse Hadamard + dequant of the 16 luma DC
/// coefficients (scattered at element offsets `block*16` within `coeffs`).
fn luma_dc_dequant_idct(coeffs: &mut [i16; 384], qp: i32) {
    const STRIDE: usize = 16;
    let qmul = (G_KUI_DEQUANT_COEFF[qp as usize][0] as i32) << 4;
    let x_off = [0usize, STRIDE, STRIDE << 2, 5 * STRIDE];
    let y_off = [0usize, STRIDE << 1, STRIDE << 3, 10 * STRIDE];
    let mut tmp = [0i32; 16];

    for i in 0..4 {
        let off = y_off[i];
        let x1 = off + x_off[2];
        let x2 = STRIDE + off;
        let x3 = off + x_off[3];
        let z0 = coeffs[off] as i32 + coeffs[x1] as i32;
        let z1 = coeffs[off] as i32 - coeffs[x1] as i32;
        let z2 = coeffs[x2] as i32 - coeffs[x3] as i32;
        let z3 = coeffs[x2] as i32 + coeffs[x3] as i32;
        tmp[i * 4] = z0 + z3;
        tmp[1 + i * 4] = z1 + z2;
        tmp[2 + i * 4] = z1 - z2;
        tmp[3 + i * 4] = z0 - z3;
    }

    for i in 0..4 {
        let off = x_off[i];
        let i4 = 4 + i;
        let z0 = tmp[i] + tmp[4 + i4];
        let z1 = tmp[i] - tmp[4 + i4];
        let z2 = tmp[i4] - tmp[8 + i4];
        let z3 = tmp[i4] + tmp[8 + i4];
        coeffs[off] = (((z0 + z3) * qmul + (1 << 5)) >> 6) as i16;
        coeffs[y_off[1] + off] = (((z1 + z2) * qmul + (1 << 5)) >> 6) as i16;
        coeffs[y_off[2] + off] = (((z1 - z2) * qmul + (1 << 5)) >> 6) as i16;
        coeffs[y_off[3] + off] = (((z0 - z3) * qmul + (1 << 5)) >> 6) as i16;
    }
}

/// `WelsChromaDcIdct`: inverse 2x2 Hadamard of the chroma DC block (samples at
/// element offsets {0,16,32,48} within `block`).
fn chroma_dc_idct(block: &mut [i16]) {
    let x = 16usize;
    let s = 32usize;
    let s1 = x + s;
    let a = block[0] as i32;
    let b = block[x] as i32;
    let c = block[s] as i32;
    let d = block[s1] as i32;
    let e = a - b;
    let a2 = a + b;
    let b2 = c - d;
    let c2 = c + d;
    block[0] = (a2 + c2) as i16;
    block[x] = (e + b2) as i16;
    block[s] = (a2 - c2) as i16;
    block[s1] = (e - b2) as i16;
}

// ===================== P-slice (inter) macroblock parse =====================

use super::context::SubMbType;
use super::mv_pred::{
    pred_inter16x8, pred_inter8x16, pred_mv, pred_p_skip_mv, REF_NOT_AVAIL, REF_NOT_IN_LIST, SCAN4,
};

// g_kuiInterCbpTable.
#[rustfmt::skip]
const INTER_CBP_TABLE: [u8; 48] = [
    0, 16,  1,  2,  4,  8, 32,  3,  5, 10, 12, 15, 47,  7, 11, 13,
    14,  6,  9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// The 30-entry list-0 neighbour MV / ref-index cache (`WelsFillCacheInter`).
struct InterCache {
    mv: [[i16; 2]; 30],
    ref_idx: [i8; 30],
}

impl InterCache {
    /// Build the neighbour cache for `mb_xy` from the surrounding decoded MBs.
    fn build(ctx: &DecoderContext, mb_xy: usize) -> Self {
        let mb_width = ctx.mb_width;
        let mb_x = mb_xy % mb_width;
        let mb_y = mb_xy / mb_width;
        let cur = ctx.slice_idc[mb_xy];

        let avail = |cond: bool, xy: usize| -> bool { cond && ctx.slice_idc[xy] == cur };
        let left = mb_x != 0 && avail(true, mb_xy - 1);
        let top = mb_y != 0 && avail(true, mb_xy - mb_width);
        let left_top = mb_x != 0 && mb_y != 0 && avail(true, mb_xy.wrapping_sub(mb_width + 1));
        let right_top =
            mb_x != mb_width - 1 && mb_y != 0 && avail(true, mb_xy + 1 - mb_width);

        let left_xy = mb_xy.wrapping_sub(1);
        let top_xy = mb_xy.wrapping_sub(mb_width);
        let left_top_xy = mb_xy.wrapping_sub(mb_width + 1);
        let right_top_xy = mb_xy + 1 - mb_width;

        let mut mv = [[0i16; 2]; 30];
        let mut ref_idx = [REF_NOT_AVAIL; 30];

        let mv_of = |xy: usize, b: usize| -> [i16; 2] {
            let base = (xy * 16 + b) * 2;
            [ctx.mv[base], ctx.mv[base + 1]]
        };
        let ref_of = |xy: usize, b: usize| -> i8 { ctx.ref_idx[xy * 16 + b] };

        // Left column (cache 6,12,18,24 <- neighbour blocks 3,7,11,15).
        if left && ctx.mb_type[left_xy].is_inter() {
            for (k, &b) in [3usize, 7, 11, 15].iter().enumerate() {
                let c = [6, 12, 18, 24][k];
                mv[c] = mv_of(left_xy, b);
                ref_idx[c] = ref_of(left_xy, b);
            }
        } else {
            let r = if left { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
            for &c in &[6usize, 12, 18, 24] {
                ref_idx[c] = r;
            }
        }
        // Left-top (cache 0 <- block 15).
        if left_top && ctx.mb_type[left_top_xy].is_inter() {
            mv[0] = mv_of(left_top_xy, 15);
            ref_idx[0] = ref_of(left_top_xy, 15);
        } else {
            ref_idx[0] = if left_top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
        }
        // Top row (cache 1,2,3,4 <- blocks 12,13,14,15).
        if top && ctx.mb_type[top_xy].is_inter() {
            for (k, &b) in [12usize, 13, 14, 15].iter().enumerate() {
                let c = 1 + k;
                mv[c] = mv_of(top_xy, b);
                ref_idx[c] = ref_of(top_xy, b);
            }
        } else {
            let r = if top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
            for c in 1..=4 {
                ref_idx[c] = r;
            }
        }
        // Right-top (cache 5 <- block 12).
        if right_top && ctx.mb_type[right_top_xy].is_inter() {
            mv[5] = mv_of(right_top_xy, 12);
            ref_idx[5] = ref_of(right_top_xy, 12);
        } else {
            ref_idx[5] = if right_top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
        }
        // Interior right-edge cells: always unavailable / zero.
        for &c in &[9usize, 11, 17, 21, 23] {
            ref_idx[c] = REF_NOT_AVAIL;
            mv[c] = [0, 0];
        }

        InterCache { mv, ref_idx }
    }
}

/// Store one motion vector + reference index into both the current MB's per-4x4
/// raster arrays (in `ctx`) and the neighbour cache, over a `w`x`h` block of
/// 4x4 cells anchored at raster index `scan4` / cache index `cache`.
#[allow(clippy::too_many_arguments)]
fn store_block(
    ctx: &mut DecoderContext,
    cache: &mut InterCache,
    mb_xy: usize,
    scan4: usize,
    cache_idx: usize,
    mv: [i16; 2],
    iref: i8,
    ref_pic_id: i32,
    w: usize,
    h: usize,
) {
    for by in 0..h {
        for bx in 0..w {
            let raster = scan4 + by * 4 + bx;
            let base = (mb_xy * 16 + raster) * 2;
            ctx.mv[base] = mv[0];
            ctx.mv[base + 1] = mv[1];
            ctx.ref_idx[mb_xy * 16 + raster] = iref;
            ctx.ref_pic_id[mb_xy * 16 + raster] = ref_pic_id;
            let c = cache_idx + by * 6 + bx;
            cache.mv[c] = mv;
            cache.ref_idx[c] = iref;
        }
    }
}

/// Parse one P-slice macroblock (CAVLC). Handles `mb_skip_run`, the inter
/// partition kinds, intra MBs appearing in P slices, cbp and residuals.
///
/// `skip_run` carries `pSlice->iMbSkipRun` across MBs (-1 == "read a fresh
/// run"). `ref_pic_ids` maps slice-local list-0 ref indices to a stable
/// reference-picture identity for the deblocker; its length is the active
/// list-0 reference count.
#[allow(clippy::too_many_arguments)]
pub fn parse_p_mb_cavlc(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    mb_xy: usize,
    pps: &Pps,
    last_mb_qp: &mut i32,
    skip_run: &mut i32,
    ref_pic_ids: &[i32],
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let ref_count = ref_pic_ids.len();

    if *skip_run == -1 {
        *skip_run = bs.read_ue()? as i32;
    }
    let old = *skip_run;
    *skip_run -= 1;
    if old != 0 {
        // P_Skip macroblock.
        let mv = pred_p_skip_mv(ctx, mb_xy);
        let ref_pic_id = if ref_count > 0 { ref_pic_ids[0] } else { -1 };
        for raster in 0..16 {
            let base = (mb_xy * 16 + raster) * 2;
            ctx.mv[base] = mv[0];
            ctx.mv[base + 1] = mv[1];
            ctx.ref_idx[mb_xy * 16 + raster] = 0;
            ctx.ref_pic_id[mb_xy * 16 + raster] = ref_pic_id;
        }
        let luma_qp = *last_mb_qp;
        commit_inter_meta(ctx, mb_xy, MbType::PSkip, 0, luma_qp, pps, &[0; 16], &[0; 8]);
        return Ok(());
    }

    let ui_mb_type = bs.read_ue()?;
    if ui_mb_type < 5 {
        parse_inter_mb(bs, ctx, mb_xy, pps, last_mb_qp, ui_mb_type, ref_pic_ids, coeffs)
    } else {
        // Intra MB inside a P slice: reuse the intra core with the -5 offset.
        parse_intra_mb_core(bs, ctx, mb_xy, pps, last_mb_qp, coeffs, ui_mb_type - 5)
    }
}

/// Commit per-MB inter metadata (type/cbp/qp/nzc) into `ctx`.
fn commit_inter_meta(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    mb_type: MbType,
    cbp: u8,
    luma_qp: i32,
    pps: &Pps,
    nzc_luma: &[i8; 16],
    nzc_chroma: &[i8; 8],
) {
    let chroma_qp = [
        CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[0], 0, 51) as usize] as i8,
        CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[1], 0, 51) as usize] as i8,
    ];
    ctx.mb_type[mb_xy] = mb_type;
    ctx.cbp[mb_xy] = cbp;
    ctx.luma_qp[mb_xy] = luma_qp as i8;
    ctx.chroma_qp[mb_xy * 2] = chroma_qp[0];
    ctx.chroma_qp[mb_xy * 2 + 1] = chroma_qp[1];
    ctx.nzc_luma[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(nzc_luma);
    ctx.nzc_chroma[mb_xy * 8..mb_xy * 8 + 8].copy_from_slice(nzc_chroma);
}

/// Parse a genuine inter macroblock (`ui_mb_type` 0..4): motion then residual.
#[allow(clippy::too_many_arguments)]
fn parse_inter_mb(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    mb_xy: usize,
    pps: &Pps,
    last_mb_qp: &mut i32,
    ui_mb_type: u32,
    ref_pic_ids: &[i32],
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let mb_type = match ui_mb_type {
        0 => MbType::Inter16x16,
        1 => MbType::Inter16x8,
        2 => MbType::Inter8x16,
        3 => MbType::Inter8x8,
        _ => MbType::Inter8x8Ref0,
    };
    let ref_count = ref_pic_ids.len();
    let mut cache = InterCache::build(ctx, mb_xy);

    parse_inter_motion(bs, ctx, &mut cache, mb_xy, mb_type, ref_count, ref_pic_ids)?;

    // coded_block_pattern (inter mapping).
    let ui_cbp = bs.read_ue()?;
    if ui_cbp > 47 {
        return Err(DecodeError::InvalidSyntax("inter cbp"));
    }
    let cbp = INTER_CBP_TABLE[ui_cbp as usize];
    let cbp_l = cbp & 0x0f;
    let cbp_c = cbp >> 4;

    // QP / residual.
    let luma_qp: i32;
    let mut cur_nzc_luma = [0i8; 16];
    let mut cur_nzc_chroma = [0i8; 8];
    if cbp != 0 {
        coeffs.iter_mut().for_each(|c| *c = 0);
        let qp_delta = bs.read_se()?;
        if !(-26..=25).contains(&qp_delta) {
            return Err(DecodeError::InvalidSyntax("mb_qp_delta"));
        }
        luma_qp = (*last_mb_qp + qp_delta + 52) % 52;
        *last_mb_qp = luma_qp;

        let neigh = ctx.neighbors(mb_xy);
        let left = NeighborSnap::from_ctx(ctx, neigh.left, neigh.left_xy);
        let top = NeighborSnap::from_ctx(ctx, neigh.top, neigh.top_xy);
        let chroma_qp = [
            CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[0], 0, 51) as usize] as i32,
            CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[1], 0, 51) as usize] as i32,
        ];
        parse_residuals(
            bs, mb_type, cbp_l, cbp_c, luma_qp, &chroma_qp, &left, &top, &mut cur_nzc_luma,
            &mut cur_nzc_chroma, coeffs,
        )?;
    } else {
        luma_qp = *last_mb_qp;
    }

    commit_inter_meta(ctx, mb_xy, mb_type, cbp, luma_qp, pps, &cur_nzc_luma, &cur_nzc_chroma);
    Ok(())
}

/// Parse `ref_idx_l0` (te) + `mvd_l0` (se×2) for each partition, reconstruct the
/// MV via the predictor, and store into `ctx` + the neighbour cache.
/// (`ParseInterInfo`).
fn parse_inter_motion(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    cache: &mut InterCache,
    mb_xy: usize,
    mb_type: MbType,
    ref_count: usize,
    ref_pic_ids: &[i32],
) -> Result<()> {
    let read_ref = |bs: &mut BitReader<'_>| -> Result<i8> {
        let v = bs.read_te(ref_count as u32)?;
        if v as usize >= ref_count {
            return Err(DecodeError::InvalidSyntax("ref_idx_l0"));
        }
        Ok(v as i8)
    };
    let read_mvd = |bs: &mut BitReader<'_>| -> Result<[i16; 2]> {
        let x = bs.read_se()? as i16;
        let y = bs.read_se()? as i16;
        Ok([x, y])
    };

    match mb_type {
        MbType::Inter16x16 => {
            let iref = read_ref(bs)?;
            let mvp = pred_mv(&cache.mv, &cache.ref_idx, 0, 4, iref);
            let mvd = read_mvd(bs)?;
            let mv = [mvp[0] + mvd[0], mvp[1] + mvd[1]];
            store_block(ctx, cache, mb_xy, 0, 0, mv, iref, ref_pic_ids[iref as usize], 4, 4);
        }
        MbType::Inter16x8 => {
            let r = [read_ref(bs)?, read_ref(bs)?];
            for i in 0..2 {
                let part_idx = i << 3;
                let mvp = pred_inter16x8(&cache.mv, &cache.ref_idx, part_idx, r[i]);
                let mvd = read_mvd(bs)?;
                let mv = [mvp[0] + mvd[0], mvp[1] + mvd[1]];
                update_p16x8(ctx, cache, mb_xy, part_idx, mv, r[i], ref_pic_ids[r[i] as usize]);
            }
        }
        MbType::Inter8x16 => {
            let r = [read_ref(bs)?, read_ref(bs)?];
            for i in 0..2 {
                let part_idx = i << 2;
                let mvp = pred_inter8x16(&cache.mv, &cache.ref_idx, part_idx, r[i]);
                let mvd = read_mvd(bs)?;
                let mv = [mvp[0] + mvd[0], mvp[1] + mvd[1]];
                update_p8x16(ctx, cache, mb_xy, part_idx, mv, r[i], ref_pic_ids[r[i] as usize]);
            }
        }
        MbType::Inter8x8 | MbType::Inter8x8Ref0 => {
            let ref0 = mb_type == MbType::Inter8x8Ref0;
            let eff_ref_count = if ref0 { 1 } else { ref_count };
            let mut subs = [SubMbType::P8x8; 4];
            for s in subs.iter_mut() {
                let st = bs.read_ue()?;
                if st >= 4 {
                    return Err(DecodeError::InvalidSyntax("sub_mb_type"));
                }
                *s = match st {
                    0 => SubMbType::P8x8,
                    1 => SubMbType::P8x4,
                    2 => SubMbType::P4x8,
                    _ => SubMbType::P4x4,
                };
            }
            ctx.sub_mb_type[mb_xy * 4..mb_xy * 4 + 4].copy_from_slice(&subs);

            // ref_idx for each 8x8.
            let mut iref = [0i8; 4];
            if !ref0 {
                for r in iref.iter_mut() {
                    let v = bs.read_te(eff_ref_count as u32)?;
                    if v as usize >= eff_ref_count {
                        return Err(DecodeError::InvalidSyntax("ref_idx_l0 8x8"));
                    }
                    *r = v as i8;
                }
            }

            for i in 0..4 {
                let i_idx = i << 2; // block-scan index of the 8x8's top-left
                let scan4_8 = SCAN4[i_idx];
                let cache8 = CACHE30_SCAN_IDX[i_idx];
                let ref_pic = ref_pic_ids[iref[i] as usize];
                // Reference index fills the whole 8x8 (cache + ctx) up front.
                cache.ref_idx[cache8] = iref[i];
                cache.ref_idx[cache8 + 1] = iref[i];
                cache.ref_idx[cache8 + 6] = iref[i];
                cache.ref_idx[cache8 + 7] = iref[i];
                for &raster in &[scan4_8, scan4_8 + 1, scan4_8 + 4, scan4_8 + 5] {
                    ctx.ref_idx[mb_xy * 16 + raster] = iref[i];
                    ctx.ref_pic_id[mb_xy * 16 + raster] = ref_pic;
                }

                let (part_count, part_w) = subs[i].part_info();
                for j in 0..part_count {
                    let part_idx = i_idx + j * part_w;
                    let scan4 = SCAN4[part_idx];
                    let cache_idx = CACHE30_SCAN_IDX[part_idx];
                    let mvp = pred_mv(&cache.mv, &cache.ref_idx, part_idx, part_w, iref[i]);
                    let x = bs.read_se()? as i16;
                    let y = bs.read_se()? as i16;
                    let mv = [mvp[0] + x, mvp[1] + y];
                    let (w, h) = match subs[i] {
                        SubMbType::P8x8 => (2, 2),
                        SubMbType::P8x4 => (2, 1),
                        SubMbType::P4x8 => (1, 2),
                        SubMbType::P4x4 => (1, 1),
                    };
                    store_mv_only(ctx, cache, mb_xy, scan4, cache_idx, mv, w, h);
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Store an MV (no ref) across a `w`x`h` 4x4 block (`ctx` + cache).
fn store_mv_only(
    ctx: &mut DecoderContext,
    cache: &mut InterCache,
    mb_xy: usize,
    scan4: usize,
    cache_idx: usize,
    mv: [i16; 2],
    w: usize,
    h: usize,
) {
    for by in 0..h {
        for bx in 0..w {
            let raster = scan4 + by * 4 + bx;
            let base = (mb_xy * 16 + raster) * 2;
            ctx.mv[base] = mv[0];
            ctx.mv[base + 1] = mv[1];
            let c = cache_idx + by * 6 + bx;
            cache.mv[c] = mv;
        }
    }
}

/// `UpdateP16x8MotionInfo` for partition `part_idx` (0 or 8).
fn update_p16x8(
    ctx: &mut DecoderContext,
    cache: &mut InterCache,
    mb_xy: usize,
    part_idx: usize,
    mv: [i16; 2],
    iref: i8,
    ref_pic_id: i32,
) {
    let mut p = part_idx;
    for _ in 0..2 {
        let scan4 = SCAN4[p];
        let cache_idx = CACHE30_SCAN_IDX[p];
        store_block(ctx, cache, mb_xy, scan4, cache_idx, mv, iref, ref_pic_id, 2, 2);
        p += 4;
    }
}

/// `UpdateP8x16MotionInfo` for partition `part_idx` (0 or 4).
fn update_p8x16(
    ctx: &mut DecoderContext,
    cache: &mut InterCache,
    mb_xy: usize,
    part_idx: usize,
    mv: [i16; 2],
    iref: i8,
    ref_pic_id: i32,
) {
    let mut p = part_idx;
    for _ in 0..2 {
        let scan4 = SCAN4[p];
        let cache_idx = CACHE30_SCAN_IDX[p];
        store_block(ctx, cache, mb_xy, scan4, cache_idx, mv, iref, ref_pic_id, 2, 2);
        p += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nc_average_matches_macro() {
        assert_eq!(nc_average(-1, -1), 0);
        assert_eq!(nc_average(5, -1), 5);
        assert_eq!(nc_average(-1, 3), 3);
        assert_eq!(nc_average(5, 3), 4); // (5+3+1)>>1
        assert_eq!(nc_average(0, 0), 0); // (0+0+1)>>1
    }

    #[test]
    fn chroma_dc_idct_dc_only() {
        // A single DC term propagates to all four outputs unchanged in sign.
        let mut blk = [0i16; 64];
        blk[0] = 10;
        chroma_dc_idct(&mut blk);
        assert_eq!(blk[0], 10);
        assert_eq!(blk[16], 10);
        assert_eq!(blk[32], 10);
        assert_eq!(blk[48], 10);
    }

    #[test]
    fn i16_dc_mode_resolves_dc128_without_neighbors() {
        let mut m = 2i8;
        check_intra16x16_mode(0, &mut m).unwrap();
        assert_eq!(m, 6); // DC_128
        let mut m = 2i8;
        check_intra16x16_mode(0x04 | 0x01, &mut m).unwrap();
        assert_eq!(m, 2); // both available -> plain DC
    }

    #[test]
    fn block_raster_is_a_permutation() {
        let mut seen = [false; 16];
        for &r in &BLOCK_RASTER {
            assert!(!seen[r]);
            seen[r] = true;
        }
        // bx/by agree with raster.
        for i in 0..16 {
            assert_eq!(BLOCK_BY[i] * 4 + BLOCK_BX[i], BLOCK_RASTER[i]);
        }
    }
}
