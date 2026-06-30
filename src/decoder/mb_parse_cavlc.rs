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
    G_KUI_CHROMA_DC_SCAN, G_KUI_DEQUANT_COEFF, G_KUI_DEQUANT_COEFF8X8, G_KUI_LUMA_DC_ZIGZAG_SCAN,
    G_KUI_ZIGZAG_SCAN, G_KUI_ZIGZAG_SCAN8X8,
};
use crate::error::DecodeError;

use super::cavlc::residual_block_cavlc;
use super::context::{DecoderContext, MbType, NeighborAvail};
use super::params::Pps;

type Result<T> = core::result::Result<T, DecodeError>;

// ---------------------------------------------------------------------------
// Parameter-bundle structs (keep parse signatures <= 7 args). Each is
// destructured back into the original locals on the first line(s) of the body,
// leaving all arithmetic unchanged.
// ---------------------------------------------------------------------------

/// `ctx` + per-MB location + active PPS.
pub struct MbCtx<'a> {
    pub ctx: &'a mut DecoderContext,
    pub mb_xy: usize,
    pub pps: &'a Pps,
}

/// Residual-decode inputs (mb_type, luma/chroma cbp split, qp + chroma qp).
#[derive(Clone, Copy)]
struct ResidualParams {
    mb_type: MbType,
    cbp_l: u8,
    cbp_c: u8,
    luma_qp: i32,
    chroma_qp: [i32; 2],
    /// Luma residual uses the 8x8 transform (`transform_size_8x8_flag`).
    transform_8x8: bool,
}

/// Left/top neighbour snapshots for nC derivation.
#[derive(Clone, Copy)]
struct Neighbours<'a> {
    left: &'a NeighborSnap,
    top: &'a NeighborSnap,
}

/// Mutable per-MB residual outputs (nzc counts).
struct ResidualOut<'a> {
    cur_nzc_luma: &'a mut [i8; 16],
    cur_nzc_chroma: &'a mut [i8; 8],
}

/// Placement of a `w`x`h` block of 4x4 cells: raster anchor + 30-cache anchor.
#[derive(Clone, Copy)]
struct BlockPlace {
    scan4: usize,
    cache_idx: usize,
    w: usize,
    h: usize,
}

// --- Per-MB 4x4 block geometry (luma) -------------------------------------
// Block scan index i (the order residuals are coded, == g_kuiScan8 order) maps
// to a raster position (bx,by) within the macroblock.
pub(super) const BLOCK_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
pub(super) const BLOCK_BX: [usize; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
pub(super) const BLOCK_BY: [usize; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];

// g_kuiCache30ScanIdx: block i -> position in the 30-entry (6-wide) sample
// availability grid used by CheckIntraNxNPredMode.
pub(super) const CACHE30_SCAN_IDX: [usize; 16] =
    [7, 8, 13, 14, 9, 10, 15, 16, 19, 20, 25, 26, 21, 22, 27, 28];

// g_kuiI16CbpTable.
pub(super) const I16_CBP_TABLE: [u8; 6] = [0, 16, 32, 15, 31, 47];

// g_kuiIntra4x4CbpTable (chroma_format_idc != 0).
#[rustfmt::skip]
pub(super) const INTRA4X4_CBP_TABLE: [u8; 48] = [
    47, 31, 15,  0, 23, 27, 29, 30,  7, 11, 13, 14, 39, 43, 45, 46,
    16,  3,  5, 10, 12, 19, 21, 26, 28, 35, 37, 42, 44,  1,  2,  4,
     8, 17, 18, 20, 24,  6,  9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

// g_kuiChromaQpTable.
#[rustfmt::skip]
pub(super) const CHROMA_QP_TABLE: [u8; 52] = [
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
pub(super) fn clip3(x: i32, lo: i32, hi: i32) -> i32 {
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
        s.best_mode
            .copy_from_slice(&ctx.i4_best_mode[xy * 16..xy * 16 + 16]);
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

/// Apply `constrained_intra_pred_flag`: a neighbouring macroblock that is not
/// intra-coded is unavailable for intra prediction (spec 8.3). Returns the
/// availability with inter neighbours masked out; a no-op when `constrained`
/// is false or in I slices (all neighbours intra).
pub(super) fn constrain_intra_avail(
    ctx: &DecoderContext,
    mb_xy: usize,
    mut n: NeighborAvail,
    constrained: bool,
) -> NeighborAvail {
    if !constrained {
        return n;
    }
    let mb_width = ctx.mb_width;
    if n.left && !ctx.mb_type[n.left_xy].is_intra() {
        n.left = false;
    }
    if n.top && !ctx.mb_type[n.top_xy].is_intra() {
        n.top = false;
    }
    if n.top_left && !ctx.mb_type[mb_xy - mb_width - 1].is_intra() {
        n.top_left = false;
    }
    if n.top_right && !ctx.mb_type[mb_xy - mb_width + 1].is_intra() {
        n.top_right = false;
    }
    n
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
    // Geometric snapshots: used for residual non-zero-count (nC) derivation,
    // which is NOT affected by constrained_intra_pred.
    let left = NeighborSnap::from_ctx(ctx, neigh.left, neigh.left_xy);
    let top = NeighborSnap::from_ctx(ctx, neigh.top, neigh.top_xy);
    // Constrained availability for intra prediction: with
    // constrained_intra_pred_flag, inter-coded neighbours are unavailable
    // (spec 8.3.x / WelsFillCache*Constrain1*).
    let neigh_c = constrain_intra_avail(ctx, mb_xy, neigh, pps.constrained_intra_pred_flag);
    let left_c = NeighborSnap::from_ctx(ctx, neigh_c.left, neigh.left_xy);
    let top_c = NeighborSnap::from_ctx(ctx, neigh_c.top, neigh.top_xy);

    // Current-MB block state, accumulated locally then written back to ctx.
    let mut cur_nzc_luma = [0i8; 16];
    let mut cur_nzc_chroma = [0i8; 8];
    let mut best_mode = [-1i8; 16];
    let mut final_mode = [2i8; 16];

    let mb_type;
    let mut i16_mode = 0i8;
    let mut chroma_mode;
    let cbp: u8;
    let mut transform_8x8 = false;
    let mut i8_avail = 0u8;

    if ui_mb_type > 25 {
        return Err(DecodeError::InvalidSyntax("intra mb_type"));
    }
    if ui_mb_type == 25 {
        return parse_pcm_mb_cavlc(bs, ctx, mb_xy);
    }

    if ui_mb_type == 0 {
        // I_NxN: transform_size_8x8 (I_8x8) is High-profile only.
        if pps.transform_8x8_mode_flag {
            transform_8x8 = bs.read_flag()?;
        }
        mb_type = MbType::Intra4x4;
        if transform_8x8 {
            let (cm, avail8) = parse_intra8x8(
                bs,
                &neigh_c,
                &left_c,
                &top_c,
                &mut best_mode,
                &mut final_mode,
            )?;
            chroma_mode = cm;
            i8_avail = avail8;
        } else {
            chroma_mode = parse_intra4x4(
                bs,
                &neigh_c,
                &left_c,
                &top_c,
                &mut best_mode,
                &mut final_mode,
            )?;
        }

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

        let neigh_avail =
            ((neigh_c.left as i32) << 2) | ((neigh_c.top_left as i32) << 1) | (neigh_c.top as i32);
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
            ResidualParams {
                mb_type,
                cbp_l,
                cbp_c,
                luma_qp,
                chroma_qp,
                transform_8x8,
            },
            Neighbours {
                left: &left,
                top: &top,
            },
            ResidualOut {
                cur_nzc_luma: &mut cur_nzc_luma,
                cur_nzc_chroma: &mut cur_nzc_chroma,
            },
            coeffs,
        )?;
    }

    // Commit MB state to the context.
    ctx.mb_type[mb_xy] = mb_type;
    ctx.transform_8x8[mb_xy] = transform_8x8;
    ctx.i8_avail[mb_xy] = i8_avail;
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

/// Commit per-MB state for an I_PCM macroblock (spec 8.5 / Rec. 9.2.1): QP = 0,
/// nnz = 16 per block, no motion. Shared by the CAVLC and CABAC parse paths.
pub(super) fn commit_pcm_state(ctx: &mut DecoderContext, mb_xy: usize) {
    ctx.mb_type[mb_xy] = MbType::IPcm;
    ctx.i16_mode[mb_xy] = 0;
    ctx.chroma_mode[mb_xy] = 0;
    ctx.cbp[mb_xy] = 0;
    ctx.luma_qp[mb_xy] = 0;
    ctx.chroma_qp[mb_xy * 2] = 0;
    ctx.chroma_qp[mb_xy * 2 + 1] = 0;
    ctx.nzc_luma[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&[16i8; 16]);
    ctx.nzc_chroma[mb_xy * 8..mb_xy * 8 + 8].copy_from_slice(&[16i8; 8]);
    // CABAC coded_block_flag context: a PCM neighbour contributes condTermFlag=1
    // for every transform block (spec 9.3.3.1.1.9). The nnz=16 above covers the
    // AC/4x4 path; set every DC cbf bit so the DC path reads 1 too. (Unused on
    // the CAVLC path, harmless to set.)
    ctx.cbf_dc[mb_xy] = 0xFFFF;
    ctx.i4_best_mode[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&[-1i8; 16]);
    ctx.i4_final_mode[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&[2i8; 16]);
    for b in 0..16 {
        ctx.ref_idx[mb_xy * 16 + b] = -1;
        ctx.ref_pic_id[mb_xy * 16 + b] = -1;
        ctx.mv[(mb_xy * 16 + b) * 2] = 0;
        ctx.mv[(mb_xy * 16 + b) * 2 + 1] = 0;
    }
}

/// Parse an I_PCM macroblock (CAVLC, spec 7.3.5): byte-align past
/// `pcm_alignment_zero_bit`, then read 256 luma + 64 Cb + 64 Cr raw samples and
/// copy them straight into the reconstructed picture.
fn parse_pcm_mb_cavlc(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    mb_xy: usize,
) -> Result<()> {
    bs.align_to_byte();
    let mut luma = [0u8; 256];
    for s in luma.iter_mut() {
        *s = bs.read_u8()?;
    }
    let mut cb = [0u8; 64];
    for s in cb.iter_mut() {
        *s = bs.read_u8()?;
    }
    let mut cr = [0u8; 64];
    for s in cr.iter_mut() {
        *s = bs.read_u8()?;
    }
    super::recon_intra::recon_pcm_mb(ctx, mb_xy, &luma, &cb, &cr);
    commit_pcm_state(ctx, mb_xy);
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
        [
            top.best_mode[12],
            top.best_mode[13],
            top.best_mode[14],
            top.best_mode[15],
        ]
    } else if top.avail {
        [2; 4]
    } else {
        [-1; 4]
    };
    let left_modes: [i8; 4] = if left.avail && left.is_nxn {
        [
            left.best_mode[3],
            left.best_mode[7],
            left.best_mode[11],
            left.best_mode[15],
        ]
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

    let chroma_neigh_avail = (sample_avail[6] << 2) | (sample_avail[0] << 1) | sample_avail[1];

    for i in 0..16 {
        let raster = BLOCK_RASTER[i];
        let bx = BLOCK_BX[i];
        let by = BLOCK_BY[i];

        let prev_flag = bs.read_flag()?;
        let top_mode = if by > 0 {
            best_mode[(by - 1) * 4 + bx]
        } else {
            top_modes[bx]
        };
        let left_mode = if bx > 0 {
            best_mode[by * 4 + bx - 1]
        } else {
            left_modes[by]
        };
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

        let cur_final = check_intra_nxn_mode(&sample_avail, cur_best, i, false)?;

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

/// `ParseIntra8x8Mode` (CAVLC): four 8x8 luma prediction modes (one per 8x8
/// block, replicated to its four 4x4 sub-blocks) plus the chroma mode. Returns
/// `(chroma_mode, i8_avail_flag)`. Mirrors [`parse_intra4x4`] but iterates the
/// four 8x8 blocks in raster order and uses the 8x8 right-top neighbour rule.
fn parse_intra8x8(
    bs: &mut BitReader<'_>,
    neigh: &NeighborAvail,
    left: &NeighborSnap,
    top: &NeighborSnap,
    best_mode: &mut [i8; 16],
    final_mode: &mut [i8; 16],
) -> Result<(i8, u8)> {
    let top_modes: [i8; 4] = if top.avail && top.is_nxn {
        [
            top.best_mode[12],
            top.best_mode[13],
            top.best_mode[14],
            top.best_mode[15],
        ]
    } else if top.avail {
        [2; 4]
    } else {
        [-1; 4]
    };
    let left_modes: [i8; 4] = if left.avail && left.is_nxn {
        [
            left.best_mode[3],
            left.best_mode[7],
            left.best_mode[11],
            left.best_mode[15],
        ]
    } else if left.avail {
        [2; 4]
    } else {
        [-1; 4]
    };

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

    // I_8x8 neighbour-availability flag (Top-Right:Left:Top-Left:Top).
    let avail8 = ((sample_avail[5] as u8) << 3)
        | ((sample_avail[6] as u8) << 2)
        | ((sample_avail[0] as u8) << 1)
        | (sample_avail[1] as u8);
    let chroma_neigh_avail = (sample_avail[6] << 2) | (sample_avail[0] << 1) | sample_avail[1];

    for i8 in 0..4 {
        let bx8 = i8 & 1;
        let by8 = i8 >> 1;
        let top_mode = if by8 > 0 {
            best_mode[(by8 * 2 - 1) * 4 + bx8 * 2]
        } else {
            top_modes[bx8 * 2]
        };
        let left_mode = if bx8 > 0 {
            best_mode[(by8 * 2) * 4 + bx8 * 2 - 1]
        } else {
            left_modes[by8 * 2]
        };
        let pred_mode = if left_mode == -1 || top_mode == -1 {
            2
        } else {
            left_mode.min(top_mode)
        };

        let prev_flag = bs.read_flag()?;
        let cur_best = if prev_flag {
            pred_mode
        } else {
            let rem = bs.read_bits(3)? as i8;
            rem + if rem >= pred_mode { 1 } else { 0 }
        };
        let cur_final = check_intra_nxn_mode(&sample_avail, cur_best, i8 << 2, true)?;

        for j in 0..4 {
            let sub = (i8 << 2) + j;
            let raster = BLOCK_RASTER[sub];
            best_mode[raster] = cur_best;
            final_mode[raster] = cur_final;
            sample_avail[CACHE30_SCAN_IDX[sub]] = 1;
        }
    }

    let cm = bs.read_ue()?;
    if cm > 3 {
        return Err(DecodeError::InvalidSyntax("intra_chroma_pred_mode"));
    }
    let mut chroma_mode = cm as i8;
    check_intra_chroma_mode(chroma_neigh_avail, &mut chroma_mode)?;
    Ok((chroma_mode, avail8))
}

/// `CheckIntraNxNPredMode` (4x4): validate `mode` against availability, return
/// the final mode (with DC / DDL_TOP / VL_TOP variants resolved).
pub(super) fn check_intra_nxn_mode(
    sample_avail: &[i32; 30],
    mode: i8,
    i: usize,
    b8x8: bool,
) -> Result<i8> {
    if !(0..=8).contains(&mode) {
        return Err(DecodeError::InvalidSyntax("intra4x4 pred mode"));
    }
    let idx = CACHE30_SCAN_IDX[i];
    let left_avail = sample_avail[idx - 1];
    let top_avail = sample_avail[idx - 6];
    let left_top_avail = sample_avail[idx - 7];
    // The right-top sample sits one cell further left in the 8x8 grid.
    let right_top_avail = sample_avail[idx - if b8x8 { 4 } else { 5 }];

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
pub(super) fn check_intra16x16_mode(neigh_avail: i32, mode: &mut i8) -> Result<()> {
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
pub(super) fn check_intra_chroma_mode(neigh_avail: i32, mode: &mut i8) -> Result<()> {
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

fn parse_residuals(
    bs: &mut BitReader<'_>,
    params: ResidualParams,
    neigh: Neighbours,
    out: ResidualOut,
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let ResidualParams {
        mb_type,
        cbp_l,
        cbp_c,
        luma_qp,
        chroma_qp,
        transform_8x8,
    } = params;
    let Neighbours { left, top } = neigh;
    let ResidualOut {
        cur_nzc_luma,
        cur_nzc_chroma,
    } = out;
    let deq_l = &G_KUI_DEQUANT_COEFF[luma_qp as usize];
    let mut out = [0i32; 16];

    if transform_8x8 {
        // I_8x8 / inter 8x8 luma: four 8x8 blocks, each assembled from four
        // interleaved 4x4 CAVLC sub-blocks into 64 coeffs (spec 8.5.6).
        decode_luma_8x8(bs, cbp_l, luma_qp, left, top, cur_nzc_luma, coeffs)?;
    } else if mb_type == MbType::Intra16x16 {
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
            for &scan in &G_KUI_CHROMA_DC_SCAN {
                let j = cbase + scan as usize;
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

/// Decode the four 8x8 luma residual blocks (`WelsResidualBlockCavlc8x8`,
/// spec 8.5.6). Each 8x8 block is the interleave of four 4x4 CAVLC sub-blocks:
/// sub-block `id4`'s scan coefficient `s` lands at zig-zag-8x8 position
/// `(s<<2)+id4`. Dequant uses the flat 8x8 table (no scaling list in the corpus
/// streams) with the High-profile qp-dependent shift.
fn decode_luma_8x8(
    bs: &mut BitReader<'_>,
    cbp_l: u8,
    luma_qp: i32,
    left: &NeighborSnap,
    top: &NeighborSnap,
    cur_nzc_luma: &mut [i8; 16],
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let deq8 = &G_KUI_DEQUANT_COEFF8X8[luma_qp as usize];
    let qbits = luma_qp / 6;
    let mut out = [0i32; 16];
    for id8 in 0..4 {
        if cbp_l & (1 << id8) == 0 {
            continue;
        }
        let cbase = id8 * 64;
        for id4 in 0..4 {
            let i = id8 * 4 + id4;
            let raster = BLOCK_RASTER[i];
            let nc = nc_luma(cur_nzc_luma, left, top, BLOCK_BX[i], BLOCK_BY[i]);
            out.fill(0);
            let total = residual_block_cavlc(bs, nc, 16, &mut out)?;
            for (s, &lvl) in out.iter().enumerate() {
                if lvl != 0 {
                    let j = G_KUI_ZIGZAG_SCAN8X8[(s << 2) + id4] as usize;
                    let d = deq8[j] as i32;
                    coeffs[cbase + j] = dequant8x8(lvl, d, qbits) as i16;
                }
            }
            cur_nzc_luma[raster] = total as i8;
        }
    }
    Ok(())
}

/// One 8x8 dequant step (`pTCoeff[j]`): the High-profile qp-dependent scale.
#[inline]
pub(super) fn dequant8x8(level: i32, deq: i32, qbits: i32) -> i32 {
    if qbits >= 6 {
        level * deq * (1 << (qbits - 6))
    } else {
        (level * deq + (1 << (5 - qbits))) >> (6 - qbits)
    }
}

/// `WelsLumaDcDequantIdct`: inverse Hadamard + dequant of the 16 luma DC
/// coefficients (scattered at element offsets `block*16` within `coeffs`).
pub(super) fn luma_dc_dequant_idct(coeffs: &mut [i16; 384], qp: i32) {
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
pub(super) fn chroma_dc_idct(block: &mut [i16]) {
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
    REF_NOT_AVAIL, REF_NOT_IN_LIST, SCAN4, pred_inter8x16, pred_inter16x8, pred_mv, pred_p_skip_mv,
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
            mb_x != mb_width - 1 && mb_y != 0 && avail(true, (mb_xy + 1).wrapping_sub(mb_width));

        let left_xy = mb_xy.wrapping_sub(1);
        let top_xy = mb_xy.wrapping_sub(mb_width);
        let left_top_xy = mb_xy.wrapping_sub(mb_width + 1);
        let right_top_xy = (mb_xy + 1).wrapping_sub(mb_width);

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
            ref_idx[0] = if left_top {
                REF_NOT_IN_LIST
            } else {
                REF_NOT_AVAIL
            };
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
            for slot in &mut ref_idx[1..=4] {
                *slot = r;
            }
        }
        // Right-top (cache 5 <- block 12).
        if right_top && ctx.mb_type[right_top_xy].is_inter() {
            mv[5] = mv_of(right_top_xy, 12);
            ref_idx[5] = ref_of(right_top_xy, 12);
        } else {
            ref_idx[5] = if right_top {
                REF_NOT_IN_LIST
            } else {
                REF_NOT_AVAIL
            };
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
fn store_block(
    ctx: &mut DecoderContext,
    cache: &mut InterCache,
    mb_xy: usize,
    place: BlockPlace,
    mv: [i16; 2],
    iref: i8,
    ref_pic_id: i32,
) {
    let BlockPlace {
        scan4,
        cache_idx,
        w,
        h,
    } = place;
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
pub fn parse_p_mb_cavlc(
    bs: &mut BitReader<'_>,
    mb: MbCtx,
    last_mb_qp: &mut i32,
    skip_run: &mut i32,
    ref_pic_ids: &[i32],
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let MbCtx { ctx, mb_xy, pps } = mb;
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
        commit_inter_meta(
            MbCtx { ctx, mb_xy, pps },
            MbType::PSkip,
            0,
            luma_qp,
            &[0; 16],
            &[0; 8],
        );
        return Ok(());
    }

    let ui_mb_type = bs.read_ue()?;
    if ui_mb_type < 5 {
        parse_inter_mb(
            bs,
            MbCtx { ctx, mb_xy, pps },
            last_mb_qp,
            ui_mb_type,
            ref_pic_ids,
            coeffs,
        )
    } else {
        // Intra MB inside a P slice: reuse the intra core with the -5 offset.
        parse_intra_mb_core(bs, ctx, mb_xy, pps, last_mb_qp, coeffs, ui_mb_type - 5)
    }
}

/// Commit per-MB inter metadata (type/cbp/qp/nzc) into `ctx`.
fn commit_inter_meta(
    mb: MbCtx,
    mb_type: MbType,
    cbp: u8,
    luma_qp: i32,
    nzc_luma: &[i8; 16],
    nzc_chroma: &[i8; 8],
) {
    let MbCtx { ctx, mb_xy, pps } = mb;
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
fn parse_inter_mb(
    bs: &mut BitReader<'_>,
    mb: MbCtx,
    last_mb_qp: &mut i32,
    ui_mb_type: u32,
    ref_pic_ids: &[i32],
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let MbCtx { ctx, mb_xy, pps } = mb;
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

    // transform_size_8x8_flag (High profile): present for 16x16/16x8/8x16, or an
    // 8x8 MB whose sub-partitions are all 8x8, when cbp luma != 0.
    let transform_8x8 = parse_inter_t8_flag(bs, ctx, mb_xy, mb_type, cbp_l, pps)?;

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
            bs,
            ResidualParams {
                mb_type,
                cbp_l,
                cbp_c,
                luma_qp,
                chroma_qp,
                transform_8x8,
            },
            Neighbours {
                left: &left,
                top: &top,
            },
            ResidualOut {
                cur_nzc_luma: &mut cur_nzc_luma,
                cur_nzc_chroma: &mut cur_nzc_chroma,
            },
            coeffs,
        )?;
    } else {
        luma_qp = *last_mb_qp;
    }
    ctx.transform_8x8[mb_xy] = transform_8x8;

    commit_inter_meta(
        MbCtx { ctx, mb_xy, pps },
        mb_type,
        cbp,
        luma_qp,
        &cur_nzc_luma,
        &cur_nzc_chroma,
    );
    Ok(())
}

/// `transform_size_8x8_flag` for an inter MB (CAVLC). Read when the MB is
/// 16x16/16x8/8x16 (or 8x8 with every sub-partition 8x8) and `cbp_l != 0` and
/// the PPS enables the 8x8 transform.
fn parse_inter_t8_flag(
    bs: &mut BitReader<'_>,
    ctx: &DecoderContext,
    mb_xy: usize,
    mb_type: MbType,
    cbp_l: u8,
    pps: &Pps,
) -> Result<bool> {
    if !pps.transform_8x8_mode_flag || cbp_l == 0 {
        return Ok(false);
    }
    let no_sub_lt_8x8 = match mb_type {
        MbType::Inter8x8 | MbType::Inter8x8Ref0 | MbType::B8x8 => {
            (0..4).all(|i| ctx.sub_mb_type[mb_xy * 4 + i] == SubMbType::P8x8)
        }
        _ => false,
    };
    let big_part = matches!(
        mb_type,
        MbType::Inter16x16
            | MbType::Inter16x8
            | MbType::Inter8x16
            | MbType::B16x16
            | MbType::B16x8
            | MbType::B8x16
            | MbType::BDirect16x16
    );
    if big_part || no_sub_lt_8x8 {
        bs.read_flag()
    } else {
        Ok(false)
    }
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
            store_block(
                ctx,
                cache,
                mb_xy,
                BlockPlace {
                    scan4: 0,
                    cache_idx: 0,
                    w: 4,
                    h: 4,
                },
                mv,
                iref,
                ref_pic_ids[iref as usize],
            );
        }
        MbType::Inter16x8 => {
            let r = [read_ref(bs)?, read_ref(bs)?];
            for i in 0..2 {
                let part_idx = i << 3;
                let mvp = pred_inter16x8(&cache.mv, &cache.ref_idx, part_idx, r[i]);
                let mvd = read_mvd(bs)?;
                let mv = [mvp[0] + mvd[0], mvp[1] + mvd[1]];
                update_p16x8(
                    ctx,
                    cache,
                    mb_xy,
                    part_idx,
                    mv,
                    r[i],
                    ref_pic_ids[r[i] as usize],
                );
            }
        }
        MbType::Inter8x16 => {
            let r = [read_ref(bs)?, read_ref(bs)?];
            for i in 0..2 {
                let part_idx = i << 2;
                let mvp = pred_inter8x16(&cache.mv, &cache.ref_idx, part_idx, r[i]);
                let mvd = read_mvd(bs)?;
                let mv = [mvp[0] + mvd[0], mvp[1] + mvd[1]];
                update_p8x16(
                    ctx,
                    cache,
                    mb_xy,
                    part_idx,
                    mv,
                    r[i],
                    ref_pic_ids[r[i] as usize],
                );
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
                    store_mv_only(
                        ctx,
                        cache,
                        mb_xy,
                        BlockPlace {
                            scan4,
                            cache_idx,
                            w,
                            h,
                        },
                        mv,
                    );
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
    place: BlockPlace,
    mv: [i16; 2],
) {
    let BlockPlace {
        scan4,
        cache_idx,
        w,
        h,
    } = place;
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
        store_block(
            ctx,
            cache,
            mb_xy,
            BlockPlace {
                scan4,
                cache_idx,
                w: 2,
                h: 2,
            },
            mv,
            iref,
            ref_pic_id,
        );
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
        store_block(
            ctx,
            cache,
            mb_xy,
            BlockPlace {
                scan4,
                cache_idx,
                w: 2,
                h: 2,
            },
            mv,
            iref,
            ref_pic_id,
        );
        p += 8;
    }
}

// ===================== B-slice (bi-predictive) macroblock parse =============

use super::bdirect::{
    ColRef, DirectInfo, Part8x8, b_direct_spatial, b_direct_temporal_sub, fill_direct_8x8,
    fill_direct_16x16,
};

/// B macroblock partition shape (`g_ksInterBMbTypeInfo` geometry).
#[derive(Clone, Copy, PartialEq)]
pub(super) enum BShape {
    Direct,
    P16x16,
    P16x8,
    P8x16,
    P8x8,
}

/// One B mb_type entry: partition shape + per-partition `(uses_l0, uses_l1)`.
pub(super) struct BMbInfo {
    pub(super) shape: BShape,
    pub(super) dir: [(bool, bool); 2],
}

#[rustfmt::skip]
pub(super) const B_MB_INFO: [BMbInfo; 23] = [
    BMbInfo { shape: BShape::Direct, dir: [(false,false),(false,false)] }, // 0 B_Direct_16x16
    BMbInfo { shape: BShape::P16x16, dir: [(true,false),(false,false)] },  // 1 B_L0_16x16
    BMbInfo { shape: BShape::P16x16, dir: [(false,true),(false,false)] },  // 2 B_L1_16x16
    BMbInfo { shape: BShape::P16x16, dir: [(true,true),(false,false)] },   // 3 B_Bi_16x16
    BMbInfo { shape: BShape::P16x8,  dir: [(true,false),(true,false)] },   // 4 B_L0_L0_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(true,false),(true,false)] },   // 5 B_L0_L0_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(false,true),(false,true)] },   // 6 B_L1_L1_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(false,true),(false,true)] },   // 7 B_L1_L1_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(true,false),(false,true)] },   // 8 B_L0_L1_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(true,false),(false,true)] },   // 9 B_L0_L1_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(false,true),(true,false)] },   // 10 B_L1_L0_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(false,true),(true,false)] },   // 11 B_L1_L0_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(true,false),(true,true)] },    // 12 B_L0_Bi_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(true,false),(true,true)] },    // 13 B_L0_Bi_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(false,true),(true,true)] },    // 14 B_L1_Bi_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(false,true),(true,true)] },    // 15 B_L1_Bi_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(true,true),(true,false)] },    // 16 B_Bi_L0_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(true,true),(true,false)] },    // 17 B_Bi_L0_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(true,true),(false,true)] },    // 18 B_Bi_L1_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(true,true),(false,true)] },    // 19 B_Bi_L1_8x16
    BMbInfo { shape: BShape::P16x8,  dir: [(true,true),(true,true)] },     // 20 B_Bi_Bi_16x8
    BMbInfo { shape: BShape::P8x16,  dir: [(true,true),(true,true)] },     // 21 B_Bi_Bi_8x16
    BMbInfo { shape: BShape::P8x8,   dir: [(false,false),(false,false)] }, // 22 B_8x8
];

/// One B sub_mb_type entry (`g_ksInterBSubMbTypeInfo`).
pub(super) struct BSubInfo {
    pub(super) direct: bool,
    pub(super) sub: SubMbType,
    pub(super) dir: (bool, bool),
    pub(super) part_count: usize,
    pub(super) part_w: usize,
}

#[rustfmt::skip]
pub(super) const B_SUB_INFO: [BSubInfo; 13] = [
    BSubInfo { direct: true,  sub: SubMbType::P8x8, dir: (false,false), part_count: 1, part_w: 2 }, // 0 B_Direct_8x8
    BSubInfo { direct: false, sub: SubMbType::P8x8, dir: (true,false),  part_count: 1, part_w: 2 }, // 1 B_L0_8x8
    BSubInfo { direct: false, sub: SubMbType::P8x8, dir: (false,true),  part_count: 1, part_w: 2 }, // 2 B_L1_8x8
    BSubInfo { direct: false, sub: SubMbType::P8x8, dir: (true,true),   part_count: 1, part_w: 2 }, // 3 B_Bi_8x8
    BSubInfo { direct: false, sub: SubMbType::P8x4, dir: (true,false),  part_count: 2, part_w: 2 }, // 4 B_L0_8x4
    BSubInfo { direct: false, sub: SubMbType::P4x8, dir: (true,false),  part_count: 2, part_w: 1 }, // 5 B_L0_4x8
    BSubInfo { direct: false, sub: SubMbType::P8x4, dir: (false,true),  part_count: 2, part_w: 2 }, // 6 B_L1_8x4
    BSubInfo { direct: false, sub: SubMbType::P4x8, dir: (false,true),  part_count: 2, part_w: 1 }, // 7 B_L1_4x8
    BSubInfo { direct: false, sub: SubMbType::P8x4, dir: (true,true),   part_count: 2, part_w: 2 }, // 8 B_Bi_8x4
    BSubInfo { direct: false, sub: SubMbType::P4x8, dir: (true,true),   part_count: 2, part_w: 1 }, // 9 B_Bi_4x8
    BSubInfo { direct: false, sub: SubMbType::P4x4, dir: (true,false),  part_count: 4, part_w: 1 }, // 10 B_L0_4x4
    BSubInfo { direct: false, sub: SubMbType::P4x4, dir: (false,true),  part_count: 4, part_w: 1 }, // 11 B_L1_4x4
    BSubInfo { direct: false, sub: SubMbType::P4x4, dir: (true,true),   part_count: 4, part_w: 1 }, // 12 B_Bi_4x4
];

/// Reference data for one B slice, passed to the MB parse.
pub struct BRefs<'a> {
    /// Resolved reference-picture ids for `[list0, list1]` (deblock identity).
    pub ref_pic_ids: [&'a [i32]; 2],
    /// Active reference counts for `[list0, list1]`.
    pub ref_count: [usize; 2],
    /// `direct_spatial_mv_pred_flag`.
    pub direct_spatial: bool,
    /// Colocated picture (`list1[0]`) for direct prediction.
    pub col: ColRef<'a>,
}

/// Motion of one prediction list to store: list index, vector, reference index
/// (slice-local) and the reference picture's decode id.
#[derive(Clone, Copy)]
struct Motion {
    list: usize,
    mv: [i16; 2],
    iref: i8,
    ref_pic: i32,
}

/// The 30-entry list-0 + list-1 neighbour MV / ref-index cache for B
/// (`WelsFillCacheInter`, both lists).
struct BInterCache {
    mv: [[[i16; 2]; 30]; 2],
    ref_idx: [[i8; 30]; 2],
}

impl BInterCache {
    fn build(ctx: &DecoderContext, mb_xy: usize) -> Self {
        let mb_width = ctx.mb_width;
        let mb_x = mb_xy % mb_width;
        let mb_y = mb_xy / mb_width;
        let cur = ctx.slice_idc[mb_xy];
        let avail = |xy: usize| ctx.slice_idc[xy] == cur;
        let left = mb_x != 0 && avail(mb_xy - 1);
        let top = mb_y != 0 && avail(mb_xy - mb_width);
        let left_top = mb_x != 0 && mb_y != 0 && avail(mb_xy - mb_width - 1);
        let right_top = mb_x != mb_width - 1 && mb_y != 0 && avail(mb_xy - mb_width + 1);
        let left_xy = mb_xy.wrapping_sub(1);
        let top_xy = mb_xy.wrapping_sub(mb_width);
        let left_top_xy = mb_xy.wrapping_sub(mb_width + 1);
        let right_top_xy = (mb_xy + 1).wrapping_sub(mb_width);

        let mut mv = [[[0i16; 2]; 30]; 2];
        let mut ref_idx = [[REF_NOT_AVAIL; 30]; 2];

        for list in 0..2 {
            let mv_of = |xy: usize, b: usize| -> [i16; 2] {
                if list == 0 {
                    [ctx.mv[(xy * 16 + b) * 2], ctx.mv[(xy * 16 + b) * 2 + 1]]
                } else {
                    [
                        ctx.mv_l1[(xy * 16 + b) * 2],
                        ctx.mv_l1[(xy * 16 + b) * 2 + 1],
                    ]
                }
            };
            let ref_of = |xy: usize, b: usize| -> i8 {
                if list == 0 {
                    ctx.ref_idx[xy * 16 + b]
                } else {
                    ctx.ref_idx_l1[xy * 16 + b]
                }
            };
            let m = &mut mv[list];
            let r = &mut ref_idx[list];
            // Left column.
            if left && ctx.mb_type[left_xy].is_inter() {
                for (k, &b) in [3usize, 7, 11, 15].iter().enumerate() {
                    let c = [6, 12, 18, 24][k];
                    m[c] = mv_of(left_xy, b);
                    r[c] = ref_of(left_xy, b);
                }
            } else {
                let v = if left { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
                for &c in &[6usize, 12, 18, 24] {
                    r[c] = v;
                }
            }
            // Left-top.
            if left_top && ctx.mb_type[left_top_xy].is_inter() {
                m[0] = mv_of(left_top_xy, 15);
                r[0] = ref_of(left_top_xy, 15);
            } else {
                r[0] = if left_top {
                    REF_NOT_IN_LIST
                } else {
                    REF_NOT_AVAIL
                };
            }
            // Top row.
            if top && ctx.mb_type[top_xy].is_inter() {
                for (k, &b) in [12usize, 13, 14, 15].iter().enumerate() {
                    m[1 + k] = mv_of(top_xy, b);
                    r[1 + k] = ref_of(top_xy, b);
                }
            } else {
                let v = if top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
                for slot in &mut r[1..=4] {
                    *slot = v;
                }
            }
            // Right-top.
            if right_top && ctx.mb_type[right_top_xy].is_inter() {
                m[5] = mv_of(right_top_xy, 12);
                r[5] = ref_of(right_top_xy, 12);
            } else {
                r[5] = if right_top {
                    REF_NOT_IN_LIST
                } else {
                    REF_NOT_AVAIL
                };
            }
            for &c in &[9usize, 11, 17, 21, 23] {
                r[c] = REF_NOT_AVAIL;
                m[c] = [0, 0];
            }
        }
        BInterCache { mv, ref_idx }
    }

    /// Store one block of motion for `list` into both `ctx` and the cache.
    fn store(&mut self, ctx: &mut DecoderContext, mb_xy: usize, place: BlockPlace, m: Motion) {
        let BlockPlace {
            scan4,
            cache_idx,
            w,
            h,
        } = place;
        let Motion {
            list,
            mv,
            iref,
            ref_pic,
        } = m;
        for by in 0..h {
            for bx in 0..w {
                let raster = scan4 + by * 4 + bx;
                let b = (mb_xy * 16 + raster) * 2;
                if list == 0 {
                    ctx.mv[b] = mv[0];
                    ctx.mv[b + 1] = mv[1];
                    ctx.ref_idx[mb_xy * 16 + raster] = iref;
                    ctx.ref_pic_id[mb_xy * 16 + raster] = if iref >= 0 { ref_pic } else { -1 };
                } else {
                    ctx.mv_l1[b] = mv[0];
                    ctx.mv_l1[b + 1] = mv[1];
                    ctx.ref_idx_l1[mb_xy * 16 + raster] = iref;
                    ctx.ref_pic_id_l1[mb_xy * 16 + raster] = if iref >= 0 { ref_pic } else { -1 };
                }
                let c = cache_idx + by * 6 + bx;
                self.mv[list][c] = mv;
                self.ref_idx[list][c] = iref;
            }
        }
    }
}

/// Parse one B-slice macroblock (CAVLC). Handles `mb_skip_run`, the B inter
/// partition kinds, direct/skip, intra MBs appearing in B slices, cbp and
/// residuals.
pub fn parse_b_mb_cavlc(
    bs: &mut BitReader<'_>,
    mb: MbCtx,
    last_mb_qp: &mut i32,
    skip_run: &mut i32,
    bref: &BRefs,
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let MbCtx { ctx, mb_xy, pps } = mb;

    if *skip_run == -1 {
        *skip_run = bs.read_ue()? as i32;
    }
    let old = *skip_run;
    *skip_run -= 1;
    if old != 0 {
        // B_Skip: direct prediction, cbp = 0.
        ctx.mb_type[mb_xy] = MbType::BSkip;
        apply_b_direct(
            ctx,
            mb_xy,
            bref.ref_pic_ids,
            &bref.col,
            true,
            bref.direct_spatial,
        );
        let luma_qp = *last_mb_qp;
        commit_inter_meta(
            MbCtx { ctx, mb_xy, pps },
            MbType::BSkip,
            0,
            luma_qp,
            &[0; 16],
            &[0; 8],
        );
        return Ok(());
    }

    let ui_mb_type = bs.read_ue()?;
    if ui_mb_type >= 23 {
        // Intra MB inside a B slice (offset 23).
        let v = ui_mb_type - 23;
        if v > 25 {
            return Err(DecodeError::InvalidSyntax("B intra mb_type"));
        }
        return parse_intra_mb_core(bs, ctx, mb_xy, pps, last_mb_qp, coeffs, v);
    }

    let info = &B_MB_INFO[ui_mb_type as usize];
    let mb_type = match info.shape {
        BShape::Direct => MbType::BDirect16x16,
        BShape::P16x16 => MbType::B16x16,
        BShape::P16x8 => MbType::B16x8,
        BShape::P8x16 => MbType::B8x16,
        BShape::P8x8 => MbType::B8x8,
    };
    ctx.mb_type[mb_xy] = mb_type;

    if info.shape == BShape::Direct {
        apply_b_direct(
            ctx,
            mb_xy,
            bref.ref_pic_ids,
            &bref.col,
            true,
            bref.direct_spatial,
        );
    } else {
        let mut cache = BInterCache::build(ctx, mb_xy);
        parse_b_motion(bs, ctx, &mut cache, mb_xy, ui_mb_type, bref)?;
    }

    // coded_block_pattern (inter mapping).
    let ui_cbp = bs.read_ue()?;
    if ui_cbp > 47 {
        return Err(DecodeError::InvalidSyntax("B inter cbp"));
    }
    let cbp = INTER_CBP_TABLE[ui_cbp as usize];
    let cbp_l = cbp & 0x0f;
    let cbp_c = cbp >> 4;

    let transform_8x8 = parse_inter_t8_flag(bs, ctx, mb_xy, mb_type, cbp_l, pps)?;

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
            bs,
            ResidualParams {
                mb_type,
                cbp_l,
                cbp_c,
                luma_qp,
                chroma_qp,
                transform_8x8,
            },
            Neighbours {
                left: &left,
                top: &top,
            },
            ResidualOut {
                cur_nzc_luma: &mut cur_nzc_luma,
                cur_nzc_chroma: &mut cur_nzc_chroma,
            },
            coeffs,
        )?;
    } else {
        luma_qp = *last_mb_qp;
    }
    ctx.transform_8x8[mb_xy] = transform_8x8;

    commit_inter_meta(
        MbCtx { ctx, mb_xy, pps },
        mb_type,
        cbp,
        luma_qp,
        &cur_nzc_luma,
        &cur_nzc_chroma,
    );
    Ok(())
}

/// Compute and store B direct prediction (spatial only; temporal direct is not
/// used by the corpus B streams — see `bdirect.rs`). `whole_mb` true for
/// skip / B_Direct_16x16.
pub(super) fn apply_b_direct(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    ref_pic_ids: [&[i32]; 2],
    col: &ColRef,
    whole_mb: bool,
    direct_spatial: bool,
) {
    if !direct_spatial {
        // `b_direct_temporal` already stores the correct per-block reference
        // identities: L0 = `cur_ref0_ids[ref0]` (the `MapColToList0` result, which
        // need not be index 0) and L1 = `col_id` (== list1[0]). The previous
        // unconditional `ref_pic_id = ref_pic_ids[0][0]` overwrite clobbered the
        // L0 identity to list-0 index 0, breaking the deblock boundary-strength
        // reference comparison when `ref0 != 0`.
        super::bdirect::b_direct_temporal(ctx, mb_xy, !whole_mb, col);
        return;
    }
    ctx.direct_8x8[mb_xy] = col.resolved_8x8(mb_xy, !whole_mb);
    let info: DirectInfo = b_direct_spatial(ctx, mb_xy, !whole_mb);
    let ref_pic = [
        if info.iref[0] >= 0 && (info.iref[0] as usize) < ref_pic_ids[0].len() {
            ref_pic_ids[0][info.iref[0] as usize]
        } else {
            -1
        },
        if info.iref[1] >= 0 && (info.iref[1] as usize) < ref_pic_ids[1].len() {
            ref_pic_ids[1][info.iref[1] as usize]
        } else {
            -1
        },
    ];
    // The colZeroFlag granularity follows the *resolved* partition mode: a whole-MB
    // spatial-direct MB whose colocated MB forces 8x8 resolution (`direct_8x8`)
    // must evaluate colZero per 8x8 sub-block, not once from block 0. Using
    // `info.mb16x16` (which only reflects the syntax 16x16/8x8) zeroed the whole MB
    // uniformly and lost the per-8x8 mvL1 variation the spec/oracle produce.
    if info.mb16x16 && !ctx.direct_8x8[mb_xy] {
        fill_direct_16x16(ctx, mb_xy, &info, col, ref_pic);
    } else {
        // Fill each 8x8 with its own colZero (B_8x8 direct, or a 16x16 direct MB
        // resolved to 8x8 by its colocated MB).
        for i in 0..4 {
            fill_direct_8x8(
                ctx,
                mb_xy,
                Part8x8 {
                    idx8: i,
                    part_count: 1,
                    part_w: 2,
                },
                &info,
                col,
                ref_pic,
            );
            ctx.sub_mb_type[mb_xy * 4 + i] = SubMbType::P8x8;
        }
    }
}

/// Parse `ParseInterBInfo`: ref_idx + mvd for the non-direct B partition kinds.
fn parse_b_motion(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    cache: &mut BInterCache,
    mb_xy: usize,
    ui_mb_type: u32,
    bref: &BRefs,
) -> Result<()> {
    let info = &B_MB_INFO[ui_mb_type as usize];
    let read_ref = |bs: &mut BitReader<'_>, list: usize| -> Result<i8> {
        let n = bref.ref_count[list];
        let v = bs.read_te(n as u32)?;
        if v as usize >= n {
            return Err(DecodeError::InvalidSyntax("B ref_idx"));
        }
        Ok(v as i8)
    };

    match info.shape {
        BShape::P16x16 => {
            let mut iref = [0i8; 2];
            for (list, slot) in iref.iter_mut().enumerate() {
                if dir_uses(info.dir[0], list) {
                    *slot = read_ref(bs, list)?;
                }
            }
            for (list, &iref_l) in iref.iter().enumerate() {
                if dir_uses(info.dir[0], list) {
                    let mvp = pred_mv(&cache.mv[list], &cache.ref_idx[list], 0, 4, iref_l);
                    let dx = bs.read_se()? as i16;
                    let dy = bs.read_se()? as i16;
                    let mv = [mvp[0] + dx, mvp[1] + dy];
                    cache.store(
                        ctx,
                        mb_xy,
                        BlockPlace {
                            scan4: 0,
                            cache_idx: 0,
                            w: 4,
                            h: 4,
                        },
                        Motion {
                            list,
                            mv,
                            iref: iref_l,
                            ref_pic: bref.ref_pic_ids[list][iref_l as usize],
                        },
                    );
                } else {
                    cache.store(
                        ctx,
                        mb_xy,
                        BlockPlace {
                            scan4: 0,
                            cache_idx: 0,
                            w: 4,
                            h: 4,
                        },
                        Motion {
                            list,
                            mv: [0, 0],
                            iref: REF_NOT_IN_LIST,
                            ref_pic: -1,
                        },
                    );
                }
            }
        }
        BShape::P16x8 | BShape::P8x16 => {
            let is16x8 = info.shape == BShape::P16x8;
            // `iref[list][p]` (list-major to match the bitstream read order).
            let mut iref = [[REF_NOT_IN_LIST; 2]; 2];
            for (list, row) in iref.iter_mut().enumerate() {
                for (p, slot) in row.iter_mut().enumerate() {
                    if dir_uses(info.dir[p], list) {
                        *slot = read_ref(bs, list)?;
                    }
                }
            }
            // Partition size in 4x4 units: 16x8 = 4 wide x 2 tall, 8x16 = 2x4.
            let (pw, ph) = if is16x8 { (4, 2) } else { (2, 4) };
            for (list, row) in iref.iter().enumerate() {
                for (p, &r) in row.iter().enumerate() {
                    let part_idx = if is16x8 { p << 3 } else { p << 2 };
                    let scan4 = SCAN4[part_idx];
                    let cache_idx = CACHE30_SCAN_IDX[part_idx];
                    if dir_uses(info.dir[p], list) {
                        let mvp = if is16x8 {
                            pred_inter16x8(&cache.mv[list], &cache.ref_idx[list], part_idx, r)
                        } else {
                            pred_inter8x16(&cache.mv[list], &cache.ref_idx[list], part_idx, r)
                        };
                        let dx = bs.read_se()? as i16;
                        let dy = bs.read_se()? as i16;
                        let mv = [mvp[0] + dx, mvp[1] + dy];
                        cache.store(
                            ctx,
                            mb_xy,
                            BlockPlace {
                                scan4,
                                cache_idx,
                                w: pw,
                                h: ph,
                            },
                            Motion {
                                list,
                                mv,
                                iref: r,
                                ref_pic: bref.ref_pic_ids[list][r as usize],
                            },
                        );
                    } else {
                        cache.store(
                            ctx,
                            mb_xy,
                            BlockPlace {
                                scan4,
                                cache_idx,
                                w: pw,
                                h: ph,
                            },
                            Motion {
                                list,
                                mv: [0, 0],
                                iref: REF_NOT_IN_LIST,
                                ref_pic: -1,
                            },
                        );
                    }
                }
            }
        }
        BShape::P8x8 => {
            parse_b_8x8(bs, ctx, cache, mb_xy, bref)?;
        }
        BShape::Direct => unreachable!(),
    }
    Ok(())
}

#[inline]
pub(super) fn dir_uses(d: (bool, bool), list: usize) -> bool {
    if list == 0 { d.0 } else { d.1 }
}

/// Parse the four 8x8 sub-partitions of a B_8x8 macroblock.
fn parse_b_8x8(
    bs: &mut BitReader<'_>,
    ctx: &mut DecoderContext,
    cache: &mut BInterCache,
    mb_xy: usize,
    bref: &BRefs,
) -> Result<()> {
    // sub_mb_type for each 8x8.
    let mut subs = [0usize; 4];
    for s in subs.iter_mut() {
        let st = bs.read_ue()?;
        if st >= 13 {
            return Err(DecodeError::InvalidSyntax("B sub_mb_type"));
        }
        *s = st as usize;
    }

    // Direct prediction for direct sub-partitions. Spatial: a single shared
    // `DirectInfo`; temporal: per-8x8-sub colocated MV scaling.
    let any_direct = subs.iter().any(|&s| B_SUB_INFO[s].direct);
    let direct = if any_direct && bref.direct_spatial {
        Some(b_direct_spatial(ctx, mb_xy, true))
    } else {
        None
    };
    let direct_refpic = direct.as_ref().map(|d| {
        [
            if d.iref[0] >= 0 && (d.iref[0] as usize) < bref.ref_pic_ids[0].len() {
                bref.ref_pic_ids[0][d.iref[0] as usize]
            } else {
                -1
            },
            if d.iref[1] >= 0 && (d.iref[1] as usize) < bref.ref_pic_ids[1].len() {
                bref.ref_pic_ids[1][d.iref[1] as usize]
            } else {
                -1
            },
        ]
    });

    // Fill direct 8x8 sub-partitions first.
    for (i, &s) in subs.iter().enumerate() {
        let sinfo = &B_SUB_INFO[s];
        ctx.sub_mb_type[mb_xy * 4 + i] = sinfo.sub;
        if sinfo.direct {
            if bref.direct_spatial {
                let d = direct.as_ref().unwrap();
                fill_direct_8x8(
                    ctx,
                    mb_xy,
                    Part8x8 {
                        idx8: i,
                        part_count: 1,
                        part_w: 2,
                    },
                    d,
                    &bref.col,
                    direct_refpic.unwrap(),
                );
            } else {
                b_direct_temporal_sub(ctx, mb_xy, i, &bref.col);
            }
            // sync the cache for this 8x8's blocks.
            sync_cache_8x8(cache, ctx, mb_xy, i);
        }
    }

    // ref_idx for non-direct sub-partitions, per list.
    let mut iref = [[REF_NOT_IN_LIST; 4]; 2];
    for (list, irefs) in iref.iter_mut().enumerate() {
        for (i, &s) in subs.iter().enumerate() {
            let sinfo = &B_SUB_INFO[s];
            if sinfo.direct {
                if bref.direct_spatial {
                    let d = direct.as_ref().unwrap();
                    irefs[i] = d.iref[list];
                    set_8x8_ref(
                        ctx,
                        mb_xy,
                        i,
                        list,
                        d.iref[list],
                        direct_refpic.unwrap()[list],
                    );
                } else {
                    // Temporal: the neighbour-prediction cache treats a direct
                    // sub-partition as not-in-list (the C leaves `ref_idx_list`
                    // at its -1 init), so later explicit subs exclude its MV from
                    // prediction. `ctx` keeps the real ref for recon/deblock.
                    irefs[i] = REF_NOT_IN_LIST;
                }
            } else if dir_uses(sinfo.dir, list) {
                let n = bref.ref_count[list];
                let v = bs.read_te(n as u32)?;
                if v as usize >= n {
                    return Err(DecodeError::InvalidSyntax("B 8x8 ref_idx"));
                }
                irefs[i] = v as i8;
                set_8x8_ref(
                    ctx,
                    mb_xy,
                    i,
                    list,
                    v as i8,
                    bref.ref_pic_ids[list][v as usize],
                );
            } else {
                set_8x8_ref(ctx, mb_xy, i, list, REF_NOT_IN_LIST, -1);
            }
        }
    }

    // mvd for non-direct sub-partitions.
    for (list, irefs) in iref.iter().enumerate() {
        for (i, &s) in subs.iter().enumerate() {
            // Set this 8x8's neighbour-cache ref-index now (C sets it per-8x8 at
            // the start of the mvd loop, so a later 8x8 doesn't affect this one).
            let cache8 = CACHE30_SCAN_IDX[i << 2];
            for &c in &[cache8, cache8 + 1, cache8 + 6, cache8 + 7] {
                cache.ref_idx[list][c] = irefs[i];
            }
            let sinfo = &B_SUB_INFO[s];
            if sinfo.direct {
                continue;
            }
            let r = irefs[i];
            let uses = dir_uses(sinfo.dir, list);
            for j in 0..sinfo.part_count {
                let part_idx = (i << 2) + j * sinfo.part_w;
                let scan4 = SCAN4[part_idx];
                let cache_idx = CACHE30_SCAN_IDX[part_idx];
                let (w, h) = match sinfo.sub {
                    SubMbType::P8x8 => (2, 2),
                    SubMbType::P8x4 => (2, 1),
                    SubMbType::P4x8 => (1, 2),
                    SubMbType::P4x4 => (1, 1),
                };
                if uses {
                    let mvp = pred_mv(
                        &cache.mv[list],
                        &cache.ref_idx[list],
                        part_idx,
                        sinfo.part_w,
                        r,
                    );
                    let dx = bs.read_se()? as i16;
                    let dy = bs.read_se()? as i16;
                    let mv = [mvp[0] + dx, mvp[1] + dy];
                    cache.store(
                        ctx,
                        mb_xy,
                        BlockPlace {
                            scan4,
                            cache_idx,
                            w,
                            h,
                        },
                        Motion {
                            list,
                            mv,
                            iref: r,
                            ref_pic: if r >= 0 {
                                bref.ref_pic_ids[list][r as usize]
                            } else {
                                -1
                            },
                        },
                    );
                } else {
                    cache.store(
                        ctx,
                        mb_xy,
                        BlockPlace {
                            scan4,
                            cache_idx,
                            w,
                            h,
                        },
                        Motion {
                            list,
                            mv: [0, 0],
                            iref: REF_NOT_IN_LIST,
                            ref_pic: -1,
                        },
                    );
                }
            }
        }
    }
    Ok(())
}

/// Set the reference index for a whole 8x8 (4 blocks) in `ctx` only. The
/// neighbour cache's ref-index is set per-8x8 inside the mvd loop (matching the
/// C decoder, so that a later 8x8's ref does not pollute an earlier partition's
/// MV prediction).
fn set_8x8_ref(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    idx8: usize,
    list: usize,
    iref: i8,
    ref_pic: i32,
) {
    let base_part = idx8 << 2;
    let scan8 = SCAN4[base_part];
    for &raster in &[scan8, scan8 + 1, scan8 + 4, scan8 + 5] {
        if list == 0 {
            ctx.ref_idx[mb_xy * 16 + raster] = iref;
            ctx.ref_pic_id[mb_xy * 16 + raster] = if iref >= 0 { ref_pic } else { -1 };
        } else {
            ctx.ref_idx_l1[mb_xy * 16 + raster] = iref;
            ctx.ref_pic_id_l1[mb_xy * 16 + raster] = if iref >= 0 { ref_pic } else { -1 };
        }
    }
}

/// Sync the neighbour cache for one 8x8's four 4x4 blocks from `ctx` (after a
/// direct fill, so later partitions' PredMv reads the correct motion).
fn sync_cache_8x8(cache: &mut BInterCache, ctx: &DecoderContext, mb_xy: usize, idx8: usize) {
    let base_part = idx8 << 2;
    for p in 0..4 {
        let part_idx = base_part + p;
        let scan4 = SCAN4[part_idx];
        let c = CACHE30_SCAN_IDX[part_idx];
        for list in 0..2 {
            if list == 0 {
                cache.mv[0][c] = [
                    ctx.mv[(mb_xy * 16 + scan4) * 2],
                    ctx.mv[(mb_xy * 16 + scan4) * 2 + 1],
                ];
                cache.ref_idx[0][c] = ctx.ref_idx[mb_xy * 16 + scan4];
            } else {
                cache.mv[1][c] = [
                    ctx.mv_l1[(mb_xy * 16 + scan4) * 2],
                    ctx.mv_l1[(mb_xy * 16 + scan4) * 2 + 1],
                ];
                cache.ref_idx[1][c] = ctx.ref_idx_l1[mb_xy * 16 + scan4];
            }
        }
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
