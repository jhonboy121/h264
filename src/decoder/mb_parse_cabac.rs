//! Macroblock-level CABAC syntax decoding (I + P slices, 4:2:0 8-bit baseline/
//! Main profile, no transform_8x8 / PCM).
//!
//! Faithful port of the MB-syntax CABAC decoders from Cisco OpenH264
//! `reference/codec/decoder/core/src/parse_mb_syn_cabac.cpp` together with the
//! I/P CABAC MB drivers in `decode_slice.cpp`
//! (`WelsDecodeMbCabacISliceBaseMode0` / `...PSliceBaseMode0`). It populates the
//! exact same [`DecoderContext`] fields the existing CAVLC parse fills, so the
//! shared reconstruction + deblocking paths are reused unchanged.
//!
//! Deferred (rejected with [`DecodeError::Unsupported`]): I_PCM, transform_8x8
//! (I_8x8), B slices, weighted prediction, monochrome.

use crate::dsp::tables::{
    G_KUI_CHROMA_DC_SCAN, G_KUI_DEQUANT_COEFF, G_KUI_LUMA_DC_ZIGZAG_SCAN, G_KUI_ZIGZAG_SCAN,
};
use crate::error::DecodeError;

use super::cabac::{CabacContexts, CabacDecoder};
use super::cabac_mb::{
    coded_block_flag, residual_block_cabac, CHROMA_AC_U, CHROMA_AC_V, CHROMA_DC_U, CHROMA_DC_V,
    I16_LUMA_AC, I16_LUMA_DC, LUMA_DC_AC,
};
use super::context::{DecoderContext, MbType, SubMbType};
use super::mb_parse_cavlc::{
    check_intra16x16_mode, check_intra_chroma_mode, check_intra_nxn_mode, chroma_dc_idct, clip3,
    luma_dc_dequant_idct, BLOCK_BX, BLOCK_BY, BLOCK_RASTER, CACHE30_SCAN_IDX, CHROMA_QP_TABLE,
    I16_CBP_TABLE,
};
use super::mv_pred::{pred_inter16x8, pred_inter8x16, pred_mv, pred_p_skip_mv, SCAN4};
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

/// Running QP predictors carried across MBs (`pSlice->iLastMbQp` + last delta).
pub struct QpState<'a> {
    pub last_mb_qp: &'a mut i32,
    pub last_delta_qp: &'a mut i32,
}

/// Residual-decode inputs (mb_type, luma/chroma cbp split, qp + chroma qp).
#[derive(Clone, Copy)]
struct ResidualParams {
    mb_type: MbType,
    cbp_l: u8,
    cbp_c: u8,
    luma_qp: i32,
    chroma_qp: [i32; 2],
}

/// Mutable per-MB residual outputs (nzc counts + cbf-dc bitmask).
struct ResidualOut<'a> {
    cur_nzc_luma: &'a mut [i8; 16],
    cur_nzc_chroma: &'a mut [i8; 8],
    cur_cbf_dc: &'a mut u16,
}

/// Read-only neighbour snapshot + reference-picture identities for inter parse.
#[derive(Clone, Copy)]
struct InterRefs<'a> {
    n: &'a Neigh,
    ref_pic_ids: &'a [i32],
}

/// Mutable inter motion working state (neighbour cache + raster ref indices).
struct MotionState<'a> {
    cache: &'a mut InterCacheC,
    cur_ref: &'a mut [i8; 16],
}

/// Placement of a `w`x`h` block of 4x4 cells: raster anchor + 30-cache anchor.
#[derive(Clone, Copy)]
struct BlockPlace {
    scan4: usize,
    cache_idx: usize,
    w: usize,
    h: usize,
}

// --- CABAC context base offsets (`NEW_CTX_OFFSET_*`,
// reference/codec/decoder/core/inc/decoder_context.h). ---
const NEW_CTX_OFFSET_MB_TYPE_I: usize = 3;
const NEW_CTX_OFFSET_SKIP: usize = 11;
const NEW_CTX_OFFSET_SUBMB_TYPE: usize = 21;
const NEW_CTX_OFFSET_MVD: usize = 40;
const NEW_CTX_OFFSET_REF_NO: usize = 54;
const NEW_CTX_OFFSET_DELTA_QP: usize = 60;
const NEW_CTX_OFFSET_CIPR: usize = 64;
const NEW_CTX_OFFSET_IPR: usize = 68;
const NEW_CTX_OFFSET_CBP: usize = 73;
const NEW_CTX_OFFSET_TS_8X8_FLAG: usize = 399;
const CTX_NUM_MVD: usize = 7;
const CTX_NUM_CBP: usize = 4;

/// `g_kMvdBinPos2Ctx` (cabac_decoder.cpp): prefix-bin position -> ctxIdxInc for
/// the truncated-unary mvd prefix.
const MVD_BIN_POS2CTX: [usize; 8] = [0, 1, 2, 3, 3, 3, 3, 3];

/// `g_kCacheNzcScanIdx` (decoder_data_tables.cpp): block-scan index -> position
/// in the 8x6 (48-entry) non-zero-count cache. 16 luma, then Cb (4), Cr (4),
/// then luma-DC + 2 chroma-DC.
const CACHE_NZC_SCAN_IDX: [usize; 27] = [
    9, 10, 17, 18, 11, 12, 19, 20, 25, 26, 33, 34, 27, 28, 35, 36, // luma
    14, 15, 22, 23, // Cb
    38, 39, 46, 47, // Cr
    41, 42, 43, // luma-DC, chroma-DC
];

/// Snapshot of the four spatial neighbours, mirroring `SWelsNeighAvail`
/// (`GetNeighborAvailMbType`) plus the per-block neighbour state the CABAC
/// derivations read. Captured up front so `ctx` can be mutated freely after.
struct Neigh {
    left_avail: bool,
    top_avail: bool,
    left_top_avail: bool,
    right_top_avail: bool,
    left_type: Option<MbType>,
    top_type: Option<MbType>,
    left_cbp: u8,
    top_cbp: u8,
    left_cbf_dc: u16,
    top_cbf_dc: u16,
    left_nzc_luma: [i8; 16],
    left_nzc_chroma: [i8; 8],
    top_nzc_luma: [i8; 16],
    top_nzc_chroma: [i8; 8],
    left_best: [i8; 16],
    left_is_nxn: bool,
    top_best: [i8; 16],
    top_is_nxn: bool,
    left_chroma_mode: i8,
    top_chroma_mode: i8,
    /// `pRefIndex[block]` of the left/top neighbours used by skip/ref ctx.
    left_skip: bool,
    top_skip: bool,
}

impl Neigh {
    fn build(ctx: &DecoderContext, mb_xy: usize) -> Self {
        let na = ctx.neighbors(mb_xy);
        let mb_x = mb_xy % ctx.mb_width;
        let cur = ctx.slice_idc[mb_xy];
        // right-top availability (slice-gated), as GetNeighborAvailMbType.
        let right_top_avail = mb_xy >= ctx.mb_width
            && mb_x != ctx.mb_width - 1
            && ctx.slice_idc[mb_xy - ctx.mb_width + 1] == cur;

        type SnapResult = ([i8; 16], [i8; 8], [i8; 16], bool, i8, u8, u16, MbType, bool);
        let snap = |avail: bool, xy: usize| -> SnapResult {
            if avail {
                let mut nl = [0i8; 16];
                let mut nc = [0i8; 8];
                let mut bm = [0i8; 16];
                nl.copy_from_slice(ctx.nzc_luma_mb(xy));
                nc.copy_from_slice(ctx.nzc_chroma_mb(xy));
                bm.copy_from_slice(&ctx.i4_best_mode[xy * 16..xy * 16 + 16]);
                let t = ctx.mb_type[xy];
                (nl, nc, bm, t.is_intra_nxn(), ctx.chroma_mode[xy], ctx.cbp[xy], ctx.cbf_dc[xy], t, t.is_skip())
            } else {
                ([-1; 16], [-1; 8], [-1; 16], false, 0, 0, 0, MbType::Intra4x4, false)
            }
        };

        let (lnl, lnc, lbm, lnxn, lcm, lcbp, lcbf, ltype, lskip) = snap(na.left, na.left_xy);
        let (tnl, tnc, tbm, tnxn, tcm, tcbp, tcbf, ttype, tskip) = snap(na.top, na.top_xy);

        Neigh {
            left_avail: na.left,
            top_avail: na.top,
            left_top_avail: na.top_left,
            right_top_avail,
            left_type: if na.left { Some(ltype) } else { None },
            top_type: if na.top { Some(ttype) } else { None },
            left_cbp: lcbp,
            top_cbp: tcbp,
            left_cbf_dc: lcbf,
            top_cbf_dc: tcbf,
            left_nzc_luma: lnl,
            left_nzc_chroma: lnc,
            top_nzc_luma: tnl,
            top_nzc_chroma: tnc,
            left_best: lbm,
            left_is_nxn: lnxn,
            top_best: tbm,
            top_is_nxn: tnxn,
            left_chroma_mode: lcm,
            top_chroma_mode: tcm,
            left_skip: lskip,
            top_skip: tskip,
        }
    }
}

// ---------------------------------------------------------------------------
// Low-level CABAC binarisation helpers (cabac_decoder.cpp).
// ---------------------------------------------------------------------------

/// `DecodeUnaryBinCabac`: a first bin against `ctx0`, and if set, a unary tail
/// against `ctx1`.
fn decode_unary(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, ctx0: usize, ctx1: usize) -> u32 {
    if dec.decode_decision(ctxs.ctx(ctx0)) == 0 {
        return 0;
    }
    let mut sym = 0u32;
    loop {
        let c = dec.decode_decision(ctxs.ctx(ctx1));
        sym += 1;
        if c == 0 {
            return sym;
        }
    }
}

/// `DecodeUEGMvCabac`: truncated-unary prefix (ctx via `g_kMvdBinPos2Ctx`) then a
/// 3rd-order Exp-Golomb bypass suffix.
fn decode_ueg_mv(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, base: usize) -> u32 {
    if dec.decode_decision(ctxs.ctx(base + MVD_BIN_POS2CTX[0])) == 0 {
        return 0;
    }
    let mut code = 0u32;
    let mut count = 1usize;
    let mut tmp;
    loop {
        tmp = dec.decode_decision(ctxs.ctx(base + MVD_BIN_POS2CTX[count]));
        count += 1;
        code += 1;
        if tmp == 0 || count == 8 {
            break;
        }
    }
    if tmp != 0 {
        code += dec.decode_exp_bypass(3) + 1;
    }
    code
}

// ---------------------------------------------------------------------------
// MB-type / mode decoders.
// ---------------------------------------------------------------------------

/// `ParseMBTypeISliceCabac`: returns the I-slice mb_type code (0 = I_NxN,
/// 1..24 = I16x16, 25 = I_PCM).
fn parse_mb_type_i(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, n: &Neigh) -> u32 {
    let base = NEW_CTX_OFFSET_MB_TYPE_I;
    let idx_a = (n.left_avail && !is_i4x4(n.left_type)) as usize;
    let idx_b = (n.top_avail && !is_i4x4(n.top_type)) as usize;
    let ctx_inc = idx_a + idx_b;
    if dec.decode_decision(ctxs.ctx(base + ctx_inc)) == 0 {
        return 0; // I_NxN
    }
    if dec.decode_terminate() == 1 {
        return 25; // I_PCM
    }
    let mut val = 1u32;
    val += 12 * dec.decode_decision(ctxs.ctx(base + 3));
    if dec.decode_decision(ctxs.ctx(base + 4)) != 0 {
        val += 4;
        if dec.decode_decision(ctxs.ctx(base + 5)) != 0 {
            val += 4;
        }
    }
    val += dec.decode_decision(ctxs.ctx(base + 6)) << 1;
    val += dec.decode_decision(ctxs.ctx(base + 7));
    val
}

#[inline]
fn is_i4x4(t: Option<MbType>) -> bool {
    // MB_TYPE_INTRA4x4 / INTRA8x8: in our model only Intra4x4 (8x8 unsupported).
    matches!(t, Some(MbType::Intra4x4))
}

/// `ParseIntraPredModeLumaCabac`: returns -1 (use predicted) or an explicit
/// 0..7 mode.
fn parse_ipr_luma(dec: &mut CabacDecoder, ctxs: &mut CabacContexts) -> i32 {
    if dec.decode_decision(ctxs.ctx(NEW_CTX_OFFSET_IPR)) == 1 {
        return -1;
    }
    let mut v = 0i32;
    v |= dec.decode_decision(ctxs.ctx(NEW_CTX_OFFSET_IPR + 1)) as i32;
    v |= (dec.decode_decision(ctxs.ctx(NEW_CTX_OFFSET_IPR + 1)) as i32) << 1;
    v |= (dec.decode_decision(ctxs.ctx(NEW_CTX_OFFSET_IPR + 1)) as i32) << 2;
    v
}

/// `ParseIntraPredModeChromaCabac`: returns chroma pred mode 0..3.
fn parse_ipr_chroma(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, n: &Neigh) -> i32 {
    let idx_b = (n.top_avail && (1..=3).contains(&n.top_chroma_mode)) as usize;
    let idx_a = (n.left_avail && (1..=3).contains(&n.left_chroma_mode)) as usize;
    let ctx_inc = idx_a + idx_b;
    let base = NEW_CTX_OFFSET_CIPR;
    if dec.decode_decision(ctxs.ctx(base + ctx_inc)) == 0 {
        return 0;
    }
    if dec.decode_decision(ctxs.ctx(base + 3)) == 0 {
        return 1;
    }
    let mut sym = 1i32;
    if dec.decode_decision(ctxs.ctx(base + 3)) != 0 {
        sym += 1;
    }
    sym + 1
}

/// `ParseCbpInfoCabac`: 4 luma 8x8 bits + 2 chroma bits -> packed cbp.
fn parse_cbp(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, n: &Neigh) -> u32 {
    let base = NEW_CTX_OFFSET_CBP;
    let not_pcm_l = n.left_type.is_some();
    let not_pcm_t = n.top_type.is_some();
    let b_top0 = (n.top_avail && not_pcm_t && (n.top_cbp & (1 << 2)) == 0) as usize;
    let b_top1 = (n.top_avail && not_pcm_t && (n.top_cbp & (1 << 3)) == 0) as usize;
    let a_left0 = (n.left_avail && not_pcm_l && (n.left_cbp & (1 << 1)) == 0) as usize;
    let a_left1 = (n.left_avail && not_pcm_l && (n.left_cbp & (1 << 3)) == 0) as usize;

    let mut cbp = 0u32;
    let bit0 = dec.decode_decision(ctxs.ctx(base + a_left0 + (b_top0 << 1)));
    if bit0 != 0 {
        cbp += 0x01;
    }
    let bit1 = dec.decode_decision(ctxs.ctx(base + (bit0 == 0) as usize + (b_top1 << 1)));
    if bit1 != 0 {
        cbp += 0x02;
    }
    let bit2 = dec.decode_decision(ctxs.ctx(base + a_left1 + (((bit0 == 0) as usize) << 1)));
    if bit2 != 0 {
        cbp += 0x04;
    }
    let bit3 =
        dec.decode_decision(ctxs.ctx(base + (bit2 == 0) as usize + (((bit1 == 0) as usize) << 1)));
    if bit3 != 0 {
        cbp += 0x08;
    }

    // Chroma (4:2:0).
    let idx_b = (n.top_avail && (!not_pcm_t || (n.top_cbp >> 4) != 0)) as usize;
    let idx_a = (n.left_avail && (!not_pcm_l || (n.left_cbp >> 4) != 0)) as usize;
    let bit4 = dec.decode_decision(ctxs.ctx(base + CTX_NUM_CBP + idx_a + (idx_b << 1)));
    if bit4 != 0 {
        let idx_b = (n.top_avail && (!not_pcm_t || (n.top_cbp >> 4) == 2)) as usize;
        let idx_a = (n.left_avail && (!not_pcm_l || (n.left_cbp >> 4) == 2)) as usize;
        let bit5 = dec.decode_decision(ctxs.ctx(base + 2 * CTX_NUM_CBP + idx_a + (idx_b << 1)));
        cbp += 1 << (4 + bit5);
    }
    cbp
}

/// `ParseDeltaQpCabac`: signed mb_qp_delta.
fn parse_delta_qp(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, last_delta_qp: &mut i32) -> i32 {
    let base = NEW_CTX_OFFSET_DELTA_QP;
    let ctx_inc = (*last_delta_qp != 0) as usize;
    let mut delta = 0i32;
    if dec.decode_decision(ctxs.ctx(base + ctx_inc)) != 0 {
        let mut code = decode_unary(dec, ctxs, base + 2, base + 3) + 1;
        code += 1;
        delta = (code as i32) >> 1;
        if (code & 1) == 0 {
            delta = -delta;
        }
    }
    *last_delta_qp = delta;
    delta
}

/// `ParseEndOfSliceCabac`.
pub fn parse_end_of_slice(dec: &mut CabacDecoder) -> bool {
    dec.decode_terminate() != 0
}

// ---------------------------------------------------------------------------
// Non-zero-count cache + cbf neighbour derivation.
// ---------------------------------------------------------------------------

/// `WelsFillCacheNonZeroCount`: seed the 48-entry nzc cache border cells from the
/// left/top neighbours (`-1` == unavailable, the `0xff` sentinel in the C).
fn fill_nzc_cache(cache: &mut [i16; 48], n: &Neigh) {
    // Top row (luma cache[1..4], chroma cache[6,7] Cb / cache[30,31] Cr).
    if n.top_avail {
        for k in 0..4 {
            cache[1 + k] = n.top_nzc_luma[12 + k] as i16;
        }
        cache[6] = n.top_nzc_chroma[2] as i16;
        cache[7] = n.top_nzc_chroma[3] as i16;
        cache[30] = n.top_nzc_chroma[6] as i16;
        cache[31] = n.top_nzc_chroma[7] as i16;
    } else {
        for k in 0..4 {
            cache[1 + k] = -1;
        }
        cache[6] = -1;
        cache[7] = -1;
        cache[30] = -1;
        cache[31] = -1;
    }
    // Left column (luma cache[8,16,24,32], chroma Cb cache[13,21] / Cr [37,45]).
    if n.left_avail {
        cache[8] = n.left_nzc_luma[3] as i16;
        cache[16] = n.left_nzc_luma[7] as i16;
        cache[24] = n.left_nzc_luma[11] as i16;
        cache[32] = n.left_nzc_luma[15] as i16;
        cache[13] = n.left_nzc_chroma[1] as i16;
        cache[21] = n.left_nzc_chroma[3] as i16;
        cache[37] = n.left_nzc_chroma[5] as i16;
        cache[45] = n.left_nzc_chroma[7] as i16;
    } else {
        cache[8] = -1;
        cache[16] = -1;
        cache[24] = -1;
        cache[32] = -1;
        cache[13] = -1;
        cache[21] = -1;
        cache[37] = -1;
        cache[45] = -1;
    }
}

/// `ParseCbfInfoCabac` cbf bin for an AC / 4x4 block: neighbour ctx from the nzc
/// cache. `cur_intra` is `IS_INTRA` of the current MB (default nA/nB). PCM is
/// never produced, so the PCM term is always false.
fn cbf_ac(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    cache: &[i16; 48],
    z_index: usize,
    res_property: usize,
    cur_intra: bool,
) -> u32 {
    let pos = CACHE_NZC_SCAN_IDX[z_index];
    let mut na = cur_intra as u32;
    let mut nb = cur_intra as u32;
    let top = cache[pos - 8];
    if top != -1 {
        nb = (top != 0) as u32;
    }
    let left = cache[pos - 1];
    if left != -1 {
        na = (left != 0) as u32;
    }
    coded_block_flag(dec, ctxs, res_property, na + (nb << 1))
}

/// `ParseCbfInfoCabac` cbf bin for a DC block (luma DC / chroma DC), via the
/// per-MB `pCbfDc` neighbour state.
fn cbf_dc(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    n: &Neigh,
    res_property: usize,
    cur_intra: bool,
    cur_cbf_dc: &mut u16,
) -> u32 {
    let mut na = cur_intra as u32;
    let mut nb = cur_intra as u32;
    if n.top_avail {
        nb = ((n.top_cbf_dc >> res_property) & 1) as u32;
    }
    if n.left_avail {
        na = ((n.left_cbf_dc >> res_property) & 1) as u32;
    }
    let bit = coded_block_flag(dec, ctxs, res_property, na + (nb << 1));
    if bit != 0 {
        *cur_cbf_dc |= 1 << res_property;
    }
    bit
}

// ---------------------------------------------------------------------------
// Residual decode (mirrors WelsDecodeMbCabacISliceBaseMode0 residual section).
// ---------------------------------------------------------------------------

fn parse_residuals_cabac(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    n: &Neigh,
    params: ResidualParams,
    out: ResidualOut,
    coeffs: &mut [i16; 384],
) {
    let ResidualParams { mb_type, cbp_l, cbp_c, luma_qp, chroma_qp } = params;
    let ResidualOut { cur_nzc_luma, cur_nzc_chroma, cur_cbf_dc } = out;
    let mut cache = [0i16; 48];
    fill_nzc_cache(&mut cache, n);
    let cur_intra = mb_type.is_intra();
    let deq_l = &G_KUI_DEQUANT_COEFF[luma_qp as usize];
    let mut out = [0i32; 16];

    if mb_type == MbType::Intra16x16 {
        // Luma DC (always present; cbf may be 0).
        let bit = cbf_dc(dec, ctxs, n, I16_LUMA_DC, cur_intra, cur_cbf_dc);
        out.fill(0);
        let total = if bit != 0 {
            residual_block_cabac(dec, ctxs, I16_LUMA_DC, 16, &mut out).unwrap()
        } else {
            0
        };
        // cache update for luma DC block-scan index 0 (overwritten by AC).
        cache[CACHE_NZC_SCAN_IDX[0]] = total as i16;
        if bit != 0 {
            for (s, &lvl) in out.iter().enumerate() {
                coeffs[G_KUI_LUMA_DC_ZIGZAG_SCAN[s] as usize] = lvl as i16;
            }
            luma_dc_dequant_idct(coeffs, luma_qp);
        }

        // Luma AC.
        if cbp_l != 0 {
            for i in 0..16 {
                let raster = BLOCK_RASTER[i];
                let bit = cbf_ac(dec, ctxs, &cache, i, I16_LUMA_AC, cur_intra);
                out.fill(0);
                let total = if bit != 0 {
                    residual_block_cabac(dec, ctxs, I16_LUMA_AC, 16, &mut out).unwrap()
                } else {
                    0
                };
                cache[CACHE_NZC_SCAN_IDX[i]] = total as i16;
                cur_nzc_luma[raster] = total as i8;
                if bit != 0 {
                    let cbase = i * 16;
                    for s in 0..15 {
                        if out[s] != 0 {
                            let j = G_KUI_ZIGZAG_SCAN[s + 1] as usize;
                            coeffs[cbase + j] = (out[s] * deq_l[j & 7] as i32) as i16;
                        }
                    }
                }
            }
        }
    } else {
        // I_4x4: per 8x8, four 4x4 blocks (DC+AC together, 16-coeff zig-zag).
        for id8 in 0..4 {
            if cbp_l & (1 << id8) == 0 {
                continue;
            }
            for id4 in 0..4 {
                let i = id8 * 4 + id4;
                let raster = BLOCK_RASTER[i];
                let bit = cbf_ac(dec, ctxs, &cache, i, LUMA_DC_AC, cur_intra);
                out.fill(0);
                let total = if bit != 0 {
                    residual_block_cabac(dec, ctxs, LUMA_DC_AC, 16, &mut out).unwrap()
                } else {
                    0
                };
                cache[CACHE_NZC_SCAN_IDX[i]] = total as i16;
                cur_nzc_luma[raster] = total as i8;
                if bit != 0 {
                    let cbase = i * 16;
                    for s in 0..16 {
                        if out[s] != 0 {
                            let j = G_KUI_ZIGZAG_SCAN[s] as usize;
                            coeffs[cbase + j] = (out[s] * deq_l[j & 7] as i32) as i16;
                        }
                    }
                }
            }
        }
    }

    // Chroma DC (cbp_c 1 or 2).
    if cbp_c == 1 || cbp_c == 2 {
        for c in 0..2 {
            let res = if c == 0 { CHROMA_DC_U } else { CHROMA_DC_V };
            let cbase = 256 + c * 64;
            let bit = cbf_dc(dec, ctxs, n, res, cur_intra, cur_cbf_dc);
            out.fill(0);
            if bit != 0 {
                residual_block_cabac(dec, ctxs, res, 4, &mut out).unwrap();
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
    }

    // Chroma AC (cbp_c == 2).
    if cbp_c == 2 {
        for c in 0..2 {
            let res = if c == 0 { CHROMA_AC_U } else { CHROMA_AC_V };
            let deq_c = &G_KUI_DEQUANT_COEFF[chroma_qp[c] as usize];
            for b in 0..4 {
                let z = 16 + c * 4 + b;
                let bit = cbf_ac(dec, ctxs, &cache, z, res, cur_intra);
                out.fill(0);
                let total = if bit != 0 {
                    residual_block_cabac(dec, ctxs, res, 16, &mut out).unwrap()
                } else {
                    0
                };
                cache[CACHE_NZC_SCAN_IDX[z]] = total as i16;
                cur_nzc_chroma[c * 4 + b] = total as i8;
                if bit != 0 {
                    let cbase = 256 + c * 64 + b * 16;
                    for s in 0..15 {
                        if out[s] != 0 {
                            let j = G_KUI_ZIGZAG_SCAN[s + 1] as usize;
                            coeffs[cbase + j] = (out[s] * deq_c[j & 7] as i32) as i16;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// I-slice MB driver.
// ---------------------------------------------------------------------------

/// Decode one I-slice macroblock in CABAC into `ctx` + `coeffs`. Returns the
/// `end_of_slice_flag`.
pub fn decode_mb_cabac_islice(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    mb: MbCtx,
    qp: QpState,
    coeffs: &mut [i16; 384],
) -> Result<bool> {
    let MbCtx { ctx, mb_xy, pps } = mb;
    let n = Neigh::build(ctx, mb_xy);
    let ui_mb_type = parse_mb_type_i(dec, ctxs, &n);
    decode_intra_mb_body(dec, ctxs, MbCtx { ctx, mb_xy, pps }, qp, coeffs, &n, ui_mb_type)?;
    Ok(parse_end_of_slice(dec))
}

/// Shared intra-MB body (I-slice mb_type already decoded; P-slice intra hands in
/// the same `ui_mb_type` code via the I-slice numbering).
fn decode_intra_mb_body(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    mb: MbCtx,
    qp: QpState,
    coeffs: &mut [i16; 384],
    n: &Neigh,
    ui_mb_type: u32,
) -> Result<()> {
    let MbCtx { ctx, mb_xy, pps } = mb;
    let QpState { last_mb_qp, last_delta_qp } = qp;
    if ui_mb_type == 25 {
        return Err(DecodeError::Unsupported("I_PCM (CABAC)"));
    }
    if ui_mb_type > 25 {
        return Err(DecodeError::InvalidSyntax("intra mb_type (cabac)"));
    }

    let mut cur_nzc_luma = [0i8; 16];
    let mut cur_nzc_chroma = [0i8; 8];
    let mut best_mode = [-1i8; 16];
    let mut final_mode = [2i8; 16];
    let mut cur_cbf_dc = 0u16;

    let mb_type;
    let mut i16_mode = 0i8;
    let mut chroma_mode;
    let cbp: u8;

    if ui_mb_type == 0 {
        // I_NxN. transform_size_8x8 (I_8x8) is unsupported.
        if pps.transform_8x8_mode_flag {
            let t8 = parse_transform_size_8x8(dec, ctxs, ctx, mb_xy, n);
            if t8 {
                return Err(DecodeError::Unsupported("transform_size_8x8 / I_8x8 (CABAC)"));
            }
        }
        mb_type = MbType::Intra4x4;
        chroma_mode = parse_intra4x4_cabac(dec, ctxs, ctx, mb_xy, n, &mut best_mode, &mut final_mode)?;
        let cbp_v = parse_cbp(dec, ctxs, n);
        cbp = cbp_v as u8;
    } else {
        mb_type = MbType::Intra16x16;
        i16_mode = ((ui_mb_type - 1) & 3) as i8;
        cbp = I16_CBP_TABLE[((ui_mb_type - 1) >> 2) as usize];
        let neigh_avail = ((n.left_avail as i32) << 2)
            | ((n.left_top_avail as i32) << 1)
            | (n.top_avail as i32);
        check_intra16x16_mode(neigh_avail, &mut i16_mode)?;
        let cm = parse_ipr_chroma(dec, ctxs, n);
        chroma_mode = cm as i8;
        check_intra_chroma_mode(neigh_avail, &mut chroma_mode)?;
    }

    let cbp_l = cbp & 0x0f;
    let cbp_c = cbp >> 4;

    // QP / residual.
    let luma_qp: i32;
    if cbp == 0 && mb_type == MbType::Intra4x4 {
        *last_delta_qp = 0;
        luma_qp = *last_mb_qp;
    } else {
        let qp_delta = parse_delta_qp(dec, ctxs, last_delta_qp);
        if !(-26..=25).contains(&qp_delta) {
            return Err(DecodeError::InvalidSyntax("mb_qp_delta (cabac)"));
        }
        luma_qp = (*last_mb_qp + qp_delta + 52) % 52;
        *last_mb_qp = luma_qp;
    }
    let chroma_qp = [
        CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[0], 0, 51) as usize] as i32,
        CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[1], 0, 51) as usize] as i32,
    ];

    if cbp != 0 || mb_type == MbType::Intra16x16 {
        coeffs.iter_mut().for_each(|c| *c = 0);
        parse_residuals_cabac(
            dec,
            ctxs,
            n,
            ResidualParams { mb_type, cbp_l, cbp_c, luma_qp, chroma_qp },
            ResidualOut {
                cur_nzc_luma: &mut cur_nzc_luma,
                cur_nzc_chroma: &mut cur_nzc_chroma,
                cur_cbf_dc: &mut cur_cbf_dc,
            },
            coeffs,
        );
    }

    // Commit MB state to the context (same fields the CAVLC parse writes).
    ctx.mb_type[mb_xy] = mb_type;
    ctx.i16_mode[mb_xy] = i16_mode;
    ctx.chroma_mode[mb_xy] = chroma_mode;
    ctx.cbp[mb_xy] = cbp;
    ctx.cbf_dc[mb_xy] = cur_cbf_dc;
    ctx.luma_qp[mb_xy] = luma_qp as i8;
    ctx.chroma_qp[mb_xy * 2] = chroma_qp[0] as i8;
    ctx.chroma_qp[mb_xy * 2 + 1] = chroma_qp[1] as i8;
    ctx.nzc_luma[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&cur_nzc_luma);
    ctx.nzc_chroma[mb_xy * 8..mb_xy * 8 + 8].copy_from_slice(&cur_nzc_chroma);
    ctx.i4_best_mode[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&best_mode);
    ctx.i4_final_mode[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&final_mode);
    for b in 0..16 {
        ctx.ref_idx[mb_xy * 16 + b] = -1;
        ctx.ref_pic_id[mb_xy * 16 + b] = -1;
        ctx.mv[(mb_xy * 16 + b) * 2] = 0;
        ctx.mv[(mb_xy * 16 + b) * 2 + 1] = 0;
    }
    Ok(())
}

/// `ParseTransformSize8x8FlagCabac`.
fn parse_transform_size_8x8(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    ctx: &DecoderContext,
    mb_xy: usize,
    n: &Neigh,
) -> bool {
    // Neighbour transform_size_8x8 flags are not tracked (we never produce
    // I_8x8); treat both as 0 -> ctx_inc 0, matching a stream without 8x8.
    let _ = (ctx, mb_xy, n);
    let base = NEW_CTX_OFFSET_TS_8X8_FLAG;
    dec.decode_decision(ctxs.ctx(base)) != 0
}

/// CABAC `ParseIntra4x4Mode`: 16 luma modes (each via
/// `ParseIntraPredModeLuma`) + the chroma mode. Mirrors the CAVLC
/// `parse_intra4x4`, swapping the bit reads for CABAC. Returns the checked
/// chroma mode.
fn parse_intra4x4_cabac(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    _ctx: &DecoderContext,
    _mb_xy: usize,
    n: &Neigh,
    best_mode: &mut [i8; 16],
    final_mode: &mut [i8; 16],
) -> Result<i8> {
    let top_modes: [i8; 4] = if n.top_avail && n.top_is_nxn {
        [n.top_best[12], n.top_best[13], n.top_best[14], n.top_best[15]]
    } else if n.top_avail {
        [2; 4]
    } else {
        [-1; 4]
    };
    let left_modes: [i8; 4] = if n.left_avail && n.left_is_nxn {
        [n.left_best[3], n.left_best[7], n.left_best[11], n.left_best[15]]
    } else if n.left_avail {
        [2; 4]
    } else {
        [-1; 4]
    };

    let mut sample_avail = [0i32; 30];
    if n.left_avail {
        sample_avail[6] = 1;
        sample_avail[12] = 1;
        sample_avail[18] = 1;
        sample_avail[24] = 1;
    }
    if n.left_top_avail {
        sample_avail[0] = 1;
    }
    if n.top_avail {
        sample_avail[1] = 1;
        sample_avail[2] = 1;
        sample_avail[3] = 1;
        sample_avail[4] = 1;
    }
    if n.right_top_avail {
        sample_avail[5] = 1;
    }

    for i in 0..16 {
        let raster = BLOCK_RASTER[i];
        let bx = BLOCK_BX[i];
        let by = BLOCK_BY[i];

        let code = parse_ipr_luma(dec, ctxs);
        let top_mode = if by > 0 { best_mode[(by - 1) * 4 + bx] } else { top_modes[bx] };
        let left_mode = if bx > 0 { best_mode[by * 4 + bx - 1] } else { left_modes[by] };
        let pred_mode = if left_mode == -1 || top_mode == -1 {
            2
        } else {
            left_mode.min(top_mode)
        };
        let cur_best = if code == -1 {
            pred_mode
        } else {
            (code as i8) + if code as i8 >= pred_mode { 1 } else { 0 }
        };
        let cur_final = check_intra_nxn_mode(&sample_avail, cur_best, i)?;
        best_mode[raster] = cur_best;
        final_mode[raster] = cur_final;
        sample_avail[CACHE30_SCAN_IDX[i]] = 1;
    }

    let cm = parse_ipr_chroma(dec, ctxs, n);
    let mut chroma_mode = cm as i8;
    let chroma_neigh_avail =
        ((sample_avail[6]) << 2) | ((sample_avail[0]) << 1) | sample_avail[1];
    check_intra_chroma_mode(chroma_neigh_avail, &mut chroma_mode)?;
    Ok(chroma_mode)
}

// ---------------------------------------------------------------------------
// P-slice MB driver.
// ---------------------------------------------------------------------------

/// Inter neighbour MV/ref cache (30-entry), CABAC variant; also tracks the per-
/// block mvd cache the mvd-ctx derivation reads.
struct InterCacheC {
    mv: [[i16; 2]; 30],
    ref_idx: [i8; 30],
    mvd: [[i16; 2]; 30],
}

const REF_NOT_AVAIL_C: i8 = -2;
const REF_NOT_IN_LIST_C: i8 = -1;

impl InterCacheC {
    fn build(ctx: &DecoderContext, mb_xy: usize) -> Self {
        let mb_width = ctx.mb_width;
        let mb_x = mb_xy % mb_width;
        let mb_y = mb_xy / mb_width;
        let cur = ctx.slice_idc[mb_xy];
        let avail = |cond: bool, xy: usize| cond && ctx.slice_idc[xy] == cur;
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
        let mut ref_idx = [REF_NOT_AVAIL_C; 30];
        let mut mvd = [[0i16; 2]; 30];
        let mv_of = |xy: usize, b: usize| [ctx.mv[(xy * 16 + b) * 2], ctx.mv[(xy * 16 + b) * 2 + 1]];
        let mvd_of = |xy: usize, b: usize| [ctx.mvd[(xy * 16 + b) * 2], ctx.mvd[(xy * 16 + b) * 2 + 1]];
        let ref_of = |xy: usize, b: usize| ctx.ref_idx[xy * 16 + b];

        if left && ctx.mb_type[left_xy].is_inter() {
            for (k, &b) in [3usize, 7, 11, 15].iter().enumerate() {
                let c = [6, 12, 18, 24][k];
                mv[c] = mv_of(left_xy, b);
                mvd[c] = mvd_of(left_xy, b);
                ref_idx[c] = ref_of(left_xy, b);
            }
        } else {
            let r = if left { REF_NOT_IN_LIST_C } else { REF_NOT_AVAIL_C };
            for &c in &[6usize, 12, 18, 24] {
                ref_idx[c] = r;
            }
        }
        if left_top && ctx.mb_type[left_top_xy].is_inter() {
            mv[0] = mv_of(left_top_xy, 15);
            mvd[0] = mvd_of(left_top_xy, 15);
            ref_idx[0] = ref_of(left_top_xy, 15);
        } else {
            ref_idx[0] = if left_top { REF_NOT_IN_LIST_C } else { REF_NOT_AVAIL_C };
        }
        if top && ctx.mb_type[top_xy].is_inter() {
            for (k, &b) in [12usize, 13, 14, 15].iter().enumerate() {
                let c = 1 + k;
                mv[c] = mv_of(top_xy, b);
                mvd[c] = mvd_of(top_xy, b);
                ref_idx[c] = ref_of(top_xy, b);
            }
        } else {
            let r = if top { REF_NOT_IN_LIST_C } else { REF_NOT_AVAIL_C };
            for slot in &mut ref_idx[1..=4] {
                *slot = r;
            }
        }
        if right_top && ctx.mb_type[right_top_xy].is_inter() {
            mv[5] = mv_of(right_top_xy, 12);
            mvd[5] = mvd_of(right_top_xy, 12);
            ref_idx[5] = ref_of(right_top_xy, 12);
        } else {
            ref_idx[5] = if right_top { REF_NOT_IN_LIST_C } else { REF_NOT_AVAIL_C };
        }
        for &c in &[9usize, 11, 17, 21, 23] {
            ref_idx[c] = REF_NOT_AVAIL_C;
            mv[c] = [0, 0];
        }
        InterCacheC { mv, ref_idx, mvd }
    }
}

/// `ParseMBTypePSliceCabac`: returns the P-slice mb_type code. 0..4 = inter
/// partitions; 5 = intra-NxN; >=5 maps to the intra I-slice numbering via -5.
fn parse_mb_type_p(dec: &mut CabacDecoder, ctxs: &mut CabacContexts) -> Result<u32> {
    let base = NEW_CTX_OFFSET_SKIP;
    if dec.decode_decision(ctxs.ctx(base + 3)) != 0 {
        // Intra MB inside P slice.
        if dec.decode_decision(ctxs.ctx(base + 6)) != 0 {
            // Intra 16x16.
            if dec.decode_terminate() != 0 {
                return Ok(30); // I_PCM
            }
            let mut t = 6u32;
            t += 12 * dec.decode_decision(ctxs.ctx(base + 7));
            if dec.decode_decision(ctxs.ctx(base + 8)) != 0 {
                t += 4;
                if dec.decode_decision(ctxs.ctx(base + 8)) != 0 {
                    t += 4;
                }
            }
            t += dec.decode_decision(ctxs.ctx(base + 9)) << 1;
            t += dec.decode_decision(ctxs.ctx(base + 9));
            Ok(t)
        } else {
            Ok(5) // Intra 4x4
        }
    } else {
        if dec.decode_decision(ctxs.ctx(base + 4)) != 0 {
            if dec.decode_decision(ctxs.ctx(base + 6)) != 0 {
                Ok(1)
            } else {
                Ok(2)
            }
        } else if dec.decode_decision(ctxs.ctx(base + 5)) != 0 {
            Ok(3)
        } else {
            Ok(0)
        }
    }
}

/// `ParseSkipFlagCabac` (P slice).
fn parse_skip_flag(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, n: &Neigh) -> bool {
    let mut ctx_inc = NEW_CTX_OFFSET_SKIP;
    ctx_inc += (n.left_avail && !n.left_skip) as usize + (n.top_avail && !n.top_skip) as usize;
    dec.decode_decision(ctxs.ctx(ctx_inc)) != 0
}

/// `ParseSubMBTypeCabac` (P slice): returns 0..3 (P_8x8/8x4/4x8/4x4).
fn parse_sub_mb_type_p(dec: &mut CabacDecoder, ctxs: &mut CabacContexts) -> u32 {
    let base = NEW_CTX_OFFSET_SUBMB_TYPE;
    if dec.decode_decision(ctxs.ctx(base)) != 0 {
        0
    } else if dec.decode_decision(ctxs.ctx(base + 1)) != 0 {
        3 - dec.decode_decision(ctxs.ctx(base + 2))
    } else {
        1
    }
}

/// `ParseRefIdxCabac`: ref_idx_l0 with neighbour ctx from the ref-index cache.
fn parse_ref_idx(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    cache: &InterCacheC,
    cur_ref_in_mb: &[i8; 16],
    n: &Neigh,
    z_index: usize,
    active_ref: usize,
) -> u32 {
    if active_ref == 1 {
        return 0;
    }
    let c = CACHE30_SCAN_IDX[z_index];
    let scan = SCAN4[z_index];
    let (idx_a, idx_b);
    if z_index == 0 {
        idx_b = (n.top_avail && cache.ref_idx[c - 6] > 0) as u32;
        idx_a = (n.left_avail && cache.ref_idx[c - 1] > 0) as u32;
    } else if z_index == 4 {
        idx_b = (n.top_avail && cache.ref_idx[c - 6] > 0) as u32;
        idx_a = (cur_ref_in_mb[scan - 1] > 0) as u32;
    } else if z_index == 8 {
        idx_b = (cur_ref_in_mb[scan - 4] > 0) as u32;
        idx_a = (n.left_avail && cache.ref_idx[c - 1] > 0) as u32;
    } else {
        idx_b = (cur_ref_in_mb[scan - 4] > 0) as u32;
        idx_a = (cur_ref_in_mb[scan - 1] > 0) as u32;
    }
    let ctx_inc = idx_a + (idx_b << 1);
    let base = NEW_CTX_OFFSET_REF_NO;
    let mut code = dec.decode_decision(ctxs.ctx(base + ctx_inc as usize));
    if code != 0 {
        code = decode_unary(dec, ctxs, base + 4, base + 5) + 1;
    }
    code
}

/// `ParseMvdInfoCabac` for one component: neighbour ctx from the mvd cache.
fn parse_mvd(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    cache: &InterCacheC,
    z_index: usize,
    comp: usize,
) -> i16 {
    let c = CACHE30_SCAN_IDX[z_index];
    let mut sum = 0i32;
    if cache.ref_idx[c - 6] >= 0 {
        sum += (cache.mvd[c - 6][comp] as i32).abs();
    }
    if cache.ref_idx[c - 1] >= 0 {
        sum += (cache.mvd[c - 1][comp] as i32).abs();
    }
    let ctx_inc = if sum >= 3 { 1 + (sum > 32) as usize } else { 0 };
    let base = NEW_CTX_OFFSET_MVD + comp * CTX_NUM_MVD;
    if dec.decode_decision(ctxs.ctx(base + ctx_inc)) == 0 {
        return 0;
    }
    let mag = decode_ueg_mv(dec, ctxs, base + 3) + 1;
    let mut val = mag as i16;
    if dec.decode_bypass() != 0 {
        val = -val;
    }
    val
}

/// Decode one P-slice macroblock in CABAC. Returns `end_of_slice_flag`.
pub fn decode_mb_cabac_pslice(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    mb: MbCtx,
    qp: QpState,
    ref_pic_ids: &[i32],
    coeffs: &mut [i16; 384],
) -> Result<bool> {
    let MbCtx { ctx, mb_xy, pps } = mb;
    let n = Neigh::build(ctx, mb_xy);
    let ref_count = ref_pic_ids.len();

    if !parse_skip_flag(dec, ctxs, &n) {
        let ui_mb_type = parse_mb_type_p(dec, ctxs)?;
        if ui_mb_type < 5 {
            parse_inter_mb_cabac(
                dec,
                ctxs,
                MbCtx { ctx, mb_xy, pps },
                qp,
                ui_mb_type,
                InterRefs { n: &n, ref_pic_ids },
                coeffs,
            )?;
        } else if ui_mb_type == 30 {
            return Err(DecodeError::Unsupported("I_PCM (CABAC P)"));
        } else {
            decode_intra_mb_body(dec, ctxs, MbCtx { ctx, mb_xy, pps }, qp, coeffs, &n, ui_mb_type - 5)?;
        }
    } else {
        // P_Skip.
        let QpState { last_mb_qp, last_delta_qp } = qp;
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
        *last_delta_qp = 0;
        commit_inter_meta(
            MbCtx { ctx: &mut *ctx, mb_xy, pps },
            MbType::PSkip,
            0,
            luma_qp,
            &[0; 16],
            &[0; 8],
        );
        ctx.cbf_dc[mb_xy] = 0;
    }
    Ok(parse_end_of_slice(dec))
}

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

fn parse_inter_mb_cabac(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    mb: MbCtx,
    qp: QpState,
    ui_mb_type: u32,
    refs: InterRefs,
    coeffs: &mut [i16; 384],
) -> Result<()> {
    let MbCtx { ctx, mb_xy, pps } = mb;
    let QpState { last_mb_qp, last_delta_qp } = qp;
    let InterRefs { n, ref_pic_ids } = refs;
    let mb_type = match ui_mb_type {
        0 => MbType::Inter16x16,
        1 => MbType::Inter16x8,
        2 => MbType::Inter8x16,
        3 => MbType::Inter8x8,
        _ => MbType::Inter8x8Ref0,
    };
    let ref_count = ref_pic_ids.len();
    let mut cache = InterCacheC::build(ctx, mb_xy);
    // current-MB ref index (raster) used by ref-idx ctx derivation.
    let mut cur_ref = [-1i8; 16];

    parse_inter_motion_cabac(
        dec,
        ctxs,
        MbCtx { ctx: &mut *ctx, mb_xy, pps },
        MotionState { cache: &mut cache, cur_ref: &mut cur_ref },
        mb_type,
        ref_count,
        refs,
    )?;

    let cbp_v = parse_cbp(dec, ctxs, n);
    let cbp = cbp_v as u8;
    let cbp_l = cbp & 0x0f;
    let cbp_c = cbp >> 4;

    let luma_qp: i32;
    let mut cur_nzc_luma = [0i8; 16];
    let mut cur_nzc_chroma = [0i8; 8];
    let mut cur_cbf_dc = 0u16;
    if cbp != 0 {
        coeffs.iter_mut().for_each(|c| *c = 0);
        let qp_delta = parse_delta_qp(dec, ctxs, last_delta_qp);
        if !(-26..=25).contains(&qp_delta) {
            return Err(DecodeError::InvalidSyntax("mb_qp_delta (cabac P)"));
        }
        luma_qp = (*last_mb_qp + qp_delta + 52) % 52;
        *last_mb_qp = luma_qp;
        let chroma_qp = [
            CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[0], 0, 51) as usize] as i32,
            CHROMA_QP_TABLE[clip3(luma_qp + pps.chroma_qp_index_offset[1], 0, 51) as usize] as i32,
        ];
        parse_residuals_cabac(
            dec,
            ctxs,
            n,
            ResidualParams { mb_type, cbp_l, cbp_c, luma_qp, chroma_qp },
            ResidualOut {
                cur_nzc_luma: &mut cur_nzc_luma,
                cur_nzc_chroma: &mut cur_nzc_chroma,
                cur_cbf_dc: &mut cur_cbf_dc,
            },
            coeffs,
        );
    } else {
        *last_delta_qp = 0;
        luma_qp = *last_mb_qp;
    }
    commit_inter_meta(
        MbCtx { ctx: &mut *ctx, mb_xy, pps },
        mb_type,
        cbp,
        luma_qp,
        &cur_nzc_luma,
        &cur_nzc_chroma,
    );
    ctx.cbf_dc[mb_xy] = cur_cbf_dc;
    Ok(())
}

fn parse_inter_motion_cabac(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    mb: MbCtx,
    motion: MotionState,
    mb_type: MbType,
    ref_count: usize,
    refs: InterRefs,
) -> Result<()> {
    let MbCtx { ctx, mb_xy, pps: _ } = mb;
    let MotionState { cache, cur_ref } = motion;
    let InterRefs { n, ref_pic_ids } = refs;
    match mb_type {
        MbType::Inter16x16 => {
            let iref = parse_ref_idx(dec, ctxs, cache, cur_ref, n, 0, ref_count) as i8;
            store_ref_block(
                ctx,
                cache,
                cur_ref,
                mb_xy,
                BlockPlace { scan4: 0, cache_idx: CACHE30_SCAN_IDX[0], w: 4, h: 4 },
                iref,
                ref_pic_ids[iref as usize],
            );
            let mvp = pred_mv(&cache.mv, &cache.ref_idx, 0, 4, iref);
            let mx = parse_mvd(dec, ctxs, cache, 0, 0);
            let my = parse_mvd(dec, ctxs, cache, 0, 1);
            let mv = [mvp[0] + mx, mvp[1] + my];
            store_mvmvd_block(
                ctx,
                cache,
                mb_xy,
                BlockPlace { scan4: 0, cache_idx: CACHE30_SCAN_IDX[0], w: 4, h: 4 },
                mv,
                [mx, my],
            );
        }
        MbType::Inter16x8 => {
            let mut r = [0i8; 2];
            for i in 0..2 {
                let part = i << 3;
                r[i] = parse_ref_idx(dec, ctxs, cache, cur_ref, n, part, ref_count) as i8;
                store_ref_block(
                    ctx,
                    cache,
                    cur_ref,
                    mb_xy,
                    BlockPlace { scan4: SCAN4[part], cache_idx: CACHE30_SCAN_IDX[part], w: 4, h: 2 },
                    r[i],
                    ref_pic_ids[r[i] as usize],
                );
            }
            for (i, &ri) in r.iter().enumerate() {
                let part = i << 3;
                let mvp = pred_inter16x8(&cache.mv, &cache.ref_idx, part, ri);
                let mx = parse_mvd(dec, ctxs, cache, part, 0);
                let my = parse_mvd(dec, ctxs, cache, part, 1);
                let mv = [mvp[0] + mx, mvp[1] + my];
                store_mvmvd_block(
                    ctx,
                    cache,
                    mb_xy,
                    BlockPlace { scan4: SCAN4[part], cache_idx: CACHE30_SCAN_IDX[part], w: 4, h: 2 },
                    mv,
                    [mx, my],
                );
            }
        }
        MbType::Inter8x16 => {
            let mut r = [0i8; 2];
            for i in 0..2 {
                let part = i << 2;
                r[i] = parse_ref_idx(dec, ctxs, cache, cur_ref, n, part, ref_count) as i8;
                store_ref_block(
                    ctx,
                    cache,
                    cur_ref,
                    mb_xy,
                    BlockPlace { scan4: SCAN4[part], cache_idx: CACHE30_SCAN_IDX[part], w: 2, h: 4 },
                    r[i],
                    ref_pic_ids[r[i] as usize],
                );
            }
            for (i, &ri) in r.iter().enumerate() {
                let part = i << 2;
                let mvp = pred_inter8x16(&cache.mv, &cache.ref_idx, part, ri);
                let mx = parse_mvd(dec, ctxs, cache, part, 0);
                let my = parse_mvd(dec, ctxs, cache, part, 1);
                let mv = [mvp[0] + mx, mvp[1] + my];
                store_mvmvd_block(
                    ctx,
                    cache,
                    mb_xy,
                    BlockPlace { scan4: SCAN4[part], cache_idx: CACHE30_SCAN_IDX[part], w: 2, h: 4 },
                    mv,
                    [mx, my],
                );
            }
        }
        MbType::Inter8x8 | MbType::Inter8x8Ref0 => {
            let ref0 = mb_type == MbType::Inter8x8Ref0;
            let eff = if ref0 { 1 } else { ref_count };
            let mut subs = [SubMbType::P8x8; 4];
            for s in subs.iter_mut() {
                let st = parse_sub_mb_type_p(dec, ctxs);
                *s = match st {
                    0 => SubMbType::P8x8,
                    1 => SubMbType::P8x4,
                    2 => SubMbType::P4x8,
                    _ => SubMbType::P4x4,
                };
            }
            ctx.sub_mb_type[mb_xy * 4..mb_xy * 4 + 4].copy_from_slice(&subs);
            // ref pass: fills only the raster per-MB ref (cur_ref + ctx), NOT the
            // 30-entry cache (UpdateP8x8RefIdxCabac). The cache ref is filled per
            // 8x8 inside the mv loop, just before that 8x8's mvs are predicted.
            let mut iref = [0i8; 4];
            for (i, ir) in iref.iter_mut().enumerate() {
                let z = i << 2;
                *ir = if ref0 { 0 } else { parse_ref_idx(dec, ctxs, cache, cur_ref, n, z, eff) as i8 };
                let s8 = SCAN4[z];
                let rp = ref_pic_ids[*ir as usize];
                for &rr in &[s8, s8 + 1, s8 + 4, s8 + 5] {
                    cur_ref[rr] = *ir;
                    ctx.ref_idx[mb_xy * 16 + rr] = *ir;
                    ctx.ref_pic_id[mb_xy * 16 + rr] = rp;
                }
            }
            for i in 0..4 {
                let z8 = i << 2;
                // Fill the cache ref for this 8x8 only now (UpdateP8x8RefCacheIdx).
                let c8 = CACHE30_SCAN_IDX[z8];
                cache.ref_idx[c8] = iref[i];
                cache.ref_idx[c8 + 1] = iref[i];
                cache.ref_idx[c8 + 6] = iref[i];
                cache.ref_idx[c8 + 7] = iref[i];
                let (pc, pw) = subs[i].part_info();
                let (w, h) = match subs[i] {
                    SubMbType::P8x8 => (2, 2),
                    SubMbType::P8x4 => (2, 1),
                    SubMbType::P4x8 => (1, 2),
                    SubMbType::P4x4 => (1, 1),
                };
                for j in 0..pc {
                    let part = z8 + j * pw;
                    let mvp = pred_mv(&cache.mv, &cache.ref_idx, part, pw, iref[i]);
                    let mx = parse_mvd(dec, ctxs, cache, part, 0);
                    let my = parse_mvd(dec, ctxs, cache, part, 1);
                    let mv = [mvp[0] + mx, mvp[1] + my];
                    store_mvmvd_block(
                        ctx,
                        cache,
                        mb_xy,
                        BlockPlace { scan4: SCAN4[part], cache_idx: CACHE30_SCAN_IDX[part], w, h },
                        mv,
                        [mx, my],
                    );
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Store a reference index across a `w`x`h` 4x4 block (raster `ctx` + `cur_ref`
/// + the 30-entry cache).
fn store_ref_block(
    ctx: &mut DecoderContext,
    cache: &mut InterCacheC,
    cur_ref: &mut [i8; 16],
    mb_xy: usize,
    place: BlockPlace,
    iref: i8,
    ref_pic: i32,
) {
    let BlockPlace { scan4, cache_idx, w, h } = place;
    for by in 0..h {
        for bx in 0..w {
            let raster = scan4 + by * 4 + bx;
            cur_ref[raster] = iref;
            ctx.ref_idx[mb_xy * 16 + raster] = iref;
            ctx.ref_pic_id[mb_xy * 16 + raster] = ref_pic;
            cache.ref_idx[cache_idx + by * 6 + bx] = iref;
        }
    }
}

/// Store a motion vector + mvd across a `w`x`h` 4x4 block (raster `ctx.mv` /
/// `ctx.mvd` + the 30-entry mv/mvd caches).
fn store_mvmvd_block(
    ctx: &mut DecoderContext,
    cache: &mut InterCacheC,
    mb_xy: usize,
    place: BlockPlace,
    mv: [i16; 2],
    mvd: [i16; 2],
) {
    let BlockPlace { scan4, cache_idx, w, h } = place;
    for by in 0..h {
        for bx in 0..w {
            let raster = scan4 + by * 4 + bx;
            let base = (mb_xy * 16 + raster) * 2;
            ctx.mv[base] = mv[0];
            ctx.mv[base + 1] = mv[1];
            ctx.mvd[base] = mvd[0];
            ctx.mvd[base + 1] = mvd[1];
            let c = cache_idx + by * 6 + bx;
            cache.mv[c] = mv;
            cache.mvd[c] = mvd;
        }
    }
}
