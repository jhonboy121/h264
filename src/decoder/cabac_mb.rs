//! CABAC residual-block syntax decoders.
//!
//! Ported from Cisco OpenH264
//! `reference/codec/decoder/core/src/parse_mb_syn_cabac.cpp`:
//! - `ParseCbfInfoCabac` (the `coded_block_flag` bin),
//! - `ParseSignificantMapCabac` (`significant_coeff_flag` /
//!   `last_significant_coeff_flag` decode),
//! - `ParseSignificantCoeffCabac` (`coeff_abs_level_minus1` via the `coeff_abs`
//!   contexts + the UEGk suffix `DecodeUEGLevelCabac`, then the bypass sign),
//! - `ParseResidualBlockCabac` (reassembly into a residual block).
//!
//! The CABAC analog of [`super::cavlc::residual_block_cavlc`]. As with the CAVLC
//! port, the in-loop dequant / IDCT and the zig-zag scan remap are left to P3
//! reconstruction — `out_level` receives the signed levels at their coefficient
//! (scan) positions.
//!
//! The `coded_block_flag` context-increment derivation needs decoded-neighbour
//! state (`nA`/`nB` from adjacent blocks), which is MB-level context deferred to
//! P3; [`coded_block_flag`] therefore takes the already-computed `ctx_inc`
//! (`nA + (nB << 1)`, 0..=3). The higher MB-type / sub-MB-type / mvd / ref-idx /
//! cbp CABAC decoders (`ParseMBType*Cabac`, `ParseMvdInfoCabac`,
//! `ParseRefIdxCabac`, `ParseCbpInfoCabac`, `ParseDeltaQpCabac`, ...) all need
//! the same neighbour context and are deferred to P3.

use crate::error::DecodeError;

use super::cabac::{CabacContexts, CabacDecoder};

type Result<T> = core::result::Result<T, DecodeError>;

// --- OpenH264 residual-property codes (`iResProperty`, the ctxBlockCat analog),
// from `reference/codec/decoder/core/inc/wels_common_basis.h`. ---
/// `I16_LUMA_DC`.
pub const I16_LUMA_DC: usize = 1;
/// `I16_LUMA_AC`.
pub const I16_LUMA_AC: usize = 2;
/// `LUMA_DC_AC` (4x4 luma).
pub const LUMA_DC_AC: usize = 3;
/// `CHROMA_DC`.
pub const CHROMA_DC: usize = 4;
/// `CHROMA_AC`.
pub const CHROMA_AC: usize = 5;
/// `LUMA_DC_AC_8` (8x8 luma).
pub const LUMA_DC_AC_8: usize = 6;
/// `CHROMA_DC_U`.
pub const CHROMA_DC_U: usize = 7;
/// `CHROMA_DC_V`.
pub const CHROMA_DC_V: usize = 8;
/// `CHROMA_AC_U`.
pub const CHROMA_AC_U: usize = 9;
/// `CHROMA_AC_V`.
pub const CHROMA_AC_V: usize = 10;

// --- Context base offsets (`NEW_CTX_OFFSET_*`) from
// `reference/codec/decoder/core/inc/decoder_context.h`. ---
const NEW_CTX_OFFSET_CBF: usize = 85;
const NEW_CTX_OFFSET_MAP: usize = 105;
const NEW_CTX_OFFSET_LAST: usize = 166;
const NEW_CTX_OFFSET_ONE: usize = 227;
const NEW_CTX_OFFSET_ABS: usize = 232;
const NEW_CTX_OFFSET_MAP_8X8: usize = 402;
const NEW_CTX_OFFSET_LAST_8X8: usize = 417;
const NEW_CTX_OFFSET_ONE_8X8: usize = 426;
const NEW_CTX_OFFSET_ABS_8X8: usize = 431;

// --- Per-residual-property tables (`g_k*[]`) from parse_mb_syn_cabac.cpp.
// Index 0 is the unused (`IDX_UNUSED`) sentinel slot. ---

/// `g_kMaxPos`: highest coefficient scan position for each residual property.
pub static MAX_POS: [i32; 11] = [-1, 15, 14, 15, 3, 14, 63, 3, 3, 14, 14];
/// `g_kMaxC2`: cap on the `coeff_abs` context counter `c2`.
pub static MAX_C2: [i32; 11] = [-1, 4, 4, 4, 3, 4, 4, 3, 3, 4, 4];
/// `g_kBlockCat2CtxOffsetCBF`.
pub static CTX_OFFSET_CBF: [usize; 11] = [0, 0, 4, 8, 12, 16, 0, 12, 12, 16, 16];
/// `g_kBlockCat2CtxOffsetMap`.
pub static CTX_OFFSET_MAP: [usize; 11] = [0, 0, 15, 29, 44, 47, 0, 44, 44, 47, 47];
/// `g_kBlockCat2CtxOffsetLast`.
pub static CTX_OFFSET_LAST: [usize; 11] = [0, 0, 15, 29, 44, 47, 0, 44, 44, 47, 47];
/// `g_kBlockCat2CtxOffsetOne`.
pub static CTX_OFFSET_ONE: [usize; 11] = [0, 0, 10, 20, 30, 39, 0, 30, 30, 39, 39];
/// `g_kBlockCat2CtxOffsetAbs`.
pub static CTX_OFFSET_ABS: [usize; 11] = [0, 0, 10, 20, 30, 39, 0, 30, 30, 39, 39];

/// `g_kuiIdx2CtxSignificantCoeffFlag8x8` (spec Table 9-43): position ->
/// `significant_coeff_flag` ctxIdxInc for the 8x8 transform.
pub static IDX2CTX_SIG_8X8: [usize; 64] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8, 7,
    7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12, 13,
    11, 14, 10, 12, 14,
];

/// `g_kuiIdx2CtxLastSignificantCoeffFlag8x8` (spec Table 9-43): position ->
/// `last_significant_coeff_flag` ctxIdxInc for the 8x8 transform.
pub static IDX2CTX_LAST_8X8: [usize; 64] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
];

/// Decode `coded_block_flag` for residual property `res_property` given the
/// already-derived neighbour context increment `ctx_inc` (`nA + (nB << 1)`,
/// 0..=3). Port of the `DecodeBinCabac` call inside `ParseCbfInfoCabac`.
pub fn coded_block_flag(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    res_property: usize,
    ctx_inc: u32,
) -> u32 {
    let idx = NEW_CTX_OFFSET_CBF + CTX_OFFSET_CBF[res_property] + ctx_inc as usize;
    dec.decode_decision(ctxs.ctx(idx))
}

/// `DecodeUEGLevelCabac`: the UEG0 suffix used by `coeff_abs_level_minus1`. The
/// truncated-unary prefix bins are context-coded against `ctx_idx`; once the
/// prefix saturates (13 bins) the remainder is an Exp-Golomb(0) bypass suffix.
fn decode_ueg_level(dec: &mut CabacDecoder, ctxs: &mut CabacContexts, ctx_idx: usize) -> u32 {
    let mut code = dec.decode_decision(ctxs.ctx(ctx_idx));
    if code == 0 {
        return 0;
    }
    let mut count: u32 = 1;
    code = 0;
    let mut tmp;
    loop {
        tmp = dec.decode_decision(ctxs.ctx(ctx_idx));
        code += 1;
        count += 1;
        if tmp == 0 || count == 13 {
            break;
        }
    }
    if tmp != 0 {
        let exp = dec.decode_exp_bypass(0);
        code += exp + 1;
    }
    code
}

/// `ParseSignificantMapCabac`: decode the `significant_coeff_flag` /
/// `last_significant_coeff_flag` map into `out` (1 = significant, 0 = not).
/// Returns the number of significant coefficients (`uiCoeffNum`).
fn significant_map(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    res_property: usize,
    out: &mut [i32],
) -> i32 {
    let is_8x8 = res_property == LUMA_DC_AC_8;
    let map_base =
        (if is_8x8 { NEW_CTX_OFFSET_MAP_8X8 } else { NEW_CTX_OFFSET_MAP }) + CTX_OFFSET_MAP[res_property];
    let last_base = (if is_8x8 { NEW_CTX_OFFSET_LAST_8X8 } else { NEW_CTX_OFFSET_LAST })
        + CTX_OFFSET_LAST[res_property];
    let i1 = MAX_POS[res_property] as usize;
    let mut coeff_num = 0;

    for i in 0..i1 {
        let ctx_sig = if is_8x8 { IDX2CTX_SIG_8X8[i] } else { i };
        let code = dec.decode_decision(ctxs.ctx(map_base + ctx_sig));
        if code != 0 {
            out[i] = 1;
            coeff_num += 1;
            let ctx_last = if is_8x8 { IDX2CTX_LAST_8X8[i] } else { i };
            let last = dec.decode_decision(ctxs.ctx(last_base + ctx_last));
            if last != 0 {
                // Remaining positions (i+1..=i1) are not significant.
                for o in out.iter_mut().take(i1 + 1).skip(i + 1) {
                    *o = 0;
                }
                return coeff_num;
            }
        } else {
            out[i] = 0;
        }
    }
    // The final position is significant by construction when reached.
    out[i1] = 1;
    coeff_num + 1
}

/// `ParseSignificantCoeffCabac`: walk the significant positions from high to
/// low, decoding `coeff_abs_level_minus1` (>1 via the UEG0 suffix) and the
/// bypass sign, writing signed levels back into `out`.
fn significant_coeff(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    res_property: usize,
    out: &mut [i32],
) {
    let is_8x8 = res_property == LUMA_DC_AC_8;
    let one_base =
        (if is_8x8 { NEW_CTX_OFFSET_ONE_8X8 } else { NEW_CTX_OFFSET_ONE }) + CTX_OFFSET_ONE[res_property];
    let abs_base =
        (if is_8x8 { NEW_CTX_OFFSET_ABS_8X8 } else { NEW_CTX_OFFSET_ABS }) + CTX_OFFSET_ABS[res_property];
    let max_type = MAX_C2[res_property];
    let max_pos = MAX_POS[res_property] as usize;

    let mut c1: i32 = 1;
    let mut c2: i32 = 0;
    for i in (0..=max_pos).rev() {
        if out[i] != 0 {
            let code = dec.decode_decision(ctxs.ctx(one_base + c1 as usize));
            out[i] += code as i32;
            if out[i] == 2 {
                let ueg = decode_ueg_level(dec, ctxs, abs_base + c2 as usize);
                out[i] += ueg as i32;
                c2 = (c2 + 1).min(max_type);
                c1 = 0;
            } else if c1 != 0 {
                c1 = (c1 + 1).min(4);
            }
            if dec.decode_bypass() != 0 {
                out[i] = -out[i];
            }
        }
    }
}

/// Decode one residual block in CABAC, returning the number of significant
/// coefficients (the `total_coeff` analog). Port of the syntax core of
/// `ParseResidualBlockCabac` (significance map + level decode), assuming
/// `coded_block_flag` has already been decoded as 1 by the caller (see
/// [`coded_block_flag`]).
///
/// `out_level` receives the signed coefficient levels in scan position order;
/// it must be at least `max_num_coeff` entries and zero-initialised by the
/// caller. The dequant/IDCT and zig-zag remap are applied later in P3.
pub fn residual_block_cabac(
    dec: &mut CabacDecoder,
    ctxs: &mut CabacContexts,
    res_property: usize,
    max_num_coeff: usize,
    out_level: &mut [i32],
) -> Result<i32> {
    debug_assert!(out_level.len() >= max_num_coeff);
    debug_assert!((1..=CHROMA_AC_V).contains(&res_property));
    let coeff_num = significant_map(dec, ctxs, res_property, out_level);
    significant_coeff(dec, ctxs, res_property, out_level);
    Ok(coeff_num)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::slice_header::SliceType;

    #[test]
    fn ctx_selection_tables_match_reference() {
        // g_kMaxPos / g_kMaxC2 spot checks.
        assert_eq!(MAX_POS[LUMA_DC_AC], 15);
        assert_eq!(MAX_POS[CHROMA_DC], 3);
        assert_eq!(MAX_POS[LUMA_DC_AC_8], 63);
        assert_eq!(MAX_C2[CHROMA_DC], 3);
        assert_eq!(MAX_C2[LUMA_DC_AC], 4);

        // Block-cat -> ctx offset tables.
        assert_eq!(CTX_OFFSET_CBF[CHROMA_DC_U], 12);
        assert_eq!(CTX_OFFSET_MAP[I16_LUMA_AC], 15);
        assert_eq!(CTX_OFFSET_ONE[CHROMA_AC], 39);

        // 8x8 significance/last maps (spec Table 9-43) corners.
        assert_eq!(IDX2CTX_SIG_8X8.len(), 64);
        assert_eq!(IDX2CTX_SIG_8X8[0], 0);
        assert_eq!(IDX2CTX_SIG_8X8[63], 14);
        assert_eq!(IDX2CTX_LAST_8X8[0], 0);
        assert_eq!(IDX2CTX_LAST_8X8[63], 8);
    }

    // A clean re-implementation of the residual decode driven by a fixed
    // "decoder" model, used to validate the real port's control flow and ctx
    // selection bit-for-bit.
    //
    // On an all-zero byte stream the engine offset stays 0 forever, so every
    // `decode_decision` takes the MPS path and returns the context's MPS (which
    // never flips, since an LPS is never taken), and every `decode_bypass`
    // returns 0. This makes the entire decode a pure function of the initial
    // per-context MPS values — which we snapshot and replay here.
    struct Model {
        mps: [u8; 460],
    }
    impl Model {
        fn from(ctxs: &CabacContexts) -> Self {
            let mut mps = [0u8; 460];
            for (i, m) in mps.iter_mut().enumerate() {
                *m = ctxs.get(i).mps;
            }
            Self { mps }
        }
        fn dec(&self, idx: usize) -> u32 {
            self.mps[idx] as u32
        }
        fn bypass(&self) -> u32 {
            0
        }
        fn ueg(&self, ctx_idx: usize) -> u32 {
            // decode_ueg_level on a constant-MPS, all-bypass-0 model.
            if self.dec(ctx_idx) == 0 {
                return 0;
            }
            // prefix bin == 1; subsequent prefix bins == same MPS.
            let mut count = 1u32;
            let mut code = 0u32;
            loop {
                let tmp = self.dec(ctx_idx);
                code += 1;
                count += 1;
                if tmp == 0 || count == 13 {
                    if tmp != 0 {
                        // exp bypass(0) on all-zero bypass -> 0, plus the +1.
                        code += 1;
                    }
                    return code;
                }
            }
        }

        fn significant_map(&self, res: usize, out: &mut [i32]) -> i32 {
            let is_8x8 = res == LUMA_DC_AC_8;
            let map_base = (if is_8x8 { NEW_CTX_OFFSET_MAP_8X8 } else { NEW_CTX_OFFSET_MAP })
                + CTX_OFFSET_MAP[res];
            let last_base = (if is_8x8 { NEW_CTX_OFFSET_LAST_8X8 } else { NEW_CTX_OFFSET_LAST })
                + CTX_OFFSET_LAST[res];
            let i1 = MAX_POS[res] as usize;
            let mut coeff_num = 0;
            for i in 0..i1 {
                let ctx_sig = if is_8x8 { IDX2CTX_SIG_8X8[i] } else { i };
                if self.dec(map_base + ctx_sig) != 0 {
                    out[i] = 1;
                    coeff_num += 1;
                    let ctx_last = if is_8x8 { IDX2CTX_LAST_8X8[i] } else { i };
                    if self.dec(last_base + ctx_last) != 0 {
                        return coeff_num;
                    }
                } else {
                    out[i] = 0;
                }
            }
            out[i1] = 1;
            coeff_num + 1
        }

        fn significant_coeff(&self, res: usize, out: &mut [i32]) {
            let is_8x8 = res == LUMA_DC_AC_8;
            let one_base = (if is_8x8 { NEW_CTX_OFFSET_ONE_8X8 } else { NEW_CTX_OFFSET_ONE })
                + CTX_OFFSET_ONE[res];
            let abs_base = (if is_8x8 { NEW_CTX_OFFSET_ABS_8X8 } else { NEW_CTX_OFFSET_ABS })
                + CTX_OFFSET_ABS[res];
            let max_type = MAX_C2[res];
            let max_pos = MAX_POS[res] as usize;
            let mut c1: i32 = 1;
            let mut c2: i32 = 0;
            for i in (0..=max_pos).rev() {
                if out[i] != 0 {
                    out[i] += self.dec(one_base + c1 as usize) as i32;
                    if out[i] == 2 {
                        out[i] += self.ueg(abs_base + c2 as usize) as i32;
                        c2 = (c2 + 1).min(max_type);
                        c1 = 0;
                    } else if c1 != 0 {
                        c1 = (c1 + 1).min(4);
                    }
                    if self.bypass() != 0 {
                        out[i] = -out[i];
                    }
                }
            }
        }

        fn residual(&self, res: usize, out: &mut [i32]) -> i32 {
            let n = self.significant_map(res, out);
            self.significant_coeff(res, out);
            n
        }
    }

    fn run_case(slice: SliceType, init_idc: u32, qp: i32, res: usize, max_coeff: usize) {
        let zeros = [0u8; 64];

        // Reference (model) result.
        let ctxs_ref = CabacContexts::init(slice, init_idc, qp);
        let model = Model::from(&ctxs_ref);
        let mut expect = [0i32; 64];
        let exp_count = model.residual(res, &mut expect[..max_coeff]);

        // Real engine + port.
        let mut ctxs = CabacContexts::init(slice, init_idc, qp);
        let mut dec = CabacDecoder::new(&zeros, 0).unwrap();
        let mut got = [0i32; 64];
        let got_count =
            residual_block_cabac(&mut dec, &mut ctxs, res, max_coeff, &mut got[..max_coeff]).unwrap();

        assert_eq!(got_count, exp_count, "coeff count mismatch res={res}");
        assert_eq!(&got[..max_coeff], &expect[..max_coeff], "levels mismatch res={res}");
    }

    #[test]
    fn residual_block_matches_model_all_cases() {
        // Exercise every residual property and several context inits. The
        // all-zero stream makes the decode deterministic (see Model docs), so
        // this is a true end-to-end check of significance map + level + sign +
        // UEG control flow and context selection.
        for &(slice, idc) in &[
            (SliceType::I, 0u32),
            (SliceType::P, 0),
            (SliceType::P, 1),
            (SliceType::B, 2),
        ] {
            for &qp in &[0i32, 12, 26, 37, 51] {
                run_case(slice, idc, qp, I16_LUMA_DC, 16);
                run_case(slice, idc, qp, I16_LUMA_AC, 16);
                run_case(slice, idc, qp, LUMA_DC_AC, 16);
                run_case(slice, idc, qp, CHROMA_DC, 4);
                run_case(slice, idc, qp, CHROMA_AC, 16);
                run_case(slice, idc, qp, CHROMA_DC_U, 4);
                run_case(slice, idc, qp, CHROMA_DC_V, 4);
                run_case(slice, idc, qp, CHROMA_AC_U, 16);
                run_case(slice, idc, qp, CHROMA_AC_V, 16);
                run_case(slice, idc, qp, LUMA_DC_AC_8, 64);
            }
        }
    }

    #[test]
    fn coded_block_flag_selects_expected_context() {
        // On the zero stream coded_block_flag returns the selected context MPS.
        let zeros = [0u8; 16];
        let ctxs_ref = CabacContexts::init(SliceType::I, 0, 26);
        let mut ctxs = CabacContexts::init(SliceType::I, 0, 26);
        let mut dec = CabacDecoder::new(&zeros, 0).unwrap();
        let res = LUMA_DC_AC;
        let ctx_inc = 2u32;
        let idx = NEW_CTX_OFFSET_CBF + CTX_OFFSET_CBF[res] + ctx_inc as usize;
        let expect = ctxs_ref.get(idx).mps as u32;
        assert_eq!(coded_block_flag(&mut dec, &mut ctxs, res, ctx_inc), expect);
    }
}
