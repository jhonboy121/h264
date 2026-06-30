//! Intra macroblock reconstruction: prediction + inverse transform add.
//!
//! Port of `RecI16x16Mb` / `RecI4x4Mb` / `RecChroma`
//! (`reference/codec/decoder/core/src/rec_mb.cpp`) and the `IdctFourResAddPred`
//! driver (`reference/codec/decoder/core/src/decoder.cpp`). Prediction reads the
//! already-reconstructed neighbours straight from the [`Picture`] planes (which
//! carry a padding border), and each 4x4 residual block is added with
//! [`idct4x4_add`]. For I_4x4 every block is predicted and reconstructed before
//! the next, so later blocks see the correct neighbour samples.

use crate::dsp::intra_pred as ip;
use crate::dsp::transform::{idct4x4_add, idct8x8_add};

use super::context::{DecoderContext, MbType};

// Block scan index -> raster (bx,by) within the MB; same tables as the parser.
const BLOCK_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
const BLOCK_BX: [usize; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
const BLOCK_BY: [usize; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];

/// Dispatch a luma 16x16 prediction (`I16_PRED_*`, 0..=6).
fn luma16_pred(mode: i8, plane: &mut [u8], off: usize, stride: usize) {
    match mode {
        0 => ip::i16x16_luma_pred_v(plane, off, stride),
        1 => ip::i16x16_luma_pred_h(plane, off, stride),
        2 => ip::i16x16_luma_pred_dc(plane, off, stride),
        3 => ip::i16x16_luma_pred_plane(plane, off, stride),
        4 => ip::i16x16_luma_pred_dc_left(plane, off, stride),
        5 => ip::i16x16_luma_pred_dc_top(plane, off, stride),
        _ => ip::i16x16_luma_pred_dc_na(plane, off, stride),
    }
}

/// Dispatch a luma 4x4 prediction (`I4_PRED_*`, 0..=13).
fn luma4_pred(mode: i8, plane: &mut [u8], off: usize, stride: usize) {
    match mode {
        0 => ip::i4x4_luma_pred_v(plane, off, stride),
        1 => ip::i4x4_luma_pred_h(plane, off, stride),
        2 => ip::i4x4_luma_pred_dc(plane, off, stride),
        3 => ip::i4x4_luma_pred_ddl(plane, off, stride),
        4 => ip::i4x4_luma_pred_ddr(plane, off, stride),
        5 => ip::i4x4_luma_pred_vr(plane, off, stride),
        6 => ip::i4x4_luma_pred_hd(plane, off, stride),
        7 => ip::i4x4_luma_pred_vl(plane, off, stride),
        8 => ip::i4x4_luma_pred_hu(plane, off, stride),
        9 => ip::i4x4_luma_pred_dc_left(plane, off, stride),
        10 => ip::i4x4_luma_pred_dc_top(plane, off, stride),
        11 => ip::i4x4_luma_pred_dc_na(plane, off, stride),
        12 => ip::i4x4_luma_pred_ddl_top(plane, off, stride),
        _ => ip::i4x4_luma_pred_vl_top(plane, off, stride),
    }
}

/// Dispatch a luma 8x8 prediction (`I8_PRED_*`, same numbering as 4x4).
fn luma8_pred(mode: i8, plane: &mut [u8], off: usize, stride: usize, b_tl: bool, b_tr: bool) {
    match mode {
        0 => ip::i8x8_luma_pred_v(plane, off, stride, b_tl, b_tr),
        1 => ip::i8x8_luma_pred_h(plane, off, stride, b_tl, b_tr),
        2 => ip::i8x8_luma_pred_dc(plane, off, stride, b_tl, b_tr),
        3 => ip::i8x8_luma_pred_ddl(plane, off, stride, b_tl, b_tr),
        4 => ip::i8x8_luma_pred_ddr(plane, off, stride, b_tl, b_tr),
        5 => ip::i8x8_luma_pred_vr(plane, off, stride, b_tl, b_tr),
        6 => ip::i8x8_luma_pred_hd(plane, off, stride, b_tl, b_tr),
        7 => ip::i8x8_luma_pred_vl(plane, off, stride, b_tl, b_tr),
        8 => ip::i8x8_luma_pred_hu(plane, off, stride, b_tl, b_tr),
        9 => ip::i8x8_luma_pred_dc_left(plane, off, stride, b_tl, b_tr),
        10 => ip::i8x8_luma_pred_dc_top(plane, off, stride, b_tl, b_tr),
        11 => ip::i8x8_luma_pred_dc_na(plane, off, stride, b_tl, b_tr),
        12 => ip::i8x8_luma_pred_ddl_top(plane, off, stride, b_tl, b_tr),
        _ => ip::i8x8_luma_pred_vl_top(plane, off, stride, b_tl, b_tr),
    }
}

/// Dispatch a chroma prediction (`C_PRED_*`, 0..=6).
fn chroma_pred(mode: i8, plane: &mut [u8], off: usize, stride: usize) {
    match mode {
        0 => ip::i_chroma_pred_dc(plane, off, stride),
        1 => ip::i_chroma_pred_h(plane, off, stride),
        2 => ip::i_chroma_pred_v(plane, off, stride),
        3 => ip::i_chroma_pred_plane(plane, off, stride),
        4 => ip::i_chroma_pred_dc_left(plane, off, stride),
        5 => ip::i_chroma_pred_dc_top(plane, off, stride),
        _ => ip::i_chroma_pred_dc_na(plane, off, stride),
    }
}

#[inline]
fn block16(coeffs: &[i16; 384], base: usize) -> [i16; 16] {
    let mut b = [0i16; 16];
    b.copy_from_slice(&coeffs[base..base + 16]);
    b
}

#[inline]
fn block64(coeffs: &[i16; 384], base: usize) -> [i16; 64] {
    let mut b = [0i16; 64];
    b.copy_from_slice(&coeffs[base..base + 64]);
    b
}

/// Reconstruct intra macroblock `mb_xy` into `ctx.picture` using the parsed
/// modes/cbp/non-zero-counts and the 384-entry dequantised `coeffs`.
pub fn recon_intra_mb(ctx: &mut DecoderContext, mb_xy: usize, coeffs: &[i16; 384]) {
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;

    let mb_type = ctx.mb_type[mb_xy];
    if mb_type == MbType::IPcm {
        // I_PCM samples are copied straight into the picture during parse; there
        // is no prediction or residual to reconstruct here.
        return;
    }
    let i16_mode = ctx.i16_mode[mb_xy];
    let chroma_mode = ctx.chroma_mode[mb_xy];
    let cbp_c = ctx.cbp[mb_xy] >> 4;

    let mut nzc_luma = [0i8; 16];
    nzc_luma.copy_from_slice(ctx.nzc_luma_mb(mb_xy));
    let mut nzc_chroma = [0i8; 8];
    nzc_chroma.copy_from_slice(ctx.nzc_chroma_mb(mb_xy));
    let mut final_mode = [0i8; 16];
    final_mode.copy_from_slice(&ctx.i4_final_mode[mb_xy * 16..mb_xy * 16 + 16]);
    let transform_8x8 = ctx.transform_8x8[mb_xy];
    let i8_avail = ctx.i8_avail[mb_xy];

    let ystride = ctx.picture.luma_stride;
    let cstride = ctx.picture.chroma_stride;
    let y_off = ctx.picture.luma_mb_offset(mb_x, mb_y);
    let c_off = ctx.picture.chroma_mb_offset(mb_x, mb_y);
    let pic = &mut ctx.picture;

    match mb_type {
        MbType::Intra16x16 => {
            luma16_pred(i16_mode, &mut pic.y, y_off, ystride);
            for i in 0..16 {
                let raster = BLOCK_RASTER[i];
                let off = y_off + BLOCK_BY[i] * 4 * ystride + BLOCK_BX[i] * 4;
                let base = i * 16;
                // DC may be non-zero even when AC count is zero.
                if nzc_luma[raster] != 0 || coeffs[base] != 0 {
                    idct4x4_add(&mut pic.y[off..], ystride, &block16(coeffs, base));
                }
            }
        }
        MbType::Intra4x4 if transform_8x8 => {
            // I_8x8: four 8x8 blocks in scan order (TL, TR, BL, BR). Per-block
            // top-left / top-right reference availability (`RecI8x8Luma`).
            let b_tl = [
                i8_avail & 0x02 != 0,
                i8_avail & 0x01 != 0,
                i8_avail & 0x04 != 0,
                true,
            ];
            let b_tr = [
                i8_avail & 0x01 != 0,
                i8_avail & 0x08 != 0,
                true,
                false,
            ];
            for i8 in 0..4 {
                let bx8 = i8 & 1;
                let by8 = i8 >> 1;
                let off = y_off + by8 * 8 * ystride + bx8 * 8;
                let mode = final_mode[BLOCK_RASTER[i8 * 4]];
                luma8_pred(mode, &mut pic.y, off, ystride, b_tl[i8], b_tr[i8]);
                let any_nz = (0..4).any(|j| nzc_luma[BLOCK_RASTER[i8 * 4 + j]] != 0);
                if any_nz {
                    idct8x8_add(&mut pic.y[off..], ystride, &block64(coeffs, i8 * 64));
                }
            }
        }
        MbType::Intra4x4 => {
            for i in 0..16 {
                let raster = BLOCK_RASTER[i];
                let off = y_off + BLOCK_BY[i] * 4 * ystride + BLOCK_BX[i] * 4;
                luma4_pred(final_mode[raster], &mut pic.y, off, ystride);
                if nzc_luma[raster] != 0 {
                    let base = i * 16;
                    idct4x4_add(&mut pic.y[off..], ystride, &block16(coeffs, base));
                }
            }
        }
        _ => unreachable!("recon_intra_mb called on an inter macroblock"),
    }

    // Chroma (identical for both intra MB kinds).
    chroma_pred(chroma_mode, &mut pic.u, c_off, cstride);
    chroma_pred(chroma_mode, &mut pic.v, c_off, cstride);
    if cbp_c == 1 || cbp_c == 2 {
        for c in 0..2 {
            let plane = if c == 0 { &mut pic.u } else { &mut pic.v };
            for b in 0..4 {
                let base = 256 + c * 64 + b * 16;
                if nzc_chroma[c * 4 + b] != 0 || coeffs[base] != 0 {
                    let off = c_off + (b / 2) * 4 * cstride + (b % 2) * 4;
                    idct4x4_add(&mut plane[off..], cstride, &block16(coeffs, base));
                }
            }
        }
    }
}

/// Copy raw I_PCM samples straight into the reconstructed picture (spec 8.5):
/// 256 luma samples (16x16, raster order), then 64 Cb and 64 Cr (8x8, raster
/// order) for 4:2:0. No prediction, transform, or deblock-residual is involved;
/// the caller has already tagged the MB as [`MbType::IPcm`] with QP = 0 and
/// nnz = 16 per block so deblocking applies the intra rules.
pub fn recon_pcm_mb(ctx: &mut DecoderContext, mb_xy: usize, luma: &[u8; 256], cb: &[u8; 64], cr: &[u8; 64]) {
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;
    let ystride = ctx.picture.luma_stride;
    let cstride = ctx.picture.chroma_stride;
    let y_off = ctx.picture.luma_mb_offset(mb_x, mb_y);
    let c_off = ctx.picture.chroma_mb_offset(mb_x, mb_y);
    let pic = &mut ctx.picture;

    for row in 0..16 {
        let dst = y_off + row * ystride;
        pic.y[dst..dst + 16].copy_from_slice(&luma[row * 16..row * 16 + 16]);
    }
    for row in 0..8 {
        let dst = c_off + row * cstride;
        pic.u[dst..dst + 8].copy_from_slice(&cb[row * 8..row * 8 + 8]);
        pic.v[dst..dst + 8].copy_from_slice(&cr[row * 8..row * 8 + 8]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn i16x16_dc128_flat_fills_grey() {
        // No neighbours -> DC_128 mode (6) with zero residual leaves the MB at
        // 128 across both luma and chroma.
        let mut ctx = DecoderContext::new(Vec::new(), Vec::new(), 1, 1);
        ctx.slice_idc[0] = 0;
        ctx.mb_type[0] = MbType::Intra16x16;
        ctx.i16_mode[0] = 6; // DC_128
        ctx.chroma_mode[0] = 6; // DC_128
        ctx.cbp[0] = 0;
        let coeffs = [0i16; 384];
        recon_intra_mb(&mut ctx, 0, &coeffs);

        let o = ctx.picture.luma_mb_offset(0, 0);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(ctx.picture.y[o + y * ctx.picture.luma_stride + x], 128);
            }
        }
        let co = ctx.picture.chroma_mb_offset(0, 0);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(ctx.picture.u[co + y * ctx.picture.chroma_stride + x], 128);
            }
        }
    }

    #[test]
    fn i16x16_dc_residual_offsets_block() {
        // DC_128 prediction (128) + a single luma-DC coefficient in block 0
        // shifts that 4x4 block uniformly by (32 + dc) >> 6.
        let mut ctx = DecoderContext::new(Vec::new(), Vec::new(), 1, 1);
        ctx.slice_idc[0] = 0;
        ctx.mb_type[0] = MbType::Intra16x16;
        ctx.i16_mode[0] = 6;
        ctx.chroma_mode[0] = 6;
        ctx.cbp[0] = 0;
        let mut coeffs = [0i16; 384];
        coeffs[0] = 64; // block-0 DC residual
        recon_intra_mb(&mut ctx, 0, &coeffs);

        let o = ctx.picture.luma_mb_offset(0, 0);
        let expect = (128 + ((32 + 64) >> 6)) as u8;
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(ctx.picture.y[o + y * ctx.picture.luma_stride + x], expect);
            }
        }
        // Neighbouring block (to the right) stays at the flat prediction.
        assert_eq!(ctx.picture.y[o + 4], 128);
    }
}
