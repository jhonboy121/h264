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
use crate::dsp::transform::idct4x4_add;

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

/// Reconstruct intra macroblock `mb_xy` into `ctx.picture` using the parsed
/// modes/cbp/non-zero-counts and the 384-entry dequantised `coeffs`.
pub fn recon_intra_mb(ctx: &mut DecoderContext, mb_xy: usize, coeffs: &[i16; 384]) {
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;

    let mb_type = ctx.mb_type[mb_xy];
    let i16_mode = ctx.i16_mode[mb_xy];
    let chroma_mode = ctx.chroma_mode[mb_xy];
    let cbp_c = ctx.cbp[mb_xy] >> 4;

    let mut nzc_luma = [0i8; 16];
    nzc_luma.copy_from_slice(ctx.nzc_luma_mb(mb_xy));
    let mut nzc_chroma = [0i8; 8];
    nzc_chroma.copy_from_slice(ctx.nzc_chroma_mb(mb_xy));
    let mut final_mode = [0i8; 16];
    final_mode.copy_from_slice(&ctx.i4_final_mode[mb_xy * 16..mb_xy * 16 + 16]);

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
