//! Inter macroblock reconstruction: motion compensation + residual add.
//!
//! Port of the inter path of `GetInterPred` / `BaseMC` (`rec_mb.cpp`): for each
//! list-0 partition the reference block is motion-compensated (luma 6-tap,
//! chroma bilinear via [`crate::dsp::mc`]) into the current picture's MB area,
//! then the dequantised residual is added with [`idct4x4_add`] exactly as in the
//! intra path. P_Skip is MC-only.

use crate::dsp::mc::{mc_chroma, mc_luma};
use crate::dsp::transform::idct4x4_add;

use super::context::{DecoderContext, MbType, SubMbType};
use super::picture::{Picture, PADDING};

const BLOCK_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
const BLOCK_BX: [usize; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
const BLOCK_BY: [usize; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];

/// `BaseMC`: clip the full-pel MV into the padded reference, then motion
/// compensate one `w`x`h` luma block (+ `w/2`x`h/2` chroma) from `ref_pic`
/// into the current picture at MB-relative offset `(dx, dy)` luma pixels.
#[allow(clippy::too_many_arguments)]
fn base_mc(
    pic: &mut Picture,
    ref_pic: &Picture,
    mb_x: usize,
    mb_y: usize,
    dx: usize,
    dy: usize,
    mv: [i16; 2],
    w: usize,
    h: usize,
) {
    let pic_w = pic.width as i32;
    let pic_h = pic.height as i32;
    let abs_x = ((mb_x * 16 + dx) as i32) << 2;
    let abs_y = ((mb_y * 16 + dy) as i32) << 2;
    let pad = PADDING as i32;
    let full_mvx = (abs_x + mv[0] as i32).clamp((-pad + 2) << 2, (pic_w + pad - 19) << 2);
    let full_mvy = (abs_y + mv[1] as i32).clamp((-pad + 2) << 2, (pic_h + pad - 19) << 2);

    // Luma.
    let ls = pic.luma_stride;
    let dst_l = pic.luma_origin() + (mb_y * 16 + dy) * ls + (mb_x * 16 + dx);
    let src_l = (ref_pic.luma_origin() as i32 + (full_mvx >> 2) + (full_mvy >> 2) * ls as i32) as usize;
    mc_luma(
        &mut pic.y[dst_l..],
        ls,
        &ref_pic.y,
        src_l,
        ls,
        full_mvx as i16,
        full_mvy as i16,
        w,
        h,
    );

    // Chroma (half resolution; MV in quarter-luma == eighth-chroma units).
    let cs = pic.chroma_stride;
    let cdx = dx / 2;
    let cdy = dy / 2;
    let dst_c = pic.chroma_origin() + (mb_y * 8 + cdy) * cs + (mb_x * 8 + cdx);
    let src_c = (ref_pic.chroma_origin() as i32 + (full_mvx >> 3) + (full_mvy >> 3) * cs as i32) as usize;
    let (cw, ch) = (w / 2, h / 2);
    mc_chroma(&mut pic.u[dst_c..], cs, &ref_pic.u, src_c, cs, full_mvx as i16, full_mvy as i16, cw, ch);
    mc_chroma(&mut pic.v[dst_c..], cs, &ref_pic.v, src_c, cs, full_mvx as i16, full_mvy as i16, cw, ch);
}

#[inline]
fn block16(coeffs: &[i16; 384], base: usize) -> [i16; 16] {
    let mut b = [0i16; 16];
    b.copy_from_slice(&coeffs[base..base + 16]);
    b
}

/// Reconstruct inter macroblock `mb_xy`: motion-compensate every partition from
/// `ref_pics` (indexed by list-0 ref index) then add the residual.
pub fn recon_inter_mb(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    coeffs: &[i16; 384],
    ref_pics: &[&Picture],
) {
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;
    let mb_type = ctx.mb_type[mb_xy];

    // Snapshot per-4x4 motion (raster order) and the sub-mb partition kinds.
    let mut mv = [[0i16; 2]; 16];
    let mut ref_idx = [0i8; 16];
    for r in 0..16 {
        let base = (mb_xy * 16 + r) * 2;
        mv[r] = [ctx.mv[base], ctx.mv[base + 1]];
        ref_idx[r] = ctx.ref_idx[mb_xy * 16 + r];
    }
    let mut subs = [SubMbType::P8x8; 4];
    subs.copy_from_slice(&ctx.sub_mb_type[mb_xy * 4..mb_xy * 4 + 4]);

    let rp = |i: i8| -> &Picture { ref_pics[i.max(0) as usize] };

    // ---- Motion compensation ----
    match mb_type {
        MbType::Inter16x16 | MbType::PSkip => {
            base_mc(&mut ctx.picture, rp(ref_idx[0]), mb_x, mb_y, 0, 0, mv[0], 16, 16);
        }
        MbType::Inter16x8 => {
            base_mc(&mut ctx.picture, rp(ref_idx[0]), mb_x, mb_y, 0, 0, mv[0], 16, 8);
            base_mc(&mut ctx.picture, rp(ref_idx[8]), mb_x, mb_y, 0, 8, mv[8], 16, 8);
        }
        MbType::Inter8x16 => {
            base_mc(&mut ctx.picture, rp(ref_idx[0]), mb_x, mb_y, 0, 0, mv[0], 8, 16);
            base_mc(&mut ctx.picture, rp(ref_idx[2]), mb_x, mb_y, 8, 0, mv[2], 8, 16);
        }
        MbType::Inter8x8 | MbType::Inter8x8Ref0 => {
            for (i, &sub) in subs.iter().enumerate() {
                let i_idx = ((i >> 1) << 3) + ((i & 1) << 1); // raster top-left of 8x8
                let blk8x = (i & 1) << 3;
                let blk8y = (i >> 1) << 3;
                let r = rp(ref_idx[i_idx]);
                match sub {
                    SubMbType::P8x8 => {
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, blk8x, blk8y, mv[i_idx], 8, 8);
                    }
                    SubMbType::P8x4 => {
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, blk8x, blk8y, mv[i_idx], 8, 4);
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, blk8x, blk8y + 4, mv[i_idx + 4], 8, 4);
                    }
                    SubMbType::P4x8 => {
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, blk8x, blk8y, mv[i_idx], 4, 8);
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, blk8x + 4, blk8y, mv[i_idx + 1], 4, 8);
                    }
                    SubMbType::P4x4 => {
                        for j in 0..4 {
                            let j_idx = ((j >> 1) << 2) + (j & 1);
                            let b4x = (j & 1) << 2;
                            let b4y = (j >> 1) << 2;
                            base_mc(
                                &mut ctx.picture, r, mb_x, mb_y, blk8x + b4x, blk8y + b4y,
                                mv[i_idx + j_idx], 4, 4,
                            );
                        }
                    }
                }
            }
        }
        _ => unreachable!("recon_inter_mb called on an intra macroblock"),
    }

    // ---- Residual add ----
    let cbp_c = ctx.cbp[mb_xy] >> 4;
    if ctx.cbp[mb_xy] == 0 {
        return; // P_Skip or cbp == 0: MC only.
    }
    let mut nzc_luma = [0i8; 16];
    nzc_luma.copy_from_slice(ctx.nzc_luma_mb(mb_xy));
    let mut nzc_chroma = [0i8; 8];
    nzc_chroma.copy_from_slice(ctx.nzc_chroma_mb(mb_xy));

    let ystride = ctx.picture.luma_stride;
    let cstride = ctx.picture.chroma_stride;
    let y_off = ctx.picture.luma_mb_offset(mb_x, mb_y);
    let c_off = ctx.picture.chroma_mb_offset(mb_x, mb_y);
    let pic = &mut ctx.picture;

    for i in 0..16 {
        let raster = BLOCK_RASTER[i];
        if nzc_luma[raster] != 0 {
            let off = y_off + BLOCK_BY[i] * 4 * ystride + BLOCK_BX[i] * 4;
            idct4x4_add(&mut pic.y[off..], ystride, &block16(coeffs, i * 16));
        }
    }

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
