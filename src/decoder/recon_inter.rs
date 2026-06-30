//! Inter macroblock reconstruction: motion compensation + residual add.
//!
//! Port of the inter path of `GetInterPred` / `BaseMC` (`rec_mb.cpp`): for each
//! list-0 partition the reference block is motion-compensated (luma 6-tap,
//! chroma bilinear via [`crate::dsp::mc`]) into the current picture's MB area,
//! then the dequantised residual is added with [`idct4x4_add`] exactly as in the
//! intra path. P_Skip is MC-only.

use crate::dsp::mc::{mc_chroma, mc_luma};
use crate::dsp::transform::idct4x4_add;
use crate::dsp::{Dim, Mv};

use super::context::{DecoderContext, MbType, SubMbType};
use super::picture::{Picture, PADDING};

const BLOCK_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
const BLOCK_BX: [usize; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
const BLOCK_BY: [usize; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];

/// Destination placement of a partition inside its macroblock: pixel offset
/// `(dx, dy)` from the macroblock's top-left luma sample.
struct PartPos {
    dx: usize,
    dy: usize,
}

/// `BaseMC`: clip the full-pel MV into the padded reference, then motion
/// compensate one `dim` luma block (+ `dim/2` chroma) from `ref_pic`
/// into the current picture at MB-relative offset `(pos.dx, pos.dy)` luma pixels.
fn base_mc(pic: &mut Picture, ref_pic: &Picture, mb_x: usize, mb_y: usize, pos: PartPos, mv: [i16; 2], dim: Dim) {
    let PartPos { dx, dy } = pos;
    let (w, h) = (dim.w, dim.h);
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
        Mv { x: full_mvx as i16, y: full_mvy as i16 },
        Dim { w, h },
    );

    // Chroma (half resolution; MV in quarter-luma == eighth-chroma units).
    let cs = pic.chroma_stride;
    let cdx = dx / 2;
    let cdy = dy / 2;
    let dst_c = pic.chroma_origin() + (mb_y * 8 + cdy) * cs + (mb_x * 8 + cdx);
    let src_c = (ref_pic.chroma_origin() as i32 + (full_mvx >> 3) + (full_mvy >> 3) * cs as i32) as usize;
    let (cw, ch) = (w / 2, h / 2);
    let cmv = Mv { x: full_mvx as i16, y: full_mvy as i16 };
    let cdim = Dim { w: cw, h: ch };
    mc_chroma(&mut pic.u[dst_c..], cs, &ref_pic.u, src_c, cs, cmv, cdim);
    mc_chroma(&mut pic.v[dst_c..], cs, &ref_pic.v, src_c, cs, cmv, cdim);
}

/// Motion-compensate one `dim` luma block (+ chroma) from `ref_pic` at
/// MB-relative offset `(dx, dy)` into the destination buffers (`dst_*`), which
/// may be the current picture or a temporary bi-prediction scratch buffer.
/// Mirrors `BaseMC` (the MV clip + source addressing are identical to
/// [`base_mc`]).
#[allow(clippy::too_many_arguments)]
fn mc_to(
    ref_pic: &Picture,
    mb_x: usize,
    mb_y: usize,
    dx: usize,
    dy: usize,
    mv: [i16; 2],
    dim: Dim,
    dst_y: &mut [u8],
    dys: usize,
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    dcs: usize,
) {
    let (w, h) = (dim.w, dim.h);
    let pic_w = ref_pic.width as i32;
    let pic_h = ref_pic.height as i32;
    let abs_x = ((mb_x * 16 + dx) as i32) << 2;
    let abs_y = ((mb_y * 16 + dy) as i32) << 2;
    let pad = PADDING as i32;
    let full_mvx = (abs_x + mv[0] as i32).clamp((-pad + 2) << 2, (pic_w + pad - 19) << 2);
    let full_mvy = (abs_y + mv[1] as i32).clamp((-pad + 2) << 2, (pic_h + pad - 19) << 2);

    let ls = ref_pic.luma_stride;
    let src_l = (ref_pic.luma_origin() as i32 + (full_mvx >> 2) + (full_mvy >> 2) * ls as i32) as usize;
    mc_luma(dst_y, dys, &ref_pic.y, src_l, ls, Mv { x: full_mvx as i16, y: full_mvy as i16 }, Dim { w, h });

    let cs = ref_pic.chroma_stride;
    let src_c = (ref_pic.chroma_origin() as i32 + (full_mvx >> 3) + (full_mvy >> 3) * cs as i32) as usize;
    let cmv = Mv { x: full_mvx as i16, y: full_mvy as i16 };
    let cdim = Dim { w: w / 2, h: h / 2 };
    mc_chroma(dst_u, dcs, &ref_pic.u, src_c, cs, cmv, cdim);
    mc_chroma(dst_v, dcs, &ref_pic.v, src_c, cs, cmv, cdim);
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
            base_mc(&mut ctx.picture, rp(ref_idx[0]), mb_x, mb_y, PartPos { dx: 0, dy: 0 }, mv[0], Dim { w: 16, h: 16 });
        }
        MbType::Inter16x8 => {
            base_mc(&mut ctx.picture, rp(ref_idx[0]), mb_x, mb_y, PartPos { dx: 0, dy: 0 }, mv[0], Dim { w: 16, h: 8 });
            base_mc(&mut ctx.picture, rp(ref_idx[8]), mb_x, mb_y, PartPos { dx: 0, dy: 8 }, mv[8], Dim { w: 16, h: 8 });
        }
        MbType::Inter8x16 => {
            base_mc(&mut ctx.picture, rp(ref_idx[0]), mb_x, mb_y, PartPos { dx: 0, dy: 0 }, mv[0], Dim { w: 8, h: 16 });
            base_mc(&mut ctx.picture, rp(ref_idx[2]), mb_x, mb_y, PartPos { dx: 8, dy: 0 }, mv[2], Dim { w: 8, h: 16 });
        }
        MbType::Inter8x8 | MbType::Inter8x8Ref0 => {
            for (i, &sub) in subs.iter().enumerate() {
                let i_idx = ((i >> 1) << 3) + ((i & 1) << 1); // raster top-left of 8x8
                let blk8x = (i & 1) << 3;
                let blk8y = (i >> 1) << 3;
                let r = rp(ref_idx[i_idx]);
                match sub {
                    SubMbType::P8x8 => {
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, PartPos { dx: blk8x, dy: blk8y }, mv[i_idx], Dim { w: 8, h: 8 });
                    }
                    SubMbType::P8x4 => {
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, PartPos { dx: blk8x, dy: blk8y }, mv[i_idx], Dim { w: 8, h: 4 });
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, PartPos { dx: blk8x, dy: blk8y + 4 }, mv[i_idx + 4], Dim { w: 8, h: 4 });
                    }
                    SubMbType::P4x8 => {
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, PartPos { dx: blk8x, dy: blk8y }, mv[i_idx], Dim { w: 4, h: 8 });
                        base_mc(&mut ctx.picture, r, mb_x, mb_y, PartPos { dx: blk8x + 4, dy: blk8y }, mv[i_idx + 1], Dim { w: 4, h: 8 });
                    }
                    SubMbType::P4x4 => {
                        for j in 0..4 {
                            let j_idx = ((j >> 1) << 2) + (j & 1);
                            let b4x = (j & 1) << 2;
                            let b4y = (j >> 1) << 2;
                            base_mc(
                                &mut ctx.picture, r, mb_x, mb_y, PartPos { dx: blk8x + b4x, dy: blk8y + b4y },
                                mv[i_idx + j_idx], Dim { w: 4, h: 4 },
                            );
                        }
                    }
                }
            }
        }
        _ => unreachable!("recon_inter_mb called on an intra macroblock"),
    }

    add_inter_residual(ctx, mb_xy, coeffs);
}

/// Add the dequantised inter residual (luma 4x4 + chroma DC/AC) to the
/// motion-compensated prediction. Shared by the P and B inter paths.
pub(super) fn add_inter_residual(ctx: &mut DecoderContext, mb_xy: usize, coeffs: &[i16; 384]) {
    let cbp_c = ctx.cbp[mb_xy] >> 4;
    if ctx.cbp[mb_xy] == 0 {
        return; // skip / cbp == 0: MC only.
    }
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;
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

/// One B macroblock MC partition (luma pixel rect, MB-relative).
struct BPart {
    dx: usize,
    dy: usize,
    w: usize,
    h: usize,
    /// OpenH264 16x8/8x16 bi-prediction quirk index: -1 = none (16x16/8x8 do a
    /// proper bi-average); 0 = first partition (a bi partition reduces to L1);
    /// 1 = second partition (a bi partition reduces to L0). This replicates
    /// `GetInterBPred`'s destination-pointer over-advance for `IS_INTER_16x8` /
    /// `IS_INTER_8x16`, where the bi-average lands outside the partition and the
    /// L0 (part 1) or L1 (part 0) prediction is the one that survives.
    quirk: i8,
}

/// Enumerate the motion-compensation partitions of a B macroblock from its
/// type + sub_mb_type (geometry only; list usage is read per-partition).
fn b_partitions(ctx: &DecoderContext, mb_xy: usize) -> alloc::vec::Vec<BPart> {
    use alloc::vec::Vec;
    let mut parts: Vec<BPart> = Vec::new();
    match ctx.mb_type[mb_xy] {
        MbType::BSkip | MbType::BDirect16x16 | MbType::B16x16 => {
            parts.push(BPart { dx: 0, dy: 0, w: 16, h: 16, quirk: -1 });
        }
        MbType::B16x8 => {
            parts.push(BPart { dx: 0, dy: 0, w: 16, h: 8, quirk: 0 });
            parts.push(BPart { dx: 0, dy: 8, w: 16, h: 8, quirk: 1 });
        }
        MbType::B8x16 => {
            parts.push(BPart { dx: 0, dy: 0, w: 8, h: 16, quirk: 0 });
            parts.push(BPart { dx: 8, dy: 0, w: 8, h: 16, quirk: 1 });
        }
        MbType::B8x8 => {
            for i in 0..4 {
                let blk8x = (i & 1) * 8;
                let blk8y = (i >> 1) * 8;
                match ctx.sub_mb_type[mb_xy * 4 + i] {
                    SubMbType::P8x8 => parts.push(BPart { dx: blk8x, dy: blk8y, w: 8, h: 8, quirk: -1 }),
                    SubMbType::P8x4 => {
                        parts.push(BPart { dx: blk8x, dy: blk8y, w: 8, h: 4, quirk: -1 });
                        parts.push(BPart { dx: blk8x, dy: blk8y + 4, w: 8, h: 4, quirk: -1 });
                    }
                    SubMbType::P4x8 => {
                        parts.push(BPart { dx: blk8x, dy: blk8y, w: 4, h: 8, quirk: -1 });
                        parts.push(BPart { dx: blk8x + 4, dy: blk8y, w: 4, h: 8, quirk: -1 });
                    }
                    SubMbType::P4x4 => {
                        for j in 0..4 {
                            parts.push(BPart {
                                dx: blk8x + (j & 1) * 4,
                                dy: blk8y + (j >> 1) * 4,
                                w: 4,
                                h: 4,
                                quirk: -1,
                            });
                        }
                    }
                }
            }
        }
        _ => unreachable!("recon_b_mb on a non-B macroblock"),
    }
    parts
}

/// Reconstruct a B-slice inter macroblock: bi/uni motion-compensation from the
/// list-0 and list-1 reference pictures (default `(p0+p1+1)>>1` bi-averaging),
/// then add the residual (`GetInterBPred` + `BiPrediction`).
pub fn recon_b_mb(
    ctx: &mut DecoderContext,
    mb_xy: usize,
    coeffs: &[i16; 384],
    ref_pics: [&[&Picture]; 2],
) {
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;
    let parts = b_partitions(ctx, mb_xy);

    for p in parts {
        let rep = (p.dy / 4) * 4 + (p.dx / 4);
        let r0 = ctx.ref_idx[mb_xy * 16 + rep];
        let r1 = ctx.ref_idx_l1[mb_xy * 16 + rep];
        let mv0 = [ctx.mv[(mb_xy * 16 + rep) * 2], ctx.mv[(mb_xy * 16 + rep) * 2 + 1]];
        let mv1 = [ctx.mv_l1[(mb_xy * 16 + rep) * 2], ctx.mv_l1[(mb_xy * 16 + rep) * 2 + 1]];
        let mut l0 = r0 >= 0;
        let mut l1 = r1 >= 0;
        // OpenH264 16x8/8x16 bi quirk: the bi-average is discarded, so a bi
        // partition reduces to a single list (part 0 -> L1, part 1 -> L0).
        if l0 && l1 {
            match p.quirk {
                0 => l0 = false, // first partition keeps L1
                1 => l1 = false, // second partition keeps L0
                _ => {}
            }
        }
        let dim = Dim { w: p.w, h: p.h };

        let ls = ctx.picture.luma_stride;
        let cs = ctx.picture.chroma_stride;
        let y_off = ctx.picture.luma_mb_offset(mb_x, mb_y) + p.dy * ls + p.dx;
        let c_off = ctx.picture.chroma_mb_offset(mb_x, mb_y) + (p.dy / 2) * cs + (p.dx / 2);

        if l0 && l1 {
            // Both lists into scratch buffers, then average into the picture.
            let ref0 = ref_pics[0][r0 as usize];
            let ref1 = ref_pics[1][r1 as usize];
            let mut t0y = [0u8; 256];
            let mut t0u = [0u8; 64];
            let mut t0v = [0u8; 64];
            mc_to(ref0, mb_x, mb_y, p.dx, p.dy, mv0, dim, &mut t0y, 16, &mut t0u, &mut t0v, 8);
            let mut ty = [0u8; 256];
            let mut tu = [0u8; 64];
            let mut tv = [0u8; 64];
            mc_to(ref1, mb_x, mb_y, p.dx, p.dy, mv1, dim, &mut ty, 16, &mut tu, &mut tv, 8);
            let pic = &mut ctx.picture;
            let cdim = Dim { w: p.w / 2, h: p.h / 2 };
            for i in 0..dim.h {
                for j in 0..dim.w {
                    pic.y[y_off + i * ls + j] = ((t0y[i * 16 + j] as i32 + ty[i * 16 + j] as i32 + 1) >> 1) as u8;
                }
            }
            for i in 0..cdim.h {
                for j in 0..cdim.w {
                    pic.u[c_off + i * cs + j] = ((t0u[i * 8 + j] as i32 + tu[i * 8 + j] as i32 + 1) >> 1) as u8;
                    pic.v[c_off + i * cs + j] = ((t0v[i * 8 + j] as i32 + tv[i * 8 + j] as i32 + 1) >> 1) as u8;
                }
            }
        } else {
            let (rp, mv) = if l0 { (ref_pics[0][r0 as usize], mv0) } else { (ref_pics[1][r1 as usize], mv1) };
            let pic = &mut ctx.picture;
            mc_to(rp, mb_x, mb_y, p.dx, p.dy, mv, dim, &mut pic.y[y_off..], ls, &mut pic.u[c_off..], &mut pic.v[c_off..], cs);
        }
    }

    add_inter_residual(ctx, mb_xy, coeffs);
}

