//! Motion compensation (inter prediction), ported from the scalar `_c` kernels
//! in `reference/codec/common/src/mc.cpp` (`McLuma_c`, `McChroma_c`, and the 16
//! quarter-pel `McHorVerXY_c` helpers).
//!
//! Luma uses the H.264 6-tap half-pel filter `(A -5B +20C +20D -5E +F)` with
//! quarter-pel positions formed by averaging a half-pel/full-pel pair; chroma
//! uses bilinear weighting from the `g_kuiABCD` table.
//!
//! Source addressing follows the C convention that the source pointer "has
//! already been offset by the integer part of the motion vector". In Rust we
//! pass the source plane as a slice `src` plus the linear index `src_off` of the
//! block's top-left sample, so the 6-tap kernels can reach the surrounding halo
//! (up to 2 samples left/above and 3 right/below) without negative indexing.

use crate::dsp::clip1;

/// Chroma bilinear weights: `g_kuiABCD[dy][dx] = [A, B, C, D]` where
/// `A=(8-dx)(8-dy)`, `B=dx(8-dy)`, `C=(8-dx)dy`, `D=dx*dy`.
#[rustfmt::skip]
static G_ABCD: [[[u8; 4]; 8]; 8] = [
    [[64,0,0,0],[56,8,0,0],[48,16,0,0],[40,24,0,0],[32,32,0,0],[24,40,0,0],[16,48,0,0],[8,56,0,0]],
    [[56,0,8,0],[49,7,7,1],[42,14,6,2],[35,21,5,3],[28,28,4,4],[21,35,3,5],[14,42,2,6],[7,49,1,7]],
    [[48,0,16,0],[42,6,14,2],[36,12,12,4],[30,18,10,6],[24,24,8,8],[18,30,6,10],[12,36,4,12],[6,42,2,14]],
    [[40,0,24,0],[35,5,21,3],[30,10,18,6],[25,15,15,9],[20,20,12,12],[15,25,9,15],[10,30,6,18],[5,35,3,21]],
    [[32,0,32,0],[28,4,28,4],[24,8,24,8],[20,12,20,12],[16,16,16,16],[12,20,12,20],[8,24,8,24],[4,28,4,28]],
    [[24,0,40,0],[21,3,35,5],[18,6,30,10],[15,9,25,15],[12,12,20,20],[9,15,15,25],[6,18,10,30],[3,21,5,35]],
    [[16,0,48,0],[14,2,42,6],[12,4,36,12],[10,6,30,18],[8,8,24,24],[6,10,18,30],[4,12,12,36],[2,14,6,42]],
    [[8,0,56,0],[7,1,49,7],[6,2,42,14],[5,3,35,21],[4,4,28,28],[3,5,21,35],[2,6,14,42],[1,7,7,49]],
];

/// 6-tap filter on 8-bit samples with arbitrary element offset (`off`).
/// `off == 1` filters horizontally, `off == src_stride` vertically. Mirrors
/// `FilterInput8bitWithStride_c`: `(P[-2] + P[3]) - 5*(P[-1] + P[2]) + 20*(P[0] + P[1])`.
#[inline(always)]
fn filter_input_8bit(src: &[u8], pos: usize, off: isize) -> i32 {
    let p = pos as isize;
    let s = |d: isize| src[(p + d) as usize] as i32;
    let pix05 = s(-2 * off) + s(3 * off);
    let pix14 = s(-off) + s(2 * off);
    let pix23 = s(0) + s(off);
    pix05 - 5 * pix14 + 20 * pix23
}

/// 6-tap filter over the 16-bit intermediate row (`HorFilterInput16bit_c`),
/// reading `tmp[k..k+6]`.
#[inline(always)]
fn hor_filter_input_16bit(tmp: &[i16], k: usize) -> i32 {
    let pix05 = tmp[k] as i32 + tmp[k + 5] as i32;
    let pix14 = tmp[k + 1] as i32 + tmp[k + 4] as i32;
    let pix23 = tmp[k + 2] as i32 + tmp[k + 3] as i32;
    pix05 - pix14 * 5 + pix23 * 20
}

/// `(a + b + 1) >> 1` rounding average over a `w`x`h` block (`PixelAvg_c`).
///
/// Dispatches to a bit-exact NEON kernel under `--features simd` on aarch64.
#[allow(clippy::too_many_arguments)]
#[allow(unreachable_code)]
pub(crate) fn pixel_avg(
    dst: &mut [u8],
    do_: usize,
    ds: usize,
    a: &[u8],
    ao: usize,
    as_: usize,
    b: &[u8],
    bo: usize,
    bs: usize,
    w: usize,
    h: usize,
) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return crate::dsp::simd::neon::pixel_avg(dst, do_, ds, a, ao, as_, b, bo, bs, w, h);
    pixel_avg_scalar(dst, do_, ds, a, ao, as_, b, bo, bs, w, h)
}

/// Scalar reference for [`pixel_avg`] (the conformance baseline / SIMD fallback).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixel_avg_scalar(
    dst: &mut [u8],
    do_: usize,
    ds: usize,
    a: &[u8],
    ao: usize,
    as_: usize,
    b: &[u8],
    bo: usize,
    bs: usize,
    w: usize,
    h: usize,
) {
    for i in 0..h {
        for j in 0..w {
            dst[do_ + i * ds + j] = ((a[ao + i * as_ + j] as i32 + b[bo + i * bs + j] as i32 + 1) >> 1) as u8;
        }
    }
}

/// Full-pel copy of a `w`x`h` block (`McCopy_c`).
pub fn mc_copy(
    dst: &mut [u8],
    do_: usize,
    ds: usize,
    src: &[u8],
    so: usize,
    ss: usize,
    w: usize,
    h: usize,
) {
    for i in 0..h {
        for j in 0..w {
            dst[do_ + i * ds + j] = src[so + i * ss + j];
        }
    }
}

/// Horizontal half-pel filter, quarter-pel position (2,0) (`McHorVer20_c`).
///
/// Dispatches to a bit-exact NEON kernel under `--features simd` on aarch64.
#[allow(unreachable_code)]
pub fn mc_hor_ver20(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return crate::dsp::simd::neon::mc_hor_ver20(dst, do_, ds, src, so, ss, w, h);
    mc_hor_ver20_scalar(dst, do_, ds, src, so, ss, w, h)
}

/// Scalar reference for [`mc_hor_ver20`].
pub(crate) fn mc_hor_ver20_scalar(
    dst: &mut [u8],
    do_: usize,
    ds: usize,
    src: &[u8],
    so: usize,
    ss: usize,
    w: usize,
    h: usize,
) {
    for i in 0..h {
        for j in 0..w {
            let v = filter_input_8bit(src, so + i * ss + j, 1);
            dst[do_ + i * ds + j] = clip1((v + 16) >> 5);
        }
    }
}

/// Vertical half-pel filter, quarter-pel position (0,2) (`McHorVer02_c`).
///
/// Dispatches to a bit-exact NEON kernel under `--features simd` on aarch64.
#[allow(unreachable_code)]
pub fn mc_hor_ver02(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return crate::dsp::simd::neon::mc_hor_ver02(dst, do_, ds, src, so, ss, w, h);
    mc_hor_ver02_scalar(dst, do_, ds, src, so, ss, w, h)
}

/// Scalar reference for [`mc_hor_ver02`].
pub(crate) fn mc_hor_ver02_scalar(
    dst: &mut [u8],
    do_: usize,
    ds: usize,
    src: &[u8],
    so: usize,
    ss: usize,
    w: usize,
    h: usize,
) {
    for i in 0..h {
        for j in 0..w {
            let v = filter_input_8bit(src, so + i * ss + j, ss as isize);
            dst[do_ + i * ds + j] = clip1((v + 16) >> 5);
        }
    }
}

/// Center half-pel filter, quarter-pel position (2,2) (`McHorVer22_c`): vertical
/// 6-tap into a 16-bit row, then horizontal 6-tap with `(x+512)>>10` rounding.
pub fn mc_hor_ver22(
    dst: &mut [u8],
    do_: usize,
    ds: usize,
    src: &[u8],
    so: usize,
    ss: usize,
    w: usize,
    h: usize,
) {
    let mut tmp = [0i16; 16 + 5];
    for i in 0..h {
        for (j, t) in tmp.iter_mut().enumerate().take(w + 5) {
            // C: FilterInput8bitWithStride_c(pSrc - 2 + j, iSrcStride)
            let pos = (so + i * ss) as isize - 2 + j as isize;
            *t = filter_input_8bit(src, pos as usize, ss as isize) as i16;
        }
        for k in 0..w {
            dst[do_ + i * ds + k] = clip1((hor_filter_input_16bit(&tmp, k) + 512) >> 10);
        }
    }
}

// ----- the 16 quarter-pel luma positions ---------------------------------
// Temp planes are 16x16 with stride 16 (matching the C `uiTmp[256]` scratch).

fn mc_hor_ver01(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut t = [0u8; 256];
    mc_hor_ver02(&mut t, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, src, so, ss, &t, 0, 16, w, h);
}
fn mc_hor_ver03(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut t = [0u8; 256];
    mc_hor_ver02(&mut t, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, src, so + ss, ss, &t, 0, 16, w, h);
}
fn mc_hor_ver10(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut t = [0u8; 256];
    mc_hor_ver20(&mut t, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, src, so, ss, &t, 0, 16, w, h);
}
fn mc_hor_ver11(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut hor = [0u8; 256];
    let mut ver = [0u8; 256];
    mc_hor_ver20(&mut hor, 0, 16, src, so, ss, w, h);
    mc_hor_ver02(&mut ver, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, &hor, 0, 16, &ver, 0, 16, w, h);
}
fn mc_hor_ver12(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut ver = [0u8; 256];
    let mut ctr = [0u8; 256];
    mc_hor_ver02(&mut ver, 0, 16, src, so, ss, w, h);
    mc_hor_ver22(&mut ctr, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, &ver, 0, 16, &ctr, 0, 16, w, h);
}
fn mc_hor_ver13(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut hor = [0u8; 256];
    let mut ver = [0u8; 256];
    mc_hor_ver20(&mut hor, 0, 16, src, so + ss, ss, w, h);
    mc_hor_ver02(&mut ver, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, &hor, 0, 16, &ver, 0, 16, w, h);
}
fn mc_hor_ver21(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut hor = [0u8; 256];
    let mut ctr = [0u8; 256];
    mc_hor_ver20(&mut hor, 0, 16, src, so, ss, w, h);
    mc_hor_ver22(&mut ctr, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, &hor, 0, 16, &ctr, 0, 16, w, h);
}
fn mc_hor_ver23(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut hor = [0u8; 256];
    let mut ctr = [0u8; 256];
    mc_hor_ver20(&mut hor, 0, 16, src, so + ss, ss, w, h);
    mc_hor_ver22(&mut ctr, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, &hor, 0, 16, &ctr, 0, 16, w, h);
}
fn mc_hor_ver30(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut hor = [0u8; 256];
    mc_hor_ver20(&mut hor, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, src, so + 1, ss, &hor, 0, 16, w, h);
}
fn mc_hor_ver31(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut hor = [0u8; 256];
    let mut ver = [0u8; 256];
    mc_hor_ver20(&mut hor, 0, 16, src, so, ss, w, h);
    mc_hor_ver02(&mut ver, 0, 16, src, so + 1, ss, w, h);
    pixel_avg(dst, do_, ds, &hor, 0, 16, &ver, 0, 16, w, h);
}
fn mc_hor_ver32(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut ver = [0u8; 256];
    let mut ctr = [0u8; 256];
    mc_hor_ver02(&mut ver, 0, 16, src, so + 1, ss, w, h);
    mc_hor_ver22(&mut ctr, 0, 16, src, so, ss, w, h);
    pixel_avg(dst, do_, ds, &ver, 0, 16, &ctr, 0, 16, w, h);
}
fn mc_hor_ver33(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, w: usize, h: usize) {
    let mut hor = [0u8; 256];
    let mut ver = [0u8; 256];
    mc_hor_ver20(&mut hor, 0, 16, src, so + ss, ss, w, h);
    mc_hor_ver02(&mut ver, 0, 16, src, so + 1, ss, w, h);
    pixel_avg(dst, do_, ds, &hor, 0, 16, &ver, 0, 16, w, h);
}

/// Luma motion compensation dispatcher (`McLuma_c`). `src_off` is the linear
/// index of the (already integer-MV-offset) block top-left in `src`; the
/// fractional position is `(mvx & 3, mvy & 3)`.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    mvx: i16,
    mvy: i16,
    w: usize,
    h: usize,
) {
    let x = (mvx & 0x03) as usize;
    let y = (mvy & 0x03) as usize;
    let f = match (x, y) {
        (0, 0) => mc_copy,
        (0, 1) => mc_hor_ver01,
        (0, 2) => mc_hor_ver02,
        (0, 3) => mc_hor_ver03,
        (1, 0) => mc_hor_ver10,
        (1, 1) => mc_hor_ver11,
        (1, 2) => mc_hor_ver12,
        (1, 3) => mc_hor_ver13,
        (2, 0) => mc_hor_ver20,
        (2, 1) => mc_hor_ver21,
        (2, 2) => mc_hor_ver22,
        (2, 3) => mc_hor_ver23,
        (3, 0) => mc_hor_ver30,
        (3, 1) => mc_hor_ver31,
        (3, 2) => mc_hor_ver32,
        _ => mc_hor_ver33,
    };
    f(dst, 0, dst_stride, src, src_off, src_stride, w, h);
}

/// Chroma fractional MC with the bilinear weights (`McChromaWithFragMv_c`).
#[allow(clippy::too_many_arguments)]
fn mc_chroma_frag(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    so: usize,
    ss: usize,
    mvx: i16,
    mvy: i16,
    w: usize,
    h: usize,
) {
    let abcd = &G_ABCD[(mvy & 0x07) as usize][(mvx & 0x07) as usize];
    let (ia, ib, ic, id) = (abcd[0] as i32, abcd[1] as i32, abcd[2] as i32, abcd[3] as i32);
    let mut row = so;
    let mut next = so + ss;
    for i in 0..h {
        for j in 0..w {
            let v = ia * src[row + j] as i32
                + ib * src[row + j + 1] as i32
                + ic * src[next + j] as i32
                + id * src[next + j + 1] as i32;
            dst[i * ds + j] = ((v + 32) >> 6) as u8;
        }
        row = next;
        next += ss;
    }
}

/// Chroma motion compensation dispatcher (`McChroma_c`). Fractional position is
/// `(mvx & 7, mvy & 7)`; an all-zero fraction is a full-pel copy.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    mvx: i16,
    mvy: i16,
    w: usize,
    h: usize,
) {
    if (mvx & 0x07) == 0 && (mvy & 0x07) == 0 {
        mc_copy(dst, 0, dst_stride, src, src_off, src_stride, w, h);
    } else {
        mc_chroma_frag(dst, dst_stride, src, src_off, src_stride, mvx, mvy, w, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const SS: usize = 32; // source stride
    const DS: usize = 32; // dest stride
    const HT: usize = 30; // buffer height
    const PAD: usize = 4; // top/left padding so the 6-tap halo stays in bounds

    struct Lcg(u32);
    impl Lcg {
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 24) as u8
        }
    }

    // --- Independent reference (mirrors EncUT_MotionCompensation.cpp anchors) ---

    fn clip255(x: i32) -> u8 {
        x.clamp(0, 255) as u8
    }

    /// 6-tap on `buf[pos]` with the given element stride (FILTER6TAP).
    fn filter6tap(buf: &[u8], pos: usize, stride: isize) -> i32 {
        let p = pos as isize;
        let s = |d: isize| buf[(p + d) as usize] as i32;
        s(-2 * stride) + s(3 * stride) - 5 * (s(-stride) + s(2 * stride)) + 20 * (s(0) + s(stride))
    }
    fn filter6tap16(buf: &[i16], pos: usize) -> i32 {
        let p = pos as isize;
        let s = |d: isize| buf[(p + d) as usize] as i32;
        s(-2) + s(3) - 5 * (s(-1) + s(2)) + 20 * (s(0) + s(1))
    }

    /// Builds the H, V and HV half-pel planes from `src` over a `(w)x(h)` region
    /// rooted at `base` (MCHalfPelFilterAnchor).
    fn half_pel_anchor(
        dst_h: &mut [u8],
        dst_v: &mut [u8],
        dst_hv: &mut [u8],
        src: &[u8],
        base: usize,
        stride: usize,
        w: usize,
        h: usize,
    ) {
        let mut buf = [0i16; 64];
        let pb = 2usize; // pBuf base (pBuf+4 inside C, indexed [x+2])
        for y in 0..h {
            for x in 0..w {
                let v = filter6tap(src, base + y * stride + x, 1);
                dst_h[base + y * stride + x] = clip255((v + 16) >> 5);
            }
            for x in -2i32..(w as i32 + 3) {
                let pos = (base + y * stride) as i32 + x;
                let v = filter6tap(src, pos as usize, stride as isize);
                if x >= 0 && (x as usize) < w {
                    dst_v[base + y * stride + x as usize] = clip255((v + 16) >> 5);
                }
                buf[(pb as i32 + x + 2) as usize] = v as i16;
            }
            for x in 0..w {
                let v = filter6tap16(&buf, pb + 2 + x);
                dst_hv[base + y * stride + x] = clip255((v + 512) >> 10);
            }
        }
    }

    fn pixel_avg_anchor(
        dst: &mut [u8],
        s1: &[u8],
        o1: usize,
        s2: &[u8],
        o2: usize,
        stride: usize,
        w: usize,
        h: usize,
    ) {
        for y in 0..h {
            for x in 0..w {
                dst[y * DS + x] = ((s1[o1 + y * stride + x] as i32 + s2[o2 + y * stride + x] as i32 + 1) >> 1) as u8;
            }
        }
    }

    #[rustfmt::skip]
    const Q_NEEDED: [[bool; 4]; 4] = [
        [false, true, false, true],
        [true,  true,  true, true],
        [false, true, false, true],
        [true,  true,  true, true],
    ];
    #[rustfmt::skip]
    const HREF0: [[usize; 4]; 4] = [
        [0, 1, 1, 1], [0, 1, 1, 1], [2, 3, 3, 3], [0, 1, 1, 1],
    ];
    #[rustfmt::skip]
    const HREF1: [[usize; 4]; 4] = [
        [0, 0, 0, 0], [2, 2, 3, 2], [2, 2, 3, 2], [2, 2, 3, 2],
    ];

    /// Independent luma MC reference selecting among the four half-pel planes
    /// (MCLumaAnchor). `planes[0]` is full-pel; `1,2,3` are H, V, HV.
    #[allow(clippy::too_many_arguments)]
    fn luma_anchor(
        dst: &mut [u8],
        planes: &[&[u8]; 4],
        base: usize,
        stride: usize,
        mvx: i32,
        mvy: i32,
        w: usize,
        h: usize,
    ) {
        let xi = (mvx & 3) as usize;
        let yi = (mvy & 3) as usize;
        let offset = (mvy >> 2) * stride as i32 + (mvx >> 2);
        let o1 = (base as i32 + offset + if yi == 3 { stride as i32 } else { 0 }) as usize;
        let s1 = planes[HREF0[yi][xi]];
        if Q_NEEDED[yi][xi] {
            let o2 = (base as i32 + offset + if xi == 3 { 1 } else { 0 }) as usize;
            let s2 = planes[HREF1[yi][xi]];
            pixel_avg_anchor(dst, s1, o1, s2, o2, stride, w, h);
        } else {
            for y in 0..h {
                for x in 0..w {
                    dst[y * DS + x] = s1[o1 + y * stride + x];
                }
            }
        }
    }

    #[test]
    fn luma_matches_anchor_all_positions() {
        let mut lcg = Lcg(0x1234_5678);
        // Build plane 0 (full-pel) random; planes 1-3 derived by the anchor.
        let mut p0 = vec![0u8; HT * SS];
        for v in p0.iter_mut() {
            *v = lcg.next_u8();
        }
        let base = PAD * SS + PAD;

        for &(w, h) in &[(4usize, 4usize), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16)] {
            // Anchor planes filled for a (w+1)x(h+1) region as in the C test.
            let mut ph = vec![0u8; HT * SS];
            let mut pv = vec![0u8; HT * SS];
            let mut phv = vec![0u8; HT * SS];
            half_pel_anchor(&mut ph, &mut pv, &mut phv, &p0, base, SS, w + 1, h + 1);
            let planes: [&[u8]; 4] = [&p0, &ph, &pv, &phv];

            for a in 0..4i16 {
                for b in 0..4i16 {
                    let mut want = vec![0u8; HT * DS];
                    luma_anchor(&mut want, &planes, base, SS, a as i32, b as i32, w, h);

                    let mut got = vec![0u8; HT * DS];
                    mc_luma(&mut got, DS, &p0, base, SS, a, b, w, h);

                    assert_eq!(got, want, "luma {w}x{h} mv=({a},{b})");
                }
            }
        }
    }

    /// Independent chroma reference (MCChromaAnchor), bilinear from `iBiPara`.
    #[allow(clippy::too_many_arguments)]
    fn chroma_anchor(dst: &mut [u8], src: &[u8], base: usize, stride: usize, mvx: i32, mvy: i32, w: usize, h: usize) {
        let xi = mvx & 7;
        let yi = mvy & 7;
        let b0 = (8 - xi) * (8 - yi);
        let b1 = xi * (8 - yi);
        let b2 = (8 - xi) * yi;
        let b3 = xi * yi;
        let mut row = base;
        let mut next = base + stride;
        for y in 0..h {
            for x in 0..w {
                let v = b0 * src[row + x] as i32
                    + b1 * src[row + x + 1] as i32
                    + b2 * src[next + x] as i32
                    + b3 * src[next + x + 1] as i32;
                dst[y * DS + x] = ((v + 32) >> 6) as u8;
            }
            row = next;
            next += stride;
        }
    }

    #[test]
    fn chroma_matches_anchor_all_fractions() {
        let mut lcg = Lcg(0x0BAD_F00D);
        let mut src = vec![0u8; HT * SS];
        for v in src.iter_mut() {
            *v = lcg.next_u8();
        }
        // Block rooted at top-left (matches the C test's uSrcTest[0]).
        for &(w, h) in &[(2usize, 2usize), (2, 4), (4, 2), (4, 4), (4, 8), (8, 4), (8, 8)] {
            for a in 0..8i16 {
                for b in 0..8i16 {
                    let mut want = vec![0u8; HT * DS];
                    chroma_anchor(&mut want, &src, 0, SS, a as i32, b as i32, w, h);

                    let mut got = vec![0u8; HT * DS];
                    mc_chroma(&mut got, DS, &src, 0, SS, a, b, w, h);

                    assert_eq!(got, want, "chroma {w}x{h} mv=({a},{b})");
                }
            }
        }
    }

    #[test]
    fn full_pel_copy_is_exact() {
        let mut lcg = Lcg(0xDEAD_BEEF);
        let src: Vec<u8> = (0..HT * SS).map(|_| lcg.next_u8()).collect();
        let base = PAD * SS + PAD;
        let mut got = vec![0u8; HT * DS];
        mc_luma(&mut got, DS, &src, base, SS, 0, 0, 16, 16);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(got[y * DS + x], src[base + y * SS + x]);
            }
        }
    }

    #[test]
    fn horizontal_halfpel_known_values() {
        // Flat input -> filter returns 32*v, (32v+16)>>5 == v.
        let src = vec![100u8; HT * SS];
        let base = PAD * SS + PAD;
        let mut got = vec![0u8; HT * DS];
        mc_hor_ver20(&mut got, 0, DS, &src, base, SS, 8, 8);
        assert!(got[..8].iter().all(|&p| p == 100));
    }
}
