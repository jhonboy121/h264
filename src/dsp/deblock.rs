//! In-loop deblocking edge filters, ported from the scalar `_c` kernels in
//! `reference/codec/common/src/deblocking_common.cpp`.
//!
//! These are the buffer-level edge filters only: the luma/chroma `Lt4` (bS < 4,
//! §8.7.2.3) and `Eq4` (bS == 4, §8.7.2.4) kernels with their vertical/horizontal
//! wrappers. The high-level `WelsDeblockingFilterMB` driver (boundary-strength
//! derivation, MB iteration) needs decoder context and is ported in a later phase.
//!
//! As in the C code, the destination pointer addresses the first `q` sample of
//! an edge; the filter reaches `p` samples at negative offsets. In Rust we pass
//! the plane as a slice `pix` plus the linear index `off` of that sample, so the
//! negative reaches stay in bounds via signed offset arithmetic.

use crate::dsp::{clip1, clip3};

#[inline(always)]
fn abs_i32(x: i32) -> i32 {
    x.abs()
}

/// Luma `Lt4` edge filter, generic strides (`DeblockLumaLt4_c`). `stride_x` steps
/// across the edge (p/q direction), `stride_y` along it; 16 lines, `tc[line>>2]`.
#[allow(clippy::too_many_arguments)]
pub fn deblock_luma_lt4(
    pix: &mut [u8],
    off: usize,
    stride_x: isize,
    stride_y: isize,
    alpha: i32,
    beta: i32,
    tc: &[i8],
) {
    for i in 0..16i32 {
        let tc0 = tc[(i >> 2) as usize] as i32;
        if tc0 >= 0 {
            let base = off as isize + i as isize * stride_y;
            let at = |d: isize| pix[(base + d) as usize] as i32;
            let p0 = at(-stride_x);
            let p1 = at(-2 * stride_x);
            let p2 = at(-3 * stride_x);
            let q0 = at(0);
            let q1 = at(stride_x);
            let q2 = at(2 * stride_x);
            if abs_i32(p0 - q0) < alpha && abs_i32(p1 - p0) < beta && abs_i32(q1 - q0) < beta {
                let mut itc = tc0;
                if abs_i32(p2 - p0) < beta {
                    let v = p1 + clip3((p2 + ((p0 + q0 + 1) >> 1) - (p1 * 2)) >> 1, -tc0, tc0);
                    pix[(base - 2 * stride_x) as usize] = v as u8;
                    itc += 1;
                }
                if abs_i32(q2 - q0) < beta {
                    let v = q1 + clip3((q2 + ((p0 + q0 + 1) >> 1) - (q1 * 2)) >> 1, -tc0, tc0);
                    pix[(base + stride_x) as usize] = v as u8;
                    itc += 1;
                }
                let ideta = clip3((((q0 - p0) * 4) + (p1 - q1) + 4) >> 3, -itc, itc);
                pix[(base - stride_x) as usize] = clip1(p0 + ideta);
                pix[base as usize] = clip1(q0 - ideta);
            }
        }
    }
}

/// Luma `Eq4` (strong) edge filter, generic strides (`DeblockLumaEq4_c`).
pub fn deblock_luma_eq4(
    pix: &mut [u8],
    off: usize,
    stride_x: isize,
    stride_y: isize,
    alpha: i32,
    beta: i32,
) {
    for i in 0..16i32 {
        let base = off as isize + i as isize * stride_y;
        let at = |d: isize| pix[(base + d) as usize] as i32;
        let p0 = at(-stride_x);
        let p1 = at(-2 * stride_x);
        let p2 = at(-3 * stride_x);
        let q0 = at(0);
        let q1 = at(stride_x);
        let q2 = at(2 * stride_x);
        // Read the §8.7.2.4 edge taps up front so the shared borrow ends before writes.
        let p3 = at(-4 * stride_x);
        let q3 = at(3 * stride_x);
        let deta_p0q0 = abs_i32(p0 - q0);
        if deta_p0q0 < alpha && abs_i32(p1 - p0) < beta && abs_i32(q1 - q0) < beta {
            if deta_p0q0 < ((alpha >> 2) + 2) {
                if abs_i32(p2 - p0) < beta {
                    pix[(base - stride_x) as usize] = ((p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + 4) >> 3) as u8;
                    pix[(base - 2 * stride_x) as usize] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                    pix[(base - 3 * stride_x) as usize] = ((p3 * 2 + p2 + p2 * 2 + p1 + p0 + q0 + 4) >> 3) as u8;
                } else {
                    pix[(base - stride_x) as usize] = ((p1 * 2 + p0 + q1 + 2) >> 2) as u8;
                }
                if abs_i32(q2 - q0) < beta {
                    pix[base as usize] = ((p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + 4) >> 3) as u8;
                    pix[(base + stride_x) as usize] = ((p0 + q0 + q1 + q2 + 2) >> 2) as u8;
                    pix[(base + 2 * stride_x) as usize] = ((q3 * 2 + q2 + q2 * 2 + q1 + q0 + p0 + 4) >> 3) as u8;
                } else {
                    pix[base as usize] = ((q1 * 2 + q0 + p1 + 2) >> 2) as u8;
                }
            } else {
                pix[(base - stride_x) as usize] = ((p1 * 2 + p0 + q1 + 2) >> 2) as u8;
                pix[base as usize] = ((q1 * 2 + q0 + p1 + 2) >> 2) as u8;
            }
        }
    }
}

/// Vertical luma `Lt4` edge (`DeblockLumaLt4V_c`): `stride_x = stride`, `stride_y = 1`.
pub fn deblock_luma_lt4_v(pix: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc: &[i8]) {
    deblock_luma_lt4(pix, off, stride as isize, 1, alpha, beta, tc);
}
/// Horizontal luma `Lt4` edge (`DeblockLumaLt4H_c`): `stride_x = 1`, `stride_y = stride`.
pub fn deblock_luma_lt4_h(pix: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc: &[i8]) {
    deblock_luma_lt4(pix, off, 1, stride as isize, alpha, beta, tc);
}
/// Vertical luma `Eq4` edge (`DeblockLumaEq4V_c`).
pub fn deblock_luma_eq4_v(pix: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32) {
    deblock_luma_eq4(pix, off, stride as isize, 1, alpha, beta);
}
/// Horizontal luma `Eq4` edge (`DeblockLumaEq4H_c`).
pub fn deblock_luma_eq4_h(pix: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32) {
    deblock_luma_eq4(pix, off, 1, stride as isize, alpha, beta);
}

/// One chroma component's `Lt4` filter at a given line base (shared by the
/// two-plane and single-plane variants).
#[inline(always)]
fn chroma_lt4_one(pix: &mut [u8], base: isize, stride_x: isize, alpha: i32, beta: i32, tc0: i32) {
    let at = |d: isize| pix[(base + d) as usize] as i32;
    let p0 = at(-stride_x);
    let p1 = at(-2 * stride_x);
    let q0 = at(0);
    let q1 = at(stride_x);
    if abs_i32(p0 - q0) < alpha && abs_i32(p1 - p0) < beta && abs_i32(q1 - q0) < beta {
        let ideta = clip3((((q0 - p0) * 4) + (p1 - q1) + 4) >> 3, -tc0, tc0);
        pix[(base - stride_x) as usize] = clip1(p0 + ideta);
        pix[base as usize] = clip1(q0 - ideta);
    }
}

#[inline(always)]
fn chroma_eq4_one(pix: &mut [u8], base: isize, stride_x: isize, alpha: i32, beta: i32) {
    let at = |d: isize| pix[(base + d) as usize] as i32;
    let p0 = at(-stride_x);
    let p1 = at(-2 * stride_x);
    let q0 = at(0);
    let q1 = at(stride_x);
    if abs_i32(p0 - q0) < alpha && abs_i32(p1 - p0) < beta && abs_i32(q1 - q0) < beta {
        pix[(base - stride_x) as usize] = ((p1 * 2 + p0 + q1 + 2) >> 2) as u8;
        pix[base as usize] = ((q1 * 2 + q0 + p1 + 2) >> 2) as u8;
    }
}

/// Two-plane chroma `Lt4` edge filter (`DeblockChromaLt4_c`); 8 lines, `tc[line>>1]`,
/// applied only when `tc0 > 0`.
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_lt4(
    cb: &mut [u8],
    cb_off: usize,
    cr: &mut [u8],
    cr_off: usize,
    stride_x: isize,
    stride_y: isize,
    alpha: i32,
    beta: i32,
    tc: &[i8],
) {
    for i in 0..8i32 {
        let tc0 = tc[(i >> 1) as usize] as i32;
        if tc0 > 0 {
            let bb = cb_off as isize + i as isize * stride_y;
            let bc = cr_off as isize + i as isize * stride_y;
            chroma_lt4_one(cb, bb, stride_x, alpha, beta, tc0);
            chroma_lt4_one(cr, bc, stride_x, alpha, beta, tc0);
        }
    }
}

/// Two-plane chroma `Eq4` edge filter (`DeblockChromaEq4_c`).
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_eq4(
    cb: &mut [u8],
    cb_off: usize,
    cr: &mut [u8],
    cr_off: usize,
    stride_x: isize,
    stride_y: isize,
    alpha: i32,
    beta: i32,
) {
    for i in 0..8i32 {
        let bb = cb_off as isize + i as isize * stride_y;
        let bc = cr_off as isize + i as isize * stride_y;
        chroma_eq4_one(cb, bb, stride_x, alpha, beta);
        chroma_eq4_one(cr, bc, stride_x, alpha, beta);
    }
}

/// Vertical two-plane chroma `Lt4` (`DeblockChromaLt4V_c`).
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_lt4_v(cb: &mut [u8], cb_off: usize, cr: &mut [u8], cr_off: usize, stride: usize, alpha: i32, beta: i32, tc: &[i8]) {
    deblock_chroma_lt4(cb, cb_off, cr, cr_off, stride as isize, 1, alpha, beta, tc);
}
/// Horizontal two-plane chroma `Lt4` (`DeblockChromaLt4H_c`).
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_lt4_h(cb: &mut [u8], cb_off: usize, cr: &mut [u8], cr_off: usize, stride: usize, alpha: i32, beta: i32, tc: &[i8]) {
    deblock_chroma_lt4(cb, cb_off, cr, cr_off, 1, stride as isize, alpha, beta, tc);
}
/// Vertical two-plane chroma `Eq4` (`DeblockChromaEq4V_c`).
pub fn deblock_chroma_eq4_v(cb: &mut [u8], cb_off: usize, cr: &mut [u8], cr_off: usize, stride: usize, alpha: i32, beta: i32) {
    deblock_chroma_eq4(cb, cb_off, cr, cr_off, stride as isize, 1, alpha, beta);
}
/// Horizontal two-plane chroma `Eq4` (`DeblockChromaEq4H_c`).
pub fn deblock_chroma_eq4_h(cb: &mut [u8], cb_off: usize, cr: &mut [u8], cr_off: usize, stride: usize, alpha: i32, beta: i32) {
    deblock_chroma_eq4(cb, cb_off, cr, cr_off, 1, stride as isize, alpha, beta);
}

/// Single-plane chroma `Lt4` edge filter (`DeblockChromaLt42_c`), for interleaved
/// or single-component chroma.
pub fn deblock_chroma_lt42(pix: &mut [u8], off: usize, stride_x: isize, stride_y: isize, alpha: i32, beta: i32, tc: &[i8]) {
    for i in 0..8i32 {
        let tc0 = tc[(i >> 1) as usize] as i32;
        if tc0 > 0 {
            let base = off as isize + i as isize * stride_y;
            chroma_lt4_one(pix, base, stride_x, alpha, beta, tc0);
        }
    }
}
/// Single-plane chroma `Eq4` edge filter (`DeblockChromaEq42_c`).
pub fn deblock_chroma_eq42(pix: &mut [u8], off: usize, stride_x: isize, stride_y: isize, alpha: i32, beta: i32) {
    for i in 0..8i32 {
        let base = off as isize + i as isize * stride_y;
        chroma_eq4_one(pix, base, stride_x, alpha, beta);
    }
}

/// `WelsNonZeroCount_c`: clamp each of the 24 NZC entries to 0/1.
pub fn wels_non_zero_count(nzc: &mut [i8]) {
    for v in nzc.iter_mut().take(24) {
        *v = (*v != 0) as i8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0 >> 8
        }
        fn r(&mut self, m: u32) -> u32 {
            self.next() % m
        }
    }

    fn clip255(x: i32) -> u8 {
        x.clamp(0, 255) as u8
    }
    fn c3(x: i32, lo: i32, hi: i32) -> i32 {
        x.clamp(lo, hi)
    }

    // --- Independent references (DecUT_DeblockCommon.cpp anchors) ---

    fn anchor_luma_normal(pix: &mut [u8], off: usize, sx: isize, sy: isize, alpha: i32, beta: i32, tc: &[i8]) {
        for line in 0..16i32 {
            let itc_idx = (line >> 2) as usize;
            let mut itc = tc[itc_idx] as i32;
            let base = off as isize + line as isize * sy;
            let at = |b: &[u8], d: isize| b[(base + d) as usize] as i32;
            let p: [i32; 3] = [at(pix, -sx), at(pix, -2 * sx), at(pix, -3 * sx)];
            let q: [i32; 3] = [at(pix, 0), at(pix, sx), at(pix, 2 * sx)];
            if (p[0] - q[0]).abs() < alpha && (p[1] - p[0]).abs() < beta && (q[1] - q[0]).abs() < beta {
                if (p[2] - p[0]).abs() < beta {
                    let v = c3(
                        p[1] + c3((p[2] + ((p[0] + q[0] + 1) >> 1) - (p[1] << 1)) >> 1, -(tc[itc_idx] as i32), tc[itc_idx] as i32),
                        0,
                        255,
                    );
                    pix[(base - 2 * sx) as usize] = v as u8;
                    itc += 1;
                }
                if (q[2] - q[0]).abs() < beta {
                    let v = c3(
                        q[1] + c3((q[2] + ((p[0] + q[0] + 1) >> 1) - (q[1] << 1)) >> 1, -(tc[itc_idx] as i32), tc[itc_idx] as i32),
                        0,
                        255,
                    );
                    pix[(base + sx) as usize] = v as u8;
                    itc += 1;
                }
                let idelta = c3((((q[0] - p[0]) * 4) + (p[1] - q[1]) + 4) >> 3, -itc, itc);
                pix[(base - sx) as usize] = clip255(p[0] + idelta);
                pix[base as usize] = clip255(q[0] - idelta);
            }
        }
    }

    fn anchor_luma_intra(pix: &mut [u8], off: usize, sx: isize, sy: isize, alpha: i32, beta: i32) {
        for line in 0..16i32 {
            let base = off as isize + line as isize * sy;
            let at = |b: &[u8], d: isize| b[(base + d) as usize] as i32;
            let p: [i32; 4] = [at(pix, -sx), at(pix, -2 * sx), at(pix, -3 * sx), at(pix, -4 * sx)];
            let q: [i32; 4] = [at(pix, 0), at(pix, sx), at(pix, 2 * sx), at(pix, 3 * sx)];
            if (p[0] - q[0]).abs() < alpha && (p[1] - p[0]).abs() < beta && (q[1] - q[0]).abs() < beta {
                if (p[2] - p[0]).abs() < beta && (p[0] - q[0]).abs() < ((alpha >> 2) + 2) {
                    pix[(base - sx) as usize] = ((p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) >> 3) as u8;
                    pix[(base - 2 * sx) as usize] = ((p[2] + p[1] + p[0] + q[0] + 2) >> 2) as u8;
                    pix[(base - 3 * sx) as usize] = ((2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) >> 3) as u8;
                } else {
                    pix[(base - sx) as usize] = ((2 * p[1] + p[0] + q[1] + 2) >> 2) as u8;
                }
                if (q[2] - q[0]).abs() < beta && (p[0] - q[0]).abs() < ((alpha >> 2) + 2) {
                    pix[base as usize] = ((p[1] + 2 * p[0] + 2 * q[0] + 2 * q[1] + q[2] + 4) >> 3) as u8;
                    pix[(base + sx) as usize] = ((p[0] + q[0] + q[1] + q[2] + 2) >> 2) as u8;
                    pix[(base + 2 * sx) as usize] = ((2 * q[3] + 3 * q[2] + q[1] + q[0] + p[0] + 4) >> 3) as u8;
                } else {
                    pix[base as usize] = ((2 * q[1] + q[0] + p[1] + 2) >> 2) as u8;
                }
            }
        }
    }

    fn anchor_chroma_normal(cb: &mut [u8], cr: &mut [u8], off: usize, sx: isize, sy: isize, alpha: i32, beta: i32, tc: &[i8]) {
        for line in 0..8i32 {
            let itc = tc[(line >> 1) as usize] as i32;
            let base = off as isize + line as isize * sy;
            for plane in [&mut *cb, &mut *cr] {
                let at = |b: &[u8], d: isize| b[(base + d) as usize] as i32;
                let p0 = at(plane, -sx);
                let p1 = at(plane, -2 * sx);
                let q0 = at(plane, 0);
                let q1 = at(plane, sx);
                if (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta {
                    let idelta = c3((((q0 - p0) * 4) + (p1 - q1) + 4) >> 3, -itc, itc);
                    plane[(base - sx) as usize] = clip255(p0 + idelta);
                    plane[base as usize] = clip255(q0 - idelta);
                }
            }
        }
    }

    fn anchor_chroma_intra(cb: &mut [u8], cr: &mut [u8], off: usize, sx: isize, sy: isize, alpha: i32, beta: i32) {
        for line in 0..8i32 {
            let base = off as isize + line as isize * sy;
            for plane in [&mut *cb, &mut *cr] {
                let at = |b: &[u8], d: isize| b[(base + d) as usize] as i32;
                let p0 = at(plane, -sx);
                let p1 = at(plane, -2 * sx);
                let q0 = at(plane, 0);
                let q1 = at(plane, sx);
                if (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta {
                    plane[(base - sx) as usize] = clip255((2 * p1 + p0 + q1 + 2) >> 2);
                    plane[base as usize] = clip255((2 * q1 + q0 + p1 + 2) >> 2);
                }
            }
        }
    }

    /// Builds a `width`x`width` plane and the filter params per the
    /// GENERATE_DATA_DEBLOCKING macro. Returns `(buf, alpha, beta, tc)`.
    fn generate(lcg: &mut Lcg, num: u32, width: usize) -> (Vec<u8>, i32, i32, [i8; 4]) {
        let n = width * width;
        let mut buf = vec![0u8; n];
        let (alpha, beta, tc);
        if num == 0 {
            alpha = 255;
            beta = 18;
            tc = [25i8; 4];
            buf[0] = 128;
            for i in 1..n {
                buf[i] = (buf[i - 1] as i32 - 16 + lcg.r(32) as i32).clamp(0, 255) as u8;
            }
        } else if num == 1 {
            alpha = 4;
            beta = 2;
            tc = [9i8; 4];
            buf[0] = 128;
            for i in 1..n {
                buf[i] = (buf[i - 1] as i32 - 4 + lcg.r(8) as i32).clamp(0, 255) as u8;
            }
        } else {
            alpha = lcg.r(256) as i32;
            beta = lcg.r(19) as i32;
            tc = core::array::from_fn(|_| lcg.r(26) as i8);
            for v in buf.iter_mut() {
                *v = lcg.r(256) as u8;
            }
        }
        (buf, alpha, beta, tc)
    }

    #[test]
    fn luma_lt4_matches_anchor() {
        let mut lcg = Lcg(0xC0FF_EE01);
        for num in 0..600u32 {
            let n = num % 3;
            // Horizontal: offset 8*1, sx=1, sy=16.
            let (mut base, a, b, tc) = generate(&mut lcg, n, 16);
            let mut refb = base.clone();
            anchor_luma_normal(&mut base, 8, 1, 16, a, b, &tc);
            deblock_luma_lt4(&mut refb, 8, 1, 16, a, b, &tc);
            assert_eq!(base, refb, "luma lt4 H num={num}");

            // Vertical: offset 8*16, sx=16, sy=1.
            let (mut base, a, b, tc) = generate(&mut lcg, n, 16);
            let mut refb = base.clone();
            anchor_luma_normal(&mut base, 128, 16, 1, a, b, &tc);
            deblock_luma_lt4(&mut refb, 128, 16, 1, a, b, &tc);
            assert_eq!(base, refb, "luma lt4 V num={num}");
        }
    }

    #[test]
    fn luma_eq4_matches_anchor() {
        let mut lcg = Lcg(0xC0FF_EE02);
        for num in 0..600u32 {
            let n = num % 3;
            let (mut base, a, b, _tc) = generate(&mut lcg, n, 16);
            let mut refb = base.clone();
            anchor_luma_intra(&mut base, 8, 1, 16, a, b);
            deblock_luma_eq4(&mut refb, 8, 1, 16, a, b);
            assert_eq!(base, refb, "luma eq4 H num={num}");

            let (mut base, a, b, _tc) = generate(&mut lcg, n, 16);
            let mut refb = base.clone();
            anchor_luma_intra(&mut base, 128, 16, 1, a, b);
            deblock_luma_eq4(&mut refb, 128, 16, 1, a, b);
            assert_eq!(base, refb, "luma eq4 V num={num}");
        }
    }

    #[test]
    fn chroma_lt4_matches_anchor() {
        let mut lcg = Lcg(0xC0FF_EE03);
        for num in 0..600u32 {
            let n = num % 3;
            // Horizontal: offset 4*1, sx=1, sy=8.
            let (mut cb_base, a, b, tc) = generate(&mut lcg, n, 8);
            let (mut cr_base, _, _, _) = generate(&mut lcg, n, 8);
            let (mut cb_ref, mut cr_ref) = (cb_base.clone(), cr_base.clone());
            anchor_chroma_normal(&mut cb_base, &mut cr_base, 4, 1, 8, a, b, &tc);
            deblock_chroma_lt4(&mut cb_ref, 4, &mut cr_ref, 4, 1, 8, a, b, &tc);
            assert_eq!(cb_base, cb_ref, "chroma lt4 H cb num={num}");
            assert_eq!(cr_base, cr_ref, "chroma lt4 H cr num={num}");

            // Vertical: offset 4*8, sx=8, sy=1.
            let (mut cb_base, a, b, tc) = generate(&mut lcg, n, 8);
            let (mut cr_base, _, _, _) = generate(&mut lcg, n, 8);
            let (mut cb_ref, mut cr_ref) = (cb_base.clone(), cr_base.clone());
            anchor_chroma_normal(&mut cb_base, &mut cr_base, 32, 8, 1, a, b, &tc);
            deblock_chroma_lt4(&mut cb_ref, 32, &mut cr_ref, 32, 8, 1, a, b, &tc);
            assert_eq!(cb_base, cb_ref, "chroma lt4 V cb num={num}");
            assert_eq!(cr_base, cr_ref, "chroma lt4 V cr num={num}");
        }
    }

    #[test]
    fn chroma_eq4_matches_anchor() {
        let mut lcg = Lcg(0xC0FF_EE04);
        for num in 0..600u32 {
            let n = num % 3;
            let (mut cb_base, a, b, _tc) = generate(&mut lcg, n, 8);
            let (mut cr_base, _, _, _) = generate(&mut lcg, n, 8);
            let (mut cb_ref, mut cr_ref) = (cb_base.clone(), cr_base.clone());
            anchor_chroma_intra(&mut cb_base, &mut cr_base, 4, 1, 8, a, b);
            deblock_chroma_eq4(&mut cb_ref, 4, &mut cr_ref, 4, 1, 8, a, b);
            assert_eq!(cb_base, cb_ref, "chroma eq4 H cb num={num}");
            assert_eq!(cr_base, cr_ref, "chroma eq4 H cr num={num}");

            let (mut cb_base, a, b, _tc) = generate(&mut lcg, n, 8);
            let (mut cr_base, _, _, _) = generate(&mut lcg, n, 8);
            let (mut cb_ref, mut cr_ref) = (cb_base.clone(), cr_base.clone());
            anchor_chroma_intra(&mut cb_base, &mut cr_base, 32, 8, 1, a, b);
            deblock_chroma_eq4(&mut cb_ref, 32, &mut cr_ref, 32, 8, 1, a, b);
            assert_eq!(cb_base, cb_ref, "chroma eq4 V cb num={num}");
            assert_eq!(cr_base, cr_ref, "chroma eq4 V cr num={num}");
        }
    }

    #[test]
    fn vh_wrappers_consistent() {
        // The V/H wrappers must equal the generic kernel with the matching strides.
        let mut lcg = Lcg(0x1357_9BDF);
        let (buf, a, b, tc) = generate(&mut lcg, 2, 16);
        let mut g = buf.clone();
        let mut w = buf.clone();
        deblock_luma_lt4(&mut g, 128, 16, 1, a, b, &tc);
        deblock_luma_lt4_v(&mut w, 128, 16, a, b, &tc);
        assert_eq!(g, w);
        let mut g = buf.clone();
        let mut w = buf;
        deblock_luma_lt4(&mut g, 8, 1, 16, a, b, &tc);
        deblock_luma_lt4_h(&mut w, 8, 16, a, b, &tc);
        assert_eq!(g, w);
    }

    #[test]
    fn non_zero_count_clamps() {
        let mut nzc = [0i8, 1, 5, -3, 0, 127, 2];
        let mut expanded = vec![0i8; 24];
        expanded[..7].copy_from_slice(&nzc);
        wels_non_zero_count(&mut expanded);
        assert_eq!(&expanded[..7], &[0, 1, 1, 1, 0, 1, 1]);
        // sanity: original short slice ignored beyond logic
        let _ = &mut nzc;
    }
}
