//! Bit-exact tests for the encoder intra predictors. The reference functions
//! port the `_ref`/anchor logic of `EncUT_GetIntraPredictor.cpp` (and the H.264
//! spec for the modes that test only exercises indirectly), evaluated against
//! the same random reference plane the kernels read.

use super::*;
use alloc::vec::Vec;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 23) as u8
    }
}

// A reference plane large enough that every neighbour read is in bounds for an
// `roff` placed well inside it.
fn make_plane(r: &mut Lcg, stride: usize) -> (Vec<u8>, usize) {
    let plane: Vec<u8> = (0..stride * 48).map(|_| r.byte()).collect();
    let roff = stride * 24 + 24;
    (plane, roff)
}

/// Neighbour reader: `g(d)` reads `plane[roff + d]` as i32.
fn reader(plane: &[u8], roff: usize) -> impl Fn(isize) -> i32 + '_ {
    move |d: isize| plane[(roff as isize + d) as usize] as i32
}

const STRIDES: [usize; 2] = [32, 24];

// ===========================================================================
// Luma 4x4
// ===========================================================================

fn avg3(a: i32, b: i32, c: i32) -> u8 {
    ((2 + a + c + (b << 1)) >> 2) as u8
}
fn avg2(a: i32, b: i32) -> u8 {
    ((1 + a + b) >> 1) as u8
}

#[test]
fn i4x4_modes_match_reference() {
    let mut r = Lcg(0x1111_2222);
    for &stride in &STRIDES {
        let s = stride as isize;
        for _ in 0..200 {
            let (plane, roff) = make_plane(&mut r, stride);
            let g = reader(&plane, roff);
            let mut pred = [0u8; 16];

            // V
            i4x4_luma_pred_v(&mut pred, &plane, roff, stride);
            let t: [u8; 4] = core::array::from_fn(|x| g(-s + x as isize) as u8);
            for row in 0..4 {
                assert_eq!(&pred[row * 4..row * 4 + 4], &t);
            }

            // H
            i4x4_luma_pred_h(&mut pred, &plane, roff, stride);
            for row in 0..4 {
                let l = g(row as isize * s - 1) as u8;
                assert!(pred[row * 4..row * 4 + 4].iter().all(|&p| p == l));
            }

            // DC / DcLeft / DcTop / DcNA
            let left4 = g(-1) + g(s - 1) + g(2 * s - 1) + g(3 * s - 1);
            let top4 = g(-s) + g(1 - s) + g(2 - s) + g(3 - s);
            i4x4_luma_pred_dc(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == ((left4 + top4 + 4) >> 3) as u8));
            i4x4_luma_pred_dc_left(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == ((left4 + 2) >> 2) as u8));
            i4x4_luma_pred_dc_top(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == ((top4 + 2) >> 2) as u8));
            i4x4_luma_pred_dc_na(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == 0x80));

            // Directional anchors (ported from EncUT_GetIntraPredictor).
            let tk: [i32; 8] = core::array::from_fn(|k| g(-s + k as isize));
            let lk: [i32; 4] = core::array::from_fn(|k| g(k as isize * s - 1));
            let lt = g(-s - 1);

            // DDL
            i4x4_luma_pred_ddl(&mut pred, &plane, roff, stride);
            let mut e = [0u8; 16];
            let d = [
                avg3(tk[0], tk[1], tk[2]),
                avg3(tk[1], tk[2], tk[3]),
                avg3(tk[2], tk[3], tk[4]),
                avg3(tk[3], tk[4], tk[5]),
                avg3(tk[4], tk[5], tk[6]),
                avg3(tk[5], tk[6], tk[7]),
                avg3(tk[6], tk[7], tk[7]),
            ];
            e[0] = d[0];
            e[1] = d[1];
            e[4] = d[1];
            e[2] = d[2];
            e[5] = d[2];
            e[8] = d[2];
            e[3] = d[3];
            e[6] = d[3];
            e[9] = d[3];
            e[12] = d[3];
            e[7] = d[4];
            e[10] = d[4];
            e[13] = d[4];
            e[11] = d[5];
            e[14] = d[5];
            e[15] = d[6];
            assert_eq!(pred, e, "DDL stride={stride}");

            // DDLTop
            i4x4_luma_pred_ddl_top(&mut pred, &plane, roff, stride);
            let dlt0 = avg3(tk[0], tk[1], tk[2]);
            let dlt1 = avg3(tk[1], tk[2], tk[3]);
            let dlt2 = avg3(tk[2], tk[3], tk[3]);
            let dlt3 = ((2 + (tk[3] << 2)) >> 2) as u8;
            let mut e = [dlt3; 16];
            e[0] = dlt0;
            e[1] = dlt1;
            e[4] = dlt1;
            e[2] = dlt2;
            e[5] = dlt2;
            e[8] = dlt2;
            e[3] = dlt3;
            assert_eq!(pred, e, "DDLTop stride={stride}");

            // DDR
            i4x4_luma_pred_ddr(&mut pred, &plane, roff, stride);
            let tl0 = 1 + lt + lk[0];
            let lt0 = 1 + lt + tk[0];
            let t01 = 1 + tk[0] + tk[1];
            let t12 = 1 + tk[1] + tk[2];
            let t23 = 1 + tk[2] + tk[3];
            let l01 = 1 + lk[0] + lk[1];
            let l12 = 1 + lk[1] + lk[2];
            let l23 = 1 + lk[2] + lk[3];
            let ddr = [
                ((tl0 + lt0) >> 2) as u8,
                ((lt0 + t01) >> 2) as u8,
                ((t01 + t12) >> 2) as u8,
                ((t12 + t23) >> 2) as u8,
                ((tl0 + l01) >> 2) as u8,
                ((l01 + l12) >> 2) as u8,
                ((l12 + l23) >> 2) as u8,
            ];
            let mut e = [0u8; 16];
            for i in [0, 5, 10, 15] {
                e[i] = ddr[0];
            }
            for i in [1, 6, 11] {
                e[i] = ddr[1];
            }
            e[2] = ddr[2];
            e[7] = ddr[2];
            e[3] = ddr[3];
            for i in [4, 9, 14] {
                e[i] = ddr[4];
            }
            e[8] = ddr[5];
            e[13] = ddr[5];
            e[12] = ddr[6];
            assert_eq!(pred, e, "DDR stride={stride}");

            // VL
            i4x4_luma_pred_vl(&mut pred, &plane, roff, stride);
            let vl = [
                avg2(tk[0], tk[1]),
                avg2(tk[1], tk[2]),
                avg2(tk[2], tk[3]),
                avg2(tk[3], tk[4]),
                avg2(tk[4], tk[5]),
                avg3(tk[0], tk[1], tk[2]),
                avg3(tk[1], tk[2], tk[3]),
                avg3(tk[2], tk[3], tk[4]),
                avg3(tk[3], tk[4], tk[5]),
                avg3(tk[4], tk[5], tk[6]),
            ];
            let mut e = [0u8; 16];
            e[0] = vl[0];
            e[1] = vl[1];
            e[8] = vl[1];
            e[2] = vl[2];
            e[9] = vl[2];
            e[3] = vl[3];
            e[10] = vl[3];
            e[4] = vl[5];
            e[5] = vl[6];
            e[12] = vl[6];
            e[6] = vl[7];
            e[13] = vl[7];
            e[7] = vl[8];
            e[14] = vl[8];
            e[11] = vl[4];
            e[15] = vl[9];
            assert_eq!(pred, e, "VL stride={stride}");

            // VLTop
            i4x4_luma_pred_vl_top(&mut pred, &plane, roff, stride);
            let vlt = [
                avg2(tk[0], tk[1]),
                avg2(tk[1], tk[2]),
                avg2(tk[2], tk[3]),
                ((1 + (tk[3] << 1)) >> 1) as u8,
                avg3(tk[0], tk[1], tk[2]),
                avg3(tk[1], tk[2], tk[3]),
                avg3(tk[2], tk[3], tk[3]),
                ((2 + (tk[3] << 2)) >> 2) as u8,
            ];
            let mut e = [0u8; 16];
            e[0] = vlt[0];
            e[1] = vlt[1];
            e[8] = vlt[1];
            e[2] = vlt[2];
            e[9] = vlt[2];
            e[3] = vlt[3];
            e[10] = vlt[3];
            e[11] = vlt[3];
            e[4] = vlt[4];
            e[5] = vlt[5];
            e[12] = vlt[5];
            e[6] = vlt[6];
            e[13] = vlt[6];
            e[7] = vlt[7];
            e[14] = vlt[7];
            e[15] = vlt[7];
            assert_eq!(pred, e, "VLTop stride={stride}");

            // VR
            i4x4_luma_pred_vr(&mut pred, &plane, roff, stride);
            let vr = [
                ((1 + lt + tk[0]) >> 1) as u8,
                ((1 + tk[0] + tk[1]) >> 1) as u8,
                ((1 + tk[1] + tk[2]) >> 1) as u8,
                ((1 + tk[2] + tk[3]) >> 1) as u8,
                ((2 + lk[0] + (lt << 1) + tk[0]) >> 2) as u8,
                ((2 + lt + (tk[0] << 1) + tk[1]) >> 2) as u8,
                ((2 + tk[0] + (tk[1] << 1) + tk[2]) >> 2) as u8,
                ((2 + tk[1] + (tk[2] << 1) + tk[3]) >> 2) as u8,
                ((2 + lt + (lk[0] << 1) + lk[1]) >> 2) as u8,
                ((2 + lk[0] + (lk[1] << 1) + lk[2]) >> 2) as u8,
            ];
            let mut e = [0u8; 16];
            e[0] = vr[0];
            e[9] = vr[0];
            e[1] = vr[1];
            e[10] = vr[1];
            e[2] = vr[2];
            e[11] = vr[2];
            e[3] = vr[3];
            e[4] = vr[4];
            e[13] = vr[4];
            e[5] = vr[5];
            e[14] = vr[5];
            e[6] = vr[6];
            e[15] = vr[6];
            e[7] = vr[7];
            e[8] = vr[8];
            e[12] = vr[9];
            assert_eq!(pred, e, "VR stride={stride}");

            // HU
            i4x4_luma_pred_hu(&mut pred, &plane, roff, stride);
            let l01 = 1 + lk[0] + lk[1];
            let l12 = 1 + lk[1] + lk[2];
            let l23 = 1 + lk[2] + lk[3];
            let hu = [
                (l01 >> 1) as u8,
                ((l01 + l12) >> 2) as u8,
                (l12 >> 1) as u8,
                ((l12 + l23) >> 2) as u8,
                (l23 >> 1) as u8,
                ((1 + l23 + (lk[3] << 1)) >> 2) as u8,
            ];
            let mut e = [lk[3] as u8; 16];
            e[0] = hu[0];
            e[1] = hu[1];
            e[2] = hu[2];
            e[4] = hu[2];
            e[3] = hu[3];
            e[5] = hu[3];
            e[6] = hu[4];
            e[8] = hu[4];
            e[7] = hu[5];
            e[9] = hu[5];
            assert_eq!(pred, e, "HU stride={stride}");

            // HD
            i4x4_luma_pred_hd(&mut pred, &plane, roff, stride);
            let hd = [
                ((1 + lt + lk[0]) >> 1) as u8,
                ((2 + lk[0] + (lt << 1) + tk[0]) >> 2) as u8,
                ((2 + lt + (tk[0] << 1) + tk[1]) >> 2) as u8,
                ((2 + tk[0] + (tk[1] << 1) + tk[2]) >> 2) as u8,
                ((1 + lk[0] + lk[1]) >> 1) as u8,
                ((2 + lt + (lk[0] << 1) + lk[1]) >> 2) as u8,
                ((1 + lk[1] + lk[2]) >> 1) as u8,
                ((2 + lk[0] + (lk[1] << 1) + lk[2]) >> 2) as u8,
                ((1 + lk[2] + lk[3]) >> 1) as u8,
                ((2 + lk[1] + (lk[2] << 1) + lk[3]) >> 2) as u8,
            ];
            let mut e = [0u8; 16];
            e[0] = hd[0];
            e[6] = hd[0];
            e[1] = hd[1];
            e[7] = hd[1];
            e[2] = hd[2];
            e[3] = hd[3];
            e[4] = hd[4];
            e[10] = hd[4];
            e[5] = hd[5];
            e[11] = hd[5];
            e[8] = hd[6];
            e[14] = hd[6];
            e[9] = hd[7];
            e[15] = hd[7];
            e[12] = hd[8];
            e[13] = hd[9];
            assert_eq!(pred, e, "HD stride={stride}");
        }
    }
}

// ===========================================================================
// Chroma 8x8
// ===========================================================================

#[test]
fn chroma_modes_match_reference() {
    let mut r = Lcg(0x3333_4444);
    for &stride in &STRIDES {
        let s = stride as isize;
        for _ in 0..200 {
            let (plane, roff) = make_plane(&mut r, stride);
            let g = reader(&plane, roff);
            let mut pred = [0u8; 64];

            // V
            chroma_pred_v(&mut pred, &plane, roff, stride);
            let top: [u8; 8] = core::array::from_fn(|x| g(-s + x as isize) as u8);
            for row in 0..8 {
                assert_eq!(&pred[row * 8..row * 8 + 8], &top);
            }

            // H
            chroma_pred_h(&mut pred, &plane, roff, stride);
            for row in 0..8 {
                let l = g(row as isize * s - 1) as u8;
                assert!(pred[row * 8..row * 8 + 8].iter().all(|&p| p == l));
            }

            // DC
            chroma_pred_dc(&mut pred, &plane, roff, stride);
            let lk: [i32; 8] = core::array::from_fn(|k| g(k as isize * s - 1));
            let tk: [i32; 8] = core::array::from_fn(|k| g(-s + k as isize));
            let m1 = ((tk[0] + tk[1] + tk[2] + tk[3] + lk[0] + lk[1] + lk[2] + lk[3] + 4) >> 3) as u8;
            let sum2 = tk[4] + tk[5] + tk[6] + tk[7];
            let sum3 = lk[4] + lk[5] + lk[6] + lk[7];
            let m2 = ((sum2 + 2) >> 2) as u8;
            let m3 = ((sum3 + 2) >> 2) as u8;
            let m4 = ((sum2 + sum3 + 4) >> 3) as u8;
            let etop = [m1, m1, m1, m1, m2, m2, m2, m2];
            let ebot = [m3, m3, m3, m3, m4, m4, m4, m4];
            for row in 0..4 {
                assert_eq!(&pred[row * 8..row * 8 + 8], &etop);
            }
            for row in 4..8 {
                assert_eq!(&pred[row * 8..row * 8 + 8], &ebot);
            }

            // DcLeft
            chroma_pred_dc_left(&mut pred, &plane, roff, stride);
            let top = ((lk[0] + lk[1] + lk[2] + lk[3] + 2) >> 2) as u8;
            let bot = ((lk[4] + lk[5] + lk[6] + lk[7] + 2) >> 2) as u8;
            for row in 0..4 {
                assert!(pred[row * 8..row * 8 + 8].iter().all(|&p| p == top));
            }
            for row in 4..8 {
                assert!(pred[row * 8..row * 8 + 8].iter().all(|&p| p == bot));
            }

            // DcTop
            chroma_pred_dc_top(&mut pred, &plane, roff, stride);
            let m1 = ((tk[0] + tk[1] + tk[2] + tk[3] + 2) >> 2) as u8;
            let m2 = ((tk[4] + tk[5] + tk[6] + tk[7] + 2) >> 2) as u8;
            let erow = [m1, m1, m1, m1, m2, m2, m2, m2];
            for row in 0..8 {
                assert_eq!(&pred[row * 8..row * 8 + 8], &erow);
            }

            // DcNA
            chroma_pred_dc_na(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == 0x80));

            // Plane (per-pixel anchor)
            chroma_pred_plane(&mut pred, &plane, roff, stride);
            let mut top_sum = 0i32;
            let mut left_sum = 0i32;
            for i in 0..4i32 {
                top_sum += (i + 1) * (g(-s + 4 + i as isize) - g(-s + 2 - i as isize));
                left_sum += (i + 1) * (g((4 + i) as isize * s - 1) - g((2 - i) as isize * s - 1));
            }
            let a = (g(7 * s - 1) + g(-s + 7)) << 4;
            let b = (17 * top_sum + 16) >> 5;
            let c = (17 * left_sum + 16) >> 5;
            for i in 0..8 {
                for j in 0..8 {
                    let want = clip1((a + b * (j as i32 - 3) + c * (i as i32 - 3) + 16) >> 5);
                    assert_eq!(pred[i * 8 + j], want, "chroma plane stride={stride}");
                }
            }
        }
    }
}

// ===========================================================================
// Luma 16x16
// ===========================================================================

#[test]
fn i16x16_modes_match_reference() {
    let mut r = Lcg(0x5555_6666);
    for &stride in &STRIDES {
        let s = stride as isize;
        for _ in 0..150 {
            let (plane, roff) = make_plane(&mut r, stride);
            let g = reader(&plane, roff);
            let mut pred = [0u8; 256];

            // V
            i16x16_luma_pred_v(&mut pred, &plane, roff, stride);
            let top: [u8; 16] = core::array::from_fn(|x| g(-s + x as isize) as u8);
            for row in 0..16 {
                assert_eq!(&pred[row * 16..row * 16 + 16], &top);
            }

            // H
            i16x16_luma_pred_h(&mut pred, &plane, roff, stride);
            for row in 0..16 {
                let l = g(row as isize * s - 1) as u8;
                assert!(pred[row * 16..row * 16 + 16].iter().all(|&p| p == l));
            }

            // DC / DcTop / DcLeft / DcNA
            let mut top_sum = 0i32;
            let mut left_sum = 0i32;
            for i in 0..16isize {
                top_sum += g(-s + i);
                left_sum += g(i * s - 1);
            }
            i16x16_luma_pred_dc(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == ((16 + top_sum + left_sum) >> 5) as u8));
            i16x16_luma_pred_dc_top(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == ((8 + top_sum) >> 4) as u8));
            i16x16_luma_pred_dc_left(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == ((8 + left_sum) >> 4) as u8));
            i16x16_luma_pred_dc_na(&mut pred, &plane, roff, stride);
            assert!(pred.iter().all(|&p| p == 0x80));

            // Plane (per-pixel anchor)
            i16x16_luma_pred_plane(&mut pred, &plane, roff, stride);
            let mut h = 0i32;
            let mut v = 0i32;
            for i in 0..8i32 {
                h += (i + 1) * (g(-s + 8 + i as isize) - g(-s + 6 - i as isize));
                v += (i + 1) * (g((8 + i) as isize * s - 1) - g((6 - i) as isize * s - 1));
            }
            let a = (g(15 * s - 1) + g(-s + 15)) << 4;
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            for i in 0..16 {
                for j in 0..16 {
                    let want = clip1((a + b * (j as i32 - 7) + c * (i as i32 - 7) + 16) >> 5);
                    assert_eq!(pred[i * 16 + j], want, "i16 plane stride={stride}");
                }
            }
        }
    }
}

// ===========================================================================
// Combined3 cost helpers: argmin / lambda wiring.
// ===========================================================================

use crate::dsp::sad::{sad16x16, sad8x8};
use crate::dsp::satd::{satd16x16, satd4x4, satd8x8};

#[test]
fn combined3_4x4_matches_recomputed() {
    let mut r = Lcg(0x7777_8888);
    let stride = 32usize;
    for _ in 0..300 {
        let (dec, dec_off) = make_plane(&mut r, stride);
        let enc: Vec<u8> = (0..stride * 16).map(|_| r.byte()).collect();
        let enc_off = 0usize;
        let (l2, l1, l0) = ((r.next() % 200) as i32, (r.next() % 200) as i32, (r.next() % 200) as i32);
        let mut dst = [0u8; 16];
        let (cost, mode) = satd_intra_4x4_combined3(&dec, dec_off, stride, &enc, enc_off, stride, &mut dst, l2, l1, l0);

        // Recompute independently.
        let e = &enc[enc_off..];
        let mut b2 = [0u8; 16];
        let mut b1 = [0u8; 16];
        let mut b0 = [0u8; 16];
        i4x4_luma_pred_dc(&mut b2, &dec, dec_off, stride);
        i4x4_luma_pred_h(&mut b1, &dec, dec_off, stride);
        i4x4_luma_pred_v(&mut b0, &dec, dec_off, stride);
        let c2 = satd4x4(&b2, 4, e, stride) + l2;
        let c1 = satd4x4(&b1, 4, e, stride) + l1;
        let c0 = satd4x4(&b0, 4, e, stride) + l0;
        // Tie-break matches C: DC, then H (strict <), then V (strict <).
        let mut best = (c2, 2usize, b2);
        if c1 < best.0 {
            best = (c1, 1, b1);
        }
        if c0 < best.0 {
            best = (c0, 0, b0);
        }
        assert_eq!(cost, best.0);
        assert_eq!(mode, best.1 as i32);
        assert_eq!(dst, best.2);
    }
}

#[test]
fn combined3_16x16_matches_recomputed() {
    let mut r = Lcg(0x99AA_BBCC);
    let stride = 32usize;
    for _ in 0..200 {
        let (dec, dec_off) = make_plane(&mut r, stride);
        let enc: Vec<u8> = (0..stride * 16).map(|_| r.byte()).collect();
        let lambda = (r.next() % 100) as i32;
        let mut dst = [0u8; 256];

        for use_sad in [false, true] {
            let (cost, mode) = if use_sad {
                sad_intra_16x16_combined3(&dec, dec_off, stride, &enc, 0, stride, &mut dst, lambda)
            } else {
                satd_intra_16x16_combined3(&dec, dec_off, stride, &enc, 0, stride, &mut dst, lambda)
            };
            let metric = |a: &[u8], b: &[u8]| -> i32 {
                if use_sad {
                    sad16x16(a, 16, b, stride) as i32
                } else {
                    satd16x16(a, 16, b, stride)
                }
            };
            let e = &enc[..];
            let mut p = [0u8; 256];
            i16x16_luma_pred_v(&mut p, &dec, dec_off, stride);
            let cv = metric(&p, e);
            i16x16_luma_pred_h(&mut p, &dec, dec_off, stride);
            let ch = metric(&p, e) + lambda * 2;
            i16x16_luma_pred_dc(&mut p, &dec, dec_off, stride);
            let cd = metric(&p, e) + lambda * 2;
            let mut best = (cv, 0);
            if ch < best.0 {
                best = (ch, 1);
            }
            if cd < best.0 {
                best = (cd, 2);
            }
            assert_eq!((cost, mode), best, "use_sad={use_sad}");
        }
    }
}

#[test]
fn combined3_8x8_chroma_matches_recomputed() {
    let mut r = Lcg(0xCCDD_EEFF);
    let stride = 32usize;
    for _ in 0..200 {
        let (dec_cb, off_cb) = make_plane(&mut r, stride);
        let (dec_cr, off_cr) = make_plane(&mut r, stride);
        let enc_cb: Vec<u8> = (0..stride * 16).map(|_| r.byte()).collect();
        let enc_cr: Vec<u8> = (0..stride * 16).map(|_| r.byte()).collect();
        let lambda = (r.next() % 100) as i32;
        let mut dst = [0u8; 128];

        for use_sad in [false, true] {
            let (cost, mode) = if use_sad {
                sad_intra_8x8_combined3(&dec_cb, off_cb, stride, &enc_cb, 0, stride, &mut dst, lambda, &dec_cr, off_cr, &enc_cr, 0)
            } else {
                satd_intra_8x8_combined3(&dec_cb, off_cb, stride, &enc_cb, 0, stride, &mut dst, lambda, &dec_cr, off_cr, &enc_cr, 0)
            };
            let m = |a: &[u8], b: &[u8]| -> i32 {
                if use_sad {
                    sad8x8(a, 8, b, stride) as i32
                } else {
                    satd8x8(a, 8, b, stride)
                }
            };
            let cost_both = |cb: &[u8], cr: &[u8], add: i32| m(cb, &enc_cb) + m(cr, &enc_cr) + add;
            let mut cb = [0u8; 64];
            let mut cr = [0u8; 64];
            chroma_pred_v(&mut cb, &dec_cb, off_cb, stride);
            chroma_pred_v(&mut cr, &dec_cr, off_cr, stride);
            let cv = cost_both(&cb, &cr, lambda * 2);
            chroma_pred_h(&mut cb, &dec_cb, off_cb, stride);
            chroma_pred_h(&mut cr, &dec_cr, off_cr, stride);
            let ch = cost_both(&cb, &cr, lambda * 2);
            chroma_pred_dc(&mut cb, &dec_cb, off_cb, stride);
            chroma_pred_dc(&mut cr, &dec_cr, off_cr, stride);
            let cd = cost_both(&cb, &cr, 0);
            let mut best = (cv, 2);
            if ch < best.0 {
                best = (ch, 1);
            }
            if cd < best.0 {
                best = (cd, 0);
            }
            assert_eq!((cost, mode), best, "use_sad={use_sad}");
        }
    }
}
