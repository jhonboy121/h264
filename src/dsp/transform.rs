//! Inverse transforms (IDCT) and residual add, ported from
//! `reference/codec/decoder/core/src/decode_mb_aux.cpp`
//! (`IdctResAddPred_c`, `IdctResAddPred8x8_c`).
//!
//! These take already-dequantized residuals (`int16`) and add the inverse
//! transform onto an 8-bit prediction in place. The luma-DC / chroma-DC
//! dequant-IDCT helpers (which are coupled to coefficient storage) live with
//! the residual decode in P3.

/// `WelsClip1`: clamp an `i32` sample to the 8-bit range.
#[inline(always)]
pub fn clip1(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

/// 4x4 inverse integer transform + add to prediction, in place.
///
/// `pred` is the destination/prediction plane; `stride` its row stride; `rs` the
/// 16 dequantized residual coefficients in raster order. Port of
/// `IdctResAddPred_c`.
pub fn idct4x4_add(pred: &mut [u8], stride: usize, rs: &[i16; 16]) {
    // C stores the horizontal pass into `int16_t iSrc[16]`, so it truncates to
    // 16 bits before the vertical pass — replicate with wrapping i16.
    let mut src = [0i16; 16];
    // Horizontal (rows).
    for i in 0..4 {
        let y = i << 2;
        let t0 = rs[y].wrapping_add(rs[y + 2]);
        let t1 = rs[y].wrapping_sub(rs[y + 2]);
        let t2 = (rs[y + 1] >> 1).wrapping_sub(rs[y + 3]);
        let t3 = rs[y + 1].wrapping_add(rs[y + 3] >> 1);
        src[y] = t0.wrapping_add(t3);
        src[y + 1] = t1.wrapping_add(t2);
        src[y + 2] = t1.wrapping_sub(t2);
        src[y + 3] = t0.wrapping_sub(t3);
    }
    let s1 = stride;
    let s2 = stride << 1;
    let s3 = stride + s2;
    // Vertical (columns) + add + clip. i32 here matches C (operands promoted).
    for i in 0..4 {
        let a = src[i] as i32;
        let a8 = src[i + 8] as i32;
        let a4 = src[i + 4] as i32;
        let a12 = src[i + 12] as i32;
        let mut t1 = a + a8;
        let mut t2 = a4 + (a12 >> 1);
        let t3 = (32 + t1 + t2) >> 6;
        let t4 = (32 + t1 - t2) >> 6;
        pred[i] = clip1(t3 + pred[i] as i32);
        pred[i + s3] = clip1(t4 + pred[i + s3] as i32);
        t1 = a - a8;
        t2 = (a4 >> 1) - a12;
        pred[i + s1] = clip1(((32 + t1 + t2) >> 6) + pred[i + s1] as i32);
        pred[i + s2] = clip1(((32 + t1 - t2) >> 6) + pred[i + s2] as i32);
    }
}

/// 8x8 inverse integer transform + add to prediction, in place. Port of
/// `IdctResAddPred8x8_c`. `rs` holds 64 dequantized residuals in raster order.
pub fn idct8x8_add(pred: &mut [u8], stride: usize, rs: &[i16; 64]) {
    let mut tmp = [0i16; 64];
    let mut res = [0i16; 64];
    let mut p = [0i16; 8];
    let mut b = [0i16; 8];
    let mut a = [0i16; 4];

    // Horizontal.
    for i in 0..8 {
        for j in 0..8 {
            p[j] = rs[j + (i << 3)];
        }
        idct8_1d(&p, &mut a, &mut b);
        let o = i << 3;
        tmp[o] = b[0].wrapping_add(b[7]);
        tmp[o + 1] = b[2].wrapping_sub(b[5]);
        tmp[o + 2] = b[4].wrapping_add(b[3]);
        tmp[o + 3] = b[6].wrapping_add(b[1]);
        tmp[o + 4] = b[6].wrapping_sub(b[1]);
        tmp[o + 5] = b[4].wrapping_sub(b[3]);
        tmp[o + 6] = b[2].wrapping_add(b[5]);
        tmp[o + 7] = b[0].wrapping_sub(b[7]);
    }
    // Vertical.
    for i in 0..8 {
        for j in 0..8 {
            p[j] = tmp[i + (j << 3)];
        }
        idct8_1d(&p, &mut a, &mut b);
        res[i] = b[0].wrapping_add(b[7]);
        res[8 + i] = b[2].wrapping_sub(b[5]);
        res[16 + i] = b[4].wrapping_add(b[3]);
        res[24 + i] = b[6].wrapping_add(b[1]);
        res[32 + i] = b[6].wrapping_sub(b[1]);
        res[40 + i] = b[4].wrapping_sub(b[3]);
        res[48 + i] = b[2].wrapping_add(b[5]);
        res[56 + i] = b[0].wrapping_sub(b[7]);
    }
    // Add + clip.
    for i in 0..8 {
        for j in 0..8 {
            let idx = i * stride + j;
            pred[idx] = clip1(((32 + res[(i << 3) + j] as i32) >> 6) + pred[idx] as i32);
        }
    }
}

/// One stage of the 8-point inverse transform butterfly (shared by both passes).
/// Matches the `a[]`/`b[]` computation in `IdctResAddPred8x8_c`. Uses wrapping
/// arithmetic to mirror C's `int16_t` overflow semantics exactly.
#[inline(always)]
fn idct8_1d(p: &[i16; 8], a: &mut [i16; 4], b: &mut [i16; 8]) {
    a[0] = p[0].wrapping_add(p[4]);
    a[1] = p[0].wrapping_sub(p[4]);
    a[2] = p[6].wrapping_sub(p[2] >> 1);
    a[3] = p[2].wrapping_add(p[6] >> 1);
    b[0] = a[0].wrapping_add(a[3]);
    b[2] = a[1].wrapping_sub(a[2]);
    b[4] = a[1].wrapping_add(a[2]);
    b[6] = a[0].wrapping_sub(a[3]);
    a[0] = (-p[3]).wrapping_add(p[5]).wrapping_sub(p[7]).wrapping_sub(p[7] >> 1);
    a[1] = p[1].wrapping_add(p[7]).wrapping_sub(p[3]).wrapping_sub(p[3] >> 1);
    a[2] = (-p[1]).wrapping_add(p[7]).wrapping_add(p[5]).wrapping_add(p[5] >> 1);
    a[3] = p[3].wrapping_add(p[5]).wrapping_add(p[1]).wrapping_add(p[1] >> 1);
    b[1] = a[0].wrapping_add(a[3] >> 2);
    b[3] = a[1].wrapping_add(a[2] >> 2);
    b[5] = a[2].wrapping_sub(a[1] >> 2);
    b[7] = a[3].wrapping_sub(a[0] >> 2);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent, straightforward reference for the 4x4 inverse transform,
    // used to cross-check the optimized in-place kernel.
    fn idct4x4_ref(pred: &[u8], stride: usize, rs: &[i16; 16]) -> [u8; 16] {
        let mut s16 = [0i16; 16];
        for i in 0..4 {
            let y = i * 4;
            let t0 = rs[y].wrapping_add(rs[y + 2]);
            let t1 = rs[y].wrapping_sub(rs[y + 2]);
            let t2 = (rs[y + 1] >> 1).wrapping_sub(rs[y + 3]);
            let t3 = rs[y + 1].wrapping_add(rs[y + 3] >> 1);
            s16[y] = t0.wrapping_add(t3);
            s16[y + 1] = t1.wrapping_add(t2);
            s16[y + 2] = t1.wrapping_sub(t2);
            s16[y + 3] = t0.wrapping_sub(t3);
        }
        let s: [i32; 16] = core::array::from_fn(|i| s16[i] as i32);
        let mut out = [0u8; 16];
        for col in 0..4 {
            let z0 = s[col] + s[col + 8];
            let z3 = s[col] - s[col + 8];
            let z1 = (s[col + 4] >> 1) - s[col + 12];
            let z2 = s[col + 4] + (s[col + 12] >> 1);
            let rows = [
                (32 + z0 + z2) >> 6,
                (32 + z3 + z1) >> 6,
                (32 + z3 - z1) >> 6,
                (32 + z0 - z2) >> 6,
            ];
            for (r, &val) in rows.iter().enumerate() {
                out[r * 4 + col] = clip1(val + pred[r * stride + col] as i32);
            }
        }
        out
    }

    #[test]
    fn zero_residual_is_identity() {
        let mut pred = [100u8; 16];
        let orig = pred;
        idct4x4_add(&mut pred, 4, &[0i16; 16]);
        assert_eq!(pred, orig);
    }

    #[test]
    fn dc_only_residual_4x4() {
        // Only DC coefficient -> uniform offset (32 + dc) >> 6 on every sample.
        for dc in [-200i16, -64, -32, 0, 32, 64, 200, 1000] {
            let mut pred = [80u8; 16];
            let mut rs = [0i16; 16];
            rs[0] = dc;
            idct4x4_add(&mut pred, 4, &rs);
            let expect = clip1(((32 + dc as i32) >> 6) + 80);
            assert!(pred.iter().all(|&p| p == expect), "dc={dc}");
        }
    }

    #[test]
    fn matches_reference_random_4x4() {
        // Deterministic LCG so the test is reproducible without rand.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as i16
        };
        for _ in 0..2000 {
            let mut rs = [0i16; 16];
            for r in rs.iter_mut() {
                *r = (next() % 512) - 256;
            }
            let pred0: [u8; 16] = core::array::from_fn(|_| next() as u8);
            let mut pred = pred0;
            idct4x4_add(&mut pred, 4, &rs);
            let reference = idct4x4_ref(&pred0, 4, &rs);
            assert_eq!(pred, reference, "rs={rs:?}");
        }
    }

    #[test]
    fn zero_residual_is_identity_8x8() {
        let mut pred = [123u8; 64];
        let orig = pred;
        idct8x8_add(&mut pred, 8, &[0i16; 64]);
        assert_eq!(pred, orig);
    }

    #[test]
    fn dc_only_residual_8x8() {
        // For the 8x8 transform a lone DC propagates as b[0]=p[0] through both
        // passes, giving uniform (32 + dc) >> 6.
        for dc in [-256i16, -64, 0, 64, 256] {
            let mut pred = [90u8; 64];
            let mut rs = [0i16; 64];
            rs[0] = dc;
            idct8x8_add(&mut pred, 8, &rs);
            let expect = clip1(((32 + dc as i32) >> 6) + 90);
            assert!(pred.iter().all(|&p| p == expect), "dc={dc}");
        }
    }
}
