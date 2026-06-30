//! Sum-of-absolute-transformed-differences (SATD) kernels, ported from
//! `reference/codec/encoder/core/src/sample.cpp`
//! (`WelsSampleSatd{4x4,8x4,4x8,8x8,16x8,8x16,16x16}_c`).
//!
//! SATD applies a 4x4 Hadamard transform to the residual `s1 - s2` before
//! summing magnitudes, giving a frequency-weighted distortion the encoder uses
//! for intra/inter mode decision. Larger blocks accumulate the 4x4 result over
//! their sub-blocks exactly as the C composites do. All arithmetic is `i32`,
//! matching the C kernels (no overflow for 8-bit input).

/// 4x4 SATD of the block at (`o1`, `o2`) in `s1`/`s2`. Port of `WelsSampleSatd4x4_c`.
#[inline]
fn satd4x4_at(s1: &[u8], o1: usize, st1: usize, s2: &[u8], o2: usize, st2: usize) -> i32 {
    let mut m = [[0i32; 4]; 4];
    let mut p1 = o1;
    let mut p2 = o2;
    for row in m.iter_mut() {
        for x in 0..4 {
            row[x] = s1[p1 + x] as i32 - s2[p2 + x] as i32;
        }
        p1 += st1;
        p2 += st2;
    }
    // Horizontal transform.
    for row in m.iter_mut() {
        let a0 = row[0] + row[2];
        let a1 = row[1] + row[3];
        let a2 = row[0] - row[2];
        let a3 = row[1] - row[3];
        row[0] = a0 + a1;
        row[1] = a2 + a3;
        row[2] = a2 - a3;
        row[3] = a0 - a1;
    }
    // Vertical transform and accumulate transformed magnitudes.
    let mut sum = 0i32;
    for i in 0..4 {
        let a0 = m[0][i] + m[2][i];
        let a1 = m[1][i] + m[3][i];
        let a2 = m[0][i] - m[2][i];
        let a3 = m[1][i] - m[3][i];
        sum += (a0 + a1).abs() + (a2 + a3).abs() + (a2 - a3).abs() + (a0 - a1).abs();
    }
    (sum + 1) >> 1
}

/// Port of `WelsSampleSatd4x4_c`.
pub fn satd4x4(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> i32 {
    satd4x4_at(s1, 0, st1, s2, 0, st2)
}

/// Port of `WelsSampleSatd8x4_c`.
pub fn satd8x4(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> i32 {
    satd4x4_at(s1, 0, st1, s2, 0, st2) + satd4x4_at(s1, 4, st1, s2, 4, st2)
}

/// Port of `WelsSampleSatd4x8_c`.
pub fn satd4x8(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> i32 {
    satd4x4_at(s1, 0, st1, s2, 0, st2) + satd4x4_at(s1, st1 << 2, st1, s2, st2 << 2, st2)
}

/// Port of `WelsSampleSatd8x8_c`.
pub fn satd8x8(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> i32 {
    let (q1, q2) = (st1 << 2, st2 << 2);
    satd4x4_at(s1, 0, st1, s2, 0, st2)
        + satd4x4_at(s1, 4, st1, s2, 4, st2)
        + satd4x4_at(s1, q1, st1, s2, q2, st2)
        + satd4x4_at(s1, q1 + 4, st1, s2, q2 + 4, st2)
}

/// SATD over an 8x8 area rooted at (`o1`, `o2`); building block for the wide composites.
#[inline]
fn satd8x8_at(s1: &[u8], o1: usize, st1: usize, s2: &[u8], o2: usize, st2: usize) -> i32 {
    let (q1, q2) = (st1 << 2, st2 << 2);
    satd4x4_at(s1, o1, st1, s2, o2, st2)
        + satd4x4_at(s1, o1 + 4, st1, s2, o2 + 4, st2)
        + satd4x4_at(s1, o1 + q1, st1, s2, o2 + q2, st2)
        + satd4x4_at(s1, o1 + q1 + 4, st1, s2, o2 + q2 + 4, st2)
}

/// Port of `WelsSampleSatd16x8_c`.
pub fn satd16x8(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> i32 {
    satd8x8_at(s1, 0, st1, s2, 0, st2) + satd8x8_at(s1, 8, st1, s2, 8, st2)
}

/// Port of `WelsSampleSatd8x16_c`.
pub fn satd8x16(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> i32 {
    satd8x8_at(s1, 0, st1, s2, 0, st2) + satd8x8_at(s1, st1 << 3, st1, s2, st2 << 3, st2)
}

/// Port of `WelsSampleSatd16x16_c`.
pub fn satd16x16(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> i32 {
    let (h1, h2) = (st1 << 3, st2 << 3);
    satd8x8_at(s1, 0, st1, s2, 0, st2)
        + satd8x8_at(s1, 8, st1, s2, 8, st2)
        + satd8x8_at(s1, h1, st1, s2, h2, st2)
        + satd8x8_at(s1, h1 + 8, st1, s2, h2 + 8, st2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    struct Lcg(u32);
    impl Lcg {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 24) as u8
        }
    }

    // Independent 4x4 SATD reference, mirroring the explicit Hadamard in the
    // EncUT_Sample WelsSampleSatd4x4_c test (W = diffs, T/Y full transform).
    fn satd4x4_ref(s1: &[u8], o1: usize, st1: usize, s2: &[u8], o2: usize, st2: usize) -> i32 {
        let mut w = [0i32; 16];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                w[k] = s1[o1 + i * st1 + j] as i32 - s2[o2 + i * st2 + j] as i32;
                k += 1;
            }
        }
        let mut t = [0i32; 16];
        for c in 0..4 {
            t[c] = w[c] + w[4 + c] + w[8 + c] + w[12 + c];
            t[4 + c] = w[c] + w[4 + c] - w[8 + c] - w[12 + c];
            t[8 + c] = w[c] - w[4 + c] - w[8 + c] + w[12 + c];
            t[12 + c] = w[c] - w[4 + c] + w[8 + c] - w[12 + c];
        }
        let mut y = [0i32; 16];
        for r in 0..4 {
            let b = r * 4;
            y[b] = t[b] + t[b + 1] + t[b + 2] + t[b + 3];
            y[b + 1] = t[b] + t[b + 1] - t[b + 2] - t[b + 3];
            y[b + 2] = t[b] - t[b + 1] - t[b + 2] + t[b + 3];
            y[b + 3] = t[b] - t[b + 1] + t[b + 2] - t[b + 3];
        }
        let sum: i32 = y.iter().map(|v| v.abs()).sum();
        (sum + 1) >> 1
    }

    fn satd_ref(s1: &[u8], st1: usize, s2: &[u8], st2: usize, w: usize, h: usize) -> i32 {
        let mut sum = 0;
        let mut y = 0;
        while y < h {
            let mut x = 0;
            while x < w {
                sum += satd4x4_ref(s1, y * st1 + x, st1, s2, y * st2 + x, st2);
                x += 4;
            }
            y += 4;
        }
        sum
    }

    #[test]
    fn satd_matches_reference_random() {
        let mut r = Lcg(0xCAFE_1234);
        let (st1, st2) = (40usize, 48usize);
        for _ in 0..200 {
            let a: Vec<u8> = (0..st1 * 16).map(|_| r.byte()).collect();
            let b: Vec<u8> = (0..st2 * 16).map(|_| r.byte()).collect();
            assert_eq!(satd4x4(&a, st1, &b, st2), satd_ref(&a, st1, &b, st2, 4, 4));
            assert_eq!(satd8x4(&a, st1, &b, st2), satd_ref(&a, st1, &b, st2, 8, 4));
            assert_eq!(satd4x8(&a, st1, &b, st2), satd_ref(&a, st1, &b, st2, 4, 8));
            assert_eq!(satd8x8(&a, st1, &b, st2), satd_ref(&a, st1, &b, st2, 8, 8));
            assert_eq!(satd16x8(&a, st1, &b, st2), satd_ref(&a, st1, &b, st2, 16, 8));
            assert_eq!(satd8x16(&a, st1, &b, st2), satd_ref(&a, st1, &b, st2, 8, 16));
            assert_eq!(satd16x16(&a, st1, &b, st2), satd_ref(&a, st1, &b, st2, 16, 16));
        }
    }

    #[test]
    fn satd_identical_blocks_zero() {
        let buf = [77u8; 16 * 16];
        assert_eq!(satd16x16(&buf, 16, &buf, 16), 0);
        assert_eq!(satd4x4(&buf, 16, &buf, 16), 0);
    }
}
