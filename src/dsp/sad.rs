//! Sum-of-absolute-differences kernels, ported from
//! `reference/codec/common/src/sad_common.cpp`
//! (`WelsSampleSad{4x4,8x4,4x8,8x8,16x8,8x16,16x16}_c`).
//!
//! Each computes `sum |a - b|` over a `w`x`h` block with independent strides.
//! The four-position `WelsSampleSadFour*_c` variants (used by motion estimation
//! to score the four neighbours of a candidate) are ported in [`sad_four`].

/// Generic `w`x`h` SAD with independent strides.
///
/// Dispatches to a bit-exact SIMD kernel when `--features simd` is enabled on a
/// supported target (NEON / wasm `simd128`), otherwise the scalar reference.
#[inline]
pub fn sad(s1: &[u8], st1: usize, s2: &[u8], st2: usize, w: usize, h: usize) -> u32 {
    #[cfg(simd_neon)]
    return crate::dsp::simd::neon::sad(s1, st1, s2, st2, w, h);
    #[cfg(simd_wasm128)]
    return crate::dsp::simd::wasm::sad(s1, st1, s2, st2, w, h);
    #[cfg(no_simd)]
    return sad_scalar(s1, st1, s2, st2, w, h);
}

/// Scalar reference SAD (the conformance baseline / SIMD fallback).
#[inline]
pub fn sad_scalar(s1: &[u8], st1: usize, s2: &[u8], st2: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let r1 = y * st1;
        let r2 = y * st2;
        for x in 0..w {
            sum += (s1[r1 + x] as i32 - s2[r2 + x] as i32).unsigned_abs();
        }
    }
    sum
}

macro_rules! named_sad {
    ($name:ident, $w:expr, $h:expr, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name(s1: &[u8], st1: usize, s2: &[u8], st2: usize) -> u32 {
            sad(s1, st1, s2, st2, $w, $h)
        }
    };
}

named_sad!(sad4x4, 4, 4, "Port of `WelsSampleSad4x4_c`.");
named_sad!(sad8x4, 8, 4, "Port of `WelsSampleSad8x4_c`.");
named_sad!(sad4x8, 4, 8, "Port of `WelsSampleSad4x8_c`.");
named_sad!(sad8x8, 8, 8, "Port of `WelsSampleSad8x8_c`.");
named_sad!(sad16x8, 16, 8, "Port of `WelsSampleSad16x8_c`.");
named_sad!(sad8x16, 8, 16, "Port of `WelsSampleSad8x16_c`.");
named_sad!(sad16x16, 16, 16, "Port of `WelsSampleSad16x16_c`.");

/// Four-position SAD: scores `s1` (at `o1`) against the four neighbours of the
/// candidate centred at `o2` in `s2` — up, down, left, right — returning
/// `[above, below, left, right]`. Port of the `WelsSampleSadFour*_c` family
/// (`reference/codec/common/src/sad_common.cpp`).
///
/// The caller must ensure `o2 >= st2 + 1` so the up/left neighbours are in
/// bounds (the motion-vector clamp guarantees this in the encoder).
#[inline]
pub fn sad_four(s1: &[u8], o1: usize, st1: usize, s2: &[u8], o2: usize, st2: usize, w: usize, h: usize) -> [u32; 4] {
    [
        sad(&s1[o1..], st1, &s2[o2 - st2..], st2, w, h),
        sad(&s1[o1..], st1, &s2[o2 + st2..], st2, w, h),
        sad(&s1[o1..], st1, &s2[o2 - 1..], st2, w, h),
        sad(&s1[o1..], st1, &s2[o2 + 1..], st2, w, h),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // Straightforward independent reference.
    fn sad_ref(s1: &[u8], st1: usize, s2: &[u8], st2: usize, w: usize, h: usize) -> u32 {
        let mut sum = 0u32;
        for y in 0..h {
            for x in 0..w {
                let a = s1[y * st1 + x] as i32;
                let b = s2[y * st2 + x] as i32;
                sum += (a - b).unsigned_abs();
            }
        }
        sum
    }

    #[test]
    fn identical_blocks_zero() {
        let buf = vec![123u8; 16 * 16];
        assert_eq!(sad16x16(&buf, 16, &buf, 16), 0);
    }

    #[test]
    fn known_difference() {
        let a = vec![10u8; 16 * 16];
        let b = vec![13u8; 16 * 16];
        // 16x16 samples each differ by 3.
        assert_eq!(sad16x16(&a, 16, &b, 16), 3 * 16 * 16);
        assert_eq!(sad8x8(&a, 16, &b, 16), 3 * 8 * 8);
    }

    #[test]
    fn matches_reference_random() {
        let mut state = 0xBADC0FFEu32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        };
        let stride = 24;
        let a: alloc::vec::Vec<u8> = (0..stride * 20).map(|_| next()).collect();
        let b: alloc::vec::Vec<u8> = (0..stride * 20).map(|_| next()).collect();
        for (w, h) in [(4, 4), (8, 4), (4, 8), (8, 8), (16, 8), (8, 16), (16, 16)] {
            assert_eq!(
                sad(&a, stride, &b, stride, w, h),
                sad_ref(&a, stride, &b, stride, w, h),
                "{w}x{h}"
            );
        }
    }

    #[test]
    fn sad_four_matches_reference() {
        let mut state = 0x1234_ABCDu32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        };
        let (st1, st2) = (32usize, 36usize);
        let a: alloc::vec::Vec<u8> = (0..st1 * 20).map(|_| next()).collect();
        let b: alloc::vec::Vec<u8> = (0..st2 * 20).map(|_| next()).collect();
        // Centre the candidate one row/col in so the up/left neighbours are valid.
        let (o1, o2) = (0usize, st2 + 1);
        for (w, h) in [(4, 4), (8, 4), (4, 8), (8, 8), (16, 8), (8, 16), (16, 16)] {
            let got = sad_four(&a, o1, st1, &b, o2, st2, w, h);
            let want = [
                sad_ref(&a[o1..], st1, &b[o2 - st2..], st2, w, h),
                sad_ref(&a[o1..], st1, &b[o2 + st2..], st2, w, h),
                sad_ref(&a[o1..], st1, &b[o2 - 1..], st2, w, h),
                sad_ref(&a[o1..], st1, &b[o2 + 1..], st2, w, h),
            ];
            assert_eq!(got, want, "{w}x{h}");
        }
    }
}
