//! Sum-of-absolute-differences kernels, ported from
//! `reference/codec/common/src/sad_common.cpp`
//! (`WelsSampleSad{4x4,8x4,4x8,8x8,16x8,8x16,16x16}_c`).
//!
//! Each computes `sum |a - b|` over a `w`x`h` block with independent strides.
//! The `...Four_c` variants have no scalar C reference (asm-only) and are
//! omitted here; the motion-estimation layer (P5/P6) builds four-position SAD
//! from these scalar kernels.

/// Generic `w`x`h` SAD with independent strides.
#[inline]
pub fn sad(s1: &[u8], st1: usize, s2: &[u8], st2: usize, w: usize, h: usize) -> u32 {
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
}
