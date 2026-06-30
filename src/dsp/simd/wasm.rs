//! WebAssembly `simd128` implementations of `idct4x4_add` and `sad`.
//!
//! Enabled when building `wasm32` with `target_feature = "simd128"` (e.g.
//! `RUSTFLAGS="-C target-feature=+simd128"`); without it the parent modules use
//! the scalar reference. Each kernel is a bit-exact port of the scalar reference
//! using well-defined v128 lane semantics (the same algorithm validated for the
//! NEON path). The luma MC 6-tap and deblock filters are *not* ported to wasm and
//! stay scalar there.
//!
//! As with the NEON port, every load/store stays inside the index footprint the
//! scalar reference already touches.

use core::arch::wasm32::*;

// ---------------------------------------------------------------------------
// SAD
// ---------------------------------------------------------------------------

/// Per-byte `|a - b|` via saturating subtract in both directions.
#[inline]
fn absdiff(a: v128, b: v128) -> v128 {
    v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a))
}

/// Horizontal sum of the eight `u16` lanes of `acc` (widening to avoid overflow).
#[inline]
fn hsum_u16x8(acc: v128) -> u32 {
    let w = i32x4_extadd_pairwise_u16x8(acc);
    (i32x4_extract_lane::<0>(w)
        + i32x4_extract_lane::<1>(w)
        + i32x4_extract_lane::<2>(w)
        + i32x4_extract_lane::<3>(w)) as u32
}

/// wasm `simd128` SAD for `w` in {8, 16}; other widths fall back to scalar.
/// Bit-exact (associative sum of `|a-b|`; the 16x16 worst case fits a u16 lane).
pub fn sad(s1: &[u8], st1: usize, s2: &[u8], st2: usize, w: usize, h: usize) -> u32 {
    unsafe {
        match w {
            16 => {
                let mut acc = u16x8_splat(0);
                let mut p1 = s1.as_ptr();
                let mut p2 = s2.as_ptr();
                for _ in 0..h {
                    let a = v128_load(p1 as *const v128);
                    let b = v128_load(p2 as *const v128);
                    acc = i16x8_add(acc, i16x8_extadd_pairwise_u8x16(absdiff(a, b)));
                    p1 = p1.add(st1);
                    p2 = p2.add(st2);
                }
                hsum_u16x8(acc)
            }
            8 => {
                let mut acc = u16x8_splat(0);
                let mut p1 = s1.as_ptr();
                let mut p2 = s2.as_ptr();
                for _ in 0..h {
                    // Load 8 bytes; the upper 8 lanes are zero in both operands,
                    // so they contribute nothing to the sum.
                    let a = v128_load64_zero(p1 as *const u64);
                    let b = v128_load64_zero(p2 as *const u64);
                    acc = i16x8_add(acc, i16x8_extadd_pairwise_u8x16(absdiff(a, b)));
                    p1 = p1.add(st1);
                    p2 = p2.add(st2);
                }
                hsum_u16x8(acc)
            }
            _ => crate::dsp::sad::sad_scalar(s1, st1, s2, st2, w, h),
        }
    }
}

// ---------------------------------------------------------------------------
// 4x4 inverse transform + add
// ---------------------------------------------------------------------------

/// Bit-exact wasm `simd128` port of `idct4x4_add` (see the scalar reference and
/// the NEON port for the algorithm; this mirrors them lane-for-lane).
pub fn idct4x4_add(pred: &mut [u8], stride: usize, rs: &[i16; 16]) {
    unsafe {
        let lo = v128_load(rs.as_ptr() as *const v128); // rs[0..8]
        let hi = v128_load(rs.as_ptr().add(8) as *const v128); // rs[8..16]

        // Gather columns (lane = input row): cN = [rs[N], rs[4+N], rs[8+N], rs[12+N]].
        let c0 = i16x8_shuffle::<0, 4, 8, 12, 0, 0, 0, 0>(lo, hi);
        let c1 = i16x8_shuffle::<1, 5, 9, 13, 0, 0, 0, 0>(lo, hi);
        let c2 = i16x8_shuffle::<2, 6, 10, 14, 0, 0, 0, 0>(lo, hi);
        let c3 = i16x8_shuffle::<3, 7, 11, 15, 0, 0, 0, 0>(lo, hi);

        // Horizontal butterfly (wrapping i16).
        let t0 = i16x8_add(c0, c2);
        let t1 = i16x8_sub(c0, c2);
        let t2 = i16x8_sub(i16x8_shr(c1, 1), c3);
        let t3 = i16x8_add(c1, i16x8_shr(c3, 1));
        let s0 = i16x8_add(t0, t3); // src column 0 (lane = row)
        let s1 = i16x8_add(t1, t2);
        let s2 = i16x8_sub(t1, t2);
        let s3 = i16x8_sub(t0, t3);

        // Transpose src columns -> src rows (lane = column).
        let a01 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(s0, s1);
        let b23 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(s2, s3);
        let comb0 = i32x4_shuffle::<0, 4, 1, 5>(a01, b23); // [sr0 | sr1]
        let comb1 = i32x4_shuffle::<2, 6, 3, 7>(a01, b23); // [sr2 | sr3]

        let a = i32x4_extend_low_i16x8(comb0); // sr0 (src row0)
        let a4 = i32x4_extend_high_i16x8(comb0); // sr1
        let a8 = i32x4_extend_low_i16x8(comb1); // sr2
        let a12 = i32x4_extend_high_i16x8(comb1); // sr3

        let bias = i32x4_splat(32);
        let u1 = i32x4_add(a, a8);
        let u2 = i32x4_add(a4, i32x4_shr(a12, 1));
        let row0 = i32x4_shr(i32x4_add(bias, i32x4_add(u1, u2)), 6);
        let row3 = i32x4_shr(i32x4_add(bias, i32x4_sub(u1, u2)), 6);
        let v1 = i32x4_sub(a, a8);
        let v2 = i32x4_sub(i32x4_shr(a4, 1), a12);
        let row1 = i32x4_shr(i32x4_add(bias, i32x4_add(v1, v2)), 6);
        let row2 = i32x4_shr(i32x4_add(bias, i32x4_sub(v1, v2)), 6);

        let rows = [row0, row1, row2, row3];
        for (k, &delta) in rows.iter().enumerate() {
            let ptr = pred.as_mut_ptr().add(k * stride);
            let ld = v128_load32_zero(ptr as *const u32); // 4 pred bytes
            let p16 = i16x8_extend_low_u8x16(ld);
            let p32 = i32x4_extend_low_i16x8(p16);
            let sum = i32x4_add(delta, p32);
            // Clip to [0,255]: signed-narrow i32->i16, then unsigned-narrow i16->u8.
            let i16v = i16x8_narrow_i32x4(sum, sum);
            let u8v = u8x16_narrow_i16x8(i16v, i16v);
            v128_store32_lane::<0>(u8v, ptr as *mut u32);
        }
    }
}
