//! AArch64 NEON implementations of the hot DSP kernels.
//!
//! NEON is part of the aarch64 baseline, so these are always available without
//! runtime feature detection. Each is a bit-exact replacement for the matching
//! scalar reference; correctness is enforced by the parent modules' existing
//! unit tests run with `--features simd`.
//!
//! Memory-safety note: every load/store stays within the exact index footprint
//! the scalar reference already touches (the callers guarantee those indices are
//! in bounds — e.g. the decoded planes carry a 32-sample padding border), so the
//! raw-pointer accesses here read/write the same bytes the scalar code does.

use core::arch::aarch64::*;

use crate::dsp::{Blk, BlkMut, Dim};

// ---------------------------------------------------------------------------
// SAD
// ---------------------------------------------------------------------------

/// 16-wide row SAD accumulator step: `acc += sum_pairs(|a - b|)`.
#[inline]
unsafe fn sad_row16(acc: uint16x8_t, a: *const u8, b: *const u8) -> uint16x8_t {
    unsafe {
        let va = vld1q_u8(a);
        let vb = vld1q_u8(b);
        vpadalq_u8(acc, vabdq_u8(va, vb))
    }
}

/// 8-wide row SAD accumulator step.
#[inline]
unsafe fn sad_row8(acc: uint16x4_t, a: *const u8, b: *const u8) -> uint16x4_t {
    unsafe {
        let va = vld1_u8(a);
        let vb = vld1_u8(b);
        vpadal_u8(acc, vabd_u8(va, vb))
    }
}

/// NEON SAD for `w` in {8, 16}; any other width falls back to scalar. Bit-exact
/// (a plain associative sum of `|a-b|`; the 16x16 worst case fits a u16 lane).
pub fn sad(s1: &[u8], st1: usize, s2: &[u8], st2: usize, w: usize, h: usize) -> u32 {
    unsafe {
        match w {
            16 => {
                let mut acc = vdupq_n_u16(0);
                let mut p1 = s1.as_ptr();
                let mut p2 = s2.as_ptr();
                for _ in 0..h {
                    acc = sad_row16(acc, p1, p2);
                    p1 = p1.add(st1);
                    p2 = p2.add(st2);
                }
                vaddlvq_u16(acc)
            }
            8 => {
                let mut acc = vdup_n_u16(0);
                let mut p1 = s1.as_ptr();
                let mut p2 = s2.as_ptr();
                for _ in 0..h {
                    acc = sad_row8(acc, p1, p2);
                    p1 = p1.add(st1);
                    p2 = p2.add(st2);
                }
                vaddlv_u16(acc)
            }
            _ => crate::dsp::sad::sad_scalar(s1, st1, s2, st2, w, h),
        }
    }
}

// ---------------------------------------------------------------------------
// 4x4 inverse transform + add
// ---------------------------------------------------------------------------

/// Transpose four `int16x4_t` rows into four columns (own inverse).
#[inline]
unsafe fn transpose4_s16(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x4_t,
    d: int16x4_t,
) -> (int16x4_t, int16x4_t, int16x4_t, int16x4_t) {
    unsafe {
        let t0 = vtrn_s16(a, b);
        let t1 = vtrn_s16(c, d);
        let u0 = vtrn_s32(vreinterpret_s32_s16(t0.0), vreinterpret_s32_s16(t1.0));
        let u1 = vtrn_s32(vreinterpret_s32_s16(t0.1), vreinterpret_s32_s16(t1.1));
        (
            vreinterpret_s16_s32(u0.0),
            vreinterpret_s16_s32(u1.0),
            vreinterpret_s16_s32(u0.1),
            vreinterpret_s16_s32(u1.1),
        )
    }
}

/// Load the 4 prediction bytes of row `r` (at `pred[r*stride..]`) as `i32x4`.
#[inline]
unsafe fn load_pred_row(pred: *const u8, off: usize) -> int32x4_t {
    unsafe {
        let mut tmp = [0u8; 8];
        core::ptr::copy_nonoverlapping(pred.add(off), tmp.as_mut_ptr(), 4);
        let b = vld1_u8(tmp.as_ptr());
        let u16v = vget_low_u16(vmovl_u8(b));
        vreinterpretq_s32_u32(vmovl_u16(u16v))
    }
}

/// Clip an `i32x4` of (delta + pred) to `[0,255]` and store 4 bytes at `dst+off`.
#[inline]
unsafe fn store_clip_row(dst: *mut u8, off: usize, v: int32x4_t) {
    unsafe {
        let s16 = vqmovn_s32(v); // saturate i32 -> i16
        let u8v = vqmovun_s16(vcombine_s16(s16, s16)); // saturate i16 -> u8 [0,255]
        let lane = vget_lane_u32(vreinterpret_u32_u8(u8v), 0);
        core::ptr::copy_nonoverlapping((&lane as *const u32) as *const u8, dst.add(off), 4);
    }
}

/// Bit-exact NEON port of `idct4x4_add`. Horizontal pass uses wrapping `i16`
/// arithmetic (NEON add/sub/`vshr` wrap and arithmetic-shift identically to the
/// scalar `wrapping_*`/`>>`); the vertical pass promotes to `i32` exactly as the
/// scalar code, then `(32 + .) >> 6`, add prediction, and saturating clip.
pub fn idct4x4_add(pred: &mut [u8], stride: usize, rs: &[i16; 16]) {
    unsafe {
        let r0 = vld1_s16(rs.as_ptr());
        let r1 = vld1_s16(rs.as_ptr().add(4));
        let r2 = vld1_s16(rs.as_ptr().add(8));
        let r3 = vld1_s16(rs.as_ptr().add(12));

        // Transpose so each vector lane indexes the input row.
        let (c0, c1, c2, c3) = transpose4_s16(r0, r1, r2, r3);

        // Horizontal butterfly (wrapping i16).
        let t0 = vadd_s16(c0, c2);
        let t1 = vsub_s16(c0, c2);
        let t2 = vsub_s16(vshr_n_s16(c1, 1), c3);
        let t3 = vadd_s16(c1, vshr_n_s16(c3, 1));
        let s0 = vadd_s16(t0, t3); // src column 0 (lane = row)
        let s1 = vadd_s16(t1, t2); // src column 1
        let s2 = vsub_s16(t1, t2); // src column 2
        let s3 = vsub_s16(t0, t3); // src column 3

        // Transpose src columns back to src rows (lane = column).
        let (sr0, sr1, sr2, sr3) = transpose4_s16(s0, s1, s2, s3);

        // Vertical butterfly (promote to i32), across columns in parallel.
        let a = vmovl_s16(sr0);
        let a4 = vmovl_s16(sr1);
        let a8 = vmovl_s16(sr2);
        let a12 = vmovl_s16(sr3);
        let bias = vdupq_n_s32(32);

        let u1 = vaddq_s32(a, a8);
        let u2 = vaddq_s32(a4, vshrq_n_s32(a12, 1));
        let row0 = vshrq_n_s32(vaddq_s32(bias, vaddq_s32(u1, u2)), 6);
        let row3 = vshrq_n_s32(vaddq_s32(bias, vsubq_s32(u1, u2)), 6);
        let v1 = vsubq_s32(a, a8);
        let v2 = vsubq_s32(vshrq_n_s32(a4, 1), a12);
        let row1 = vshrq_n_s32(vaddq_s32(bias, vaddq_s32(v1, v2)), 6);
        let row2 = vshrq_n_s32(vaddq_s32(bias, vsubq_s32(v1, v2)), 6);

        let pp = pred.as_ptr();
        let p0 = load_pred_row(pp, 0);
        let p1 = load_pred_row(pp, stride);
        let p2 = load_pred_row(pp, 2 * stride);
        let p3 = load_pred_row(pp, 3 * stride);

        let dp = pred.as_mut_ptr();
        store_clip_row(dp, 0, vaddq_s32(row0, p0));
        store_clip_row(dp, stride, vaddq_s32(row1, p1));
        store_clip_row(dp, 2 * stride, vaddq_s32(row2, p2));
        store_clip_row(dp, 3 * stride, vaddq_s32(row3, p3));
    }
}

// ---------------------------------------------------------------------------
// Luma motion compensation: half-pel 6-tap filters + pixel average
// ---------------------------------------------------------------------------

/// Widen 8 bytes to `int16x8_t` (values 0..255 stay positive).
#[inline]
unsafe fn widen8(p: *const u8) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p))) }
}

/// 6-tap on six `int16x8_t` tap vectors: `(a+f) - 5*(b+e) + 20*(c+d)`.
#[inline]
unsafe fn tap6(
    a: int16x8_t,
    b: int16x8_t,
    c: int16x8_t,
    d: int16x8_t,
    e: int16x8_t,
    f: int16x8_t,
) -> int16x8_t {
    unsafe {
        let s05 = vaddq_s16(a, f);
        let s14 = vaddq_s16(b, e);
        let s23 = vaddq_s16(c, d);
        // Range fits i16 (max ~10710, min ~-2550), matching the scalar i32 result.
        let t = vsubq_s16(s05, vmulq_n_s16(s14, 5));
        vaddq_s16(t, vmulq_n_s16(s23, 20))
    }
}

/// One 8-wide horizontal half-pel output row at `src[base..]` -> `dst[doff..]`.
#[inline]
unsafe fn hor8(dst: *mut u8, doff: usize, src: *const u8, base: isize) {
    unsafe {
        let a = widen8(src.offset(base - 2));
        let b = widen8(src.offset(base - 1));
        let c = widen8(src.offset(base));
        let d = widen8(src.offset(base + 1));
        let e = widen8(src.offset(base + 2));
        let f = widen8(src.offset(base + 3));
        // (t + 16) >> 5 with unsigned saturation == clip1((v + 16) >> 5).
        vst1_u8(dst.add(doff), vqrshrun_n_s16(tap6(a, b, c, d, e, f), 5));
    }
}

/// One 8-wide vertical half-pel output row at `src[base..]` -> `dst[doff..]`.
#[inline]
unsafe fn ver8(dst: *mut u8, doff: usize, src: *const u8, base: isize, ss: isize) {
    unsafe {
        let a = widen8(src.offset(base - 2 * ss));
        let b = widen8(src.offset(base - ss));
        let c = widen8(src.offset(base));
        let d = widen8(src.offset(base + ss));
        let e = widen8(src.offset(base + 2 * ss));
        let f = widen8(src.offset(base + 3 * ss));
        vst1_u8(dst.add(doff), vqrshrun_n_s16(tap6(a, b, c, d, e, f), 5));
    }
}

/// Bit-exact NEON `mc_hor_ver20` (horizontal half-pel) for w in {8,16}.
pub fn mc_hor_ver20(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, dim: Dim) {
    if dim.w != 8 && dim.w != 16 {
        crate::dsp::mc::mc_hor_ver20_scalar(dst, do_, ds, src, so, ss, dim);
        return;
    }
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..dim.h {
            let base = (so + i * ss) as isize;
            let doff = do_ + i * ds;
            hor8(dp, doff, sp, base);
            if dim.w == 16 {
                hor8(dp, doff + 8, sp, base + 8);
            }
        }
    }
}

/// Bit-exact NEON `mc_hor_ver02` (vertical half-pel) for w in {8,16}.
pub fn mc_hor_ver02(dst: &mut [u8], do_: usize, ds: usize, src: &[u8], so: usize, ss: usize, dim: Dim) {
    if dim.w != 8 && dim.w != 16 {
        crate::dsp::mc::mc_hor_ver02_scalar(dst, do_, ds, src, so, ss, dim);
        return;
    }
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let ssi = ss as isize;
        for i in 0..dim.h {
            let base = (so + i * ss) as isize;
            let doff = do_ + i * ds;
            ver8(dp, doff, sp, base, ssi);
            if dim.w == 16 {
                ver8(dp, doff + 8, sp, base + 8, ssi);
            }
        }
    }
}

/// Bit-exact NEON `pixel_avg` (`(a + b + 1) >> 1`) for w in {8,16}.
pub fn pixel_avg(dst: BlkMut, a: Blk, b: Blk, dim: Dim) {
    if dim.w != 8 && dim.w != 16 {
        crate::dsp::mc::pixel_avg_scalar(dst, a, b, dim);
        return;
    }
    unsafe {
        let ap = a.data.as_ptr();
        let bp = b.data.as_ptr();
        let dp = dst.data.as_mut_ptr();
        for i in 0..dim.h {
            let da = dp.add(dst.off + i * dst.stride);
            let pa = ap.add(a.off + i * a.stride);
            let pb = bp.add(b.off + i * b.stride);
            if dim.w == 16 {
                vst1q_u8(da, vrhaddq_u8(vld1q_u8(pa), vld1q_u8(pb)));
            } else {
                vst1_u8(da, vrhadd_u8(vld1_u8(pa), vld1_u8(pb)));
            }
        }
    }
}
