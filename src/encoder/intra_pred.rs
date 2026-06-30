//! Encoder-side intra prediction generators, ported from
//! `reference/codec/encoder/core/src/get_intra_predictor.cpp` and the I16x16
//! V/H kernels of `reference/codec/common/src/intra_pred_common.cpp`.
//!
//! Unlike the decoder kernels (which reconstruct in place), the encoder builds
//! each candidate prediction into a *separate* contiguous buffer so the mode
//! decision can score it against the source with SAD/SATD. Each generator takes:
//!
//! * `pred`: the output buffer, written row-major with a fixed stride (4 for
//!   4x4 luma, 8 for 8x8 chroma, 16 for 16x16 luma).
//! * `reff` / `roff` / `stride`: the reconstructed reference plane, the index of
//!   the block's top-left sample (`pRef[0]` in C), and the plane row stride.
//!   Neighbours are read via signed offsets exactly as the C negative indexing:
//!   `roff - stride + x` (top row), `roff - 1 + y*stride` (left column),
//!   `roff - stride - 1` (top-left).
//!
//! All kernels are bit-exact ports validated against the `EncUT_GetIntraPredictor`
//! anchors.

use crate::dsp::Blk;
use crate::dsp::clip1;
use crate::dsp::sad::{sad8x8, sad16x16};
use crate::dsp::satd::{satd4x4, satd8x8, satd16x16};

// ===========================================================================
// Luma 4x4 (output stride 4, 16 bytes)
// ===========================================================================

#[inline(always)]
fn fill16(pred: &mut [u8], v: u8) {
    pred[..16].iter_mut().for_each(|p| *p = v);
}

/// `WelsI4x4LumaPredV_c`: vertical (copy the top row down).
pub fn i4x4_luma_pred_v(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let t = top4(reff, roff, stride);
    for row in 0..4 {
        pred[row * 4..row * 4 + 4].copy_from_slice(&t);
    }
}

/// `WelsI4x4LumaPredH_c`: horizontal (copy the left column across).
pub fn i4x4_luma_pred_h(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize];
    let s = stride as isize;
    for row in 0..4 {
        let l = r(row as isize * s - 1);
        pred[row * 4..row * 4 + 4].copy_from_slice(&[l, l, l, l]);
    }
}

/// `WelsI4x4LumaPredDc_c`: DC of the top row and left column.
pub fn i4x4_luma_pred_dc(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let dc = ((r(-1)
        + r(s - 1)
        + r(2 * s - 1)
        + r(3 * s - 1)
        + r(-s)
        + r(1 - s)
        + r(2 - s)
        + r(3 - s)
        + 4)
        >> 3) as u8;
    fill16(pred, dc);
}

/// `WelsI4x4LumaPredDcLeft_c`: DC of the left column only.
pub fn i4x4_luma_pred_dc_left(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let dc = ((r(-1) + r(s - 1) + r(2 * s - 1) + r(3 * s - 1) + 2) >> 2) as u8;
    fill16(pred, dc);
}

/// `WelsI4x4LumaPredDcTop_c`: DC of the top row only.
pub fn i4x4_luma_pred_dc_top(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let dc = ((r(-s) + r(1 - s) + r(2 - s) + r(3 - s) + 2) >> 2) as u8;
    fill16(pred, dc);
}

/// `WelsI4x4LumaPredDcNA_c`: no neighbours available (mid-grey 0x80).
pub fn i4x4_luma_pred_dc_na(pred: &mut [u8], _reff: &[u8], _roff: usize, _stride: usize) {
    fill16(pred, 0x80);
}

/// `WelsI4x4LumaPredDDL_c`: diagonal down-left.
pub fn i4x4_luma_pred_ddl(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let t = top8(reff, roff, stride);
    let avg3 = |a: u8, b: u8, c: u8| ((2 + a as i32 + c as i32 + ((b as i32) << 1)) >> 2) as u8;
    let d0 = avg3(t[0], t[1], t[2]);
    let d1 = avg3(t[1], t[2], t[3]);
    let d2 = avg3(t[2], t[3], t[4]);
    let d3 = avg3(t[3], t[4], t[5]);
    let d4 = avg3(t[4], t[5], t[6]);
    let d5 = avg3(t[5], t[6], t[7]);
    let d6 = avg3(t[6], t[7], t[7]);
    let mut s = [0u8; 16];
    s[0] = d0;
    s[1] = d1;
    s[4] = d1;
    s[2] = d2;
    s[5] = d2;
    s[8] = d2;
    s[3] = d3;
    s[6] = d3;
    s[9] = d3;
    s[12] = d3;
    s[7] = d4;
    s[10] = d4;
    s[13] = d4;
    s[11] = d5;
    s[14] = d5;
    s[15] = d6;
    pred[..16].copy_from_slice(&s);
}

/// `WelsI4x4LumaPredDDLTop_c`: diagonal down-left, top-right unavailable.
pub fn i4x4_luma_pred_ddl_top(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let t = top4(reff, roff, stride);
    let avg3 = |a: u8, b: u8, c: u8| ((2 + a as i32 + c as i32 + ((b as i32) << 1)) >> 2) as u8;
    let d0 = avg3(t[0], t[1], t[2]);
    let d1 = avg3(t[1], t[2], t[3]);
    let d2 = avg3(t[2], t[3], t[3]);
    let d3 = ((2 + ((t[3] as i32) << 2)) >> 2) as u8;
    let mut s = [d3; 16];
    s[0] = d0;
    s[1] = d1;
    s[4] = d1;
    s[2] = d2;
    s[5] = d2;
    s[8] = d2;
    s[3] = d3;
    pred[..16].copy_from_slice(&s);
}

/// `WelsI4x4LumaPredDDR_c`: diagonal down-right.
pub fn i4x4_luma_pred_ddr(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let lt = r(-s - 1);
    let l0 = r(-1);
    let l1 = r(s - 1);
    let l2 = r(2 * s - 1);
    let l3 = r(3 * s - 1);
    let t0 = r(-s);
    let t1 = r(1 - s);
    let t2 = r(2 - s);
    let t3 = r(3 - s);
    let tl0 = 1 + lt + l0;
    let lt0 = 1 + lt + t0;
    let t01 = 1 + t0 + t1;
    let t12 = 1 + t1 + t2;
    let t23 = 1 + t2 + t3;
    let l01 = 1 + l0 + l1;
    let l12 = 1 + l1 + l2;
    let l23 = 1 + l2 + l3;
    let d0 = ((tl0 + lt0) >> 2) as u8;
    let d1 = ((lt0 + t01) >> 2) as u8;
    let d2 = ((t01 + t12) >> 2) as u8;
    let d3 = ((t12 + t23) >> 2) as u8;
    let d4 = ((tl0 + l01) >> 2) as u8;
    let d5 = ((l01 + l12) >> 2) as u8;
    let d6 = ((l12 + l23) >> 2) as u8;
    let mut sb = [0u8; 16];
    for i in [0, 5, 10, 15] {
        sb[i] = d0;
    }
    for i in [1, 6, 11] {
        sb[i] = d1;
    }
    sb[2] = d2;
    sb[7] = d2;
    sb[3] = d3;
    for i in [4, 9, 14] {
        sb[i] = d4;
    }
    sb[8] = d5;
    sb[13] = d5;
    sb[12] = d6;
    pred[..16].copy_from_slice(&sb);
}

/// `WelsI4x4LumaPredVL_c`: vertical-left.
pub fn i4x4_luma_pred_vl(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let t = top8(reff, roff, stride);
    let a2 = |a: u8, b: u8| ((1 + a as i32 + b as i32) >> 1) as u8;
    let a3 = |a: u8, b: u8, c: u8| ((2 + a as i32 + ((b as i32) << 1) + c as i32) >> 2) as u8;
    let v0 = a2(t[0], t[1]);
    let v1 = a2(t[1], t[2]);
    let v2 = a2(t[2], t[3]);
    let v3 = a2(t[3], t[4]);
    let v4 = a2(t[4], t[5]);
    let v5 = a3(t[0], t[1], t[2]);
    let v6 = a3(t[1], t[2], t[3]);
    let v7 = a3(t[2], t[3], t[4]);
    let v8 = a3(t[3], t[4], t[5]);
    let v9 = a3(t[4], t[5], t[6]);
    let mut s = [0u8; 16];
    s[0] = v0;
    s[1] = v1;
    s[8] = v1;
    s[2] = v2;
    s[9] = v2;
    s[3] = v3;
    s[10] = v3;
    s[4] = v5;
    s[5] = v6;
    s[12] = v6;
    s[6] = v7;
    s[13] = v7;
    s[7] = v8;
    s[14] = v8;
    s[11] = v4;
    s[15] = v9;
    pred[..16].copy_from_slice(&s);
}

/// `WelsI4x4LumaPredVLTop_c`: vertical-left, top-right unavailable.
pub fn i4x4_luma_pred_vl_top(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let t = top4(reff, roff, stride);
    let a2 = |a: u8, b: u8| ((1 + a as i32 + b as i32) >> 1) as u8;
    let a3 = |a: u8, b: u8, c: u8| ((2 + a as i32 + ((b as i32) << 1) + c as i32) >> 2) as u8;
    let v0 = a2(t[0], t[1]);
    let v1 = a2(t[1], t[2]);
    let v2 = a2(t[2], t[3]);
    let v3 = ((1 + ((t[3] as i32) << 1)) >> 1) as u8;
    let v4 = a3(t[0], t[1], t[2]);
    let v5 = a3(t[1], t[2], t[3]);
    let v6 = a3(t[2], t[3], t[3]);
    let v7 = ((2 + ((t[3] as i32) << 2)) >> 2) as u8;
    let mut s = [0u8; 16];
    s[0] = v0;
    s[1] = v1;
    s[8] = v1;
    s[2] = v2;
    s[9] = v2;
    s[3] = v3;
    s[10] = v3;
    s[11] = v3;
    s[4] = v4;
    s[5] = v5;
    s[12] = v5;
    s[6] = v6;
    s[13] = v6;
    s[7] = v7;
    s[14] = v7;
    s[15] = v7;
    pred[..16].copy_from_slice(&s);
}

/// `WelsI4x4LumaPredVR_c`: vertical-right.
pub fn i4x4_luma_pred_vr(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let lt = r(-s - 1);
    let l0 = r(-1);
    let l1 = r(s - 1);
    let l2 = r(2 * s - 1);
    let t0 = r(-s);
    let t1 = r(1 - s);
    let t2 = r(2 - s);
    let t3 = r(3 - s);
    let v0 = ((1 + lt + t0) >> 1) as u8;
    let v1 = ((1 + t0 + t1) >> 1) as u8;
    let v2 = ((1 + t1 + t2) >> 1) as u8;
    let v3 = ((1 + t2 + t3) >> 1) as u8;
    let v4 = ((2 + l0 + (lt << 1) + t0) >> 2) as u8;
    let v5 = ((2 + lt + (t0 << 1) + t1) >> 2) as u8;
    let v6 = ((2 + t0 + (t1 << 1) + t2) >> 2) as u8;
    let v7 = ((2 + t1 + (t2 << 1) + t3) >> 2) as u8;
    let v8 = ((2 + lt + (l0 << 1) + l1) >> 2) as u8;
    let v9 = ((2 + l0 + (l1 << 1) + l2) >> 2) as u8;
    let mut sb = [0u8; 16];
    sb[0] = v0;
    sb[9] = v0;
    sb[1] = v1;
    sb[10] = v1;
    sb[2] = v2;
    sb[11] = v2;
    sb[3] = v3;
    sb[4] = v4;
    sb[13] = v4;
    sb[5] = v5;
    sb[14] = v5;
    sb[6] = v6;
    sb[15] = v6;
    sb[7] = v7;
    sb[8] = v8;
    sb[12] = v9;
    pred[..16].copy_from_slice(&sb);
}

/// `WelsI4x4LumaPredHU_c`: horizontal-up.
pub fn i4x4_luma_pred_hu(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let l0 = r(-1);
    let l1 = r(s - 1);
    let l2 = r(2 * s - 1);
    let l3 = r(3 * s - 1);
    let l01 = 1 + l0 + l1;
    let l12 = 1 + l1 + l2;
    let l23 = 1 + l2 + l3;
    let h0 = (l01 >> 1) as u8;
    let h1 = ((l01 + l12) >> 2) as u8;
    let h2 = (l12 >> 1) as u8;
    let h3 = ((l12 + l23) >> 2) as u8;
    let h4 = (l23 >> 1) as u8;
    let h5 = ((1 + l23 + (l3 << 1)) >> 2) as u8;
    let mut sb = [l3 as u8; 16];
    sb[0] = h0;
    sb[1] = h1;
    sb[2] = h2;
    sb[4] = h2;
    sb[3] = h3;
    sb[5] = h3;
    sb[6] = h4;
    sb[8] = h4;
    sb[7] = h5;
    sb[9] = h5;
    pred[..16].copy_from_slice(&sb);
}

/// `WelsI4x4LumaPredHD_c`: horizontal-down.
pub fn i4x4_luma_pred_hd(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let lt = r(-s - 1);
    let l0 = r(-1);
    let l1 = r(s - 1);
    let l2 = r(2 * s - 1);
    let l3 = r(3 * s - 1);
    let t0 = r(-s);
    let t1 = r(1 - s);
    let t2 = r(2 - s);
    let h0 = ((1 + lt + l0) >> 1) as u8;
    let h1 = ((2 + l0 + (lt << 1) + t0) >> 2) as u8;
    let h2 = ((2 + lt + (t0 << 1) + t1) >> 2) as u8;
    let h3 = ((2 + t0 + (t1 << 1) + t2) >> 2) as u8;
    let h4 = ((1 + l0 + l1) >> 1) as u8;
    let h5 = ((2 + lt + (l0 << 1) + l1) >> 2) as u8;
    let h6 = ((1 + l1 + l2) >> 1) as u8;
    let h7 = ((2 + l0 + (l1 << 1) + l2) >> 2) as u8;
    let h8 = ((1 + l2 + l3) >> 1) as u8;
    let h9 = ((2 + l1 + (l2 << 1) + l3) >> 2) as u8;
    let mut sb = [0u8; 16];
    sb[0] = h0;
    sb[6] = h0;
    sb[1] = h1;
    sb[7] = h1;
    sb[2] = h2;
    sb[3] = h3;
    sb[4] = h4;
    sb[10] = h4;
    sb[5] = h5;
    sb[11] = h5;
    sb[8] = h6;
    sb[14] = h6;
    sb[9] = h7;
    sb[15] = h7;
    sb[12] = h8;
    sb[13] = h9;
    pred[..16].copy_from_slice(&sb);
}

#[inline(always)]
fn top4(reff: &[u8], roff: usize, stride: usize) -> [u8; 4] {
    let b = roff - stride;
    [reff[b], reff[b + 1], reff[b + 2], reff[b + 3]]
}

#[inline(always)]
fn top8(reff: &[u8], roff: usize, stride: usize) -> [u8; 8] {
    let b = roff - stride;
    let mut t = [0u8; 8];
    t.copy_from_slice(&reff[b..b + 8]);
    t
}

// ===========================================================================
// Chroma 8x8 (output stride 8, 64 bytes)
// ===========================================================================

/// `WelsIChromaPredV_c`: vertical (copy the top row down).
pub fn chroma_pred_v(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let top = &reff[roff - stride..roff - stride + 8];
    for row in 0..8 {
        pred[row * 8..row * 8 + 8].copy_from_slice(top);
    }
}

/// `WelsIChromaPredH_c`: horizontal (copy the left column across).
pub fn chroma_pred_h(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize];
    let s = stride as isize;
    for row in 0..8 {
        let l = r(row as isize * s - 1);
        pred[row * 8..row * 8 + 8].copy_from_slice(&[l; 8]);
    }
}

/// `WelsIChromaPredDc_c`: per-quadrant DC.
pub fn chroma_pred_dc(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let l = |k: isize| r(k * s - 1); // left column
    let m1 = ((r(-s) + r(1 - s) + r(2 - s) + r(3 - s) + l(0) + l(1) + l(2) + l(3) + 4) >> 3) as u8;
    let sum2 = r(4 - s) + r(5 - s) + r(6 - s) + r(7 - s);
    let sum3 = l(4) + l(5) + l(6) + l(7);
    let m2 = ((sum2 + 2) >> 2) as u8;
    let m3 = ((sum3 + 2) >> 2) as u8;
    let m4 = ((sum2 + sum3 + 4) >> 3) as u8;
    let top = [m1, m1, m1, m1, m2, m2, m2, m2];
    let bot = [m3, m3, m3, m3, m4, m4, m4, m4];
    for row in 0..4 {
        pred[row * 8..row * 8 + 8].copy_from_slice(&top);
    }
    for row in 4..8 {
        pred[row * 8..row * 8 + 8].copy_from_slice(&bot);
    }
}

/// `WelsIChromaPredDcLeft_c`: DC of the left column (top/bottom halves).
pub fn chroma_pred_dc_left(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let l = |k: isize| r(k * s - 1);
    let top = ((l(0) + l(1) + l(2) + l(3) + 2) >> 2) as u8;
    let bot = ((l(4) + l(5) + l(6) + l(7) + 2) >> 2) as u8;
    for row in 0..4 {
        pred[row * 8..row * 8 + 8].copy_from_slice(&[top; 8]);
    }
    for row in 4..8 {
        pred[row * 8..row * 8 + 8].copy_from_slice(&[bot; 8]);
    }
}

/// `WelsIChromaPredDcTop_c`: DC of the top row (left/right halves).
pub fn chroma_pred_dc_top(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let m1 = ((r(-s) + r(1 - s) + r(2 - s) + r(3 - s) + 2) >> 2) as u8;
    let m2 = ((r(4 - s) + r(5 - s) + r(6 - s) + r(7 - s) + 2) >> 2) as u8;
    let row = [m1, m1, m1, m1, m2, m2, m2, m2];
    for r in 0..8 {
        pred[r * 8..r * 8 + 8].copy_from_slice(&row);
    }
}

/// `WelsIChromaPredDcNA_c`: no neighbours (mid-grey 0x80).
pub fn chroma_pred_dc_na(pred: &mut [u8], _reff: &[u8], _roff: usize, _stride: usize) {
    pred[..64].iter_mut().for_each(|p| *p = 0x80);
}

/// `WelsIChromaPredPlane_c`: plane prediction.
pub fn chroma_pred_plane(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let top = |k: isize| r(-s + k);
    let left = |k: isize| r(k * s - 1);
    let mut top_sum = 0i32;
    let mut left_sum = 0i32;
    for i in 0..4i32 {
        top_sum += (i + 1) * (top(4 + i as isize) - top(2 - i as isize));
        left_sum += (i + 1) * (left(4 + i as isize) - left(2 - i as isize));
    }
    let lt_shift = (left(7) + top(7)) << 4;
    let top_shift = (17 * top_sum + 16) >> 5;
    let left_shift = (17 * left_sum + 16) >> 5;
    for i in 0..8 {
        for j in 0..8 {
            pred[i * 8 + j] = clip1(
                (lt_shift + top_shift * (j as i32 - 3) + left_shift * (i as i32 - 3) + 16) >> 5,
            );
        }
    }
}

// ===========================================================================
// Luma 16x16 (output stride 16, 256 bytes)
// ===========================================================================

/// `WelsI16x16LumaPredV_c`: vertical (copy the top row down).
pub fn i16x16_luma_pred_v(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let top = &reff[roff - stride..roff - stride + 16];
    for row in 0..16 {
        pred[row * 16..row * 16 + 16].copy_from_slice(top);
    }
}

/// `WelsI16x16LumaPredH_c`: horizontal (copy the left column across).
pub fn i16x16_luma_pred_h(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize];
    let s = stride as isize;
    for row in 0..16 {
        let l = r(row as isize * s - 1);
        pred[row * 16..row * 16 + 16].copy_from_slice(&[l; 16]);
    }
}

/// `WelsI16x16LumaPredDc_c`: DC of the top row and left column.
pub fn i16x16_luma_pred_dc(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let mut sum = 0i32;
    for i in 0..16 {
        sum += r(i * s - 1) + r(-s + i);
    }
    let mean = ((16 + sum) >> 5) as u8;
    pred[..256].iter_mut().for_each(|p| *p = mean);
}

/// `WelsI16x16LumaPredDcTop_c`: DC of the top row only.
pub fn i16x16_luma_pred_dc_top(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let mut sum = 0i32;
    for i in 0..16 {
        sum += r(-s + i);
    }
    let mean = ((8 + sum) >> 4) as u8;
    pred[..256].iter_mut().for_each(|p| *p = mean);
}

/// `WelsI16x16LumaPredDcLeft_c`: DC of the left column only.
pub fn i16x16_luma_pred_dc_left(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let mut sum = 0i32;
    for i in 0..16 {
        sum += r(i * s - 1);
    }
    let mean = ((8 + sum) >> 4) as u8;
    pred[..256].iter_mut().for_each(|p| *p = mean);
}

/// `WelsI16x16LumaPredDcNA_c`: no neighbours (mid-grey 0x80).
pub fn i16x16_luma_pred_dc_na(pred: &mut [u8], _reff: &[u8], _roff: usize, _stride: usize) {
    pred[..256].iter_mut().for_each(|p| *p = 0x80);
}

/// `WelsI16x16LumaPredPlane_c`: plane prediction.
pub fn i16x16_luma_pred_plane(pred: &mut [u8], reff: &[u8], roff: usize, stride: usize) {
    let r = |d: isize| reff[(roff as isize + d) as usize] as i32;
    let s = stride as isize;
    let top = |k: isize| r(-s + k);
    let left = |k: isize| r(k * s - 1);
    let mut top_sum = 0i32;
    let mut left_sum = 0i32;
    for i in 0..8i32 {
        top_sum += (i + 1) * (top(8 + i as isize) - top(6 - i as isize));
        left_sum += (i + 1) * (left(8 + i as isize) - left(6 - i as isize));
    }
    let lt_shift = (left(15) + top(15)) << 4;
    let top_shift = (5 * top_sum + 32) >> 6;
    let left_shift = (5 * left_sum + 32) >> 6;
    for i in 0..16 {
        for j in 0..16 {
            pred[i * 16 + j] = clip1(
                (lt_shift + top_shift * (j as i32 - 7) + left_shift * (i as i32 - 7) + 16) >> 5,
            );
        }
    }
}

// ===========================================================================
// Intra mode-decision cost helpers (predict-then-metric, pick the best mode).
// Ports of the `WelsSample{Satd,Sad}Intra*Combined3_c` from sample.cpp.
// Each returns `(best_cost, best_mode)`.
// ===========================================================================

/// `WelsSampleSatdIntra4x4Combined3_c`: scores luma 4x4 DC/H/V (modes 2/1/0).
/// The best prediction (16 bytes, stride 4) is written to `dst`.
pub fn satd_intra_4x4_combined3(
    dec: Blk,
    enc: Blk,
    dst: &mut [u8],
    lambda2: i32,
    lambda1: i32,
    lambda0: i32,
) -> (i32, i32) {
    let Blk {
        data: dec,
        off: dec_off,
        stride: dec_stride,
    } = dec;
    let Blk {
        data: enc,
        off: enc_off,
        stride: enc_stride,
    } = enc;
    let mut buf = [[0u8; 16]; 3];
    let e = &enc[enc_off..];
    i4x4_luma_pred_dc(&mut buf[2], dec, dec_off, dec_stride);
    let mut best_mode = 2usize;
    let mut best_cost = satd4x4(&buf[2], 4, e, enc_stride) + lambda2;
    i4x4_luma_pred_h(&mut buf[1], dec, dec_off, dec_stride);
    let cost = satd4x4(&buf[1], 4, e, enc_stride) + lambda1;
    if cost < best_cost {
        best_mode = 1;
        best_cost = cost;
    }
    i4x4_luma_pred_v(&mut buf[0], dec, dec_off, dec_stride);
    let cost = satd4x4(&buf[0], 4, e, enc_stride) + lambda0;
    if cost < best_cost {
        best_mode = 0;
        best_cost = cost;
    }
    dst[..16].copy_from_slice(&buf[best_mode]);
    (best_cost, best_mode as i32)
}

/// `WelsSampleSatdIntra16x16Combined3_c`: luma 16x16 V/H/DC (modes 0/1/2).
/// `dst` (256 bytes, stride 16) is left holding the last (DC) prediction.
pub fn satd_intra_16x16_combined3(dec: Blk, enc: Blk, dst: &mut [u8], lambda: i32) -> (i32, i32) {
    intra_16x16_combined3(dec, enc, dst, lambda, satd16x16)
}

/// `WelsSampleSadIntra16x16Combined3_c`: as above but using SAD.
pub fn sad_intra_16x16_combined3(dec: Blk, enc: Blk, dst: &mut [u8], lambda: i32) -> (i32, i32) {
    intra_16x16_combined3(dec, enc, dst, lambda, |a, sa, b, sb| {
        sad16x16(a, sa, b, sb) as i32
    })
}

#[inline]
fn intra_16x16_combined3(
    dec: Blk,
    enc: Blk,
    dst: &mut [u8],
    lambda: i32,
    metric: impl Fn(&[u8], usize, &[u8], usize) -> i32,
) -> (i32, i32) {
    let Blk {
        data: dec,
        off: dec_off,
        stride: dec_stride,
    } = dec;
    let Blk {
        data: enc,
        off: enc_off,
        stride: enc_stride,
    } = enc;
    let e = &enc[enc_off..];
    i16x16_luma_pred_v(dst, dec, dec_off, dec_stride);
    let mut best_cost = metric(dst, 16, e, enc_stride);
    let mut best_mode = 0;
    i16x16_luma_pred_h(dst, dec, dec_off, dec_stride);
    let cost = metric(dst, 16, e, enc_stride) + lambda * 2;
    if cost < best_cost {
        best_mode = 1;
        best_cost = cost;
    }
    i16x16_luma_pred_dc(dst, dec, dec_off, dec_stride);
    let cost = metric(dst, 16, e, enc_stride) + lambda * 2;
    if cost < best_cost {
        best_mode = 2;
        best_cost = cost;
    }
    (best_cost, best_mode)
}

/// `WelsSampleSatdIntra8x8Combined3_c`: chroma Cb+Cr V/H/DC (modes 2/1/0).
/// `dst` holds Cb at `[0..64]` and Cr at `[64..128]` (stride 8).
pub fn satd_intra_8x8_combined3(
    dec_cb: Blk,
    enc_cb: Blk,
    dst: &mut [u8],
    lambda: i32,
    dec_cr: Blk,
    enc_cr: Blk,
) -> (i32, i32) {
    intra_8x8_combined3(dec_cb, enc_cb, dst, lambda, dec_cr, enc_cr, satd8x8)
}

/// `WelsSampleSadIntra8x8Combined3_c`: as above but using SAD.
pub fn sad_intra_8x8_combined3(
    dec_cb: Blk,
    enc_cb: Blk,
    dst: &mut [u8],
    lambda: i32,
    dec_cr: Blk,
    enc_cr: Blk,
) -> (i32, i32) {
    intra_8x8_combined3(
        dec_cb,
        enc_cb,
        dst,
        lambda,
        dec_cr,
        enc_cr,
        |a, sa, b, sb| sad8x8(a, sa, b, sb) as i32,
    )
}

#[inline]
fn intra_8x8_combined3(
    dec_cb: Blk,
    enc_cb: Blk,
    dst: &mut [u8],
    lambda: i32,
    dec_cr: Blk,
    enc_cr: Blk,
    metric: impl Fn(&[u8], usize, &[u8], usize) -> i32,
) -> (i32, i32) {
    let Blk {
        data: dec_cb,
        off: dec_off_cb,
        stride: dec_stride,
    } = dec_cb;
    let Blk {
        data: enc_cb,
        off: enc_off_cb,
        stride: enc_stride,
    } = enc_cb;
    let Blk {
        data: dec_cr,
        off: dec_off_cr,
        stride: _,
    } = dec_cr;
    let Blk {
        data: enc_cr,
        off: enc_off_cr,
        stride: _,
    } = enc_cr;
    let ecb = &enc_cb[enc_off_cb..];
    let ecr = &enc_cr[enc_off_cr..];
    let cost_both = |dst: &[u8], lambda_add: i32| {
        let (cb, cr) = dst.split_at(64);
        metric(cb, 8, ecb, enc_stride) + metric(cr, 8, ecr, enc_stride) + lambda_add
    };
    // V -> mode 2
    chroma_pred_v(&mut dst[..64], dec_cb, dec_off_cb, dec_stride);
    chroma_pred_v(&mut dst[64..], dec_cr, dec_off_cr, dec_stride);
    let mut best_mode = 2;
    let mut best_cost = cost_both(dst, lambda * 2);
    // H -> mode 1
    chroma_pred_h(&mut dst[..64], dec_cb, dec_off_cb, dec_stride);
    chroma_pred_h(&mut dst[64..], dec_cr, dec_off_cr, dec_stride);
    let cost = cost_both(dst, lambda * 2);
    if cost < best_cost {
        best_mode = 1;
        best_cost = cost;
    }
    // DC -> mode 0
    chroma_pred_dc(&mut dst[..64], dec_cb, dec_off_cb, dec_stride);
    chroma_pred_dc(&mut dst[64..], dec_cr, dec_off_cr, dec_stride);
    let cost = cost_both(dst, 0);
    if cost < best_cost {
        best_mode = 0;
        best_cost = cost;
    }
    (best_cost, best_mode)
}

#[cfg(test)]
mod tests;
