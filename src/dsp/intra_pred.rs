//! H.264 intra prediction kernels, ported from
//! `reference/codec/decoder/core/src/get_intra_predictor.cpp` and
//! `reference/codec/common/src/intra_pred_common.cpp`.
//!
//! Faithful, bit-exact ports of the Cisco OpenH264 `WelsI*Pred*_c` functions.
//!
//! # Addressing model
//!
//! The C kernels receive `uint8_t* pPred` and read neighbors via negative
//! indexing (`pPred[-kiStride + x]` for the row above, `pPred[-1 + y*kiStride]`
//! for the column to the left). To mirror this faithfully and keep the code
//! testable, every kernel takes the full plane slice `plane: &mut [u8]`, an
//! `offset: usize` to the block's top-left sample, and `stride: usize`. Indices
//! are then formed exactly as in C: `offset + y*stride + x` for the block,
//! `offset - stride + x` for the top row and `offset - 1 + y*stride` for the
//! left column. The caller must supply an `offset` large enough that the
//! top-left neighbor (`offset - stride - 1`) is in bounds.
//!
//! 8x8 luma kernels keep the `b_tl` / `b_tr` (top-left / top-right available)
//! flags of the C originals, which select the reference-sample smoothing at the
//! block edges (clause 8.3.2.2.1).

use crate::dsp::clip1;

// ===========================================================================
// Luma 4x4
// ===========================================================================

/// Port of `WelsI4x4LumaPredV_c`: vertical prediction (copy top row down).
pub fn i4x4_luma_pred_v(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let t = [
        plane[(o - s) as usize],
        plane[(o - s + 1) as usize],
        plane[(o - s + 2) as usize],
        plane[(o - s + 3) as usize],
    ];
    for y in 0..4 {
        for x in 0..4 {
            plane[(o + (y as isize) * s + x as isize) as usize] = t[x];
        }
    }
}

/// Port of `WelsI4x4LumaPredH_c`: horizontal prediction (copy left column across).
pub fn i4x4_luma_pred_h(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    for y in 0..4 {
        let l = plane[(o + (y as isize) * s - 1) as usize];
        for x in 0..4 {
            plane[(o + (y as isize) * s + x as isize) as usize] = l;
        }
    }
}

/// Port of `WelsI4x4LumaPredDc_c`: DC from both top row and left column.
pub fn i4x4_luma_pred_dc(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mean = ((g(-1) + g(-1 + s) + g(-1 + 2 * s) + g(-1 + 3 * s)
        + g(-s) + g(1 - s) + g(2 - s) + g(3 - s)
        + 4)
        >> 3) as u8;
    fill4(plane, o, s, mean);
}

/// Port of `WelsI4x4LumaPredDcLeft_c`: DC from the left column only.
pub fn i4x4_luma_pred_dc_left(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mean = ((g(-1) + g(-1 + s) + g(-1 + 2 * s) + g(-1 + 3 * s) + 2) >> 2) as u8;
    fill4(plane, o, s, mean);
}

/// Port of `WelsI4x4LumaPredDcTop_c`: DC from the top row only.
pub fn i4x4_luma_pred_dc_top(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mean = ((g(-s) + g(1 - s) + g(2 - s) + g(3 - s) + 2) >> 2) as u8;
    fill4(plane, o, s, mean);
}

/// Port of `WelsI4x4LumaPredDcNA_c`: DC with no neighbors (mid-grey 128).
pub fn i4x4_luma_pred_dc_na(plane: &mut [u8], offset: usize, stride: usize) {
    fill4(plane, offset as isize, stride as isize, 0x80);
}

#[inline]
fn fill4(plane: &mut [u8], o: isize, s: isize, v: u8) {
    for y in 0..4 {
        for x in 0..4 {
            plane[(o + (y as isize) * s + x as isize) as usize] = v;
        }
    }
}

#[inline]
fn write_rows4(plane: &mut [u8], o: isize, s: isize, list: &[u8; 8], starts: [usize; 4]) {
    for (y, &k) in starts.iter().enumerate() {
        for x in 0..4 {
            plane[(o + (y as isize) * s + x as isize) as usize] = list[k + x];
        }
    }
}

/// Port of `WelsI4x4LumaPredDDL_c`: diagonal down-left (top / top-right).
pub fn i4x4_luma_pred_ddl(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 8] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let t = [
            g(-s),
            g(1 - s),
            g(2 - s),
            g(3 - s),
            g(4 - s),
            g(5 - s),
            g(6 - s),
            g(7 - s),
        ];
        [
            ((2 + t[0] + t[2] + (t[1] << 1)) >> 2) as u8,
            ((2 + t[1] + t[3] + (t[2] << 1)) >> 2) as u8,
            ((2 + t[2] + t[4] + (t[3] << 1)) >> 2) as u8,
            ((2 + t[3] + t[5] + (t[4] << 1)) >> 2) as u8,
            ((2 + t[4] + t[6] + (t[5] << 1)) >> 2) as u8,
            ((2 + t[5] + t[7] + (t[6] << 1)) >> 2) as u8,
            ((2 + t[6] + t[7] + (t[7] << 1)) >> 2) as u8,
            0,
        ]
    };
    write_rows4(plane, o, s, &list, [0, 1, 2, 3]);
}

/// Port of `WelsI4x4LumaPredDDLTop_c`: diagonal down-left (top only, no top-right).
pub fn i4x4_luma_pred_ddl_top(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 8] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let t0 = g(-s);
        let t1 = g(1 - s);
        let t2 = g(2 - s);
        let t3 = g(3 - s);
        let t01 = 1 + t0 + t1;
        let t12 = 1 + t1 + t2;
        let t23 = 1 + t2 + t3;
        let t33 = 1 + (t3 << 1);
        let d0 = ((t01 + t12) >> 2) as u8;
        let d1 = ((t12 + t23) >> 2) as u8;
        let d2 = ((t23 + t33) >> 2) as u8;
        let d3 = (t33 >> 1) as u8;
        [d0, d1, d2, d3, d3, d3, d3, d3]
    };
    write_rows4(plane, o, s, &list, [0, 1, 2, 3]);
}

/// Port of `WelsI4x4LumaPredDDR_c`: diagonal down-right.
pub fn i4x4_luma_pred_ddr(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 8] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let lt = g(-s - 1);
        let l0 = g(-1);
        let l1 = g(-1 + s);
        let l2 = g(-1 + 2 * s);
        let l3 = g(-1 + 3 * s);
        let t0 = g(-s);
        let t1 = g(1 - s);
        let t2 = g(2 - s);
        let t3 = g(3 - s);
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
        [d6, d5, d4, d0, d1, d2, d3, 0]
    };
    write_rows4(plane, o, s, &list, [3, 2, 1, 0]);
}

/// Port of `WelsI4x4LumaPredVL_c`: vertical-left (top / top-right).
pub fn i4x4_luma_pred_vl(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 10] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let t0 = g(-s);
        let t1 = g(1 - s);
        let t2 = g(2 - s);
        let t3 = g(3 - s);
        let t4 = g(4 - s);
        let t5 = g(5 - s);
        let t6 = g(6 - s);
        let t01 = 1 + t0 + t1;
        let t12 = 1 + t1 + t2;
        let t23 = 1 + t2 + t3;
        let t34 = 1 + t3 + t4;
        let t45 = 1 + t4 + t5;
        let t56 = 1 + t5 + t6;
        [
            (t01 >> 1) as u8,
            (t12 >> 1) as u8,
            (t23 >> 1) as u8,
            (t34 >> 1) as u8,
            (t45 >> 1) as u8,
            ((t01 + t12) >> 2) as u8,
            ((t12 + t23) >> 2) as u8,
            ((t23 + t34) >> 2) as u8,
            ((t34 + t45) >> 2) as u8,
            ((t45 + t56) >> 2) as u8,
        ]
    };
    write_rows4_10(plane, o, s, &list, [0, 5, 1, 6]);
}

/// Port of `WelsI4x4LumaPredVLTop_c`: vertical-left (top only).
pub fn i4x4_luma_pred_vl_top(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 10] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let t0 = g(-s);
        let t1 = g(1 - s);
        let t2 = g(2 - s);
        let t3 = g(3 - s);
        let t01 = 1 + t0 + t1;
        let t12 = 1 + t1 + t2;
        let t23 = 1 + t2 + t3;
        let t33 = 1 + (t3 << 1);
        let v0 = (t01 >> 1) as u8;
        let v1 = (t12 >> 1) as u8;
        let v2 = (t23 >> 1) as u8;
        let v3 = (t33 >> 1) as u8;
        let v4 = ((t01 + t12) >> 2) as u8;
        let v5 = ((t12 + t23) >> 2) as u8;
        let v6 = ((t23 + t33) >> 2) as u8;
        [v0, v1, v2, v3, v3, v4, v5, v6, v3, v3]
    };
    write_rows4_10(plane, o, s, &list, [0, 5, 1, 6]);
}

/// Port of `WelsI4x4LumaPredVR_c`: vertical-right.
pub fn i4x4_luma_pred_vr(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 10] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let lt = g(-s - 1);
        let l0 = g(-1);
        let l1 = g(s - 1);
        let l2 = g(2 * s - 1);
        let t0 = g(-s);
        let t1 = g(1 - s);
        let t2 = g(2 - s);
        let t3 = g(3 - s);
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
        [v8, v0, v1, v2, v3, v9, v4, v5, v6, v7]
    };
    write_rows4_10(plane, o, s, &list, [1, 6, 0, 5]);
}

/// Port of `WelsI4x4LumaPredHU_c`: horizontal-up.
pub fn i4x4_luma_pred_hu(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 10] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let l0 = g(-1);
        let l1 = g(s - 1);
        let l2 = g(2 * s - 1);
        let l3 = g(3 * s - 1);
        let l01 = 1 + l0 + l1;
        let l12 = 1 + l1 + l2;
        let l23 = 1 + l2 + l3;
        let h0 = (l01 >> 1) as u8;
        let h1 = ((l01 + l12) >> 2) as u8;
        let h2 = (l12 >> 1) as u8;
        let h3 = ((l12 + l23) >> 2) as u8;
        let h4 = (l23 >> 1) as u8;
        let h5 = ((1 + l23 + (l3 << 1)) >> 2) as u8;
        let l3u = l3 as u8;
        [h0, h1, h2, h3, h4, h5, l3u, l3u, l3u, l3u]
    };
    write_rows4_10(plane, o, s, &list, [0, 2, 4, 6]);
}

/// Port of `WelsI4x4LumaPredHD_c`: horizontal-down.
pub fn i4x4_luma_pred_hd(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let list: [u8; 10] = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let lt = g(-s - 1);
        let l0 = g(-1);
        let l1 = g(-1 + s);
        let l2 = g(-1 + 2 * s);
        let l3 = g(-1 + 3 * s);
        let t0 = g(-s);
        let t1 = g(-s + 1);
        let t2 = g(-s + 2);
        let tl0 = 1 + lt + l0;
        let lt0 = 1 + lt + t0;
        let t01 = 1 + t0 + t1;
        let t12 = 1 + t1 + t2;
        let l01 = 1 + l0 + l1;
        let l12 = 1 + l1 + l2;
        let l23 = 1 + l2 + l3;
        let h0 = (tl0 >> 1) as u8;
        let h1 = ((tl0 + lt0) >> 2) as u8;
        let h2 = ((lt0 + t01) >> 2) as u8;
        let h3 = ((t01 + t12) >> 2) as u8;
        let h4 = (l01 >> 1) as u8;
        let h5 = ((tl0 + l01) >> 2) as u8;
        let h6 = (l12 >> 1) as u8;
        let h7 = ((l01 + l12) >> 2) as u8;
        let h8 = (l23 >> 1) as u8;
        let h9 = ((l12 + l23) >> 2) as u8;
        [h8, h9, h6, h7, h4, h5, h0, h1, h2, h3]
    };
    write_rows4_10(plane, o, s, &list, [6, 4, 2, 0]);
}

#[inline]
fn write_rows4_10(plane: &mut [u8], o: isize, s: isize, list: &[u8; 10], starts: [usize; 4]) {
    for (y, &k) in starts.iter().enumerate() {
        for x in 0..4 {
            plane[(o + (y as isize) * s + x as isize) as usize] = list[k + x];
        }
    }
}

// ===========================================================================
// Luma 8x8 (reference-sample smoothing per clause 8.3.2.2.1)
// ===========================================================================

#[inline]
fn put8(plane: &mut [u8], o: isize, s: isize, y: i32, x: i32, v: i32) {
    plane[(o + (y as isize) * s + x as isize) as usize] = v as u8;
}

/// Filtered top samples, "full" profile (top-right available): indices 0..=15.
/// Used by DDL / VL. `b_tl` selects the left-edge filter; the right edge always
/// uses the 3-tap boundary form.
#[inline]
fn filt_top_full(plane: &[u8], o: isize, s: isize, b_tl: bool) -> [i32; 16] {
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut t = [0i32; 16];
    t[0] = if b_tl {
        (g(-s - 1) + (g(-s) << 1) + g(1 - s) + 2) >> 2
    } else {
        (g(-s) * 3 + g(1 - s) + 2) >> 2
    };
    for i in 1..15 {
        let id = i as isize;
        t[i] = (g(id - 1 - s) + (g(id - s) << 1) + g(id + 1 - s) + 2) >> 2;
    }
    t[15] = (g(14 - s) + g(15 - s) * 3 + 2) >> 2;
    t
}

/// Filtered top samples, "top-only" profile (top-right unavailable): indices
/// 8..=15 are replicated from sample 7. Used by DDLTop / VLTop.
#[inline]
fn filt_top_only(plane: &[u8], o: isize, s: isize, b_tl: bool) -> [i32; 16] {
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut t = [0i32; 16];
    t[0] = if b_tl {
        (g(-s - 1) + (g(-s) << 1) + g(1 - s) + 2) >> 2
    } else {
        (g(-s) * 3 + g(1 - s) + 2) >> 2
    };
    for i in 1..7 {
        let id = i as isize;
        t[i] = (g(id - 1 - s) + (g(id - s) << 1) + g(id + 1 - s) + 2) >> 2;
    }
    t[7] = (g(6 - s) + g(7 - s) * 3 + 2) >> 2;
    for i in 8..16 {
        t[i] = g(7 - s);
    }
    t
}

/// Filtered top samples, "boundary at 7" profile: indices 0..=7, with `b_tl`
/// left edge and `b_tr` right edge. Used by V / Dc / DcTop.
#[inline]
fn filt_top8(plane: &[u8], o: isize, s: isize, b_tl: bool, b_tr: bool) -> [i32; 8] {
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut t = [0i32; 8];
    t[0] = if b_tl {
        (g(-s - 1) + (g(-s) << 1) + g(1 - s) + 2) >> 2
    } else {
        (g(-s) * 3 + g(1 - s) + 2) >> 2
    };
    for i in 1..7 {
        let id = i as isize;
        t[i] = (g(id - 1 - s) + (g(id - s) << 1) + g(id + 1 - s) + 2) >> 2;
    }
    t[7] = if b_tr {
        (g(6 - s) + (g(7 - s) << 1) + g(8 - s) + 2) >> 2
    } else {
        (g(6 - s) + g(7 - s) * 3 + 2) >> 2
    };
    t
}

/// As `filt_top8` but the left edge always uses the 3-tap (top-left available)
/// form. Used by DDR / VR / HD, where top-left is guaranteed available.
#[inline]
fn filt_top8_tl(plane: &[u8], o: isize, s: isize, b_tr: bool) -> [i32; 8] {
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut t = [0i32; 8];
    t[0] = (g(-s - 1) + (g(-s) << 1) + g(1 - s) + 2) >> 2;
    for i in 1..7 {
        let id = i as isize;
        t[i] = (g(id - 1 - s) + (g(id - s) << 1) + g(id + 1 - s) + 2) >> 2;
    }
    t[7] = if b_tr {
        (g(6 - s) + (g(7 - s) << 1) + g(8 - s) + 2) >> 2
    } else {
        (g(6 - s) + g(7 - s) * 3 + 2) >> 2
    };
    t
}

/// Filtered left samples, with `b_tl` top edge. Used by H / Dc / DcLeft / HU.
#[inline]
fn filt_left8(plane: &[u8], o: isize, s: isize, b_tl: bool) -> [i32; 8] {
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut l = [0i32; 8];
    l[0] = if b_tl {
        (g(-s - 1) + (g(-1) << 1) + g(-1 + s) + 2) >> 2
    } else {
        (g(-1) * 3 + g(-1 + s) + 2) >> 2
    };
    for i in 1..7 {
        let id = i as isize;
        l[i] = (g(-1 + (id - 1) * s) + (g(-1 + id * s) << 1) + g(-1 + (id + 1) * s) + 2) >> 2;
    }
    l[7] = (g(-1 + 6 * s) + g(-1 + 7 * s) * 3 + 2) >> 2;
    l
}

/// As `filt_left8` but the top edge always uses the 3-tap form. Used by
/// DDR / VR / HD.
#[inline]
fn filt_left8_tl(plane: &[u8], o: isize, s: isize) -> [i32; 8] {
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut l = [0i32; 8];
    l[0] = (g(-s - 1) + (g(-1) << 1) + g(-1 + s) + 2) >> 2;
    for i in 1..7 {
        let id = i as isize;
        l[i] = (g(-1 + (id - 1) * s) + (g(-1 + id * s) << 1) + g(-1 + (id + 1) * s) + 2) >> 2;
    }
    l[7] = (g(-1 + 6 * s) + g(-1 + 7 * s) * 3 + 2) >> 2;
    l
}

/// Filtered top-left corner sample.
#[inline]
fn filt_tl(plane: &[u8], o: isize, s: isize) -> i32 {
    let g = |d: isize| plane[(o + d) as usize] as i32;
    (g(-1) + (g(-s - 1) << 1) + g(-s) + 2) >> 2
}

/// Port of `WelsI8x8LumaPredV_c`.
pub fn i8x8_luma_pred_v(plane: &mut [u8], offset: usize, stride: usize, b_tl: bool, b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top8(plane, o, s, b_tl, b_tr);
    for y in 0..8 {
        for x in 0..8 {
            put8(plane, o, s, y, x, t[x as usize]);
        }
    }
}

/// Port of `WelsI8x8LumaPredH_c`.
pub fn i8x8_luma_pred_h(plane: &mut [u8], offset: usize, stride: usize, b_tl: bool, _b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let l = filt_left8(plane, o, s, b_tl);
    for y in 0..8 {
        for x in 0..8 {
            put8(plane, o, s, y, x, l[y as usize]);
        }
    }
}

/// Port of `WelsI8x8LumaPredDc_c`.
pub fn i8x8_luma_pred_dc(plane: &mut [u8], offset: usize, stride: usize, b_tl: bool, b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top8(plane, o, s, b_tl, b_tr);
    let l = filt_left8(plane, o, s, b_tl);
    let mut total: i32 = 0;
    for i in 0..8 {
        total += t[i] + l[i];
    }
    let mean = ((total + 8) >> 4) as i32;
    for y in 0..8 {
        for x in 0..8 {
            put8(plane, o, s, y, x, mean);
        }
    }
}

/// Port of `WelsI8x8LumaPredDcLeft_c`.
pub fn i8x8_luma_pred_dc_left(
    plane: &mut [u8],
    offset: usize,
    stride: usize,
    b_tl: bool,
    _b_tr: bool,
) {
    let o = offset as isize;
    let s = stride as isize;
    let l = filt_left8(plane, o, s, b_tl);
    let total: i32 = l.iter().sum();
    let mean = (total + 4) >> 3;
    for y in 0..8 {
        for x in 0..8 {
            put8(plane, o, s, y, x, mean);
        }
    }
}

/// Port of `WelsI8x8LumaPredDcTop_c`.
pub fn i8x8_luma_pred_dc_top(
    plane: &mut [u8],
    offset: usize,
    stride: usize,
    b_tl: bool,
    b_tr: bool,
) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top8(plane, o, s, b_tl, b_tr);
    let total: i32 = t.iter().sum();
    let mean = (total + 4) >> 3;
    for y in 0..8 {
        for x in 0..8 {
            put8(plane, o, s, y, x, mean);
        }
    }
}

/// Port of `WelsI8x8LumaPredDcNA_c`.
pub fn i8x8_luma_pred_dc_na(
    plane: &mut [u8],
    offset: usize,
    stride: usize,
    _b_tl: bool,
    _b_tr: bool,
) {
    let o = offset as isize;
    let s = stride as isize;
    for y in 0..8 {
        for x in 0..8 {
            put8(plane, o, s, y, x, 0x80);
        }
    }
}

/// Port of `WelsI8x8LumaPredDDL_c` (top and top-right available).
pub fn i8x8_luma_pred_ddl(plane: &mut [u8], offset: usize, stride: usize, b_tl: bool, _b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top_full(plane, o, s, b_tl);
    for i in 0..8 {
        for j in 0..8 {
            let v = if i == 7 && j == 7 {
                (t[14] + 3 * t[15] + 2) >> 2
            } else {
                let k = (i + j) as usize;
                (t[k] + (t[k + 1] << 1) + t[k + 2] + 2) >> 2
            };
            put8(plane, o, s, i, j, v);
        }
    }
}

/// Port of `WelsI8x8LumaPredDDLTop_c` (top available, top-right unavailable).
pub fn i8x8_luma_pred_ddl_top(
    plane: &mut [u8],
    offset: usize,
    stride: usize,
    b_tl: bool,
    _b_tr: bool,
) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top_only(plane, o, s, b_tl);
    for i in 0..8 {
        for j in 0..8 {
            let v = if i == 7 && j == 7 {
                (t[14] + 3 * t[15] + 2) >> 2
            } else {
                let k = (i + j) as usize;
                (t[k] + (t[k + 1] << 1) + t[k + 2] + 2) >> 2
            };
            put8(plane, o, s, i, j, v);
        }
    }
}

/// Port of `WelsI8x8LumaPredDDR_c` (top-left, top, left all available).
pub fn i8x8_luma_pred_ddr(plane: &mut [u8], offset: usize, stride: usize, _b_tl: bool, b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top8_tl(plane, o, s, b_tr);
    let l = filt_left8_tl(plane, o, s);
    let tl = filt_tl(plane, o, s);
    for i in 0..8 {
        // 8-98, x < y-1
        let mut j = 0;
        while j < i - 1 {
            let v = (l[(i - j - 2) as usize] + (l[(i - j - 1) as usize] << 1) + l[(i - j) as usize]
                + 2)
                >> 2;
            put8(plane, o, s, i, j, v);
            j += 1;
        }
        // 8-98, x == y-1
        if i >= 1 {
            let j = i - 1;
            let v = (tl + (l[0] << 1) + l[1] + 2) >> 2;
            put8(plane, o, s, i, j, v);
        }
        // 8-99, x == y
        {
            let j = i;
            let v = (t[0] + (tl << 1) + l[0] + 2) >> 2;
            put8(plane, o, s, i, j, v);
        }
        // 8-97, x == y+1
        if i < 7 {
            let j = i + 1;
            let v = (tl + (t[0] << 1) + t[1] + 2) >> 2;
            put8(plane, o, s, i, j, v);
        }
        // 8-97, x > y+1
        let mut j = i + 2;
        while j < 8 {
            let v = (t[(j - i - 2) as usize] + (t[(j - i - 1) as usize] << 1) + t[(j - i) as usize]
                + 2)
                >> 2;
            put8(plane, o, s, i, j, v);
            j += 1;
        }
    }
}

/// Port of `WelsI8x8LumaPredVL_c` (top and top-right available).
pub fn i8x8_luma_pred_vl(plane: &mut [u8], offset: usize, stride: usize, b_tl: bool, _b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top_full(plane, o, s, b_tl);
    for i in 0..8 {
        for j in 0..8 {
            let k = (j + (i >> 1)) as usize;
            let v = if i & 1 == 0 {
                (t[k] + t[k + 1] + 1) >> 1
            } else {
                (t[k] + (t[k + 1] << 1) + t[k + 2] + 2) >> 2
            };
            put8(plane, o, s, i, j, v);
        }
    }
}

/// Port of `WelsI8x8LumaPredVLTop_c` (top available, top-right unavailable).
pub fn i8x8_luma_pred_vl_top(
    plane: &mut [u8],
    offset: usize,
    stride: usize,
    b_tl: bool,
    _b_tr: bool,
) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top_only(plane, o, s, b_tl);
    for i in 0..8 {
        for j in 0..8 {
            let k = (j + (i >> 1)) as usize;
            let v = if i & 1 == 0 {
                (t[k] + t[k + 1] + 1) >> 1
            } else {
                (t[k] + (t[k + 1] << 1) + t[k + 2] + 2) >> 2
            };
            put8(plane, o, s, i, j, v);
        }
    }
}

/// Port of `WelsI8x8LumaPredVR_c` (top-left, top, left all available).
pub fn i8x8_luma_pred_vr(plane: &mut [u8], offset: usize, stride: usize, _b_tl: bool, b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top8_tl(plane, o, s, b_tr);
    let l = filt_left8_tl(plane, o, s);
    let tl = filt_tl(plane, o, s);
    for i in 0..8 {
        for j in 0..8 {
            let z = (j << 1) - i; // 2x - y
            let d = j - (i >> 1);
            let v = if z >= 0 {
                if z & 1 == 0 {
                    if d > 0 {
                        (t[(d - 1) as usize] + t[d as usize] + 1) >> 1
                    } else {
                        (tl + t[0] + 1) >> 1
                    }
                } else if d > 1 {
                    (t[(d - 2) as usize] + (t[(d - 1) as usize] << 1) + t[d as usize] + 2) >> 2
                } else {
                    (tl + (t[0] << 1) + t[1] + 2) >> 2
                }
            } else if z == -1 {
                (l[0] + (tl << 1) + t[0] + 2) >> 2
            } else if z < -2 {
                (l[(-z - 1) as usize] + (l[(-z - 2) as usize] << 1) + l[(-z - 3) as usize] + 2) >> 2
            } else {
                (l[1] + (l[0] << 1) + tl + 2) >> 2
            };
            put8(plane, o, s, i, j, v);
        }
    }
}

/// Port of `WelsI8x8LumaPredHU_c`.
pub fn i8x8_luma_pred_hu(plane: &mut [u8], offset: usize, stride: usize, b_tl: bool, _b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let l = filt_left8(plane, o, s, b_tl);
    for i in 0..8 {
        for j in 0..8 {
            let z = j + (i << 1); // x + 2y
            let v = if z < 13 {
                let k = (z >> 1) as usize;
                if z & 1 == 0 {
                    (l[k] + l[k + 1] + 1) >> 1
                } else {
                    (l[k] + (l[k + 1] << 1) + l[k + 2] + 2) >> 2
                }
            } else if z == 13 {
                (l[6] + 3 * l[7] + 2) >> 2
            } else {
                l[7]
            };
            put8(plane, o, s, i, j, v);
        }
    }
}

/// Port of `WelsI8x8LumaPredHD_c` (top-left, top, left all available).
pub fn i8x8_luma_pred_hd(plane: &mut [u8], offset: usize, stride: usize, _b_tl: bool, b_tr: bool) {
    let o = offset as isize;
    let s = stride as isize;
    let t = filt_top8_tl(plane, o, s, b_tr);
    let l = filt_left8_tl(plane, o, s);
    let tl = filt_tl(plane, o, s);
    for i in 0..8 {
        for j in 0..8 {
            let z = (i << 1) - j; // 2y - x
            let d = i - (j >> 1);
            let v = if z >= 0 {
                if z & 1 == 0 {
                    if d == 0 {
                        (tl + l[0] + 1) >> 1
                    } else {
                        (l[(d - 1) as usize] + l[d as usize] + 1) >> 1
                    }
                } else if d == 1 {
                    (tl + (l[0] << 1) + l[1] + 2) >> 2
                } else {
                    (l[(d - 2) as usize] + (l[(d - 1) as usize] << 1) + l[d as usize] + 2) >> 2
                }
            } else if z == -1 {
                (l[0] + (tl << 1) + t[0] + 2) >> 2
            } else if z < -2 {
                (t[(-z - 1) as usize] + (t[(-z - 2) as usize] << 1) + t[(-z - 3) as usize] + 2) >> 2
            } else {
                (t[1] + (t[0] << 1) + tl + 2) >> 2
            };
            put8(plane, o, s, i, j, v);
        }
    }
}

// ===========================================================================
// Chroma 8x8
// ===========================================================================

/// Port of `WelsIChromaPredV_c`: copy the (unfiltered) top row down.
pub fn i_chroma_pred_v(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let mut top = [0u8; 8];
    for x in 0..8 {
        top[x] = plane[(o - s + x as isize) as usize];
    }
    for y in 0..8 {
        for x in 0..8 {
            plane[(o + (y as isize) * s + x as isize) as usize] = top[x];
        }
    }
}

/// Port of `WelsIChromaPredH_c`: copy the (unfiltered) left column across.
pub fn i_chroma_pred_h(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    for y in 0..8 {
        let l = plane[(o + (y as isize) * s - 1) as usize];
        for x in 0..8 {
            plane[(o + (y as isize) * s + x as isize) as usize] = l;
        }
    }
}

/// Port of `WelsIChromaPredPlane_c`.
pub fn i_chroma_pred_plane(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let (a, b, c) = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let mut h = 0i32;
        let mut v = 0i32;
        for i in 0..4 {
            let i = i as isize;
            h += ((i + 1) as i32) * (g(-s + 4 + i) - g(-s + 2 - i));
            v += ((i + 1) as i32) * (g(-1 + (4 + i) * s) - g(-1 + (2 - i) * s));
        }
        let a = (g(-1 + 7 * s) + g(-s + 7)) << 4;
        let b = (17 * h + 16) >> 5;
        let c = (17 * v + 16) >> 5;
        (a, b, c)
    };
    for i in 0..8 {
        for j in 0..8 {
            let tmp = (a + b * (j as i32 - 3) + c * (i as i32 - 3) + 16) >> 5;
            plane[(o + (i as isize) * s + j as isize) as usize] = clip1(tmp);
        }
    }
}

/// Port of `WelsIChromaPredDc_c`: per-quadrant DC.
pub fn i_chroma_pred_dc(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let m1 = ((g(-s) + g(1 - s) + g(2 - s) + g(3 - s)
        + g(-1) + g(s - 1) + g(2 * s - 1) + g(3 * s - 1)
        + 4)
        >> 3) as u8;
    let sum2 = g(4 - s) + g(5 - s) + g(6 - s) + g(7 - s);
    let sum3 = g(4 * s - 1) + g(5 * s - 1) + g(6 * s - 1) + g(7 * s - 1);
    let m2 = ((sum2 + 2) >> 2) as u8;
    let m3 = ((sum3 + 2) >> 2) as u8;
    let m4 = ((sum2 + sum3 + 4) >> 3) as u8;
    let up = [m1, m1, m1, m1, m2, m2, m2, m2];
    let down = [m3, m3, m3, m3, m4, m4, m4, m4];
    for y in 0..8 {
        let row = if y < 4 { &up } else { &down };
        for x in 0..8 {
            plane[(o + (y as isize) * s + x as isize) as usize] = row[x];
        }
    }
}

/// Port of `WelsIChromaPredDcLeft_c`.
pub fn i_chroma_pred_dc_left(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let up = ((g(-1) + g(-1 + s) + g(-1 + 2 * s) + g(-1 + 3 * s) + 2) >> 2) as u8;
    let down = ((g(-1 + 4 * s) + g(-1 + 5 * s) + g(-1 + 6 * s) + g(-1 + 7 * s) + 2) >> 2) as u8;
    for y in 0..8 {
        let v = if y < 4 { up } else { down };
        for x in 0..8 {
            plane[(o + (y as isize) * s + x as isize) as usize] = v;
        }
    }
}

/// Port of `WelsIChromaPredDcTop_c`.
pub fn i_chroma_pred_dc_top(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let m1 = ((g(-s) + g(1 - s) + g(2 - s) + g(3 - s) + 2) >> 2) as u8;
    let m2 = ((g(4 - s) + g(5 - s) + g(6 - s) + g(7 - s) + 2) >> 2) as u8;
    let row = [m1, m1, m1, m1, m2, m2, m2, m2];
    for y in 0..8 {
        for x in 0..8 {
            plane[(o + (y as isize) * s + x as isize) as usize] = row[x];
        }
    }
}

/// Port of `WelsIChromaPredDcNA_c`.
pub fn i_chroma_pred_dc_na(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    for y in 0..8 {
        for x in 0..8 {
            plane[(o + (y as isize) * s + x as isize) as usize] = 0x80;
        }
    }
}

// ===========================================================================
// Luma 16x16
// ===========================================================================

/// Port of `WelsI16x16LumaPredV_c`: copy the top row down (in-plane variant).
pub fn i16x16_luma_pred_v(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let mut top = [0u8; 16];
    for x in 0..16 {
        top[x] = plane[(o - s + x as isize) as usize];
    }
    for y in 0..16 {
        for x in 0..16 {
            plane[(o + (y as isize) * s + x as isize) as usize] = top[x];
        }
    }
}

/// Port of `WelsI16x16LumaPredH_c`: copy the left column across (in-plane variant).
pub fn i16x16_luma_pred_h(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    for y in 0..16 {
        let l = plane[(o + (y as isize) * s - 1) as usize];
        for x in 0..16 {
            plane[(o + (y as isize) * s + x as isize) as usize] = l;
        }
    }
}

/// Port of `WelsI16x16LumaPredPlane_c`.
pub fn i16x16_luma_pred_plane(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let (a, b, c) = {
        let g = |d: isize| plane[(o + d) as usize] as i32;
        let mut h = 0i32;
        let mut v = 0i32;
        for i in 0..8 {
            let i = i as isize;
            h += ((i + 1) as i32) * (g(-s + 8 + i) - g(-s + 6 - i));
            v += ((i + 1) as i32) * (g(-1 + (8 + i) * s) - g(-1 + (6 - i) * s));
        }
        let a = (g(-1 + 15 * s) + g(-s + 15)) << 4;
        let b = (5 * h + 32) >> 6;
        let c = (5 * v + 32) >> 6;
        (a, b, c)
    };
    for i in 0..16 {
        for j in 0..16 {
            let tmp = (a + b * (j as i32 - 7) + c * (i as i32 - 7) + 16) >> 5;
            plane[(o + (i as isize) * s + j as isize) as usize] = clip1(tmp);
        }
    }
}

/// Port of `WelsI16x16LumaPredDc_c`: DC from top row and left column.
pub fn i16x16_luma_pred_dc(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut sum = 0i32;
    for i in 0..16 {
        let i = i as isize;
        sum += g(-1 + i * s) + g(-s + i);
    }
    let mean = ((16 + sum) >> 5) as u8;
    fill16(plane, o, s, mean);
}

/// Port of `WelsI16x16LumaPredDcTop_c`.
pub fn i16x16_luma_pred_dc_top(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut sum = 0i32;
    for i in 0..16 {
        sum += g(-s + i as isize);
    }
    let mean = ((8 + sum) >> 4) as u8;
    fill16(plane, o, s, mean);
}

/// Port of `WelsI16x16LumaPredDcLeft_c`.
pub fn i16x16_luma_pred_dc_left(plane: &mut [u8], offset: usize, stride: usize) {
    let o = offset as isize;
    let s = stride as isize;
    let g = |d: isize| plane[(o + d) as usize] as i32;
    let mut sum = 0i32;
    for i in 0..16 {
        sum += g(-1 + (i as isize) * s);
    }
    let mean = ((8 + sum) >> 4) as u8;
    fill16(plane, o, s, mean);
}

/// Port of `WelsI16x16LumaPredDcNA_c`.
pub fn i16x16_luma_pred_dc_na(plane: &mut [u8], offset: usize, stride: usize) {
    fill16(plane, offset as isize, stride as isize, 0x80);
}

#[inline]
fn fill16(plane: &mut [u8], o: isize, s: isize, v: u8) {
    for y in 0..16 {
        for x in 0..16 {
            plane[(o + (y as isize) * s + x as isize) as usize] = v;
        }
    }
}

// ===========================================================================
// Luma 16x16, common "pRef" variants (intra_pred_common.cpp).
// These read neighbors from a separate reference plane `pref` (strided by
// `stride`) and write into a packed 16-wide destination block.
// ===========================================================================

/// Port of `WelsI16x16LumaPredV_c` (common: `pPred`, `pRef`, `kiStride`).
/// Reads the 16 top samples from `pref` at `pref_off - stride`, writing 16 rows
/// of a packed (stride-16) destination starting at `pred_off`.
pub fn i16x16_luma_pred_v_pref(
    pred: &mut [u8],
    pred_off: usize,
    pref: &[u8],
    pref_off: usize,
    stride: usize,
) {
    let ro = pref_off as isize;
    let s = stride as isize;
    let mut top = [0u8; 16];
    for x in 0..16 {
        top[x] = pref[(ro - s + x as isize) as usize];
    }
    for r in 0..16 {
        for x in 0..16 {
            pred[pred_off + r * 16 + x] = top[x];
        }
    }
}

/// Port of `WelsI16x16LumaPredH_c` (common: `pPred`, `pRef`, `kiStride`).
/// Row `r` is filled with `pref[pref_off + r*stride - 1]`, written to a packed
/// (stride-16) destination starting at `pred_off`.
pub fn i16x16_luma_pred_h_pref(
    pred: &mut [u8],
    pred_off: usize,
    pref: &[u8],
    pref_off: usize,
    stride: usize,
) {
    let ro = pref_off as isize;
    let s = stride as isize;
    for r in 0..16 {
        let v = pref[(ro + (r as isize) * s - 1) as usize];
        for x in 0..16 {
            pred[pred_off + r * 16 + x] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // Deterministic LCG (same constants as src/dsp/transform.rs tests).
    fn lcg() -> impl FnMut() -> u8 {
        let mut state = 0x1234_5678u32;
        move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        }
    }

    #[inline]
    fn g(p: &[u8], o: usize, d: isize) -> i32 {
        p[(o as isize + d) as usize] as i32
    }

    #[inline]
    fn st(p: &mut [u8], o: isize, s: isize, y: i32, x: i32, v: i32) {
        p[(o + (y as isize) * s + x as isize) as usize] = v as u8;
    }

    // ---- Runners ---------------------------------------------------------

    fn run4x4(k: impl Fn(&mut [u8], usize, usize), r: impl Fn(&mut [u8], usize, usize)) {
        let stride = 32usize;
        let off = 4 * stride + 4;
        let mut rnd = lcg();
        for _ in 0..1000 {
            let mut a = vec![0u8; 12 * stride];
            for i in 0..12 {
                a[3 * stride + i] = rnd();
                a[i * stride + 3] = rnd();
            }
            let mut b = a.clone();
            k(&mut a, off, stride);
            r(&mut b, off, stride);
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(
                        a[(i + 4) * stride + j + 4],
                        b[(i + 4) * stride + j + 4],
                        "mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    fn run_big(sz: usize, k: impl Fn(&mut [u8], usize, usize), r: impl Fn(&mut [u8], usize, usize)) {
        let stride = 32usize;
        let off = 2 * stride;
        let mut rnd = lcg();
        for _ in 0..1000 {
            let mut a = vec![0u8; 18 * stride];
            for i in 0..17 {
                a[stride + i] = rnd();
                a[(i + 1) * stride - 1] = rnd();
            }
            let mut b = a.clone();
            k(&mut a, off, stride);
            r(&mut b, off, stride);
            for i in 0..sz {
                for j in 0..sz {
                    assert_eq!(
                        a[(2 + i) * stride + j],
                        b[(2 + i) * stride + j],
                        "mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    fn run8x8(
        k: impl Fn(&mut [u8], usize, usize, bool, bool),
        r: impl Fn(&mut [u8], usize, usize, bool, bool),
    ) {
        for &(tl, tr) in &[(false, false), (false, true), (true, false), (true, true)] {
            run_big(8, |p, o, s| k(p, o, s, tl, tr), |p, o, s| r(p, o, s, tl, tr));
        }
    }

    // ---- 4x4 references --------------------------------------------------

    fn r4_v(p: &mut [u8], o: usize, s: usize) {
        for y in 0..4 {
            for x in 0..4 {
                p[o + y * s + x] = p[o - s + x];
            }
        }
    }
    fn r4_h(p: &mut [u8], o: usize, s: usize) {
        for y in 0..4 {
            let v = p[o + y * s - 1];
            for x in 0..4 {
                p[o + y * s + x] = v;
            }
        }
    }
    fn r4_dc(p: &mut [u8], o: usize, s: usize) {
        let mut sum = 4i32;
        for i in 0..4 {
            sum += g(p, o, -1 + (i as isize) * s as isize) + g(p, o, (i as isize) - s as isize);
        }
        let m = (sum >> 3) as u8;
        for y in 0..4 {
            for x in 0..4 {
                p[o + y * s + x] = m;
            }
        }
    }
    fn r4_dc_left(p: &mut [u8], o: usize, s: usize) {
        let mut sum = 2i32;
        for i in 0..4 {
            sum += g(p, o, -1 + (i as isize) * s as isize);
        }
        let m = (sum >> 2) as u8;
        for y in 0..4 {
            for x in 0..4 {
                p[o + y * s + x] = m;
            }
        }
    }
    fn r4_dc_top(p: &mut [u8], o: usize, s: usize) {
        let mut sum = 2i32;
        for i in 0..4 {
            sum += g(p, o, (i as isize) - s as isize);
        }
        let m = (sum >> 2) as u8;
        for y in 0..4 {
            for x in 0..4 {
                p[o + y * s + x] = m;
            }
        }
    }
    fn r4_dc_na(p: &mut [u8], o: usize, s: usize) {
        for y in 0..4 {
            for x in 0..4 {
                p[o + y * s + x] = 128;
            }
        }
    }

    // 4x4 directional refs, derived from the H.264 spec sample arrays
    // (independent of the optimized kernel's list-permutation form).
    fn top4(p: &[u8], o: usize, s: isize) -> [i32; 8] {
        let mut t = [0i32; 8];
        for x in 0..8 {
            t[x] = g(p, o, -s + x as isize);
        }
        t
    }
    fn left4(p: &[u8], o: usize, s: isize) -> [i32; 4] {
        let mut l = [0i32; 4];
        for y in 0..4 {
            l[y] = g(p, o, -1 + (y as isize) * s);
        }
        l
    }

    fn r4_ddl(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let t = top4(p, o, si);
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let k = (x + y) as usize;
                let v = if k == 6 {
                    (t[6] + 3 * t[7] + 2) >> 2
                } else {
                    (t[k] + 2 * t[k + 1] + t[k + 2] + 2) >> 2
                };
                st(p, oi, si, y, x, v);
            }
        }
    }
    fn r4_ddl_top(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let raw = top4(p, o, si);
        let t = [raw[0], raw[1], raw[2], raw[3], raw[3], raw[3], raw[3], raw[3]];
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let k = (x + y) as usize;
                let v = if k == 6 {
                    (t[6] + 3 * t[7] + 2) >> 2
                } else {
                    (t[k] + 2 * t[k + 1] + t[k + 2] + 2) >> 2
                };
                st(p, oi, si, y, x, v);
            }
        }
    }
    fn r4_ddr(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let t = top4(p, o, si);
        let l = left4(p, o, si);
        let lt = g(p, o, -si - 1);
        let gt = |k: i32| if k < 0 { lt } else { t[k as usize] };
        let gl = |k: i32| if k < 0 { lt } else { l[k as usize] };
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let v = if x > y {
                    (gt(x - y - 2) + 2 * gt(x - y - 1) + gt(x - y) + 2) >> 2
                } else if x < y {
                    (gl(y - x - 2) + 2 * gl(y - x - 1) + gl(y - x) + 2) >> 2
                } else {
                    (t[0] + 2 * lt + l[0] + 2) >> 2
                };
                st(p, oi, si, y, x, v);
            }
        }
    }
    fn r4_vr(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let t = top4(p, o, si);
        let l = left4(p, o, si);
        let lt = g(p, o, -si - 1);
        let gt = |k: i32| if k < 0 { lt } else { t[k as usize] };
        let gl = |k: i32| if k < 0 { lt } else { l[k as usize] };
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let z = 2 * x - y;
                let d = x - (y >> 1);
                let v = if z >= 0 {
                    if z & 1 == 0 {
                        (gt(d - 1) + gt(d) + 1) >> 1
                    } else {
                        (gt(d - 2) + 2 * gt(d - 1) + gt(d) + 2) >> 2
                    }
                } else if z == -1 {
                    (l[0] + 2 * lt + t[0] + 2) >> 2
                } else {
                    let e = -z;
                    (gl(e - 1) + 2 * gl(e - 2) + gl(e - 3) + 2) >> 2
                };
                st(p, oi, si, y, x, v);
            }
        }
    }
    fn r4_vl(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let t = top4(p, o, si);
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let k = (x + (y >> 1)) as usize;
                let v = if y & 1 == 0 {
                    (t[k] + t[k + 1] + 1) >> 1
                } else {
                    (t[k] + 2 * t[k + 1] + t[k + 2] + 2) >> 2
                };
                st(p, oi, si, y, x, v);
            }
        }
    }
    fn r4_vl_top(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let raw = top4(p, o, si);
        let t = [raw[0], raw[1], raw[2], raw[3], raw[3], raw[3], raw[3], raw[3]];
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let k = (x + (y >> 1)) as usize;
                let v = if y & 1 == 0 {
                    (t[k] + t[k + 1] + 1) >> 1
                } else {
                    (t[k] + 2 * t[k + 1] + t[k + 2] + 2) >> 2
                };
                st(p, oi, si, y, x, v);
            }
        }
    }
    fn r4_hu(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let l = left4(p, o, si);
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let z = x + 2 * y;
                let v = if z < 5 {
                    let k = (z >> 1) as usize;
                    if z & 1 == 0 {
                        (l[k] + l[k + 1] + 1) >> 1
                    } else {
                        (l[k] + 2 * l[k + 1] + l[k + 2] + 2) >> 2
                    }
                } else if z == 5 {
                    (l[2] + 3 * l[3] + 2) >> 2
                } else {
                    l[3]
                };
                st(p, oi, si, y, x, v);
            }
        }
    }
    fn r4_hd(p: &mut [u8], o: usize, s: usize) {
        let si = s as isize;
        let t = top4(p, o, si);
        let l = left4(p, o, si);
        let lt = g(p, o, -si - 1);
        let gt = |k: i32| if k < 0 { lt } else { t[k as usize] };
        let gl = |k: i32| if k < 0 { lt } else { l[k as usize] };
        let oi = o as isize;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let z = 2 * y - x;
                let d = y - (x >> 1);
                let v = if z >= 0 {
                    if z & 1 == 0 {
                        (gl(d - 1) + gl(d) + 1) >> 1
                    } else {
                        (gl(d - 2) + 2 * gl(d - 1) + gl(d) + 2) >> 2
                    }
                } else if z == -1 {
                    (t[0] + 2 * lt + l[0] + 2) >> 2
                } else {
                    let e = -z;
                    (gt(e - 1) + 2 * gt(e - 2) + gt(e - 3) + 2) >> 2
                };
                st(p, oi, si, y, x, v);
            }
        }
    }

    #[test]
    fn i4x4_all() {
        run4x4(i4x4_luma_pred_v, r4_v);
        run4x4(i4x4_luma_pred_h, r4_h);
        run4x4(i4x4_luma_pred_dc, r4_dc);
        run4x4(i4x4_luma_pred_dc_left, r4_dc_left);
        run4x4(i4x4_luma_pred_dc_top, r4_dc_top);
        run4x4(i4x4_luma_pred_dc_na, r4_dc_na);
        run4x4(i4x4_luma_pred_ddl, r4_ddl);
        run4x4(i4x4_luma_pred_ddl_top, r4_ddl_top);
        run4x4(i4x4_luma_pred_ddr, r4_ddr);
        run4x4(i4x4_luma_pred_vr, r4_vr);
        run4x4(i4x4_luma_pred_vl, r4_vl);
        run4x4(i4x4_luma_pred_vl_top, r4_vl_top);
        run4x4(i4x4_luma_pred_hu, r4_hu);
        run4x4(i4x4_luma_pred_hd, r4_hd);
    }

    // ====================================================================
    // 8x8 luma references (clause 8.3.2.2 with 8.3.2.2.1 reference-sample
    // smoothing). Built independently from the spec position equations.
    // ====================================================================

    // Raw neighbour samples (unfiltered).
    fn rtop(p: &[u8], o: usize, s: usize, x: isize) -> i32 {
        g(p, o, x - s as isize) // p[x, -1]
    }
    fn rleft(p: &[u8], o: usize, s: usize, y: isize) -> i32 {
        g(p, o, -1 + y * s as isize) // p[-1, y]
    }
    fn rcorner(p: &[u8], o: usize, s: usize) -> i32 {
        g(p, o, -1 - s as isize) // p[-1, -1]
    }

    /// Filtered top, 8 samples, left edge per `b_tl`, right edge per `b_tr`.
    fn ft8(p: &[u8], o: usize, s: usize, b_tl: bool, b_tr: bool) -> [i32; 8] {
        let mut t = [0i32; 8];
        t[0] = if b_tl {
            (rcorner(p, o, s) + 2 * rtop(p, o, s, 0) + rtop(p, o, s, 1) + 2) >> 2
        } else {
            (3 * rtop(p, o, s, 0) + rtop(p, o, s, 1) + 2) >> 2
        };
        for i in 1..7 {
            t[i] = (rtop(p, o, s, i as isize - 1) + 2 * rtop(p, o, s, i as isize) + rtop(p, o, s, i as isize + 1) + 2) >> 2;
        }
        t[7] = if b_tr {
            (rtop(p, o, s, 6) + 2 * rtop(p, o, s, 7) + rtop(p, o, s, 8) + 2) >> 2
        } else {
            (rtop(p, o, s, 6) + 3 * rtop(p, o, s, 7) + 2) >> 2
        };
        t
    }
    /// Filtered top, 16 samples (top-right available). For DDL/VL.
    fn ft_full(p: &[u8], o: usize, s: usize, b_tl: bool) -> [i32; 16] {
        let mut t = [0i32; 16];
        t[0] = if b_tl {
            (rcorner(p, o, s) + 2 * rtop(p, o, s, 0) + rtop(p, o, s, 1) + 2) >> 2
        } else {
            (3 * rtop(p, o, s, 0) + rtop(p, o, s, 1) + 2) >> 2
        };
        for i in 1..15 {
            t[i] = (rtop(p, o, s, i as isize - 1) + 2 * rtop(p, o, s, i as isize) + rtop(p, o, s, i as isize + 1) + 2) >> 2;
        }
        t[15] = (rtop(p, o, s, 14) + 3 * rtop(p, o, s, 15) + 2) >> 2;
        t
    }
    /// Filtered top, 16 samples (top-right unavailable: 8..15 replicate raw T7).
    fn ft_only(p: &[u8], o: usize, s: usize, b_tl: bool) -> [i32; 16] {
        let mut t = [0i32; 16];
        t[0] = if b_tl {
            (rcorner(p, o, s) + 2 * rtop(p, o, s, 0) + rtop(p, o, s, 1) + 2) >> 2
        } else {
            (3 * rtop(p, o, s, 0) + rtop(p, o, s, 1) + 2) >> 2
        };
        for i in 1..7 {
            t[i] = (rtop(p, o, s, i as isize - 1) + 2 * rtop(p, o, s, i as isize) + rtop(p, o, s, i as isize + 1) + 2) >> 2;
        }
        t[7] = (rtop(p, o, s, 6) + 3 * rtop(p, o, s, 7) + 2) >> 2;
        for i in 8..16 {
            t[i] = rtop(p, o, s, 7);
        }
        t
    }
    /// Filtered top, 8 samples, left edge forced to the top-left-available form.
    fn ft8_tl(p: &[u8], o: usize, s: usize, b_tr: bool) -> [i32; 8] {
        let mut t = [0i32; 8];
        t[0] = (rcorner(p, o, s) + 2 * rtop(p, o, s, 0) + rtop(p, o, s, 1) + 2) >> 2;
        for i in 1..7 {
            t[i] = (rtop(p, o, s, i as isize - 1) + 2 * rtop(p, o, s, i as isize) + rtop(p, o, s, i as isize + 1) + 2) >> 2;
        }
        t[7] = if b_tr {
            (rtop(p, o, s, 6) + 2 * rtop(p, o, s, 7) + rtop(p, o, s, 8) + 2) >> 2
        } else {
            (rtop(p, o, s, 6) + 3 * rtop(p, o, s, 7) + 2) >> 2
        };
        t
    }
    /// Filtered left, 8 samples, top edge per `b_tl`.
    fn fl8(p: &[u8], o: usize, s: usize, b_tl: bool) -> [i32; 8] {
        let mut l = [0i32; 8];
        l[0] = if b_tl {
            (rcorner(p, o, s) + 2 * rleft(p, o, s, 0) + rleft(p, o, s, 1) + 2) >> 2
        } else {
            (3 * rleft(p, o, s, 0) + rleft(p, o, s, 1) + 2) >> 2
        };
        for i in 1..7 {
            l[i] = (rleft(p, o, s, i as isize - 1) + 2 * rleft(p, o, s, i as isize) + rleft(p, o, s, i as isize + 1) + 2) >> 2;
        }
        l[7] = (rleft(p, o, s, 6) + 3 * rleft(p, o, s, 7) + 2) >> 2;
        l
    }
    /// Filtered left, 8 samples, top edge forced to top-left-available form.
    fn fl8_tl(p: &[u8], o: usize, s: usize) -> [i32; 8] {
        let mut l = [0i32; 8];
        l[0] = (rcorner(p, o, s) + 2 * rleft(p, o, s, 0) + rleft(p, o, s, 1) + 2) >> 2;
        for i in 1..7 {
            l[i] = (rleft(p, o, s, i as isize - 1) + 2 * rleft(p, o, s, i as isize) + rleft(p, o, s, i as isize + 1) + 2) >> 2;
        }
        l[7] = (rleft(p, o, s, 6) + 3 * rleft(p, o, s, 7) + 2) >> 2;
        l
    }
    /// Filtered top-left corner.
    fn ftl(p: &[u8], o: usize, s: usize) -> i32 {
        (rleft(p, o, s, 0) + 2 * rcorner(p, o, s) + rtop(p, o, s, 0) + 2) >> 2
    }

    fn fill8(p: &mut [u8], o: usize, s: usize, rows: &[[i32; 8]; 8]) {
        for y in 0..8 {
            for x in 0..8 {
                p[o + y * s + x] = rows[y][x] as u8;
            }
        }
    }

    fn r8_v(p: &mut [u8], o: usize, s: usize, b_tl: bool, b_tr: bool) {
        let t = ft8(p, o, s, b_tl, b_tr);
        let rows = core::array::from_fn(|_| t);
        fill8(p, o, s, &rows);
    }
    fn r8_h(p: &mut [u8], o: usize, s: usize, b_tl: bool, _b_tr: bool) {
        let l = fl8(p, o, s, b_tl);
        let rows = core::array::from_fn(|y| [l[y]; 8]);
        fill8(p, o, s, &rows);
    }
    fn r8_dc(p: &mut [u8], o: usize, s: usize, b_tl: bool, b_tr: bool) {
        let t = ft8(p, o, s, b_tl, b_tr);
        let l = fl8(p, o, s, b_tl);
        let total: i32 = t.iter().sum::<i32>() + l.iter().sum::<i32>();
        let m = (total + 8) >> 4;
        fill8(p, o, s, &[[m; 8]; 8]);
    }
    fn r8_dc_left(p: &mut [u8], o: usize, s: usize, b_tl: bool, _b_tr: bool) {
        let l = fl8(p, o, s, b_tl);
        let m = (l.iter().sum::<i32>() + 4) >> 3;
        fill8(p, o, s, &[[m; 8]; 8]);
    }
    fn r8_dc_top(p: &mut [u8], o: usize, s: usize, b_tl: bool, b_tr: bool) {
        let t = ft8(p, o, s, b_tl, b_tr);
        let m = (t.iter().sum::<i32>() + 4) >> 3;
        fill8(p, o, s, &[[m; 8]; 8]);
    }
    fn r8_dc_na(p: &mut [u8], o: usize, s: usize, _b_tl: bool, _b_tr: bool) {
        fill8(p, o, s, &[[0x80; 8]; 8]);
    }
    fn r8_ddl_inner(p: &mut [u8], o: usize, s: usize, t: &[i32; 16]) {
        for i in 0..8 {
            for j in 0..8 {
                let v = if i == 7 && j == 7 {
                    (t[14] + 3 * t[15] + 2) >> 2
                } else {
                    (t[i + j] + 2 * t[i + j + 1] + t[i + j + 2] + 2) >> 2
                };
                p[o + i * s + j] = v as u8;
            }
        }
    }
    fn r8_ddl(p: &mut [u8], o: usize, s: usize, b_tl: bool, _b_tr: bool) {
        let t = ft_full(p, o, s, b_tl);
        r8_ddl_inner(p, o, s, &t);
    }
    fn r8_ddl_top(p: &mut [u8], o: usize, s: usize, b_tl: bool, _b_tr: bool) {
        let t = ft_only(p, o, s, b_tl);
        r8_ddl_inner(p, o, s, &t);
    }
    fn r8_vl_inner(p: &mut [u8], o: usize, s: usize, t: &[i32; 16]) {
        for i in 0..8 {
            for j in 0..8 {
                let k = j + (i >> 1);
                let v = if i & 1 == 0 {
                    (t[k] + t[k + 1] + 1) >> 1
                } else {
                    (t[k] + 2 * t[k + 1] + t[k + 2] + 2) >> 2
                };
                p[o + i * s + j] = v as u8;
            }
        }
    }
    fn r8_vl(p: &mut [u8], o: usize, s: usize, b_tl: bool, _b_tr: bool) {
        let t = ft_full(p, o, s, b_tl);
        r8_vl_inner(p, o, s, &t);
    }
    fn r8_vl_top(p: &mut [u8], o: usize, s: usize, b_tl: bool, _b_tr: bool) {
        let t = ft_only(p, o, s, b_tl);
        r8_vl_inner(p, o, s, &t);
    }
    fn r8_ddr(p: &mut [u8], o: usize, s: usize, _b_tl: bool, b_tr: bool) {
        let t = ft8_tl(p, o, s, b_tr);
        let l = fl8_tl(p, o, s);
        let tl = ftl(p, o, s);
        for i in 0..8i32 {
            for j in 0..8i32 {
                let v = if j < i - 1 {
                    let k = (i - j) as usize;
                    (l[k - 2] + 2 * l[k - 1] + l[k] + 2) >> 2
                } else if j == i - 1 {
                    (tl + 2 * l[0] + l[1] + 2) >> 2
                } else if j == i {
                    (t[0] + 2 * tl + l[0] + 2) >> 2
                } else if j == i + 1 {
                    (tl + 2 * t[0] + t[1] + 2) >> 2
                } else {
                    let k = (j - i) as usize;
                    (t[k - 2] + 2 * t[k - 1] + t[k] + 2) >> 2
                };
                p[o + i as usize * s + j as usize] = v as u8;
            }
        }
    }
    fn r8_vr(p: &mut [u8], o: usize, s: usize, _b_tl: bool, b_tr: bool) {
        let t = ft8_tl(p, o, s, b_tr);
        let l = fl8_tl(p, o, s);
        let tl = ftl(p, o, s);
        for i in 0..8i32 {
            for j in 0..8i32 {
                let z = (j << 1) - i;
                let d = j - (i >> 1);
                let v = if z >= 0 {
                    if z & 1 == 0 {
                        if d > 0 {
                            (t[(d - 1) as usize] + t[d as usize] + 1) >> 1
                        } else {
                            (tl + t[0] + 1) >> 1
                        }
                    } else if d > 1 {
                        (t[(d - 2) as usize] + 2 * t[(d - 1) as usize] + t[d as usize] + 2) >> 2
                    } else {
                        (tl + 2 * t[0] + t[1] + 2) >> 2
                    }
                } else if z == -1 {
                    (l[0] + 2 * tl + t[0] + 2) >> 2
                } else if z < -2 {
                    (l[(-z - 1) as usize] + 2 * l[(-z - 2) as usize] + l[(-z - 3) as usize] + 2) >> 2
                } else {
                    (l[1] + 2 * l[0] + tl + 2) >> 2
                };
                p[o + i as usize * s + j as usize] = v as u8;
            }
        }
    }
    fn r8_hu(p: &mut [u8], o: usize, s: usize, b_tl: bool, _b_tr: bool) {
        let l = fl8(p, o, s, b_tl);
        for i in 0..8 {
            for j in 0..8 {
                let z = j + (i << 1);
                let v = if z < 13 {
                    let k = z >> 1;
                    if z & 1 == 0 {
                        (l[k] + l[k + 1] + 1) >> 1
                    } else {
                        (l[k] + 2 * l[k + 1] + l[k + 2] + 2) >> 2
                    }
                } else if z == 13 {
                    (l[6] + 3 * l[7] + 2) >> 2
                } else {
                    l[7]
                };
                p[o + i * s + j] = v as u8;
            }
        }
    }
    fn r8_hd(p: &mut [u8], o: usize, s: usize, _b_tl: bool, b_tr: bool) {
        let t = ft8_tl(p, o, s, b_tr);
        let l = fl8_tl(p, o, s);
        let tl = ftl(p, o, s);
        for i in 0..8i32 {
            for j in 0..8i32 {
                let z = (i << 1) - j;
                let d = i - (j >> 1);
                let v = if z >= 0 {
                    if z & 1 == 0 {
                        if d == 0 {
                            (tl + l[0] + 1) >> 1
                        } else {
                            (l[(d - 1) as usize] + l[d as usize] + 1) >> 1
                        }
                    } else if d == 1 {
                        (tl + 2 * l[0] + l[1] + 2) >> 2
                    } else {
                        (l[(d - 2) as usize] + 2 * l[(d - 1) as usize] + l[d as usize] + 2) >> 2
                    }
                } else if z == -1 {
                    (l[0] + 2 * tl + t[0] + 2) >> 2
                } else if z < -2 {
                    (t[(-z - 1) as usize] + 2 * t[(-z - 2) as usize] + t[(-z - 3) as usize] + 2) >> 2
                } else {
                    (t[1] + 2 * t[0] + tl + 2) >> 2
                };
                p[o + i as usize * s + j as usize] = v as u8;
            }
        }
    }

    #[test]
    fn i8x8_all() {
        run8x8(i8x8_luma_pred_v, r8_v);
        run8x8(i8x8_luma_pred_h, r8_h);
        run8x8(i8x8_luma_pred_dc, r8_dc);
        run8x8(i8x8_luma_pred_dc_left, r8_dc_left);
        run8x8(i8x8_luma_pred_dc_top, r8_dc_top);
        run8x8(i8x8_luma_pred_dc_na, r8_dc_na);
        run8x8(i8x8_luma_pred_ddl, r8_ddl);
        run8x8(i8x8_luma_pred_ddl_top, r8_ddl_top);
        run8x8(i8x8_luma_pred_ddr, r8_ddr);
        run8x8(i8x8_luma_pred_vl, r8_vl);
        run8x8(i8x8_luma_pred_vl_top, r8_vl_top);
        run8x8(i8x8_luma_pred_vr, r8_vr);
        run8x8(i8x8_luma_pred_hu, r8_hu);
        run8x8(i8x8_luma_pred_hd, r8_hd);
    }

    // ====================================================================
    // 16x16 luma references (no reference-sample smoothing).
    // ====================================================================

    fn r16_v(p: &mut [u8], o: usize, s: usize) {
        for y in 0..16 {
            for x in 0..16 {
                p[o + y * s + x] = rtop(p, o, s, x as isize) as u8;
            }
        }
    }
    fn r16_h(p: &mut [u8], o: usize, s: usize) {
        for y in 0..16 {
            let v = rleft(p, o, s, y as isize) as u8;
            for x in 0..16 {
                p[o + y * s + x] = v;
            }
        }
    }
    fn r16_dc(p: &mut [u8], o: usize, s: usize) {
        let mut sum = 16i32;
        for i in 0..16isize {
            sum += rleft(p, o, s, i) + rtop(p, o, s, i);
        }
        let m = (sum >> 5) as u8;
        for y in 0..16 {
            for x in 0..16 {
                p[o + y * s + x] = m;
            }
        }
    }
    fn r16_dc_top(p: &mut [u8], o: usize, s: usize) {
        let mut sum = 8i32;
        for i in 0..16isize {
            sum += rtop(p, o, s, i);
        }
        let m = (sum >> 4) as u8;
        for y in 0..16 {
            for x in 0..16 {
                p[o + y * s + x] = m;
            }
        }
    }
    fn r16_dc_left(p: &mut [u8], o: usize, s: usize) {
        let mut sum = 8i32;
        for i in 0..16isize {
            sum += rleft(p, o, s, i);
        }
        let m = (sum >> 4) as u8;
        for y in 0..16 {
            for x in 0..16 {
                p[o + y * s + x] = m;
            }
        }
    }
    fn r16_dc_na(p: &mut [u8], o: usize, s: usize) {
        for y in 0..16 {
            for x in 0..16 {
                p[o + y * s + x] = 0x80;
            }
        }
    }
    fn r16_plane(p: &mut [u8], o: usize, s: usize) {
        let mut h = 0i32;
        let mut v = 0i32;
        for i in 0..8isize {
            h += (i as i32 + 1) * (rtop(p, o, s, 8 + i) - rtop(p, o, s, 6 - i));
            v += (i as i32 + 1) * (rleft(p, o, s, 8 + i) - rleft(p, o, s, 6 - i));
        }
        let a = (rleft(p, o, s, 15) + rtop(p, o, s, 15)) << 4;
        let b = (5 * h + 32) >> 6;
        let c = (5 * v + 32) >> 6;
        for i in 0..16i32 {
            for j in 0..16i32 {
                let val = (a + b * (j - 7) + c * (i - 7) + 16) >> 5;
                p[o + i as usize * s + j as usize] = clip1(val);
            }
        }
    }

    #[test]
    fn i16x16_all() {
        run_big(16, i16x16_luma_pred_v, r16_v);
        run_big(16, i16x16_luma_pred_h, r16_h);
        run_big(16, i16x16_luma_pred_dc, r16_dc);
        run_big(16, i16x16_luma_pred_dc_top, r16_dc_top);
        run_big(16, i16x16_luma_pred_dc_left, r16_dc_left);
        run_big(16, i16x16_luma_pred_dc_na, r16_dc_na);
        run_big(16, i16x16_luma_pred_plane, r16_plane);
    }

    // ====================================================================
    // Chroma 8x8 references.
    // ====================================================================

    fn rc_v(p: &mut [u8], o: usize, s: usize) {
        for y in 0..8 {
            for x in 0..8 {
                p[o + y * s + x] = rtop(p, o, s, x as isize) as u8;
            }
        }
    }
    fn rc_h(p: &mut [u8], o: usize, s: usize) {
        for y in 0..8 {
            let v = rleft(p, o, s, y as isize) as u8;
            for x in 0..8 {
                p[o + y * s + x] = v;
            }
        }
    }
    fn rc_dc(p: &mut [u8], o: usize, s: usize) {
        let st: [i32; 8] = core::array::from_fn(|x| rtop(p, o, s, x as isize));
        let sl: [i32; 8] = core::array::from_fn(|y| rleft(p, o, s, y as isize));
        let st03 = st[0] + st[1] + st[2] + st[3];
        let st47 = st[4] + st[5] + st[6] + st[7];
        let sl03 = sl[0] + sl[1] + sl[2] + sl[3];
        let sl47 = sl[4] + sl[5] + sl[6] + sl[7];
        let m1 = ((st03 + sl03 + 4) >> 3) as u8;
        let m2 = ((st47 + 2) >> 2) as u8;
        let m3 = ((sl47 + 2) >> 2) as u8;
        let m4 = ((st47 + sl47 + 4) >> 3) as u8;
        let up = [m1, m1, m1, m1, m2, m2, m2, m2];
        let down = [m3, m3, m3, m3, m4, m4, m4, m4];
        for y in 0..8 {
            let row = if y < 4 { &up } else { &down };
            for x in 0..8 {
                p[o + y * s + x] = row[x];
            }
        }
    }
    fn rc_dc_left(p: &mut [u8], o: usize, s: usize) {
        let sl: [i32; 8] = core::array::from_fn(|y| rleft(p, o, s, y as isize));
        let up = ((sl[0] + sl[1] + sl[2] + sl[3] + 2) >> 2) as u8;
        let down = ((sl[4] + sl[5] + sl[6] + sl[7] + 2) >> 2) as u8;
        for y in 0..8 {
            let v = if y < 4 { up } else { down };
            for x in 0..8 {
                p[o + y * s + x] = v;
            }
        }
    }
    fn rc_dc_top(p: &mut [u8], o: usize, s: usize) {
        let st: [i32; 8] = core::array::from_fn(|x| rtop(p, o, s, x as isize));
        let m1 = ((st[0] + st[1] + st[2] + st[3] + 2) >> 2) as u8;
        let m2 = ((st[4] + st[5] + st[6] + st[7] + 2) >> 2) as u8;
        let row = [m1, m1, m1, m1, m2, m2, m2, m2];
        for y in 0..8 {
            for x in 0..8 {
                p[o + y * s + x] = row[x];
            }
        }
    }
    fn rc_dc_na(p: &mut [u8], o: usize, s: usize) {
        for y in 0..8 {
            for x in 0..8 {
                p[o + y * s + x] = 0x80;
            }
        }
    }
    fn rc_plane(p: &mut [u8], o: usize, s: usize) {
        let mut h = 0i32;
        let mut v = 0i32;
        for i in 0..4isize {
            h += (i as i32 + 1) * (rtop(p, o, s, 4 + i) - rtop(p, o, s, 2 - i));
            v += (i as i32 + 1) * (rleft(p, o, s, 4 + i) - rleft(p, o, s, 2 - i));
        }
        let a = (rleft(p, o, s, 7) + rtop(p, o, s, 7)) << 4;
        let b = (17 * h + 16) >> 5;
        let c = (17 * v + 16) >> 5;
        for i in 0..8i32 {
            for j in 0..8i32 {
                let val = (a + b * (j - 3) + c * (i - 3) + 16) >> 5;
                p[o + i as usize * s + j as usize] = clip1(val);
            }
        }
    }

    #[test]
    fn chroma_all() {
        run_big(8, i_chroma_pred_v, rc_v);
        run_big(8, i_chroma_pred_h, rc_h);
        run_big(8, i_chroma_pred_dc, rc_dc);
        run_big(8, i_chroma_pred_dc_left, rc_dc_left);
        run_big(8, i_chroma_pred_dc_top, rc_dc_top);
        run_big(8, i_chroma_pred_dc_na, rc_dc_na);
        run_big(8, i_chroma_pred_plane, rc_plane);
    }
}
