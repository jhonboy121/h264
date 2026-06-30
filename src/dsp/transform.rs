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
///
/// Dispatches to a bit-exact SIMD kernel when `--features simd` is enabled on a
/// supported target (NEON / wasm `simd128`), otherwise the scalar reference.
#[allow(unreachable_code)]
pub fn idct4x4_add(pred: &mut [u8], stride: usize, rs: &[i16; 16]) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return crate::dsp::simd::neon::idct4x4_add(pred, stride, rs);
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    return crate::dsp::simd::wasm::idct4x4_add(pred, stride, rs);
    idct4x4_add_scalar(pred, stride, rs)
}

/// Scalar reference for [`idct4x4_add`] (the conformance baseline / SIMD fallback).
pub fn idct4x4_add_scalar(pred: &mut [u8], stride: usize, rs: &[i16; 16]) {
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

// ===========================================================================
// Forward transform, quantization, DC Hadamard, and coefficient scan
// (encoder path). Ports of `reference/codec/encoder/core/src/encode_mb_aux.cpp`.
// ===========================================================================

/// Forward 4x4 integer DCT of the residual `pix1 - pix2`. Port of `WelsDctT4_c`.
///
/// `pix1`/`pix2` are the source and prediction planes; `off1`/`off2` the index
/// of each block's top-left sample and `st1`/`st2` their row strides. The 16
/// transform coefficients are written to `dct[0..16]` in raster order.
pub fn dct_t4(dct: &mut [i16], pix1: &[u8], off1: usize, st1: usize, pix2: &[u8], off2: usize, st2: usize) {
    let mut data = [0i16; 16];
    let mut p1 = off1;
    let mut p2 = off2;
    // Horizontal transform (one 4-sample row per iteration).
    for i in (0..16).step_by(4) {
        for k in 0..4 {
            data[i + k] = (pix1[p1 + k] as i32 - pix2[p2 + k] as i32) as i16;
        }
        p1 += st1;
        p2 += st2;
        let s0 = data[i] as i32 + data[i + 3] as i32;
        let s3 = data[i] as i32 - data[i + 3] as i32;
        let s1 = data[i + 1] as i32 + data[i + 2] as i32;
        let s2 = data[i + 1] as i32 - data[i + 2] as i32;
        dct[i] = (s0 + s1) as i16;
        dct[i + 2] = (s0 - s1) as i16;
        dct[i + 1] = ((s3 << 1) + s2) as i16;
        dct[i + 3] = (s3 - (s2 << 1)) as i16;
    }
    // Vertical transform (columns).
    for i in 0..4 {
        let s0 = dct[i] as i32 + dct[12 + i] as i32;
        let s3 = dct[i] as i32 - dct[12 + i] as i32;
        let s1 = dct[4 + i] as i32 + dct[8 + i] as i32;
        let s2 = dct[4 + i] as i32 - dct[8 + i] as i32;
        dct[i] = (s0 + s1) as i16;
        dct[8 + i] = (s0 - s1) as i16;
        dct[4 + i] = ((s3 << 1) + s2) as i16;
        dct[12 + i] = (s3 - (s2 << 1)) as i16;
    }
}

/// Forward 4x4 DCT of four sub-blocks (an 8x8 luma area). Port of `WelsDctFourT4_c`.
/// Writes 4 * 16 coefficients to `dct`, one 16-coeff block per 4x4 sub-block.
pub fn dct_four_t4(dct: &mut [i16; 64], pix1: &[u8], off1: usize, st1: usize, pix2: &[u8], off2: usize, st2: usize) {
    let s1x4 = st1 << 2;
    let s2x4 = st2 << 2;
    dct_t4(&mut dct[0..16], pix1, off1, st1, pix2, off2, st2);
    dct_t4(&mut dct[16..32], pix1, off1 + 4, st1, pix2, off2 + 4, st2);
    dct_t4(&mut dct[32..48], pix1, off1 + s1x4, st1, pix2, off2 + s2x4, st2);
    dct_t4(&mut dct[48..64], pix1, off1 + s1x4 + 4, st1, pix2, off2 + s2x4 + 4, st2);
}

// --- Quantization (`g_kiQuantMF` multiplier `mf` + `g_kiQuantInterFF` offset `ff`) ---

/// `WELS_NEW_QUANT`: quantize a single coefficient. `sign(dct) * ((ff + |dct|) * mf >> 16)`.
#[inline(always)]
fn wels_new_quant(dct: i16, ff: i16, mf: i16) -> i16 {
    let sign = (dct as i32) >> 31; // 0 or -1
    let abs = (sign ^ dct as i32) - sign;
    let q = ((ff as i32 + abs) * mf as i32) >> 16;
    ((sign ^ q) - sign) as i16
}

/// Port of `WelsQuant4x4_c`: quantize a 4x4 block with per-position `ff`/`mf`
/// (rows 0/2 use entries `[0..4]`, rows 1/3 use entries `[4..8]`).
pub fn quant4x4(dct: &mut [i16; 16], ff: &[i16; 8], mf: &[i16; 8]) {
    let mut i = 0;
    while i < 16 {
        let j = i & 0x07;
        for k in 0..4 {
            dct[i + k] = wels_new_quant(dct[i + k], ff[j + k], mf[j + k]);
        }
        i += 4;
    }
}

/// Port of `WelsQuant4x4Dc_c`: quantize 16 coefficients with a scalar `ff`/`mf`.
pub fn quant4x4_dc(dct: &mut [i16; 16], ff: i16, mf: i16) {
    let mut i = 0;
    while i < 16 {
        for k in 0..4 {
            dct[i + k] = wels_new_quant(dct[i + k], ff, mf);
        }
        i += 4;
    }
}

/// Port of `WelsQuantFour4x4_c`: quantize four contiguous 4x4 blocks (64 coeffs).
pub fn quant_four4x4(dct: &mut [i16; 64], ff: &[i16; 8], mf: &[i16; 8]) {
    let mut i = 0;
    while i < 64 {
        let j = i & 0x07;
        for k in 0..4 {
            dct[i + k] = wels_new_quant(dct[i + k], ff[j + k], mf[j + k]);
        }
        i += 4;
    }
}

/// Port of `WelsQuantFour4x4Max_c`: quantize four 4x4 blocks and report the max
/// (unsigned) quantized magnitude per block in `max[0..4]`.
pub fn quant_four4x4_max(dct: &mut [i16; 64], ff: &[i16; 8], mf: &[i16; 8], max: &mut [i16; 4]) {
    for (k, mk) in max.iter_mut().enumerate() {
        let mut max_abs = 0i16;
        let base = k * 16;
        for i in 0..16 {
            let j = i & 0x07;
            let orig = dct[base + i] as i32;
            let sign = orig >> 31;
            let abs = (sign ^ orig) - sign;
            let q = (((ff[j] as i32 + abs) * mf[j] as i32) >> 16) as i16;
            if max_abs < q {
                max_abs = q;
            }
            dct[base + i] = ((sign ^ q as i32) - sign) as i16;
        }
        *mk = max_abs;
    }
}

// --- Chroma DC: combined 2x2 Hadamard + quantization ---

/// Port of `WelsHadamardQuant2x2Skip_c`: returns 1 if any quantized chroma-DC
/// coefficient would exceed the deadzone threshold, else 0.
pub fn hadamard_quant2x2_skip(rs: &[i16], ff: i16, mf: i16) -> i32 {
    let threshold = ((((1i32 << 16) - 1) / mf as i32) - ff as i32) as i16 as i32;
    let s0 = rs[0].wrapping_add(rs[32]);
    let s1 = rs[0].wrapping_sub(rs[32]);
    let s2 = rs[16].wrapping_add(rs[48]);
    let s3 = rs[16].wrapping_sub(rs[48]);
    let d0 = s0.wrapping_add(s2);
    let d1 = s0.wrapping_sub(s2);
    let d2 = s1.wrapping_add(s3);
    let d3 = s1.wrapping_sub(s3);
    let abs = |x: i16| (x as i32).abs();
    ((abs(d0) > threshold) || (abs(d1) > threshold) || (abs(d2) > threshold) || (abs(d3) > threshold)) as i32
}

/// Port of `WelsHadamardQuant2x2_c`: 2x2 Hadamard of the four chroma-DC samples
/// (at `rs[0]`, `rs[16]`, `rs[32]`, `rs[48]`), zeroing them in `rs`, quantizing
/// into `dct[0..4]` and `block[0..4]`. Returns the non-zero count.
pub fn hadamard_quant2x2(rs: &mut [i16], ff: i16, mf: i16, dct: &mut [i16; 4], block: &mut [i16; 4]) -> i32 {
    let s0 = rs[0].wrapping_add(rs[32]);
    let s1 = rs[0].wrapping_sub(rs[32]);
    let s2 = rs[16].wrapping_add(rs[48]);
    let s3 = rs[16].wrapping_sub(rs[48]);
    rs[0] = 0;
    rs[16] = 0;
    rs[32] = 0;
    rs[48] = 0;
    dct[0] = s0.wrapping_add(s2);
    dct[1] = s0.wrapping_sub(s2);
    dct[2] = s1.wrapping_add(s3);
    dct[3] = s1.wrapping_sub(s3);
    for d in dct.iter_mut() {
        *d = wels_new_quant(*d, ff, mf);
    }
    block[..4].copy_from_slice(&dct[..4]);
    let mut nzc = 0;
    for &b in block.iter() {
        nzc += (b != 0) as i32;
    }
    nzc
}

/// `WelsHadamardT4Dc_c`: forward 4x4 Hadamard of the 16 luma-DC coefficients.
///
/// `dct` is the macroblock residual coefficient array; the DC of each 4x4 luma
/// block is picked up at the strided offsets used by the C kernel. The 16
/// transformed DC values (clipped to `i16`) are written to `luma_dc`.
pub fn hadamard_t4_dc(luma_dc: &mut [i16; 16], dct: &[i16]) {
    let mut p = [0i32; 16];
    let mut i = 0;
    while i < 16 {
        let idx = ((i & 0x08) << 4) + ((i & 0x04) << 3);
        let s0 = dct[idx] as i32 + dct[idx + 80] as i32;
        let s3 = dct[idx] as i32 - dct[idx + 80] as i32;
        let s1 = dct[idx + 16] as i32 + dct[idx + 64] as i32;
        let s2 = dct[idx + 16] as i32 - dct[idx + 64] as i32;
        p[i] = s0 + s1;
        p[i + 2] = s0 - s1;
        p[i + 1] = s3 + s2;
        p[i + 3] = s3 - s2;
        i += 4;
    }
    for i in 0..4 {
        let s0 = p[i] + p[i + 12];
        let s3 = p[i] - p[i + 12];
        let s1 = p[i + 4] + p[i + 8];
        let s2 = p[i + 4] - p[i + 8];
        luma_dc[i] = ((s0 + s1 + 1) >> 1).clamp(-32768, 32767) as i16;
        luma_dc[i + 8] = ((s0 - s1 + 1) >> 1).clamp(-32768, 32767) as i16;
        luma_dc[i + 4] = ((s3 + s2 + 1) >> 1).clamp(-32768, 32767) as i16;
        luma_dc[i + 12] = ((s3 - s2 + 1) >> 1).clamp(-32768, 32767) as i16;
    }
}

// --- Coefficient scan / zig-zag ---

/// Port of `WelsScan4x4DcAc_c` (and the identical `WelsScan4x4Dc`): zig-zag the
/// 16 raster-order coefficients of `dct` into scan order in `level`.
pub fn scan4x4_dcac(level: &mut [i16; 16], dct: &[i16; 16]) {
    const IDX: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
    for (l, &d) in level.iter_mut().zip(IDX.iter()) {
        *l = dct[d];
    }
}

/// Port of `WelsScan4x4Ac_c`: zig-zag the 15 AC coefficients (DC slot dropped),
/// trailing entry zeroed.
pub fn scan4x4_ac(level: &mut [i16; 16], dct: &[i16; 16]) {
    const IDX: [usize; 15] = [1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
    for (l, &d) in level[..15].iter_mut().zip(IDX.iter()) {
        *l = dct[d];
    }
    level[15] = 0;
}

/// Port of `WelsCalculateSingleCtr4x4_c` (JVT-O079 single-coefficient cost).
pub fn calculate_single_ctr4x4(dct: &[i16; 16]) -> i32 {
    const RUN_TABLE: [i32; 16] = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut single_ctr = 0;
    let mut idx: i32 = 15;
    while idx >= 0 && dct[idx as usize] == 0 {
        idx -= 1;
    }
    while idx >= 0 {
        idx -= 1;
        let run_start = idx;
        while idx >= 0 && dct[idx as usize] == 0 {
            idx -= 1;
        }
        single_ctr += RUN_TABLE[(run_start - idx) as usize];
    }
    single_ctr
}

/// Port of `WelsGetNoneZeroCount_c`: count non-zero entries in a 16-coeff block.
pub fn get_none_zero_count(level: &[i16; 16]) -> i32 {
    level.iter().filter(|&&x| x != 0).count() as i32
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

    // --- Forward transform / quant / scan: ported EncUT anchors ---

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }
        // Random i16 in the full signed range.
        fn i16(&mut self) -> i16 {
            (self.next() >> 16) as i16
        }
    }

    const FENC: usize = 16;
    const FDEC: usize = 32;

    // Port of `Sub4x4DctAnchor`: returns iDct[row][col].
    fn sub4x4_dct_anchor(pix1: &[u8], o1: usize, pix2: &[u8], o2: usize) -> [[i32; 4]; 4] {
        let mut diff = [[0i32; 4]; 4];
        for y in 0..4 {
            for x in 0..4 {
                diff[y][x] = pix1[o1 + y * FENC + x] as i32 - pix2[o2 + y * FDEC + x] as i32;
            }
        }
        let mut tmp = [[0i32; 4]; 4];
        for i in 0..4 {
            let a03 = diff[i][0] + diff[i][3];
            let a12 = diff[i][1] + diff[i][2];
            let s03 = diff[i][0] - diff[i][3];
            let s12 = diff[i][1] - diff[i][2];
            tmp[0][i] = a03 + a12;
            tmp[1][i] = 2 * s03 + s12;
            tmp[2][i] = a03 - a12;
            tmp[3][i] = s03 - 2 * s12;
        }
        let mut dct = [[0i32; 4]; 4];
        for i in 0..4 {
            let a03 = tmp[i][0] + tmp[i][3];
            let a12 = tmp[i][1] + tmp[i][2];
            let s03 = tmp[i][0] - tmp[i][3];
            let s12 = tmp[i][1] - tmp[i][2];
            dct[i][0] = a03 + a12;
            dct[i][1] = 2 * s03 + s12;
            dct[i][2] = a03 - a12;
            dct[i][3] = s03 - 2 * s12;
        }
        dct
    }

    #[test]
    fn dct_t4_matches_anchor() {
        let mut r = Lcg(0x1234_5678);
        for _ in 0..500 {
            let mut pix1 = [0u8; 16 * FENC];
            let mut pix2 = [0u8; 16 * FDEC];
            for i in 0..4 {
                for j in 0..4 {
                    pix1[i * FENC + j] = r.byte();
                    pix2[i * FDEC + j] = r.byte();
                }
            }
            let anchor = sub4x4_dct_anchor(&pix1, 0, &pix2, 0);
            let mut dct = [0i16; 16];
            dct_t4(&mut dct, &pix1, 0, FENC, &pix2, 0, FDEC);
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(anchor[j][i] as i16, dct[i * 4 + j]);
                }
            }
        }
    }

    #[test]
    fn dct_four_t4_matches_anchor() {
        let mut r = Lcg(0x9E37_79B9);
        for _ in 0..300 {
            let mut pix1 = [0u8; 16 * FENC];
            let mut pix2 = [0u8; 16 * FDEC];
            for i in 0..8 {
                for j in 0..8 {
                    pix1[i * FENC + j] = r.byte();
                    pix2[i * FDEC + j] = r.byte();
                }
            }
            let sub = [(0, 0), (4, 4), (4 * FENC, 4 * FDEC), (4 * FENC + 4, 4 * FDEC + 4)];
            let mut dct = [0i16; 64];
            dct_four_t4(&mut dct, &pix1, 0, FENC, &pix2, 0, FDEC);
            for (k, &(o1, o2)) in sub.iter().enumerate() {
                let anchor = sub4x4_dct_anchor(&pix1, o1, &pix2, o2);
                for i in 0..4 {
                    for j in 0..4 {
                        assert_eq!(anchor[j][i] as i16, dct[k * 16 + i * 4 + j]);
                    }
                }
            }
        }
    }

    #[test]
    fn dct_constant_residual_is_dc_only() {
        // A constant residual c produces D[0] = 16c, all other coefficients 0.
        for c in [1u8, 5, 17, 40] {
            let pix1 = [128u8 + c; 16 * FENC];
            let pix2 = [128u8; 16 * FDEC];
            let mut dct = [0i16; 16];
            dct_t4(&mut dct, &pix1, 0, FENC, &pix2, 0, FDEC);
            assert_eq!(dct[0], 16 * c as i16);
            assert!(dct[1..].iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn dct_roundtrip_recovers_residual() {
        // Cf Cf^T = diag(4,10,4,10), so Cf^-1 = Cf^T * diag(1/4,1/10,1/4,1/10).
        let cf = [[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, -1.0, -2.0], [1.0, -1.0, -1.0, 1.0], [1.0, -2.0, 2.0, -1.0]];
        let dg = [0.25, 0.1, 0.25, 0.1];
        let mut cinv = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                cinv[i][j] = cf[j][i] * dg[j];
            }
        }
        let matmul = |a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]| {
            let mut o = [[0.0f64; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    for k in 0..4 {
                        o[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            o
        };
        let transpose = |a: &[[f64; 4]; 4]| {
            let mut o = [[0.0f64; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    o[i][j] = a[j][i];
                }
            }
            o
        };
        let mut r = Lcg(0xDEAD_BEEF);
        for _ in 0..200 {
            // Residual in [-100, 100]; base 128 keeps pixels in range, diff exact.
            let mut res = [[0i32; 4]; 4];
            let mut pix1 = [0u8; 16 * FENC];
            let pix2 = [128u8; 16 * FDEC];
            for i in 0..4 {
                for j in 0..4 {
                    let v = (r.byte() as i32 % 201) - 100;
                    res[i][j] = v;
                    pix1[i * FENC + j] = (128 + v) as u8;
                }
            }
            let mut dct = [0i16; 16];
            dct_t4(&mut dct, &pix1, 0, FENC, &pix2, 0, FDEC);
            let mut d = [[0.0f64; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    d[i][j] = dct[i * 4 + j] as f64;
                }
            }
            let rec = matmul(&matmul(&cinv, &d), &transpose(&cinv));
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(rec[i][j].round() as i32, res[i][j]);
                }
            }
        }
    }

    // Port of `WelsHadamardT4DcAnchor`.
    fn hadamard_t4_dc_anchor(dct: &[i16]) -> [i16; 16] {
        let mut p = [0i32; 16];
        let mut out = [0i16; 16];
        let mut i = 0;
        while i < 16 {
            let idx = ((i & 0x08) << 4) + ((i & 0x04) << 3);
            let s0 = dct[idx] as i32 + dct[idx + 80] as i32;
            let s3 = dct[idx] as i32 - dct[idx + 80] as i32;
            let s1 = dct[idx + 16] as i32 + dct[idx + 64] as i32;
            let s2 = dct[idx + 16] as i32 - dct[idx + 64] as i32;
            p[i] = s0 + s1;
            p[i + 2] = s0 - s1;
            p[i + 1] = s3 + s2;
            p[i + 3] = s3 - s2;
            i += 4;
        }
        for i in 0..4 {
            let s0 = p[i] + p[i + 12];
            let s3 = p[i] - p[i + 12];
            let s1 = p[i + 4] + p[i + 8];
            let s2 = p[i + 4] - p[i + 8];
            out[i] = ((s0 + s1 + 1) >> 1).clamp(-32768, 32767) as i16;
            out[i + 8] = ((s0 - s1 + 1) >> 1).clamp(-32768, 32767) as i16;
            out[i + 4] = ((s3 + s2 + 1) >> 1).clamp(-32768, 32767) as i16;
            out[i + 12] = ((s3 - s2 + 1) >> 1).clamp(-32768, 32767) as i16;
        }
        out
    }

    #[test]
    fn hadamard_t4_dc_matches_anchor() {
        let mut r = Lcg(0x0BAD_F00D);
        for _ in 0..200 {
            let dct: alloc::vec::Vec<i16> = (0..128 * 16).map(|_| (r.next() & 32767) as i16 - 16384).collect();
            let anchor = hadamard_t4_dc_anchor(&dct);
            let mut out = [0i16; 16];
            hadamard_t4_dc(&mut out, &dct);
            assert_eq!(out, anchor);
        }
    }

    // Port of `WelsQuant4x4MaxAnchor` (returns signed quant, reports max magnitude).
    fn quant4x4_max_anchor(dct: &mut [i16; 16], ff: &[i16; 8], mf: &[i16; 8]) -> i16 {
        let mut max_abs = 0i16;
        for (i, d) in dct.iter_mut().enumerate() {
            let j = i & 0x07;
            let orig = *d as i32;
            let sign = orig >> 31;
            let q = (((ff[j] as i32 + ((sign ^ orig) - sign)) * mf[j] as i32) >> 16) as i16;
            if max_abs < q {
                max_abs = q;
            }
            *d = ((sign ^ q as i32) - sign) as i16;
        }
        max_abs
    }

    #[test]
    fn quant4x4_matches_anchor() {
        let mut r = Lcg(0x1357_9BDF);
        for _ in 0..500 {
            let ff: [i16; 8] = core::array::from_fn(|_| (r.next() & 32767) as i16);
            let mf: [i16; 8] = core::array::from_fn(|_| (r.next() & 32767) as i16);
            let dct0: [i16; 16] = core::array::from_fn(|_| r.i16());
            let mut a = dct0;
            let mut b = dct0;
            quant4x4_max_anchor(&mut a, &ff, &mf);
            quant4x4(&mut b, &ff, &mf);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn quant4x4_dc_matches_anchor() {
        let mut r = Lcg(0x2468_ACE0);
        for _ in 0..500 {
            let ff = (r.next() & 32767) as i16;
            let mf = (r.next() & 32767) as i16;
            let dct0: [i16; 16] = core::array::from_fn(|_| r.i16());
            let mut expect = dct0;
            for x in expect.iter_mut() {
                *x = wels_new_quant(*x, ff, mf);
            }
            let mut got = dct0;
            quant4x4_dc(&mut got, ff, mf);
            assert_eq!(expect, got);
        }
    }

    #[test]
    fn quant_four4x4_matches_anchor() {
        let mut r = Lcg(0xF00D_CAFE);
        for _ in 0..300 {
            let ff: [i16; 8] = core::array::from_fn(|_| (r.next() & 32767) as i16);
            let mf: [i16; 8] = core::array::from_fn(|_| (r.next() & 32767) as i16);
            let dct0: [i16; 64] = core::array::from_fn(|_| r.i16());
            let mut a = dct0;
            for k in 0..4 {
                let mut blk: [i16; 16] = a[k * 16..k * 16 + 16].try_into().unwrap();
                quant4x4_max_anchor(&mut blk, &ff, &mf);
                a[k * 16..k * 16 + 16].copy_from_slice(&blk);
            }
            let mut b = dct0;
            quant_four4x4(&mut b, &ff, &mf);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn quant_four4x4_max_matches_anchor() {
        let mut r = Lcg(0xABCD_1234);
        for _ in 0..300 {
            let ff: [i16; 8] = core::array::from_fn(|_| (r.next() & 32767) as i16);
            let mf: [i16; 8] = core::array::from_fn(|_| (r.next() & 32767) as i16);
            let dct0: [i16; 64] = core::array::from_fn(|_| (r.next() & 65535) as i16);
            let mut a = dct0;
            let mut max_a = [0i16; 4];
            for k in 0..4 {
                let mut blk: [i16; 16] = a[k * 16..k * 16 + 16].try_into().unwrap();
                max_a[k] = quant4x4_max_anchor(&mut blk, &ff, &mf);
                a[k * 16..k * 16 + 16].copy_from_slice(&blk);
            }
            let mut b = dct0;
            let mut max_b = [0i16; 4];
            quant_four4x4_max(&mut b, &ff, &mf, &mut max_b);
            assert_eq!(a, b);
            assert_eq!(max_a, max_b);
        }
    }

    // Port of `WelsHadamardQuant2x2SkipAnchor`.
    fn hadamard_quant2x2_skip_anchor(rs: &[i16], ff: i16, mf: i16) -> i32 {
        let threshold = ((((1i32 << 16) - 1) / mf as i32) - ff as i32) as i16 as i32;
        let s0 = rs[0].wrapping_add(rs[32]);
        let s1 = rs[0].wrapping_sub(rs[32]);
        let s2 = rs[16].wrapping_add(rs[48]);
        let s3 = rs[16].wrapping_sub(rs[48]);
        let d = [s0.wrapping_add(s2), s0.wrapping_sub(s2), s1.wrapping_add(s3), s1.wrapping_sub(s3)];
        d.iter().any(|&x| (x as i32).abs() > threshold) as i32
    }

    #[test]
    fn hadamard_quant2x2_skip_matches_anchor() {
        let mut r = Lcg(0x5151_5151);
        for _ in 0..500 {
            let rs: [i16; 64] = core::array::from_fn(|_| (r.next() & 32767) as i16 - 16384);
            let ff = (r.next() & 32767) as i16;
            let mf = ((r.next() & 32766) + 1) as i16; // avoid divide-by-zero
            assert_eq!(hadamard_quant2x2_skip(&rs, ff, mf), hadamard_quant2x2_skip_anchor(&rs, ff, mf));
        }
    }

    // Port of `WelsHadamardQuant2x2Anchor`.
    fn hadamard_quant2x2_anchor(rs: &mut [i16], ff: i16, mf: i16, dct: &mut [i16; 4], block: &mut [i16; 4]) -> i32 {
        let s0 = rs[0].wrapping_add(rs[32]);
        let s1 = rs[0].wrapping_sub(rs[32]);
        let s2 = rs[16].wrapping_add(rs[48]);
        let s3 = rs[16].wrapping_sub(rs[48]);
        rs[0] = 0;
        rs[16] = 0;
        rs[32] = 0;
        rs[48] = 0;
        dct[0] = s0.wrapping_add(s2);
        dct[1] = s0.wrapping_sub(s2);
        dct[2] = s1.wrapping_add(s3);
        dct[3] = s1.wrapping_sub(s3);
        for d in dct.iter_mut() {
            *d = wels_new_quant(*d, ff, mf);
        }
        block.copy_from_slice(dct);
        (0..4).map(|i| (block[i] != 0) as i32).sum()
    }

    #[test]
    fn hadamard_quant2x2_matches_anchor() {
        let mut r = Lcg(0x2727_2727);
        for _ in 0..500 {
            let rs0: [i16; 64] = core::array::from_fn(|_| (r.next() & 32767) as i16 - 16384);
            let ff = (r.next() & 32767) as i16;
            let mf = (r.next() & 32767) as i16;
            let mut rs_a = rs0;
            let mut rs_b = rs0;
            let (mut da, mut db) = ([0i16; 4], [0i16; 4]);
            let (mut ba, mut bb) = ([0i16; 4], [0i16; 4]);
            let ra = hadamard_quant2x2_anchor(&mut rs_a, ff, mf, &mut da, &mut ba);
            let rb = hadamard_quant2x2(&mut rs_b, ff, mf, &mut db, &mut bb);
            assert_eq!(ra, rb);
            assert_eq!(da, db);
            assert_eq!(ba, bb);
            assert_eq!(rs_a, rs_b);
        }
    }

    #[test]
    fn scan4x4_matches_anchor() {
        let mut r = Lcg(0x3939_3939);
        for _ in 0..500 {
            let dct: [i16; 16] = core::array::from_fn(|_| r.i16());
            let mut lvl = [0i16; 16];
            scan4x4_dcac(&mut lvl, &dct);
            // DcAc / Dc anchor (TestScan_4x4_dcc layout).
            let expect = [
                dct[0], dct[1], dct[4], dct[8], dct[5], dct[2], dct[3], dct[6], dct[9], dct[12], dct[13], dct[10],
                dct[7], dct[11], dct[14], dct[15],
            ];
            assert_eq!(lvl, expect);

            let mut ac = [0i16; 16];
            scan4x4_ac(&mut ac, &dct);
            let expect_ac = [
                dct[1], dct[4], dct[8], dct[5], dct[2], dct[3], dct[6], dct[9], dct[12], dct[13], dct[10], dct[7],
                dct[11], dct[14], dct[15], 0,
            ];
            assert_eq!(ac, expect_ac);
        }
    }

    #[test]
    fn calculate_single_ctr_and_nnz() {
        // Reference for the non-zero count.
        fn nnz_ref(level: &[i16; 16]) -> i32 {
            (16 - level.iter().filter(|&&x| x == 0).count()) as i32
        }
        // Reference for single-ctr (straightforward re-implementation).
        fn ctr_ref(dct: &[i16; 16]) -> i32 {
            const T: [i32; 16] = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mut ctr = 0;
            let mut idx: i32 = 15;
            while idx >= 0 && dct[idx as usize] == 0 {
                idx -= 1;
            }
            while idx >= 0 {
                idx -= 1;
                let start = idx;
                while idx >= 0 && dct[idx as usize] == 0 {
                    idx -= 1;
                }
                ctr += T[(start - idx) as usize];
            }
            ctr
        }
        let mut r = Lcg(0x4242_4242);
        for _ in 0..2000 {
            // Sparse-ish coefficients so runs of zeros occur.
            let dct: [i16; 16] = core::array::from_fn(|_| {
                if r.next() & 1 == 0 {
                    0
                } else {
                    r.i16()
                }
            });
            assert_eq!(calculate_single_ctr4x4(&dct), ctr_ref(&dct));
            assert_eq!(get_none_zero_count(&dct), nnz_ref(&dct));
        }
    }
}
