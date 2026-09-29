//! Picture downsampler, ported from `reference/codec/processing/src/downsample`
//! (`CDownsampling::Process` + the `_c` kernels in `downsamplefuncs.cpp`).
//!
//! As in the C, a plane is first halved (dyadic 2x2 average) while the half size
//! is still larger than the target in both dimensions; an exact half then ends
//! with one more dyadic pass, anything else with the general bilinear kernel
//! (the "fast" 16/15-bit variant for luma, the "accurate" 15-bit one for chroma,
//! matching `InitDownsampleFuncs`' scalar table).
//!
//! PORT: the C refuses to upscale; here any other size (up, or mixed) goes
//! through the same general bilinear kernel with its neighbour reads clamped to
//! the plane, which changes nothing when downscaling. The halving chain is
//! decided per plane from that plane's own size (identical for even sizes).

use alloc::vec;
use alloc::vec::Vec;

use crate::image::{I420, YuvRef};

/// Fixed-point widths of the fast (luma) general kernel.
const FAST_BITS_X: u32 = 16;
const FAST_BITS_Y: u32 = 15;
/// Fixed-point width of the accurate (chroma) general kernel.
const ACCURATE_BITS: u32 = 15;

/// Scale `src` to `dst`'s size. `dst`'s planes are resized to that size if they
/// are not already. An invalid `src` (planes shorter than their stride and size
/// imply) or an empty `dst` leaves `dst` untouched.
pub fn scale_i420(src: &YuvRef<'_>, dst: &mut I420) {
    if !src.is_valid() || dst.width == 0 || dst.height == 0 {
        return;
    }
    let (dw, dh) = (dst.width as usize, dst.height as usize);
    let (dcw, dch) = (dst.width.div_ceil(2) as usize, dst.height.div_ceil(2) as usize);
    dst.y.resize(dw * dh, 0);
    dst.u.resize(dcw * dch, 0);
    dst.v.resize(dcw * dch, 0);

    let luma = PlaneRef {
        data: src.y,
        stride: src.y_stride,
        width: src.width as usize,
        height: src.height as usize,
    };
    scale_plane(luma, &mut dst.y, dw, dh, Kernel::Fast);
    let (scw, sch) = (src.chroma_width() as usize, src.chroma_height() as usize);
    for (plane, out) in [(src.u, &mut dst.u), (src.v, &mut dst.v)] {
        let chroma = PlaneRef {
            data: plane,
            stride: src.c_stride,
            width: scw,
            height: sch,
        };
        scale_plane(chroma, out, dcw, dch, Kernel::Accurate);
    }
}

#[derive(Clone, Copy)]
struct PlaneRef<'a> {
    data: &'a [u8],
    stride: usize,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy)]
enum Kernel {
    Fast,
    Accurate,
}

/// Scale one plane into the tightly packed `dst` (`dw`x`dh`).
fn scale_plane(src: PlaneRef<'_>, dst: &mut [u8], dw: usize, dh: usize, kernel: Kernel) {
    if src.width == dw && src.height == dh {
        for (row, out) in dst.chunks_exact_mut(dw).enumerate() {
            let s = row * src.stride;
            out.copy_from_slice(&src.data[s..s + dw]);
        }
        return;
    }
    let mut owned: Option<Vec<u8>> = None;
    let (mut w, mut h) = (src.width, src.height);
    if dw < w && dh < h {
        loop {
            let view = match &owned {
                Some(buf) => PlaneRef {
                    data: buf,
                    stride: w,
                    width: w,
                    height: h,
                },
                None => src,
            };
            let (hw, hh) = (w >> 1, h >> 1);
            if hw == dw && hh == dh {
                dyadic_half(dst, dw, view);
                return;
            }
            if hw <= dw || hh <= dh {
                break;
            }
            let mut buf = vec![0u8; hw * hh];
            dyadic_half(&mut buf, hw, view);
            owned = Some(buf);
            (w, h) = (hw, hh);
        }
    }
    let view = match &owned {
        Some(buf) => PlaneRef {
            data: buf,
            stride: w,
            width: w,
            height: h,
        },
        None => src,
    };
    match kernel {
        Kernel::Fast => general_fast(dst, dw, dw, dh, view),
        Kernel::Accurate => general_accurate(dst, dw, dw, dh, view),
    }
}

/// `DyadicBilinearDownsampler_c`: each output sample is the rounded average of
/// the rounded averages of a 2x2 block's two rows.
fn dyadic_half(dst: &mut [u8], dst_stride: usize, src: PlaneRef<'_>) {
    let (dw, dh) = (src.width >> 1, src.height >> 1);
    for j in 0..dh {
        let r0 = 2 * j * src.stride;
        let r1 = r0 + src.stride;
        let out = &mut dst[j * dst_stride..j * dst_stride + dw];
        for (i, o) in out.iter_mut().enumerate() {
            let x = 2 * i;
            let t0 = (src.data[r0 + x] as u32 + src.data[r0 + x + 1] as u32 + 1) >> 1;
            let t1 = (src.data[r1 + x] as u32 + src.data[r1 + x + 1] as u32 + 1) >> 1;
            *o = ((t0 + t1 + 1) >> 1) as u8;
        }
    }
}

/// `WELS_ROUND(src / dst * (1 << bits))`, with the C's float / double steps.
fn scale_factor(src: usize, dst: usize, bits: u32) -> u32 {
    let ratio = src as f32 / dst as f32 * (1u32 << bits) as f32;
    (0.5 + ratio as f64) as u32
}

/// `GeneralBilinearFastDownsampler_c`.
fn general_fast(dst: &mut [u8], dst_stride: usize, dw: usize, dh: usize, src: PlaneRef<'_>) {
    const SCALE_X: u32 = 1 << FAST_BITS_X;
    const SCALE_Y: u32 = 1 << FAST_BITS_Y;
    let step_x = scale_factor(src.width, dw, FAST_BITS_X);
    let step_y = scale_factor(src.height, dh, FAST_BITS_Y);
    let (max_x, max_y) = (src.width - 1, src.height - 1);

    let mut y_inv = 1u32 << (FAST_BITS_Y - 1);
    for i in 0..dh - 1 {
        let yy = ((y_inv >> FAST_BITS_Y) as usize).min(max_y);
        let fv = y_inv & (SCALE_Y - 1);
        let r0 = yy * src.stride;
        let r1 = (yy + 1).min(max_y) * src.stride;
        let out = &mut dst[i * dst_stride..i * dst_stride + dw];
        let mut x_inv = 1u32 << (FAST_BITS_X - 1);
        for o in &mut out[..dw - 1] {
            let xx = ((x_inv >> FAST_BITS_X) as usize).min(max_x);
            let xx1 = (xx + 1).min(max_x);
            let fu = x_inv & (SCALE_X - 1);
            let a = src.data[r0 + xx] as u32;
            let b = src.data[r0 + xx1] as u32;
            let c = src.data[r1 + xx] as u32;
            let d = src.data[r1 + xx1] as u32;
            let mut x = (((SCALE_X - 1 - fu) * (SCALE_Y - 1 - fv)) >> FAST_BITS_X) * a;
            x += ((fu * (SCALE_Y - 1 - fv)) >> FAST_BITS_X) * b;
            x += (((SCALE_X - 1 - fu) * fv) >> FAST_BITS_X) * c;
            x += ((fu * fv) >> FAST_BITS_X) * d;
            x >>= FAST_BITS_Y - 1;
            x += 1;
            x >>= 1;
            *o = x.min(u8::MAX as u32) as u8;
            x_inv += step_x;
        }
        out[dw - 1] = src.data[r0 + ((x_inv >> FAST_BITS_X) as usize).min(max_x)];
        y_inv += step_y;
    }
    let (out, row) = last_rows(dst, dst_stride, dw, dh, src, y_inv >> FAST_BITS_Y);
    last_row(out, row, step_x, FAST_BITS_X);
}

/// `GeneralBilinearAccurateDownsampler_c`.
fn general_accurate(dst: &mut [u8], dst_stride: usize, dw: usize, dh: usize, src: PlaneRef<'_>) {
    const SCALE: i64 = 1 << ACCURATE_BITS;
    const ROUND: i64 = 1 << (2 * ACCURATE_BITS - 1);
    let step_x = scale_factor(src.width, dw, ACCURATE_BITS);
    let step_y = scale_factor(src.height, dh, ACCURATE_BITS);
    let (max_x, max_y) = (src.width - 1, src.height - 1);

    let mut y_inv = 1u32 << (ACCURATE_BITS - 1);
    for i in 0..dh - 1 {
        let yy = ((y_inv >> ACCURATE_BITS) as usize).min(max_y);
        let fv = (y_inv as i64) & (SCALE - 1);
        let r0 = yy * src.stride;
        let r1 = (yy + 1).min(max_y) * src.stride;
        let out = &mut dst[i * dst_stride..i * dst_stride + dw];
        let mut x_inv = 1u32 << (ACCURATE_BITS - 1);
        for o in &mut out[..dw - 1] {
            let xx = ((x_inv >> ACCURATE_BITS) as usize).min(max_x);
            let xx1 = (xx + 1).min(max_x);
            let fu = (x_inv as i64) & (SCALE - 1);
            let a = src.data[r0 + xx] as i64;
            let b = src.data[r0 + xx1] as i64;
            let c = src.data[r1 + xx] as i64;
            let d = src.data[r1 + xx1] as i64;
            let x = ((SCALE - 1 - fu) * (SCALE - 1 - fv) * a
                + fu * (SCALE - 1 - fv) * b
                + (SCALE - 1 - fu) * fv * c
                + fu * fv * d
                + ROUND)
                >> (2 * ACCURATE_BITS);
            *o = x.clamp(0, u8::MAX as i64) as u8;
            x_inv += step_x;
        }
        out[dw - 1] = src.data[r0 + ((x_inv >> ACCURATE_BITS) as usize).min(max_x)];
        y_inv += step_y;
    }
    let (out, row) = last_rows(dst, dst_stride, dw, dh, src, y_inv >> ACCURATE_BITS);
    last_row(out, row, step_x, ACCURATE_BITS);
}

/// The kernels' "last row special": nearest-neighbour along one source row.
fn last_row(out: &mut [u8], src_row: &[u8], step_x: u32, bits: u32) {
    let mut x_inv = 1u32 << (bits - 1);
    for o in out {
        *o = src_row[((x_inv >> bits) as usize).min(src_row.len() - 1)];
        x_inv += step_x;
    }
}

/// Destination and source rows for [`last_row`]: the last output row and
/// source row `yy` (clamped to the plane).
fn last_rows<'a, 'b>(
    dst: &'a mut [u8],
    dst_stride: usize,
    dw: usize,
    dh: usize,
    src: PlaneRef<'b>,
    yy: u32,
) -> (&'a mut [u8], &'b [u8]) {
    let r = (yy as usize).min(src.height - 1) * src.stride;
    let o = (dh - 1) * dst_stride;
    (&mut dst[o..o + dw], &src.data[r..r + src.width])
}
