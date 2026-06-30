//! Integer + sub-pel motion estimation for P-slice inter prediction.
//!
//! Ported in spirit from `reference/codec/encoder/core/src/svc_motion_estimate.cpp`
//! (`WelsDiamondSearch` + the half/quarter-pel refinement in `svc_encode_mb.cpp`).
//! The encoder has freedom in *how* it searches — only the resulting MV/mvd
//! syntax must round-trip — so this is a pragmatic small-diamond integer search
//! (with a brute-force fallback for testability) followed by half- then
//! quarter-pel refinement scored with SATD.
//!
//! All candidate predictions are formed through [`crate::dsp::mc`] using the
//! **exact** clamp the decoder applies in `recon_inter::base_mc` (PAD = 32), so
//! whatever MV is chosen, the encoder's prediction is bit-identical to the
//! decoder's reconstruction. MVs are carried in quarter-pel units throughout.

use crate::dsp::mc::mc_luma;
use crate::dsp::sad::sad;
use crate::dsp::satd::{satd16x16, satd16x8, satd4x4, satd4x8, satd8x16, satd8x4, satd8x8};
use crate::dsp::{Blk, Dim, Mv};

/// An integer pixel position `(x, y)` in coded-plane coordinates.
#[derive(Clone, Copy)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

/// Reference-plane padding, matching the decoder's `picture::PADDING`. The
/// encoder reconstruction planes carry the same border so MC reads agree.
pub const PAD: i32 = 32;

/// A read-only view of one reference luma plane for the searcher.
#[derive(Clone, Copy)]
pub struct RefView<'a> {
    pub plane: &'a [u8],
    pub stride: usize,
    /// Linear index of coded pixel (0, 0).
    pub origin: usize,
    /// Coded width / height in luma samples.
    pub pic_w: i32,
    pub pic_h: i32,
}

// --- Self-contained MV prediction (decoder-independent so the encoder feature
//     builds without the decoder). These must stay bit-identical to
//     `decoder::mv_pred`; a cross-check test guards that when both are present.

/// `REF_NOT_AVAIL` (-2): neighbour macroblock outside the picture.
pub const REF_NOT_AVAIL: i8 = -2;
/// `REF_NOT_IN_LIST` (-1): neighbour is intra (no list-0 reference).
pub const REF_NOT_IN_LIST: i8 = -1;

/// `g_kuiCache30ScanIdx`: block scan index → position in the 30-entry cache.
pub const CACHE30_SCAN_IDX: [usize; 16] = [7, 8, 13, 14, 9, 10, 15, 16, 19, 20, 25, 26, 21, 22, 27, 28];

/// `WelsMedian`: median of three.
#[inline]
pub fn median(x: i32, y: i32, z: i32) -> i32 {
    let mn = x.min(y).min(z);
    let mx = x.max(y).max(z);
    (x + y + z) - (mn + mx)
}

/// `PredMv` (spec 8.4.1.3): predict the list-0 MV for a partition rooted at
/// block-scan index `part_idx` of width `part_width` (4x4 units). Exact copy of
/// `decoder::mv_pred::pred_mv`.
pub fn pred_mv(mv: &[[i16; 2]; 30], ref_idx: &[i8; 30], part_idx: usize, part_width: usize, iref: i8) -> [i16; 2] {
    let left_idx = CACHE30_SCAN_IDX[part_idx] - 1;
    let top_idx = CACHE30_SCAN_IDX[part_idx] - 6;
    let right_top_idx = top_idx + part_width;
    let left_top_idx = top_idx - 1;

    let left_ref = ref_idx[left_idx];
    let top_ref = ref_idx[top_idx];
    let right_top_ref = ref_idx[right_top_idx];
    let left_top_ref = ref_idx[left_top_idx];

    let amv = mv[left_idx];
    let bmv = mv[top_idx];
    let mut cmv = mv[right_top_idx];
    let mut diagonal_ref = right_top_ref;
    if diagonal_ref == REF_NOT_AVAIL {
        diagonal_ref = left_top_ref;
        cmv = mv[left_top_idx];
    }

    if top_ref == REF_NOT_AVAIL && diagonal_ref == REF_NOT_AVAIL && left_ref >= REF_NOT_IN_LIST {
        return amv;
    }

    let match_ref = (iref == left_ref) as i32 + (iref == top_ref) as i32 + (iref == diagonal_ref) as i32;
    if match_ref == 1 {
        if iref == left_ref {
            amv
        } else if iref == top_ref {
            bmv
        } else {
            cmv
        }
    } else {
        [
            median(amv[0] as i32, bmv[0] as i32, cmv[0] as i32) as i16,
            median(amv[1] as i32, bmv[1] as i32, cmv[1] as i32) as i16,
        ]
    }
}

/// Bit length of `ue(code)` (Exp-Golomb).
#[inline]
fn ue_bits(code: u32) -> u32 {
    let c = code + 1;
    let n = 32 - c.leading_zeros();
    2 * n - 1
}

/// Bit length of `se(d)` — the cost (in bits) of coding one mvd component.
#[inline]
pub fn mvd_bits(d: i32) -> u32 {
    let code = if d <= 0 { (-d as u32) * 2 } else { (d as u32) * 2 - 1 };
    ue_bits(code)
}

/// Lagrangian motion lambda from QP (integer approximation; encoder freedom).
#[inline]
pub fn me_lambda(qp: i32) -> i32 {
    // Mild weighting that grows with QP, bottoming out at 1.
    (((qp - 12).max(0)) / 4 + 1).max(1)
}

/// Clamp a signalled full-pel-relative MV component to the decoder's read
/// window (spec 8.4.2.2.1 as implemented by `base_mc`). `px` is the partition's
/// luma pixel coordinate, `mv_q` the signalled MV in quarter-pel.
#[inline]
fn clamp_full(px: i32, mv_q: i32, pic: i32) -> i32 {
    let abs = px << 2;
    (abs + mv_q).clamp((-PAD + 2) << 2, (pic + PAD - 19) << 2)
}

/// SATD dispatcher over the supported partition sizes (falls back to a tiling of
/// 4x4 SATDs for any other size).
fn satd_wh(a: &[u8], sta: usize, b: &[u8], stb: usize, w: usize, h: usize) -> i32 {
    match (w, h) {
        (16, 16) => satd16x16(a, sta, b, stb),
        (16, 8) => satd16x8(a, sta, b, stb),
        (8, 16) => satd8x16(a, sta, b, stb),
        (8, 8) => satd8x8(a, sta, b, stb),
        (8, 4) => satd8x4(a, sta, b, stb),
        (4, 8) => satd4x8(a, sta, b, stb),
        (4, 4) => satd4x4(a, sta, b, stb),
        _ => {
            let mut s = 0;
            let mut y = 0;
            while y < h {
                let mut x = 0;
                while x < w {
                    s += satd4x4(&a[y * sta + x..], sta, &b[y * stb + x..], stb);
                    x += 4;
                }
                y += 4;
            }
            s
        }
    }
}

/// Integer-position SAD of the source block against the reference at signalled
/// integer MV `(mvx_q, mvy_q)` (both multiples of 4). The decoder clamp keeps
/// every read inside the bordered plane.
#[inline]
fn integer_sad(refv: &RefView, pos: Pos, mvx_q: i32, mvy_q: i32, src: Blk, dim: Dim) -> u32 {
    let fx = clamp_full(pos.x, mvx_q, refv.pic_w) >> 2;
    let fy = clamp_full(pos.y, mvy_q, refv.pic_h) >> 2;
    let ro = (refv.origin as i32 + fx + fy * refv.stride as i32) as usize;
    sad(&src.data[src.off..], src.stride, &refv.plane[ro..], refv.stride, dim.w, dim.h)
}

/// Form a (possibly sub-pel) prediction into `scratch` (stride = `w`) for the
/// signalled MV `(mvx_q, mvy_q)`, applying the decoder clamp.
fn subpel_pred(scratch: &mut [u8], refv: &RefView, pos: Pos, mvx_q: i32, mvy_q: i32, dim: Dim) {
    let fx = clamp_full(pos.x, mvx_q, refv.pic_w);
    let fy = clamp_full(pos.y, mvy_q, refv.pic_h);
    let so = (refv.origin as i32 + (fx >> 2) + (fy >> 2) * refv.stride as i32) as usize;
    mc_luma(scratch, dim.w, refv.plane, so, refv.stride, Mv { x: fx as i16, y: fy as i16 }, dim);
}

/// Result of a motion search: the chosen MV (quarter-pel) and its SATD-based
/// rate-distortion cost (`SATD + lambda * mvd_bits`).
pub struct MeResult {
    pub mv: [i16; 2],
    pub cost: i32,
}

/// Full integer-pel search over `[-range, range]^2` quarter-pel-aligned MVs.
/// Deterministic; used as a correctness oracle and a fallback. Returns the best
/// integer MV (quarter-pel units) minimizing `SAD + lambda * mvd_bits`.
pub fn full_search(refv: &RefView, pos: Pos, src: Blk, dim: Dim, mvp: [i16; 2], range: i32, lambda: i32) -> [i16; 2] {
    let mut best = [0i16; 2];
    let mut best_cost = i32::MAX;
    let mut dy = -range;
    while dy <= range {
        let mut dx = -range;
        while dx <= range {
            let mvx = dx << 2;
            let mvy = dy << 2;
            let s = integer_sad(refv, pos, mvx, mvy, src, dim) as i32;
            let bits = mvd_bits(mvx - mvp[0] as i32) + mvd_bits(mvy - mvp[1] as i32);
            let cost = s + lambda * bits as i32;
            if cost < best_cost {
                best_cost = cost;
                best = [mvx as i16, mvy as i16];
            }
            dx += 1;
        }
        dy += 1;
    }
    best
}

/// Small-diamond integer search seeded from the predictor and the origin, then
/// half- and quarter-pel refinement. Returns the final quarter-pel MV + cost.
pub fn search_mv(refv: &RefView, pos: Pos, src: Blk, dim: Dim, mvp: [i16; 2], lambda: i32) -> MeResult {
    let Dim { w, h } = dim;
    let icost = |mvx: i32, mvy: i32| -> i32 {
        let s = integer_sad(refv, pos, mvx, mvy, src, dim) as i32;
        let bits = mvd_bits(mvx - mvp[0] as i32) + mvd_bits(mvy - mvp[1] as i32);
        s + lambda * bits as i32
    };

    // Seed: predictor rounded to integer pel, and the origin.
    let mvp_i = [(mvp[0] as i32 >> 2) << 2, (mvp[1] as i32 >> 2) << 2];
    let mut best = mvp_i;
    let mut best_cost = icost(mvp_i[0], mvp_i[1]);
    let zero_cost = icost(0, 0);
    if zero_cost < best_cost {
        best = [0, 0];
        best_cost = zero_cost;
    }

    // Iterative small diamond (step = 1 integer pel = 4 qpel).
    const STEP: i32 = 4;
    const DIRS: [(i32, i32); 4] = [(0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0)];
    for _ in 0..64 {
        let mut moved = false;
        for (ddx, ddy) in DIRS {
            let mvx = best[0] + ddx;
            let mvy = best[1] + ddy;
            let c = icost(mvx, mvy);
            if c < best_cost {
                best_cost = c;
                best = [mvx, mvy];
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    // Sub-pel refinement scored with SATD.
    let mut scratch = [0u8; 256];
    let scost = |mvx: i32, mvy: i32, scratch: &mut [u8]| -> i32 {
        subpel_pred(scratch, refv, pos, mvx, mvy, dim);
        let d = satd_wh(scratch, w, &src.data[src.off..], src.stride, w, h);
        let bits = mvd_bits(mvx - mvp[0] as i32) + mvd_bits(mvy - mvp[1] as i32);
        d + lambda * bits as i32
    };

    let mut best_q = best;
    let mut best_qcost = scost(best_q[0], best_q[1], &mut scratch);

    for &step in &[2i32, 1] {
        let center = best_q;
        let mut local = best_q;
        let mut local_cost = best_qcost;
        let mut dy = -step;
        while dy <= step {
            let mut dx = -step;
            while dx <= step {
                if dx != 0 || dy != 0 {
                    let c = scost(center[0] + dx, center[1] + dy, &mut scratch);
                    if c < local_cost {
                        local_cost = c;
                        local = [center[0] + dx, center[1] + dy];
                    }
                }
                dx += step;
            }
            dy += step;
        }
        best_q = local;
        best_qcost = local_cost;
    }

    MeResult { mv: [best_q[0] as i16, best_q[1] as i16], cost: best_qcost }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    struct Lcg(u32);
    impl Lcg {
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 24) as u8
        }
    }

    // Build a bordered reference plane (PAD on all sides) filled with random
    // texture, edge-replicated into the border, for a `pic_w` x `pic_h` picture.
    fn make_ref(pic_w: usize, pic_h: usize, seed: u32) -> (Vec<u8>, usize, usize) {
        let stride = pic_w + 2 * PAD as usize;
        let height = pic_h + 2 * PAD as usize;
        let origin = PAD as usize * stride + PAD as usize;
        let mut plane = vec![0u8; stride * height];
        let mut lcg = Lcg(seed);
        for y in 0..pic_h {
            for x in 0..pic_w {
                plane[origin + y * stride + x] = lcg.next_u8();
            }
        }
        crate::dsp::expand::expand_plane(&mut plane, stride, PAD as usize, pic_w, pic_h, origin);
        (plane, stride, origin)
    }

    // Guard: the encoder's self-contained MV predictor must stay bit-identical
    // to the decoder's (the round-trip depends on it). Only runs when the
    // decoder is also compiled in.
    #[cfg(feature = "decoder")]
    #[test]
    fn pred_mv_matches_decoder() {
        use crate::decoder::mv_pred as dec;
        let mut s = 0x9E37_79B9u32;
        let mut rng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        for _ in 0..2000 {
            let mut mv = [[0i16; 2]; 30];
            let mut r = [0i8; 30];
            for j in 0..30 {
                mv[j] = [(rng() % 512) as i16 - 256, (rng() % 512) as i16 - 256];
                r[j] = (rng() % 18) as i8 - 2;
            }
            let iref = (rng() % 18) as i8 - 2;
            for &(idx, pw) in &[(0usize, 4usize), (8, 4), (0, 2), (4, 2), (5, 1)] {
                assert_eq!(pred_mv(&mv, &r, idx, pw, iref), dec::pred_mv(&mv, &r, idx, pw, iref));
            }
            assert_eq!(median(1, 2, 3), dec::median(1, 2, 3));
            assert_eq!((REF_NOT_AVAIL, REF_NOT_IN_LIST), (dec::REF_NOT_AVAIL, dec::REF_NOT_IN_LIST));
        }
    }

    #[test]
    fn mvd_bits_matches_se_length() {
        // se(0)=1bit, se(1)=3, se(-1)=3, se(2)=5, se(-2)=5, se(3)=5 ...
        assert_eq!(mvd_bits(0), 1);
        assert_eq!(mvd_bits(1), 3);
        assert_eq!(mvd_bits(-1), 3);
        assert_eq!(mvd_bits(2), 5);
        assert_eq!(mvd_bits(-2), 5);
        assert_eq!(mvd_bits(3), 5);
    }

    #[test]
    fn full_search_recovers_pure_translation() {
        // Source = a block of the reference translated by an integer MV; the
        // searcher must recover it exactly (SAD == 0 at the true MV).
        let (pic_w, pic_h) = (64usize, 64usize);
        let (refp, stride, origin) = make_ref(pic_w, pic_h, 0xC0FFEE01);
        let refv = RefView { plane: &refp, stride, origin, pic_w: pic_w as i32, pic_h: pic_h as i32 };

        for &(tx, ty) in &[(0i32, 0i32), (3, 0), (0, -2), (-4, 5), (6, -3)] {
            // Source block of size 16x16 rooted at MB (1,1) = pixel (16,16).
            let (px, py) = (16i32, 16i32);
            let mut src = vec![0u8; 64 * 64];
            let src_stride = 64;
            let src_off = (py as usize) * src_stride + px as usize;
            // Copy ref[(px+tx),(py+ty)] block into the source area.
            for y in 0..16 {
                for x in 0..16 {
                    let ro = (origin as i32 + (px + tx) + x + (py + ty + y) * stride as i32) as usize;
                    src[src_off + y as usize * src_stride + x as usize] = refp[ro];
                }
            }
            let mv = full_search(
                &refv,
                Pos { x: px, y: py },
                Blk { data: &src, off: src_off, stride: src_stride },
                Dim { w: 16, h: 16 },
                [0, 0],
                16,
                1,
            );
            assert_eq!(mv, [(tx << 2) as i16, (ty << 2) as i16], "translation ({tx},{ty})");
        }
    }

    // Build a bordered reference from a smooth radial "bowl" so the SAD surface
    // is unimodal — the regime real (smoothly moving) video presents to a greedy
    // diamond, unlike adversarial random texture.
    fn make_smooth_ref(pic_w: usize, pic_h: usize) -> (Vec<u8>, usize, usize) {
        let stride = pic_w + 2 * PAD as usize;
        let height = pic_h + 2 * PAD as usize;
        let origin = PAD as usize * stride + PAD as usize;
        let mut plane = vec![0u8; stride * height];
        let cx = pic_w as i32 / 2;
        let cy = pic_h as i32 / 2;
        for y in 0..pic_h as i32 {
            for x in 0..pic_w as i32 {
                let v = (((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 16).min(255);
                plane[origin + y as usize * stride + x as usize] = v as u8;
            }
        }
        crate::dsp::expand::expand_plane(&mut plane, stride, PAD as usize, pic_w, pic_h, origin);
        (plane, stride, origin)
    }

    #[test]
    fn diamond_reduces_cost_vs_predictor_and_finds_translation() {
        let (pic_w, pic_h) = (96usize, 96usize);
        let (refp, stride, origin) = make_smooth_ref(pic_w, pic_h);
        let refv = RefView { plane: &refp, stride, origin, pic_w: pic_w as i32, pic_h: pic_h as i32 };

        let (px, py) = (32i32, 32i32);
        let (tx, ty) = (5i32, -4i32);
        let mut src = vec![0u8; 96 * 96];
        let src_stride = 96;
        let src_off = (py as usize) * src_stride + px as usize;
        for y in 0..16 {
            for x in 0..16 {
                let ro = (origin as i32 + (px + tx) + x + (py + ty + y) * stride as i32) as usize;
                src[src_off + y as usize * src_stride + x as usize] = refp[ro];
            }
        }
        // Integer-SAD at the predictor (0,0) vs at the diamond result, same metric.
        let blk = Blk { data: &src, off: src_off, stride: src_stride };
        let pred_sad = integer_sad(&refv, Pos { x: px, y: py }, 0, 0, blk, Dim { w: 16, h: 16 });
        // With pure distortion (lambda 0) the greedy diamond walks the convex
        // SAD surface to the exact translation; a Lagrangian penalty would (by
        // design) stop short on the shallow gradient near the optimum.
        let res = search_mv(&refv, Pos { x: px, y: py }, blk, Dim { w: 16, h: 16 }, [0, 0], 0);
        assert_eq!(res.mv, [(tx << 2) as i16, (ty << 2) as i16]);
        let found_sad = integer_sad(&refv, Pos { x: px, y: py }, res.mv[0] as i32, res.mv[1] as i32, blk, Dim { w: 16, h: 16 });
        assert!(found_sad < pred_sad, "search SAD {found_sad} not below predictor SAD {pred_sad}");
    }
}
