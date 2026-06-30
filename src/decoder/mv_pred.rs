//! Motion-vector prediction for P slices.
//!
//! Faithful port of `PredMv`, `PredInter16x8Mv`, `PredInter8x16Mv` and
//! `PredPSkipMvFromNeighbor` from
//! `reference/codec/decoder/core/src/mv_pred.cpp`.
//!
//! The neighbour MV/ref-index "cache" is the 30-entry (6-wide × 5-tall) grid
//! `iMotionVector[30]` / `iRefIndex[30]` used by the reference: row 0 holds the
//! top-left / top / top-right neighbours, column 0 the left neighbours, and the
//! current MB's 16 4x4 blocks occupy the inner 4x4 region addressed through
//! [`CACHE30_SCAN_IDX`] (`g_kuiCache30ScanIdx`).

use super::context::DecoderContext;

/// `REF_NOT_AVAIL` (-2): neighbour macroblock outside the slice / picture.
pub const REF_NOT_AVAIL: i8 = -2;
/// `REF_NOT_IN_LIST` (-1): neighbour is intra (no list-0 reference).
pub const REF_NOT_IN_LIST: i8 = -1;

/// `g_kuiCache30ScanIdx`: block scan index → position in the 30-entry cache.
pub const CACHE30_SCAN_IDX: [usize; 16] =
    [7, 8, 13, 14, 9, 10, 15, 16, 19, 20, 25, 26, 21, 22, 27, 28];

/// `g_kuiScan4`: block scan index → raster position (0..15) within the MB.
pub const SCAN4: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

/// `WelsMedian`: median of three.
#[inline]
pub fn median(x: i32, y: i32, z: i32) -> i32 {
    let mn = x.min(y).min(z);
    let mx = x.max(y).max(z);
    (x + y + z) - (mn + mx)
}

/// `PredMv` (spec 8.4.1.3): predict the list-0 MV for a partition rooted at
/// block-scan index `part_idx` of width `part_width` (4x4 units), reference
/// `iref`. Reads the 30-entry neighbour cache.
pub fn pred_mv(
    mv: &[[i16; 2]; 30],
    ref_idx: &[i8; 30],
    part_idx: usize,
    part_width: usize,
    iref: i8,
) -> [i16; 2] {
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

/// `PredInter16x8Mv`: `part_idx` is 0 (top) or 8 (bottom).
pub fn pred_inter16x8(
    mv: &[[i16; 2]; 30],
    ref_idx: &[i8; 30],
    part_idx: usize,
    iref: i8,
) -> [i16; 2] {
    if part_idx == 0 {
        if iref == ref_idx[1] {
            return mv[1];
        }
    } else if iref == ref_idx[18] {
        return mv[18];
    }
    pred_mv(mv, ref_idx, part_idx, 4, iref)
}

/// `PredInter8x16Mv`: `part_idx` is 0 (left) or 4 (right).
pub fn pred_inter8x16(
    mv: &[[i16; 2]; 30],
    ref_idx: &[i8; 30],
    part_idx: usize,
    iref: i8,
) -> [i16; 2] {
    if part_idx == 0 {
        if iref == ref_idx[6] {
            return mv[6];
        }
    } else {
        let mut diagonal_ref = ref_idx[5];
        let mut index = 5;
        if diagonal_ref == REF_NOT_AVAIL {
            diagonal_ref = ref_idx[2];
            index = 2;
        }
        if iref == diagonal_ref {
            return mv[index];
        }
    }
    pred_mv(mv, ref_idx, part_idx, 2, iref)
}

/// One neighbour's contribution to the P_Skip MV derivation.
struct SkipNeighbor {
    is_inter: bool,
    mv: [i16; 2],
    ref_idx: i8,
}

impl SkipNeighbor {
    fn fetch(ctx: &DecoderContext, cur_slice: i32, xy: Option<usize>, block: usize) -> Self {
        match xy {
            Some(xy) if ctx.slice_idc[xy] == cur_slice => {
                let is_inter = ctx.mb_type[xy].is_inter();
                if is_inter {
                    let base = (xy * 16 + block) * 2;
                    SkipNeighbor {
                        is_inter: true,
                        mv: [ctx.mv[base], ctx.mv[base + 1]],
                        ref_idx: ctx.ref_idx[xy * 16 + block],
                    }
                } else {
                    SkipNeighbor { is_inter: false, mv: [0, 0], ref_idx: REF_NOT_IN_LIST }
                }
            }
            _ => SkipNeighbor { is_inter: false, mv: [0, 0], ref_idx: REF_NOT_AVAIL },
        }
    }
}

/// `PredPSkipMvFromNeighbor`: derive the P_Skip MV from the spatial neighbours.
pub fn pred_p_skip_mv(ctx: &DecoderContext, mb_xy: usize) -> [i16; 2] {
    let mb_width = ctx.mb_width;
    let mb_x = mb_xy % mb_width;
    let mb_y = mb_xy / mb_width;
    let cur_slice = ctx.slice_idc[mb_xy];

    let left_xy = if mb_x != 0 { Some(mb_xy - 1) } else { None };
    let top_xy = if mb_y != 0 { Some(mb_xy - mb_width) } else { None };
    let left_top_xy = if mb_x != 0 && mb_y != 0 { Some(mb_xy - mb_width - 1) } else { None };
    let right_top_xy = if mb_x != mb_width - 1 && mb_y != 0 { Some(mb_xy - mb_width + 1) } else { None };

    // Left (block 3), Top (block 12), RightTop (block 12), LeftTop (block 15).
    let left = SkipNeighbor::fetch(ctx, cur_slice, left_xy, 3);
    if left.ref_idx == REF_NOT_AVAIL || (left.ref_idx == 0 && left.mv == [0, 0]) {
        return [0, 0];
    }
    let top = SkipNeighbor::fetch(ctx, cur_slice, top_xy, 12);
    if top.ref_idx == REF_NOT_AVAIL || (top.ref_idx == 0 && top.mv == [0, 0]) {
        return [0, 0];
    }
    let right_top = SkipNeighbor::fetch(ctx, cur_slice, right_top_xy, 12);
    let left_top = SkipNeighbor::fetch(ctx, cur_slice, left_top_xy, 15);

    let mut mv_c = right_top.mv;
    let mut diagonal_ref = right_top.ref_idx;
    if diagonal_ref == REF_NOT_AVAIL {
        diagonal_ref = left_top.ref_idx;
        mv_c = left_top.mv;
    }
    let _ = (left.is_inter, top.is_inter); // availability already folded into ref

    if top.ref_idx == REF_NOT_AVAIL && diagonal_ref == REF_NOT_AVAIL && left.ref_idx >= REF_NOT_IN_LIST {
        return left.mv;
    }

    let match_ref =
        (left.ref_idx == 0) as i32 + (top.ref_idx == 0) as i32 + (diagonal_ref == 0) as i32;
    if match_ref == 1 {
        if left.ref_idx == 0 {
            left.mv
        } else if top.ref_idx == 0 {
            top.mv
        } else {
            mv_c
        }
    } else {
        [
            median(left.mv[0] as i32, top.mv[0] as i32, mv_c[0] as i32) as i16,
            median(left.mv[1] as i32, top.mv[1] as i32, mv_c[1] as i32) as i16,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // Deterministic LCG matching the spirit of the C rand()-driven anchors.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> i64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) & 0x7fff_ffff) as i64
        }
    }

    // ---- Independent anchors transcribed from DecUT_PredMv.cpp ----

    fn anchor_pred_mv(mv: &[[i16; 2]; 30], r: &[i8; 30], part_idx: usize, pw: usize, iref: i8) -> [i16; 2] {
        let left = CACHE30_SCAN_IDX[part_idx] - 1;
        let top = CACHE30_SCAN_IDX[part_idx] - 6;
        let rtop = top + pw;
        let ltop = top - 1;
        let lref = r[left];
        let tref = r[top];
        let rtref = r[rtop];
        let ltref = r[ltop];
        let mut diag = rtref;
        let amv = mv[left];
        let bmv = mv[top];
        let mut cmv = mv[rtop];
        if diag == REF_NOT_AVAIL {
            diag = ltref;
            cmv = mv[ltop];
        }
        let matchref = (iref == lref) as i32 + (iref == tref) as i32 + (iref == diag) as i32;
        if tref == REF_NOT_AVAIL && diag == REF_NOT_AVAIL && lref >= REF_NOT_IN_LIST {
            return amv;
        }
        if matchref == 1 {
            if iref == lref {
                amv
            } else if iref == tref {
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

    fn anchor_16x8(mv: &[[i16; 2]; 30], r: &[i8; 30], part_idx: usize, iref: i8) -> [i16; 2] {
        if part_idx == 0 {
            if iref == r[1] {
                return mv[1];
            }
        } else if iref == r[18] {
            return mv[18];
        }
        anchor_pred_mv(mv, r, part_idx, 4, iref)
    }

    fn anchor_8x16(mv: &[[i16; 2]; 30], r: &[i8; 30], part_idx: usize, iref: i8) -> [i16; 2] {
        if part_idx == 0 {
            if iref == r[6] {
                return mv[6];
            }
        } else {
            let mut diag = r[5];
            let mut index = 5;
            if diag == REF_NOT_AVAIL {
                diag = r[2];
                index = 2;
            }
            if iref == diag {
                return mv[index];
            }
        }
        anchor_pred_mv(mv, r, part_idx, 2, iref)
    }

    fn random_cache(rng: &mut Rng) -> ([[i16; 2]; 30], [i8; 30]) {
        let mut mv = [[0i16; 2]; 30];
        let mut r = [0i8; 30];
        for j in 0..30 {
            mv[j][0] = (rng.next() % 512 - 256) as i16;
            mv[j][1] = (rng.next() % 512 - 256) as i16;
            r[j] = (rng.next() % 18 - 2) as i8; // -2..=15
        }
        (mv, r)
    }

    #[test]
    fn pred_mv_matches_anchor() {
        let mut rng = Rng(0x1234_5678);
        let cases: &[(Vec<usize>, usize)] = &[
            (alloc::vec![0], 4),                 // 16x16
            (alloc::vec![0, 8], 4),              // 16x8
            (alloc::vec![0, 4], 2),              // 8x16
            (alloc::vec![0, 4, 8, 12], 2),       // 8x8
            ((0..16).collect(), 1),              // 4x4
        ];
        for (idxs, pw) in cases {
            for _ in 0..200 {
                let (mv, r) = random_cache(&mut rng);
                for &idx in idxs {
                    let iref = (rng.next() % 18 - 2) as i8;
                    assert_eq!(
                        pred_mv(&mv, &r, idx, *pw, iref),
                        anchor_pred_mv(&mv, &r, idx, *pw, iref),
                        "pred_mv idx={idx} pw={pw} iref={iref}"
                    );
                }
            }
        }
    }

    #[test]
    fn pred_inter16x8_matches_anchor() {
        let mut rng = Rng(0xBADC_0FFE);
        for _ in 0..400 {
            let (mv, r) = random_cache(&mut rng);
            let idx = ((rng.next() & 1) << 3) as usize;
            let iref = (rng.next() % 18 - 2) as i8;
            assert_eq!(pred_inter16x8(&mv, &r, idx, iref), anchor_16x8(&mv, &r, idx, iref));
        }
    }

    #[test]
    fn pred_inter8x16_matches_anchor() {
        let mut rng = Rng(0xFEED_FACE);
        for _ in 0..400 {
            let (mv, r) = random_cache(&mut rng);
            let idx = ((rng.next() & 1) << 2) as usize;
            let iref = (rng.next() % 18 - 2) as i8;
            assert_eq!(pred_inter8x16(&mv, &r, idx, iref), anchor_8x16(&mv, &r, idx, iref));
        }
    }
}
