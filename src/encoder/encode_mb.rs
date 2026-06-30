//! Intra macroblock encode + in-place reconstruct, ported from the OpenH264
//! encoder MB path (`svc_encode_mb.cpp` `WelsEncRecI16x16Y` / `WelsEncRecI4x4Y`
//! / chroma) plus the syntax write (`WelsSpatialWriteMbSyn`).
//!
//! For each MB we run an intra mode decision (I16x16 vs I4x4, plus a chroma
//! mode), forward-transform and quantise the residual, **reconstruct in place**
//! into a padded reconstruction plane so later MBs predict from the exact
//! samples the decoder will hold, and emit the MB syntax + CAVLC residual.
//!
//! Correctness is by construction: prediction uses the decoder's in-place
//! predictors ([`crate::dsp::intra_pred`]), and reconstruction replays the
//! decoder's dequant+IDCT on the very levels written to the bitstream, so the
//! encoder's reconstruction is bit-identical to the decoder's output.

use alloc::vec;
use alloc::vec::Vec;

use crate::bits::BitWriter;
use crate::dsp::intra_pred as ip;
use crate::dsp::tables::{G_KUI_CHROMA_DC_SCAN, G_KUI_DEQUANT_COEFF, G_KUI_LUMA_DC_ZIGZAG_SCAN, G_KUI_ZIGZAG_SCAN};
use crate::dsp::transform::{
    dct_four_t4, dct_t4, get_none_zero_count, hadamard_quant2x2, hadamard_t4_dc, idct4x4_add, quant4x4, quant4x4_dc,
    quant_four4x4, scan4x4_ac, scan4x4_dcac,
};

use crate::dsp::mc::{mc_chroma, mc_luma};
use crate::dsp::{Blk, Dim, Mv};

use super::cavlc_writer::write_residual_block;
use super::motion_est::{me_lambda, median, pred_mv, search_mv, Pos, RefView, REF_NOT_AVAIL, REF_NOT_IN_LIST};

// g_kuiInterCbpTable: maps the coded_block_pattern ue code -> cbp value (the
// inverse direction the decoder reads). The encoder needs cbp -> code, derived
// by inversion in [`inter_cbp_code`].
#[rustfmt::skip]
const INTER_CBP_TABLE: [u8; 48] = [
    0, 16,  1,  2,  4,  8, 32,  3,  5, 10, 12, 15, 47,  7, 11, 13,
    14,  6,  9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// Inverse of [`INTER_CBP_TABLE`]: cbp value (0..=47) -> ue code.
fn inter_cbp_code(cbp: u8) -> u32 {
    INTER_CBP_TABLE.iter().position(|&v| v == cbp).unwrap() as u32
}

/// Inter quant rounding offset row: `g_kiQuantInterFF[qp]`. The encoder stores
/// the intra table (`g_iQuantIntraFF = g_kiQuantInterFF + 6 rows`), so the inter
/// offset for `qp` is the intra row `qp-6` (a smaller deadzone, as inter uses
/// `f = 2^qbits/6` vs intra `2^qbits/3`). The rounding only affects which levels
/// result — any levels round-trip — so the `qp<6` fallback is harmless.
#[inline]
fn inter_ff(qp: usize) -> &'static [i16; 8] {
    &QUANT_INTRA_FF[qp.saturating_sub(6)]
}

// Block scan index -> raster (bx,by) within the MB (g_kuiMbCountScan4Idx order).
const BLOCK_BX: [usize; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
const BLOCK_BY: [usize; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];
const BLOCK_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
// 30-entry availability-grid scan index per block (CACHE30_SCAN_IDX).
const CACHE30_SCAN_IDX: [usize; 16] = [7, 8, 13, 14, 9, 10, 15, 16, 19, 20, 25, 26, 21, 22, 27, 28];
// g_kuiI16CbpTable: cbp class -> cbp value.
const I16_CBP_TABLE: [u8; 6] = [0, 16, 32, 15, 31, 47];
// g_kuiChromaQpTable.
#[rustfmt::skip]
const CHROMA_QP_TABLE: [u8; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];
// Inverse of the decoder's g_kuiIntra4x4CbpTable: cbp value -> coded_block_pattern
// ue code (so the decoder's table maps it back to this cbp).
#[rustfmt::skip]
const INTRA4X4_CBP_CODE: [u8; 48] = [
     3, 29, 30, 17, 31, 18, 37,  8, 32, 38, 19,  9, 20, 10, 11,  2,
    16, 33, 34, 21, 35, 22, 39,  4, 36, 40, 23,  5, 24,  6,  7,  1,
    41, 42, 43, 25, 44, 26, 46, 12, 45, 47, 27, 13, 28, 14, 15,  0,
];

// (pred_mode, need_left, need_top, need_left_top); DC handled specially.
const I16_PRED_INFO: [(i8, i32, i32, i32); 4] = [(0, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 0), (3, 1, 1, 1)];
const CHROMA_PRED_INFO: [(i8, i32, i32, i32); 4] = [(0, 0, 0, 0), (1, 1, 0, 0), (2, 0, 1, 0), (3, 1, 1, 1)];
const I4_PRED_INFO: [(i8, i32, i32, i32); 9] = [
    (0, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 0), (3, 0, 1, 0), (4, 1, 1, 1),
    (5, 1, 1, 1), (6, 1, 1, 1), (7, 0, 1, 0), (8, 1, 0, 0),
];

/// Quant multiplier table `g_kiQuantMF[52][8]`.
#[rustfmt::skip]
static QUANT_MF: [[i16; 8]; 52] = [
    [26214,16132,26214,16132,16132,10486,16132,10486],[23832,14980,23832,14980,14980,9320,14980,9320],
    [20164,13108,20164,13108,13108,8388,13108,8388],[18724,11650,18724,11650,11650,7294,11650,7294],
    [16384,10486,16384,10486,10486,6710,10486,6710],[14564,9118,14564,9118,9118,5786,9118,5786],
    [13107,8066,13107,8066,8066,5243,8066,5243],[11916,7490,11916,7490,7490,4660,7490,4660],
    [10082,6554,10082,6554,6554,4194,6554,4194],[9362,5825,9362,5825,5825,3647,5825,3647],
    [8192,5243,8192,5243,5243,3355,5243,3355],[7282,4559,7282,4559,4559,2893,4559,2893],
    [6554,4033,6554,4033,4033,2622,4033,2622],[5958,3745,5958,3745,3745,2330,3745,2330],
    [5041,3277,5041,3277,3277,2097,3277,2097],[4681,2913,4681,2913,2913,1824,2913,1824],
    [4096,2622,4096,2622,2622,1678,2622,1678],[3641,2280,3641,2280,2280,1447,2280,1447],
    [3277,2017,3277,2017,2017,1311,2017,1311],[2979,1873,2979,1873,1873,1165,1873,1165],
    [2521,1639,2521,1639,1639,1049,1639,1049],[2341,1456,2341,1456,1456,912,1456,912],
    [2048,1311,2048,1311,1311,839,1311,839],[1821,1140,1821,1140,1140,723,1140,723],
    [1638,1008,1638,1008,1008,655,1008,655],[1490,936,1490,936,936,583,936,583],
    [1260,819,1260,819,819,524,819,524],[1170,728,1170,728,728,456,728,456],
    [1024,655,1024,655,655,419,655,419],[910,570,910,570,570,362,570,362],
    [819,504,819,504,504,328,504,328],[745,468,745,468,468,291,468,291],
    [630,410,630,410,410,262,410,262],[585,364,585,364,364,228,364,228],
    [512,328,512,328,328,210,328,210],[455,285,455,285,285,181,285,181],
    [410,252,410,252,252,164,252,164],[372,234,372,234,234,146,234,146],
    [315,205,315,205,205,131,205,131],[293,182,293,182,182,114,182,114],
    [256,164,256,164,164,105,164,105],[228,142,228,142,142,90,142,90],
    [205,126,205,126,126,82,126,82],[186,117,186,117,117,73,117,73],
    [158,102,158,102,102,66,102,66],[146,91,146,91,91,57,91,57],
    [128,82,128,82,82,52,82,52],[114,71,114,71,71,45,71,45],
    [102,63,102,63,63,41,63,41],[93,59,93,59,59,36,59,36],
    [79,51,79,51,51,33,51,33],[73,46,73,46,46,28,46,28],
];

/// Intra quant offset `g_iQuantIntraFF = g_kiQuantInterFF + 6` (rows 6..=57).
#[rustfmt::skip]
static QUANT_INTRA_FF: [[i16; 8]; 52] = [
    [1,1,1,1,1,2,1,2],[1,1,1,1,1,2,1,2],[1,2,1,2,2,3,2,3],[1,2,1,2,2,3,2,3],
    [1,2,1,2,2,3,2,3],[1,2,1,2,2,4,2,4],[2,3,2,3,3,4,3,4],[2,3,2,3,3,5,3,5],
    [2,3,2,3,3,5,3,5],[2,4,2,4,4,6,4,6],[3,4,3,4,4,7,4,7],[3,5,3,5,5,8,5,8],
    [3,5,3,5,5,8,5,8],[4,6,4,6,6,9,6,9],[4,7,4,7,7,10,7,10],[5,8,5,8,8,12,8,12],
    [5,8,5,8,8,13,8,13],[6,10,6,10,10,15,10,15],[7,11,7,11,11,17,11,17],[7,12,7,12,12,19,12,19],
    [9,13,9,13,13,21,13,21],[9,15,9,15,15,24,15,24],[11,17,11,17,17,26,17,26],[12,19,12,19,19,30,19,30],
    [13,22,13,22,22,33,22,33],[15,23,15,23,23,38,23,38],[17,27,17,27,27,42,27,42],[19,30,19,30,30,48,30,48],
    [21,33,21,33,33,52,33,52],[24,38,24,38,38,60,38,60],[27,43,27,43,43,67,43,67],[29,47,29,47,47,75,47,75],
    [35,53,35,53,53,83,53,83],[37,60,37,60,60,96,60,96],[43,67,43,67,67,104,67,104],[48,77,48,77,77,121,77,121],
    [53,87,53,87,87,133,87,133],[59,93,59,93,93,150,93,150],[69,107,69,107,107,167,107,167],[75,120,75,120,120,192,120,192],
    [85,133,85,133,133,208,133,208],[96,153,96,153,153,242,153,242],[107,173,107,173,173,267,173,267],[117,187,117,187,187,300,187,300],
    [139,213,139,213,213,333,213,333],[149,240,149,240,240,383,240,383],[171,267,171,267,267,417,267,417],[192,307,192,307,307,483,307,483],
    [213,347,213,347,347,533,347,533],[235,373,235,373,373,600,373,600],[277,427,277,427,427,667,427,667],[299,480,299,480,480,767,480,767],
];

/// Reconstruction-plane border. Matches the decoder's `picture::PADDING` (32)
/// so the inter motion-compensation clamp + border-extension read exactly the
/// same samples the decoder will, guaranteeing bit-identical reconstruction.
const BORDER: usize = 32;

/// A YUV triple of borrowed plane buffers (source or reference). The frame
/// encoder reads these but never mutates them, so several per-slice encoders can
/// share the same source + reference across `std::thread::scope` workers.
#[derive(Clone, Copy)]
pub(crate) struct PlaneRefs<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

/// Frame dimensions in macroblocks.
#[derive(Clone, Copy)]
pub(crate) struct MbDims {
    pub width: usize,
    pub height: usize,
}

/// Reconstruction-plane geometry for a frame: `(ystride, cstride, ylen, clen)`,
/// each plane carrying a [`BORDER`]-wide pad on every side.
pub(crate) fn rec_dims(mb_width: usize, mb_height: usize) -> (usize, usize, usize, usize) {
    let ystride = mb_width * 16 + 2 * BORDER;
    let cstride = mb_width * 8 + 2 * BORDER;
    let ylen = ystride * (mb_height * 16 + 2 * BORDER);
    let clen = cstride * (mb_height * 8 + 2 * BORDER);
    (ystride, cstride, ylen, clen)
}

/// Border-extend reconstruction planes in place (edge replication, matching the
/// decoder's `expand_picture`).
pub(crate) fn expand_reference(rec_y: &mut [u8], rec_u: &mut [u8], rec_v: &mut [u8], dims: MbDims, ystride: usize, cstride: usize) {
    use crate::dsp::expand::expand_plane;
    let lw = dims.width * 16;
    let lh = dims.height * 16;
    let cw = dims.width * 8;
    let ch = dims.height * 8;
    let lo = BORDER * ystride + BORDER;
    let co = BORDER * cstride + BORDER;
    expand_plane(rec_y, ystride, BORDER, lw, lh, lo);
    expand_plane(rec_u, cstride, BORDER, cw, ch, co);
    expand_plane(rec_v, cstride, BORDER, cw, ch, co);
}

/// Macroblock position (in MB units).
#[derive(Clone, Copy)]
struct MbPos {
    x: usize,
    y: usize,
}

/// Left/top neighbour availability.
#[derive(Clone, Copy)]
struct Avail {
    left: bool,
    top: bool,
}

/// Full intra neighbour availability (left/top plus the two diagonals).
#[derive(Clone, Copy)]
struct NeighAvail {
    left: bool,
    top: bool,
    left_top: bool,
    top_right: bool,
}

/// Per-frame encoder state: padded reconstruction + source planes plus the
/// neighbour context (non-zero counts, I4x4 modes, MB types) needed to mirror
/// the decoder's CAVLC nC prediction and intra-mode prediction.
pub(crate) struct FrameEnc<'a> {
    pub mb_width: usize,
    mb_height: usize,
    pub(crate) rec_y: Vec<u8>,
    pub(crate) rec_u: Vec<u8>,
    pub(crate) rec_v: Vec<u8>,
    pub(crate) ystride: usize,
    pub(crate) cstride: usize,
    src_y: &'a [u8],
    src_u: &'a [u8],
    src_v: &'a [u8],
    src_ystride: usize,
    src_cstride: usize,
    qp: i32,
    /// First MB-row of the slice currently being encoded. Neighbour
    /// availability above this row is suppressed so prediction / nC / mvd never
    /// reach across a slice boundary — exactly as the decoder gates on its
    /// per-MB `slice_idc`. `0` for the whole-frame (single-slice) case.
    slice_top_y: usize,
    // Neighbour context, indexed per MB.
    nzc_luma: Vec<i8>,   // mb_count * 16 (raster)
    nzc_chroma: Vec<i8>, // mb_count * 8
    best_mode: Vec<i8>,  // mb_count * 16 (raster) I4x4 best modes (-1 if not nxn)
    is_nxn: Vec<bool>,   // per MB
    // --- Inter (P-slice) state ---
    ref_y: &'a [u8], // border-extended reference planes (empty for I frames)
    ref_u: &'a [u8],
    ref_v: &'a [u8],
    mv: Vec<[i16; 2]>, // mb_count * 16 (raster), signalled list-0 MVs (qpel)
    ref_idx: Vec<i8>,  // mb_count * 16 (raster); -1 == REF_NOT_IN_LIST (intra)
    mb_inter: Vec<bool>, // per MB: true if coded as inter (incl. P_Skip)
}

/// Per-MB scratch holding the chosen encoding (levels to write + reconstruction
/// already applied to the rec plane).
#[derive(Default)]
struct MbEnc {
    is_i16: bool,
    i16_base_mode: i8,
    chroma_mode_signaled: i8,
    i4_best: [i8; 16],   // raster, base modes (0..8)
    i4_final: [i8; 16],  // raster, final modes (0..13)
    luma_dc: [i16; 16],  // scan order (I16 only)
    has_dc: bool,
    luma_levels: [[i16; 16]; 16], // per scan-block; I16=AC(15), I4=full(16)
    nzc_luma: [i8; 16],  // raster
    cbp_l: u8,
    chroma_dc: [[i16; 4]; 2],
    chroma_ac: [[[i16; 16]; 4]; 2],
    nzc_chroma: [i8; 8],
    cbp_c: u8,
}

#[inline]
fn nc_average(na: i32, nb: i32) -> i32 {
    let mut nc = na + nb + 1;
    if na != -1 && nb != -1 {
        nc >>= 1;
    }
    if na == -1 && nb == -1 {
        nc += 1;
    }
    nc
}

impl<'a> FrameEnc<'a> {
    pub fn new(
        dims: MbDims,
        qp: i32,
        src: PlaneRefs<'a>,
        src_ystride: usize,
        src_cstride: usize,
        refs: PlaneRefs<'a>,
    ) -> Self {
        let MbDims { width: mb_width, height: mb_height } = dims;
        let PlaneRefs { y: src_y, u: src_u, v: src_v } = src;
        let PlaneRefs { y: ref_y, u: ref_u, v: ref_v } = refs;
        let (ystride, cstride, ylen, clen) = rec_dims(mb_width, mb_height);
        let mb_count = mb_width * mb_height;
        FrameEnc {
            mb_width,
            mb_height,
            rec_y: vec![0u8; ylen],
            rec_u: vec![0u8; clen],
            rec_v: vec![0u8; clen],
            ystride,
            cstride,
            src_y,
            src_u,
            src_v,
            src_ystride,
            src_cstride,
            qp,
            slice_top_y: 0,
            nzc_luma: vec![0i8; mb_count * 16],
            nzc_chroma: vec![0i8; mb_count * 8],
            best_mode: vec![-1i8; mb_count * 16],
            is_nxn: vec![false; mb_count],
            ref_y,
            ref_u,
            ref_v,
            mv: vec![[0i16; 2]; mb_count * 16],
            ref_idx: vec![-1i8; mb_count * 16],
            mb_inter: vec![false; mb_count],
        }
    }

    /// First MB-row above which neighbours are unavailable (slice boundary).
    #[inline]
    fn top_avail(&self, mb_y: usize) -> bool {
        mb_y > self.slice_top_y
    }

    /// Border-extend the reconstructed planes (edge replication, matching the
    /// decoder's `expand_picture`) and return them for use as the next frame's
    /// reference. Consumes the frame encoder.
    pub fn into_reference(mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        expand_reference(
            &mut self.rec_y,
            &mut self.rec_u,
            &mut self.rec_v,
            MbDims { width: self.mb_width, height: self.mb_height },
            self.ystride,
            self.cstride,
        );
        (self.rec_y, self.rec_u, self.rec_v)
    }

    /// Encode the macroblocks of one slice (MB-rows `[first_mb_y, last_mb_y)`)
    /// into `bw`, which already holds the slice header. `slice_top_y` is set so
    /// neighbour prediction never reaches above the slice's first row. Writes
    /// the MB layer only (the caller appends rbsp_trailing_bits). For P slices
    /// the trailing `mb_skip_run` is flushed here, at the slice end.
    pub(crate) fn encode_band(&mut self, bw: &mut BitWriter, first_mb_y: usize, last_mb_y: usize, is_p: bool) {
        self.slice_top_y = first_mb_y;
        if is_p {
            let mut pending = 0u32;
            for mb_y in first_mb_y..last_mb_y {
                for mb_x in 0..self.mb_width {
                    self.encode_mb_p(bw, mb_x, mb_y, &mut pending);
                }
            }
            if pending > 0 {
                bw.write_ue(pending);
            }
        } else {
            for mb_y in first_mb_y..last_mb_y {
                for mb_x in 0..self.mb_width {
                    self.encode_mb(bw, mb_x, mb_y);
                }
            }
        }
    }

    #[inline]
    fn y_off(&self, mb_x: usize, mb_y: usize) -> usize {
        (BORDER + mb_y * 16) * self.ystride + BORDER + mb_x * 16
    }
    #[inline]
    fn c_off(&self, mb_x: usize, mb_y: usize) -> usize {
        (BORDER + mb_y * 8) * self.cstride + BORDER + mb_x * 8
    }
    #[inline]
    fn src_y_off(&self, mb_x: usize, mb_y: usize) -> usize {
        (mb_y * 16) * self.src_ystride + mb_x * 16
    }
    #[inline]
    fn src_c_off(&self, mb_x: usize, mb_y: usize) -> usize {
        (mb_y * 8) * self.src_cstride + mb_x * 8
    }

    /// Encode one macroblock of an I (intra-only) frame: decide modes,
    /// transform/quant, reconstruct in place, then write the MB syntax.
    pub fn encode_mb(&mut self, bw: &mut BitWriter, mb_x: usize, mb_y: usize) {
        let mb_xy = mb_y * self.mb_width + mb_x;
        let left = mb_x > 0;
        let top = self.top_avail(mb_y);

        let mut enc = MbEnc::default();
        self.intra_encode(mb_x, mb_y, &mut enc);
        self.commit_intra_context(mb_xy, &enc);
        self.write_mb_syntax(bw, MbPos { x: mb_x, y: mb_y }, Avail { left, top }, &enc, 0);
    }

    /// Full intra encode of one MB (luma I16x16-vs-I4x4 decision + chroma),
    /// reconstructing in place and filling `enc`. Returns the luma residual SATD
    /// (used by the P-slice mode decision to weigh intra against inter).
    fn intra_encode(&mut self, mb_x: usize, mb_y: usize, enc: &mut MbEnc) -> i32 {
        let left = mb_x > 0;
        let top = self.top_avail(mb_y);
        let left_top = left && top;
        let top_right = top && mb_x + 1 < self.mb_width;

        let pos = MbPos { x: mb_x, y: mb_y };
        let avail = NeighAvail { left, top, left_top, top_right };
        let cost16 = self.decide_i16(mb_x, mb_y, left, top, left_top, enc);
        let i16_base = enc.i16_base_mode;
        let cost4 = self.encode_i4x4(pos, avail, enc);

        let luma_cost = if cost4 < cost16 {
            enc.is_i16 = false;
            cost4
        } else {
            self.encode_i16x16(pos, avail, i16_base, enc);
            enc.is_i16 = true;
            cost16
        };
        self.encode_chroma(mb_x, mb_y, left, top, left_top, enc);
        luma_cost
    }

    /// Commit an intra MB's neighbour context. Intra MBs carry no list-0 motion,
    /// so their 4x4 blocks are marked `REF_NOT_IN_LIST` for any inter neighbour.
    fn commit_intra_context(&mut self, mb_xy: usize, enc: &MbEnc) {
        self.is_nxn[mb_xy] = !enc.is_i16;
        self.nzc_luma[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&enc.nzc_luma);
        self.nzc_chroma[mb_xy * 8..mb_xy * 8 + 8].copy_from_slice(&enc.nzc_chroma);
        if enc.is_i16 {
            self.best_mode[mb_xy * 16..mb_xy * 16 + 16].fill(-1);
        } else {
            self.best_mode[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(&enc.i4_best);
        }
        self.mb_inter[mb_xy] = false;
        for b in 0..16 {
            self.ref_idx[mb_xy * 16 + b] = -1;
            self.mv[mb_xy * 16 + b] = [0, 0];
        }
    }

    // ===================== Inter (P-slice) =====================

    /// Encode one macroblock of a P slice: choose P_Skip / P_16x16 / intra by an
    /// RD-ish cost, reconstruct in place, and write the syntax. `pending` carries
    /// the running `mb_skip_run` (skipped MBs are emitted lazily before the next
    /// coded MB).
    pub fn encode_mb_p(&mut self, bw: &mut BitWriter, mb_x: usize, mb_y: usize, pending: &mut u32) {
        let mb_xy = mb_y * self.mb_width + mb_x;
        let left = mb_x > 0;
        let top = self.top_avail(mb_y);

        // ---- Inter candidate: predict the MV, search, score. ----
        let (cmv, cref) = self.build_inter_cache(mb_x, mb_y);
        let mvp = pred_mv(&cmv, &cref, 0, 4, 0);
        let me = self.search_inter(mb_x, mb_y, mvp);
        let skip_mv = self.pred_p_skip(mb_x, mb_y);

        // ---- Intra candidate (reconstructs rec; overwritten if inter wins). ----
        let mut intra_enc = MbEnc::default();
        let intra_cost = self.intra_encode(mb_x, mb_y, &mut intra_enc);

        // Bias toward inter when comparable (it codes fewer header bits and keeps
        // the stream small); intra is only chosen when clearly cheaper.
        let bias = me_lambda(self.qp) * 24;
        if me.cost <= intra_cost + bias {
            let mut enc = MbEnc::default();
            self.reconstruct_inter(mb_x, mb_y, me.mv, &mut enc);
            let cbp = (enc.cbp_c << 4) | enc.cbp_l;
            if cbp == 0 && me.mv == skip_mv {
                // P_Skip: pure MC, no residual, no syntax — extend the run.
                *pending += 1;
                self.commit_inter_context(mb_xy, skip_mv, 0, &[0; 16], &[0; 8]);
                return;
            }
            bw.write_ue(*pending);
            *pending = 0;
            self.commit_inter_context(mb_xy, me.mv, 0, &enc.nzc_luma, &enc.nzc_chroma);
            self.write_inter_mb_syntax(bw, MbPos { x: mb_x, y: mb_y }, Avail { left, top }, me.mv, mvp, &enc);
        } else {
            bw.write_ue(*pending);
            *pending = 0;
            self.commit_intra_context(mb_xy, &intra_enc);
            // Intra mb_type in a P slice carries the +5 offset.
            self.write_mb_syntax(bw, MbPos { x: mb_x, y: mb_y }, Avail { left, top }, &intra_enc, 5);
        }
    }

    /// Build the 30-entry list-0 neighbour MV / ref-index cache for `(mb_x, mb_y)`
    /// — the exact mirror of the decoder's `WelsFillCacheInter` (single slice, so
    /// availability is purely geometric).
    fn build_inter_cache(&self, mb_x: usize, mb_y: usize) -> ([[i16; 2]; 30], [i8; 30]) {
        let mb_width = self.mb_width;
        let mb_xy = mb_y * mb_width + mb_x;
        let top_row = self.top_avail(mb_y);
        let left = mb_x != 0;
        let top = top_row;
        let left_top = mb_x != 0 && top_row;
        let right_top = mb_x != mb_width - 1 && top_row;

        let left_xy = mb_xy.wrapping_sub(1);
        let top_xy = mb_xy.wrapping_sub(mb_width);
        let left_top_xy = mb_xy.wrapping_sub(mb_width + 1);
        let right_top_xy = (mb_xy + 1).wrapping_sub(mb_width);

        let mut mv = [[0i16; 2]; 30];
        let mut ref_idx = [REF_NOT_AVAIL; 30];
        let mv_of = |xy: usize, b: usize| self.mv[xy * 16 + b];
        let ref_of = |xy: usize, b: usize| self.ref_idx[xy * 16 + b];

        if left && self.mb_inter[left_xy] {
            for (k, &b) in [3usize, 7, 11, 15].iter().enumerate() {
                let c = [6usize, 12, 18, 24][k];
                mv[c] = mv_of(left_xy, b);
                ref_idx[c] = ref_of(left_xy, b);
            }
        } else {
            let r = if left { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
            for &c in &[6usize, 12, 18, 24] {
                ref_idx[c] = r;
            }
        }
        if left_top && self.mb_inter[left_top_xy] {
            mv[0] = mv_of(left_top_xy, 15);
            ref_idx[0] = ref_of(left_top_xy, 15);
        } else {
            ref_idx[0] = if left_top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
        }
        if top && self.mb_inter[top_xy] {
            for (k, &b) in [12usize, 13, 14, 15].iter().enumerate() {
                mv[1 + k] = mv_of(top_xy, b);
                ref_idx[1 + k] = ref_of(top_xy, b);
            }
        } else {
            let r = if top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
            ref_idx[1..=4].fill(r);
        }
        if right_top && self.mb_inter[right_top_xy] {
            mv[5] = mv_of(right_top_xy, 12);
            ref_idx[5] = ref_of(right_top_xy, 12);
        } else {
            ref_idx[5] = if right_top { REF_NOT_IN_LIST } else { REF_NOT_AVAIL };
        }
        for &c in &[9usize, 11, 17, 21, 23] {
            ref_idx[c] = REF_NOT_AVAIL;
            mv[c] = [0, 0];
        }
        (mv, ref_idx)
    }

    /// `PredPSkipMvFromNeighbor` (encoder mirror) — derive the P_Skip MV.
    fn pred_p_skip(&self, mb_x: usize, mb_y: usize) -> [i16; 2] {
        let mb_width = self.mb_width;
        let mb_xy = mb_y * mb_width + mb_x;
        let fetch = |avail: bool, xy: usize, b: usize| -> (i8, [i16; 2]) {
            if avail && self.mb_inter[xy] {
                (self.ref_idx[xy * 16 + b], self.mv[xy * 16 + b])
            } else if avail {
                (REF_NOT_IN_LIST, [0, 0])
            } else {
                (REF_NOT_AVAIL, [0, 0])
            }
        };
        let top_row = self.top_avail(mb_y);
        let left_a = mb_x != 0;
        let top_a = top_row;
        let lt_a = mb_x != 0 && top_row;
        let rt_a = mb_x != mb_width - 1 && top_row;

        let (lref, lmv) = fetch(left_a, mb_xy.wrapping_sub(1), 3);
        if lref == REF_NOT_AVAIL || (lref == 0 && lmv == [0, 0]) {
            return [0, 0];
        }
        let (tref, tmv) = fetch(top_a, mb_xy.wrapping_sub(mb_width), 12);
        if tref == REF_NOT_AVAIL || (tref == 0 && tmv == [0, 0]) {
            return [0, 0];
        }
        let (rtref, rtmv) = fetch(rt_a, (mb_xy + 1).wrapping_sub(mb_width), 12);
        let (ltref, ltmv) = fetch(lt_a, mb_xy.wrapping_sub(mb_width + 1), 15);

        let mut mvc = rtmv;
        let mut diag = rtref;
        if diag == REF_NOT_AVAIL {
            diag = ltref;
            mvc = ltmv;
        }
        if tref == REF_NOT_AVAIL && diag == REF_NOT_AVAIL && lref >= REF_NOT_IN_LIST {
            return lmv;
        }
        let m = (lref == 0) as i32 + (tref == 0) as i32 + (diag == 0) as i32;
        if m == 1 {
            if lref == 0 {
                lmv
            } else if tref == 0 {
                tmv
            } else {
                mvc
            }
        } else {
            [
                median(lmv[0] as i32, tmv[0] as i32, mvc[0] as i32) as i16,
                median(lmv[1] as i32, tmv[1] as i32, mvc[1] as i32) as i16,
            ]
        }
    }

    /// Motion-search the 16x16 partition against the reference luma plane.
    fn search_inter(&self, mb_x: usize, mb_y: usize, mvp: [i16; 2]) -> super::motion_est::MeResult {
        let pic_w = (self.mb_width * 16) as i32;
        let pic_h = (self.mb_height * 16) as i32;
        let origin = BORDER * self.ystride + BORDER;
        let refv = RefView { plane: self.ref_y, stride: self.ystride, origin, pic_w, pic_h };
        let px = (mb_x * 16) as i32;
        let py = (mb_y * 16) as i32;
        let soff = self.src_y_off(mb_x, mb_y);
        search_mv(
            &refv,
            Pos { x: px, y: py },
            Blk { data: self.src_y, off: soff, stride: self.src_ystride },
            Dim { w: 16, h: 16 },
            mvp,
            me_lambda(self.qp),
        )
    }

    /// Motion-compensate a 16x16 luma + 8x8 chroma partition from the reference
    /// into the rec planes, applying the decoder's `base_mc` clamp (so the
    /// reconstruction is bit-identical regardless of the chosen MV).
    fn mc_into_rec(&mut self, mb_x: usize, mb_y: usize, mv: [i16; 2]) {
        let pad = BORDER as i32;
        let pic_w = (self.mb_width * 16) as i32;
        let pic_h = (self.mb_height * 16) as i32;
        let ls = self.ystride;
        let origin = BORDER * ls + BORDER;
        let px = (mb_x * 16) as i32;
        let py = (mb_y * 16) as i32;
        let fx = ((px << 2) + mv[0] as i32).clamp((-pad + 2) << 2, (pic_w + pad - 19) << 2);
        let fy = ((py << 2) + mv[1] as i32).clamp((-pad + 2) << 2, (pic_h + pad - 19) << 2);

        let dst = origin + (py as usize) * ls + px as usize;
        let src = (origin as i32 + (fx >> 2) + (fy >> 2) * ls as i32) as usize;
        mc_luma(&mut self.rec_y[dst..], ls, self.ref_y, src, ls, Mv { x: fx as i16, y: fy as i16 }, Dim { w: 16, h: 16 });

        let cs = self.cstride;
        let corigin = BORDER * cs + BORDER;
        let cdst = corigin + (mb_y * 8) * cs + mb_x * 8;
        let csrc = (corigin as i32 + (fx >> 3) + (fy >> 3) * cs as i32) as usize;
        let cmv = Mv { x: fx as i16, y: fy as i16 };
        let cdim = Dim { w: 8, h: 8 };
        mc_chroma(&mut self.rec_u[cdst..], cs, self.ref_u, csrc, cs, cmv, cdim);
        mc_chroma(&mut self.rec_v[cdst..], cs, self.ref_v, csrc, cs, cmv, cdim);
    }

    /// Reconstruct an inter 16x16 MB: MC into rec, then forward-transform/quant
    /// (inter deadzone) + reconstruct the luma + chroma residual in place. Fills
    /// `enc` with the levels / nzc / cbp needed to write the syntax.
    fn reconstruct_inter(&mut self, mb_x: usize, mb_y: usize, mv: [i16; 2], enc: &mut MbEnc) {
        self.mc_into_rec(mb_x, mb_y, mv);

        let off = self.y_off(mb_x, mb_y);
        let soff = self.src_y_off(mb_x, mb_y);
        let stride = self.ystride;
        let qp = self.qp as usize;
        let mf = &QUANT_MF[qp];
        let ff = inter_ff(qp);
        let deq = &G_KUI_DEQUANT_COEFF[qp];

        let mut cbp_l = 0u8;
        for i in 0..16 {
            let raster = BLOCK_RASTER[i];
            let boff = off + BLOCK_BY[i] * 4 * stride + BLOCK_BX[i] * 4;
            let sboff = soff + BLOCK_BY[i] * 4 * self.src_ystride + BLOCK_BX[i] * 4;
            let mut dct = [0i16; 16];
            dct_t4(&mut dct, self.src_y, sboff, self.src_ystride, &self.rec_y, boff, stride);
            quant4x4(&mut dct, ff, mf);
            let mut scanned = [0i16; 16];
            scan4x4_dcac(&mut scanned, &dct);
            let nnz = get_none_zero_count(&scanned);
            enc.luma_levels[i] = scanned;
            enc.nzc_luma[raster] = nnz as i8;
            if nnz > 0 {
                cbp_l |= 1 << (i >> 2);
                let mut coeffs = [0i16; 16];
                for s in 0..16 {
                    let lvl = scanned[s] as i32;
                    if lvl != 0 {
                        let j = G_KUI_ZIGZAG_SCAN[s] as usize;
                        coeffs[j] = (lvl * deq[j & 7] as i32) as i16;
                    }
                }
                idct4x4_add(&mut self.rec_y[boff..], stride, &coeffs);
            }
        }
        enc.cbp_l = cbp_l;
        enc.is_i16 = false;
        self.chroma_residual_recon(mb_x, mb_y, true, enc);
    }

    /// Commit an inter MB's neighbour context (single ref index, uniform MV over
    /// the 16x16 partition).
    fn commit_inter_context(&mut self, mb_xy: usize, mv: [i16; 2], iref: i8, nzc_luma: &[i8; 16], nzc_chroma: &[i8; 8]) {
        self.is_nxn[mb_xy] = false;
        self.best_mode[mb_xy * 16..mb_xy * 16 + 16].fill(-1);
        self.nzc_luma[mb_xy * 16..mb_xy * 16 + 16].copy_from_slice(nzc_luma);
        self.nzc_chroma[mb_xy * 8..mb_xy * 8 + 8].copy_from_slice(nzc_chroma);
        self.mb_inter[mb_xy] = true;
        for b in 0..16 {
            self.mv[mb_xy * 16 + b] = mv;
            self.ref_idx[mb_xy * 16 + b] = iref;
        }
    }

    /// Write the syntax of a coded P_L0_16x16 macroblock (mb_type, mvd, cbp,
    /// residual). `ref_idx_l0` is omitted: single reference -> te(range 1) = 0 bits.
    fn write_inter_mb_syntax(
        &self,
        bw: &mut BitWriter,
        pos: MbPos,
        avail: Avail,
        mv: [i16; 2],
        mvp: [i16; 2],
        enc: &MbEnc,
    ) {
        let MbPos { x: mb_x, y: mb_y } = pos;
        let Avail { left, top } = avail;
        let mb_xy = mb_y * self.mb_width + mb_x;
        let cbp = (enc.cbp_c << 4) | enc.cbp_l;
        bw.write_ue(0); // mb_type = P_L0_16x16
        bw.write_se((mv[0] - mvp[0]) as i32);
        bw.write_se((mv[1] - mvp[1]) as i32);
        bw.write_ue(inter_cbp_code(cbp));
        if cbp != 0 {
            bw.write_se(0); // mb_qp_delta (constant QP)
            self.write_residuals(bw, mb_xy, left, top, enc);
        }
    }

    // ===================== I16x16 =====================

    /// Try the available I16x16 base modes, leaving the best prediction in the
    /// rec plane; returns the best residual SATD and records the base mode.
    fn decide_i16(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        left: bool,
        top: bool,
        left_top: bool,
        enc: &mut MbEnc,
    ) -> i32 {
        let off = self.y_off(mb_x, mb_y);
        let soff = self.src_y_off(mb_x, mb_y);
        let mut best_cost = i32::MAX;
        let mut best_base = 2i8;
        for base in 0..4i8 {
            let Some(actual) = map_i16_mode(base, left, top, left_top) else { continue };
            luma16_pred(actual, &mut self.rec_y, off, self.ystride);
            let cost = crate::dsp::satd::satd16x16(&self.rec_y[off..], self.ystride, &self.src_y[soff..], self.src_ystride);
            if cost < best_cost {
                best_cost = cost;
                best_base = base;
            }
        }
        enc.i16_base_mode = best_base;
        best_cost
    }

    /// Full I16x16 encode: predict, forward transform + quant (DC Hadamard +
    /// AC), reconstruct in place, and store the levels/nzc/cbp in `enc`.
    fn encode_i16x16(&mut self, pos: MbPos, avail: NeighAvail, base_mode: i8, enc: &mut MbEnc) {
        let MbPos { x: mb_x, y: mb_y } = pos;
        let NeighAvail { left, top, left_top, .. } = avail;
        let off = self.y_off(mb_x, mb_y);
        let soff = self.src_y_off(mb_x, mb_y);
        let stride = self.ystride;
        let actual = map_i16_mode(base_mode, left, top, left_top).unwrap();
        luma16_pred(actual, &mut self.rec_y, off, stride);

        let qp = self.qp as usize;
        let mf = &QUANT_MF[qp];
        let ff = &QUANT_INTRA_FF[qp];

        // Forward DCT of all 16 blocks (4 8x8 quadrants), scan-block order.
        let mut res = [0i16; 256];
        let quad_off = [0usize, 8, 8 * stride, 8 * stride + 8];
        for (g, &qo) in quad_off.iter().enumerate() {
            let mut dct = [0i16; 64];
            dct_four_t4(&mut dct, self.src_y, soff + (g / 2) * 8 * self.src_ystride + (g % 2) * 8, self.src_ystride,
                        &self.rec_y, off + qo, stride);
            res[g * 64..g * 64 + 64].copy_from_slice(&dct);
        }

        // Luma DC: Hadamard then quant with (ff[0]<<1, mf[0]>>1).
        let mut dc = [0i16; 16];
        hadamard_t4_dc(&mut dc, &res);
        quant4x4_dc(&mut dc, ff[0] << 1, mf[0] >> 1);
        scan4x4_dcac(&mut enc.luma_dc, &dc);
        let dc_nnz = get_none_zero_count(&enc.luma_dc);
        enc.has_dc = dc_nnz > 0;

        // Luma AC: quant each block, scan dropping DC, count nnz.
        let mut ac_total = 0;
        for i in 0..16 {
            let blk: &mut [i16; 16] = (&mut res[i * 16..i * 16 + 16]).try_into().unwrap();
            quant4x4(blk, ff, mf);
            let mut scanned = [0i16; 16];
            scan4x4_ac(&mut scanned, blk);
            let nnz = get_none_zero_count(&scanned);
            enc.luma_levels[i] = scanned;
            enc.nzc_luma[BLOCK_RASTER[i]] = nnz as i8;
            ac_total += nnz;
        }
        enc.cbp_l = if ac_total > 0 { 15 } else { 0 };

        // Reconstruct: replay the decoder's dequant + IDCT add onto the prediction.
        let mut coeffs = [0i16; 256];
        // DC: place via luma-DC zig-zag, inverse Hadamard + dequant.
        if enc.has_dc {
            let mut dc_raster = [0i16; 16];
            // Undo scan: enc.luma_dc is scan order; place into block DC slots.
            dc_raster.copy_from_slice(&enc.luma_dc);
            // Place levels at block*16 offsets per the zig-zag, then dequant-IDCT.
            for s in 0..16 {
                coeffs[G_KUI_LUMA_DC_ZIGZAG_SCAN[s] as usize] = dc_raster[s];
            }
            luma_dc_dequant_idct(&mut coeffs, qp);
        }
        // AC: place levels (scan->raster, dequant), keep block DC from above.
        let deq = &G_KUI_DEQUANT_COEFF[qp];
        for i in 0..16 {
            let base = i * 16;
            for s in 0..15 {
                let lvl = enc.luma_levels[i][s] as i32;
                if lvl != 0 {
                    let j = G_KUI_ZIGZAG_SCAN[s + 1] as usize;
                    coeffs[base + j] = (lvl * deq[j & 7] as i32) as i16;
                }
            }
        }
        for i in 0..16 {
            let raster = BLOCK_RASTER[i];
            let boff = off + BLOCK_BY[i] * 4 * stride + BLOCK_BX[i] * 4;
            let base = i * 16;
            if enc.nzc_luma[raster] != 0 || coeffs[base] != 0 {
                let blk: [i16; 16] = coeffs[base..base + 16].try_into().unwrap();
                idct4x4_add(&mut self.rec_y[boff..], stride, &blk);
            }
        }
    }

    // ===================== I4x4 =====================

    /// Full I4x4 luma encode: per block pick the best available mode, transform/
    /// quant, reconstruct in place (so the next block predicts correctly), and
    /// record levels/modes/nzc. Returns the summed residual SATD.
    fn encode_i4x4(&mut self, pos: MbPos, avail: NeighAvail, enc: &mut MbEnc) -> i32 {
        let MbPos { x: mb_x, y: mb_y } = pos;
        let NeighAvail { left, top, left_top, top_right } = avail;
        let off = self.y_off(mb_x, mb_y);
        let soff = self.src_y_off(mb_x, mb_y);
        let stride = self.ystride;
        let qp = self.qp as usize;
        let mf = &QUANT_MF[qp];
        let ff = &QUANT_INTRA_FF[qp];
        let deq = &G_KUI_DEQUANT_COEFF[qp];

        // Sample-availability grid (WelsMapNxNNeighToSampleNormal).
        let mut sample_avail = [0i32; 30];
        if left {
            sample_avail[6] = 1;
            sample_avail[12] = 1;
            sample_avail[18] = 1;
            sample_avail[24] = 1;
        }
        if left_top {
            sample_avail[0] = 1;
        }
        if top {
            sample_avail[1] = 1;
            sample_avail[2] = 1;
            sample_avail[3] = 1;
            sample_avail[4] = 1;
        }
        if top_right {
            sample_avail[5] = 1;
        }

        let mut total_cost = 0i32;
        let mut cbp_l = 0u8;
        for i in 0..16 {
            let raster = BLOCK_RASTER[i];
            let bx = BLOCK_BX[i];
            let by = BLOCK_BY[i];
            let boff = off + by * 4 * stride + bx * 4;
            let sboff = soff + by * 4 * self.src_ystride + bx * 4;

            // Pick the best available base mode by residual SATD.
            let mut best_cost = i32::MAX;
            let mut best_base = 2i8;
            let mut best_final = 2i8;
            for base in 0..9i8 {
                let Some(fin) = map_i4_mode(&sample_avail, base, i) else { continue };
                luma4_pred(fin, &mut self.rec_y, boff, stride);
                let cost = crate::dsp::satd::satd4x4(&self.rec_y[boff..], stride, &self.src_y[sboff..], self.src_ystride);
                if cost < best_cost {
                    best_cost = cost;
                    best_base = base;
                    best_final = fin;
                }
            }
            total_cost += best_cost;
            enc.i4_best[raster] = best_base;
            enc.i4_final[raster] = best_final;

            // Predict (final), transform + quant, scan, reconstruct.
            luma4_pred(best_final, &mut self.rec_y, boff, stride);
            let mut dct = [0i16; 16];
            dct_t4(&mut dct, self.src_y, sboff, self.src_ystride, &self.rec_y, boff, stride);
            quant4x4(&mut dct, ff, mf);
            let mut scanned = [0i16; 16];
            scan4x4_dcac(&mut scanned, &dct);
            let nnz = get_none_zero_count(&scanned);
            enc.luma_levels[i] = scanned;
            enc.nzc_luma[raster] = nnz as i8;
            if nnz > 0 {
                cbp_l |= 1 << (i >> 2);
                let mut coeffs = [0i16; 16];
                for s in 0..16 {
                    let lvl = scanned[s] as i32;
                    if lvl != 0 {
                        let j = G_KUI_ZIGZAG_SCAN[s] as usize;
                        coeffs[j] = (lvl * deq[j & 7] as i32) as i16;
                    }
                }
                idct4x4_add(&mut self.rec_y[boff..], stride, &coeffs);
            }
            sample_avail[CACHE30_SCAN_IDX[i]] = 1;
        }
        enc.cbp_l = cbp_l;
        total_cost
    }

    // ===================== Chroma =====================

    fn encode_chroma(&mut self, mb_x: usize, mb_y: usize, left: bool, top: bool, left_top: bool, enc: &mut MbEnc) {
        let coff = self.c_off(mb_x, mb_y);
        let scoff = self.src_c_off(mb_x, mb_y);
        let cstride = self.cstride;

        // Decide chroma mode (base 0=DC,1=H,2=V,3=Plane) over Cb+Cr SATD.
        let mut best_cost = i32::MAX;
        let mut best_base = 0i8;
        for base in 0..4i8 {
            let Some(actual) = map_chroma_mode(base, left, top, left_top) else { continue };
            chroma_pred(actual, &mut self.rec_u, coff, cstride);
            chroma_pred(actual, &mut self.rec_v, coff, cstride);
            let cost = crate::dsp::satd::satd8x8(&self.rec_u[coff..], cstride, &self.src_u[scoff..], self.src_cstride)
                + crate::dsp::satd::satd8x8(&self.rec_v[coff..], cstride, &self.src_v[scoff..], self.src_cstride);
            if cost < best_cost {
                best_cost = cost;
                best_base = base;
            }
        }
        enc.chroma_mode_signaled = best_base;
        let actual = map_chroma_mode(best_base, left, top, left_top).unwrap();
        // Leave the chosen prediction in the rec planes, then run the shared
        // residual + reconstruction (intra deadzone).
        chroma_pred(actual, &mut self.rec_u, coff, cstride);
        chroma_pred(actual, &mut self.rec_v, coff, cstride);
        self.chroma_residual_recon(mb_x, mb_y, false, enc);
    }

    /// Forward-transform, quantise and reconstruct chroma assuming the chosen
    /// prediction already sits in the rec planes (so it serves both the intra
    /// predictors and inter MC). `inter` selects the inter deadzone offset.
    fn chroma_residual_recon(&mut self, mb_x: usize, mb_y: usize, inter: bool, enc: &mut MbEnc) {
        let coff = self.c_off(mb_x, mb_y);
        let scoff = self.src_c_off(mb_x, mb_y);
        let cstride = self.cstride;

        let chroma_qp = CHROMA_QP_TABLE[self.qp.clamp(0, 51) as usize] as usize;
        let mf = &QUANT_MF[chroma_qp];
        let ff: &[i16; 8] = if inter { inter_ff(chroma_qp) } else { &QUANT_INTRA_FF[chroma_qp] };
        let deq = &G_KUI_DEQUANT_COEFF[chroma_qp];

        let mut dc_present = false;
        let mut ac_present = false;
        for c in 0..2 {
            let (src, srcs, rec) = if c == 0 {
                (self.src_u, self.src_cstride, &mut self.rec_u)
            } else {
                (self.src_v, self.src_cstride, &mut self.rec_v)
            };

            // Forward DCT of the four 4x4 chroma blocks (prediction already in rec).
            let mut res = [0i16; 64];
            dct_four_t4(&mut res, src, scoff, srcs, rec, coff, cstride);

            // DC: 2x2 Hadamard + quant (ff[0]<<1, mf[0]>>1); zeroes res DC slots.
            let mut dct2x2 = [0i16; 4];
            let mut block_dc = [0i16; 4];
            let dc_nnz = hadamard_quant2x2(&mut res, ff[0] << 1, mf[0] >> 1, &mut dct2x2, &mut block_dc);
            enc.chroma_dc[c] = block_dc;
            if dc_nnz > 0 {
                dc_present = true;
            }

            // AC: quant four 4x4, scan dropping DC, count nnz.
            quant_four4x4(&mut res, ff, mf);
            for b in 0..4 {
                let blk: [i16; 16] = res[b * 16..b * 16 + 16].try_into().unwrap();
                let mut scanned = [0i16; 16];
                scan4x4_ac(&mut scanned, &blk);
                let nnz = get_none_zero_count(&scanned);
                enc.chroma_ac[c][b] = scanned;
                enc.nzc_chroma[c * 4 + b] = nnz as i8;
                if nnz > 0 {
                    ac_present = true;
                }
            }
        }
        enc.cbp_c = if ac_present {
            2
        } else if dc_present {
            1
        } else {
            0
        };

        // Reconstruct chroma: replay decoder dequant + IDCT.
        for c in 0..2 {
            let rec = if c == 0 { &mut self.rec_u } else { &mut self.rec_v };
            let mut coeffs = [0i16; 64];
            if enc.cbp_c == 1 || enc.cbp_c == 2 {
                // DC: place levels, inverse 2x2 Hadamard, dequant.
                for k in 0..4 {
                    coeffs[G_KUI_CHROMA_DC_SCAN[k] as usize] = enc.chroma_dc[c][k];
                }
                chroma_dc_idct(&mut coeffs);
                let qmul = deq[0] as i32;
                for &scan in &G_KUI_CHROMA_DC_SCAN {
                    let j = scan as usize;
                    coeffs[j] = ((coeffs[j] as i32 * qmul) >> 1) as i16;
                }
            }
            if enc.cbp_c == 2 {
                for b in 0..4 {
                    let base = b * 16;
                    for s in 0..15 {
                        let lvl = enc.chroma_ac[c][b][s] as i32;
                        if lvl != 0 {
                            let j = G_KUI_ZIGZAG_SCAN[s + 1] as usize;
                            coeffs[base + j] = (lvl * deq[j & 7] as i32) as i16;
                        }
                    }
                }
            }
            for b in 0..4 {
                let base = b * 16;
                let boff = coff + (b / 2) * 4 * cstride + (b % 2) * 4;
                if enc.nzc_chroma[c * 4 + b] != 0 || coeffs[base] != 0 {
                    let blk: [i16; 16] = coeffs[base..base + 16].try_into().unwrap();
                    idct4x4_add(&mut rec[boff..], cstride, &blk);
                }
            }
        }
    }

    // ===================== Syntax write =====================

    fn write_mb_syntax(&self, bw: &mut BitWriter, pos: MbPos, avail: Avail, enc: &MbEnc, mb_type_offset: u32) {
        let MbPos { x: mb_x, y: mb_y } = pos;
        let Avail { left, top } = avail;
        let mb_xy = mb_y * self.mb_width + mb_x;
        let cbp = (enc.cbp_c << 4) | enc.cbp_l;

        if enc.is_i16 {
            // mb_type = 1 + base_mode + 4 * cbp_class (+ offset for P-slice intra).
            let cbp_class = I16_CBP_TABLE.iter().position(|&v| v == cbp).unwrap() as u32;
            let mb_type = 1 + enc.i16_base_mode as u32 + 4 * cbp_class;
            bw.write_ue(mb_type + mb_type_offset);
            bw.write_ue(enc.chroma_mode_signaled as u32);
        } else {
            bw.write_ue(mb_type_offset); // I_NxN (0, or 5 in a P slice)
            // 16 luma mode signals in scan order.
            self.write_i4_modes(bw, mb_xy, left, top, enc);
            bw.write_ue(enc.chroma_mode_signaled as u32);
            let code = INTRA4X4_CBP_CODE[cbp as usize];
            bw.write_ue(code as u32);
        }

        // mb_qp_delta + residual.
        if cbp != 0 || enc.is_i16 {
            bw.write_se(0); // constant QP -> delta 0
            self.write_residuals(bw, mb_xy, left, top, enc);
        }
    }

    fn write_i4_modes(&self, bw: &mut BitWriter, mb_xy: usize, left: bool, top: bool, enc: &MbEnc) {
        let mb_width = self.mb_width;
        let top_modes: [i8; 4] = if top {
            let txy = mb_xy - mb_width;
            if self.is_nxn[txy] {
                let m = &self.best_mode[txy * 16..txy * 16 + 16];
                [m[12], m[13], m[14], m[15]]
            } else {
                [2; 4]
            }
        } else {
            [-1; 4]
        };
        let left_modes: [i8; 4] = if left {
            let lxy = mb_xy - 1;
            if self.is_nxn[lxy] {
                let m = &self.best_mode[lxy * 16..lxy * 16 + 16];
                [m[3], m[7], m[11], m[15]]
            } else {
                [2; 4]
            }
        } else {
            [-1; 4]
        };

        // Local best-mode cache (raster) accumulated as we emit each block.
        let mut cache = [-1i8; 16];
        for i in 0..16 {
            let raster = BLOCK_RASTER[i];
            let bx = BLOCK_BX[i];
            let by = BLOCK_BY[i];
            let top_mode = if by > 0 { cache[(by - 1) * 4 + bx] } else { top_modes[bx] };
            let left_mode = if bx > 0 { cache[by * 4 + bx - 1] } else { left_modes[by] };
            let pred_mode = if left_mode == -1 || top_mode == -1 { 2 } else { left_mode.min(top_mode) };
            let cur = enc.i4_best[raster];
            if cur == pred_mode {
                bw.write_flag(true);
            } else {
                bw.write_flag(false);
                let rem = if cur > pred_mode { cur - 1 } else { cur };
                bw.write_bits(rem as u32, 3);
            }
            cache[raster] = cur;
        }
    }

    fn write_residuals(&self, bw: &mut BitWriter, mb_xy: usize, left: bool, top: bool, enc: &MbEnc) {
        let mb_width = self.mb_width;
        let lxy = mb_xy.wrapping_sub(1);
        let txy = mb_xy.wrapping_sub(mb_width);
        let nzc_l = |bx: usize, by: usize| -> i32 {
            let na = if bx > 0 {
                enc.nzc_luma[by * 4 + bx - 1] as i32
            } else if left {
                self.nzc_luma[lxy * 16 + by * 4 + 3] as i32
            } else {
                -1
            };
            let nb = if by > 0 {
                enc.nzc_luma[(by - 1) * 4 + bx] as i32
            } else if top {
                self.nzc_luma[txy * 16 + 12 + bx] as i32
            } else {
                -1
            };
            nc_average(na, nb)
        };

        if enc.is_i16 {
            // Luma DC (block 0,0 context), end_idx 15.
            let nc0 = nc_average(
                if left { self.nzc_luma[lxy * 16 + 3] as i32 } else { -1 },
                if top { self.nzc_luma[txy * 16 + 12] as i32 } else { -1 },
            );
            write_residual_block(bw, &enc.luma_dc, 15, false, nc0);
            if enc.cbp_l != 0 {
                for i in 0..16 {
                    let nc = nzc_l(BLOCK_BX[i], BLOCK_BY[i]);
                    write_residual_block(bw, &enc.luma_levels[i], 14, false, nc);
                }
            }
        } else {
            for id8 in 0..4 {
                if enc.cbp_l & (1 << id8) == 0 {
                    continue;
                }
                for id4 in 0..4 {
                    let i = id8 * 4 + id4;
                    let nc = nzc_l(BLOCK_BX[i], BLOCK_BY[i]);
                    write_residual_block(bw, &enc.luma_levels[i], 15, false, nc);
                }
            }
        }

        // Chroma DC.
        if enc.cbp_c == 1 || enc.cbp_c == 2 {
            for c in 0..2 {
                write_residual_block(bw, &enc.chroma_dc[c], 3, true, -1);
            }
        }
        // Chroma AC.
        if enc.cbp_c == 2 {
            for c in 0..2 {
                for b in 0..4 {
                    let bx = b % 2;
                    let by = b / 2;
                    let base = c * 4;
                    let na = if bx > 0 {
                        enc.nzc_chroma[base + by * 2 + bx - 1] as i32
                    } else if left {
                        self.nzc_chroma[lxy * 8 + base + by * 2 + 1] as i32
                    } else {
                        -1
                    };
                    let nb = if by > 0 {
                        enc.nzc_chroma[base + (by - 1) * 2 + bx] as i32
                    } else if top {
                        self.nzc_chroma[txy * 8 + base + 2 + bx] as i32
                    } else {
                        -1
                    };
                    let nc = nc_average(na, nb);
                    write_residual_block(bw, &enc.chroma_ac[c][b], 14, false, nc);
                }
            }
        }
    }
}

// ===================== Mode mapping (availability) =====================

/// `CheckIntra16x16PredMode`: map a signaled base mode (0..3) to the in-place
/// predictor index (0..6), or `None` if unavailable.
fn map_i16_mode(base: i8, left: bool, top: bool, left_top: bool) -> Option<i8> {
    if base == 2 {
        return Some(if left && top {
            2
        } else if left {
            4
        } else if top {
            5
        } else {
            6
        });
    }
    let (pm, nl, nt, nlt) = I16_PRED_INFO[base as usize];
    if base != pm || (left as i32) < nl || (top as i32) < nt || (left_top as i32) < nlt {
        return None;
    }
    Some(base)
}

/// `CheckIntraChromaPredMode`: base 0..3 -> predictor 0..6 or `None`.
fn map_chroma_mode(base: i8, left: bool, top: bool, left_top: bool) -> Option<i8> {
    if base == 0 {
        return Some(if left && top {
            0
        } else if left {
            4
        } else if top {
            5
        } else {
            6
        });
    }
    let (pm, nl, nt, nlt) = CHROMA_PRED_INFO[base as usize];
    if base != pm || (left as i32) < nl || (top as i32) < nt || (left_top as i32) < nlt {
        return None;
    }
    Some(base)
}

/// `CheckIntraNxNPredMode`: base 0..8 -> final predictor 0..13 or `None`.
fn map_i4_mode(sample_avail: &[i32; 30], base: i8, i: usize) -> Option<i8> {
    let idx = CACHE30_SCAN_IDX[i];
    let left_avail = sample_avail[idx - 1];
    let top_avail = sample_avail[idx - 6];
    let left_top_avail = sample_avail[idx - 7];
    let right_top_avail = sample_avail[idx - 5];
    if base == 2 {
        return Some(if left_avail != 0 && top_avail != 0 {
            2
        } else if left_avail != 0 {
            9
        } else if top_avail != 0 {
            10
        } else {
            11
        });
    }
    let (pm, nl, nt, nlt) = I4_PRED_INFO[base as usize];
    if base != pm || left_avail < nl || top_avail < nt || left_top_avail < nlt {
        return None;
    }
    let mut fin = base;
    if base == 3 && right_top_avail == 0 {
        fin = 12;
    } else if base == 7 && right_top_avail == 0 {
        fin = 13;
    }
    Some(fin)
}

// ===================== Prediction dispatch (decoder predictors) =====================

fn luma16_pred(mode: i8, plane: &mut [u8], off: usize, stride: usize) {
    match mode {
        0 => ip::i16x16_luma_pred_v(plane, off, stride),
        1 => ip::i16x16_luma_pred_h(plane, off, stride),
        2 => ip::i16x16_luma_pred_dc(plane, off, stride),
        3 => ip::i16x16_luma_pred_plane(plane, off, stride),
        4 => ip::i16x16_luma_pred_dc_left(plane, off, stride),
        5 => ip::i16x16_luma_pred_dc_top(plane, off, stride),
        _ => ip::i16x16_luma_pred_dc_na(plane, off, stride),
    }
}

fn luma4_pred(mode: i8, plane: &mut [u8], off: usize, stride: usize) {
    match mode {
        0 => ip::i4x4_luma_pred_v(plane, off, stride),
        1 => ip::i4x4_luma_pred_h(plane, off, stride),
        2 => ip::i4x4_luma_pred_dc(plane, off, stride),
        3 => ip::i4x4_luma_pred_ddl(plane, off, stride),
        4 => ip::i4x4_luma_pred_ddr(plane, off, stride),
        5 => ip::i4x4_luma_pred_vr(plane, off, stride),
        6 => ip::i4x4_luma_pred_hd(plane, off, stride),
        7 => ip::i4x4_luma_pred_vl(plane, off, stride),
        8 => ip::i4x4_luma_pred_hu(plane, off, stride),
        9 => ip::i4x4_luma_pred_dc_left(plane, off, stride),
        10 => ip::i4x4_luma_pred_dc_top(plane, off, stride),
        11 => ip::i4x4_luma_pred_dc_na(plane, off, stride),
        12 => ip::i4x4_luma_pred_ddl_top(plane, off, stride),
        _ => ip::i4x4_luma_pred_vl_top(plane, off, stride),
    }
}

fn chroma_pred(mode: i8, plane: &mut [u8], off: usize, stride: usize) {
    match mode {
        0 => ip::i_chroma_pred_dc(plane, off, stride),
        1 => ip::i_chroma_pred_h(plane, off, stride),
        2 => ip::i_chroma_pred_v(plane, off, stride),
        3 => ip::i_chroma_pred_plane(plane, off, stride),
        4 => ip::i_chroma_pred_dc_left(plane, off, stride),
        5 => ip::i_chroma_pred_dc_top(plane, off, stride),
        _ => ip::i_chroma_pred_dc_na(plane, off, stride),
    }
}

// ===================== Inverse DC transforms (decoder ports) =====================

/// `WelsLumaDcDequantIdct` (inverse Hadamard + dequant of the 16 luma DCs at
/// element offsets `block*16` within the 256-entry block store).
fn luma_dc_dequant_idct(coeffs: &mut [i16; 256], qp: usize) {
    const STRIDE: usize = 16;
    let qmul = (G_KUI_DEQUANT_COEFF[qp][0] as i32) << 4;
    let x_off = [0usize, STRIDE, STRIDE << 2, 5 * STRIDE];
    let y_off = [0usize, STRIDE << 1, STRIDE << 3, 10 * STRIDE];
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let o = y_off[i];
        let x1 = o + x_off[2];
        let x2 = STRIDE + o;
        let x3 = o + x_off[3];
        let z0 = coeffs[o] as i32 + coeffs[x1] as i32;
        let z1 = coeffs[o] as i32 - coeffs[x1] as i32;
        let z2 = coeffs[x2] as i32 - coeffs[x3] as i32;
        let z3 = coeffs[x2] as i32 + coeffs[x3] as i32;
        tmp[i * 4] = z0 + z3;
        tmp[1 + i * 4] = z1 + z2;
        tmp[2 + i * 4] = z1 - z2;
        tmp[3 + i * 4] = z0 - z3;
    }
    for i in 0..4 {
        let o = x_off[i];
        let i4 = 4 + i;
        let z0 = tmp[i] + tmp[4 + i4];
        let z1 = tmp[i] - tmp[4 + i4];
        let z2 = tmp[i4] - tmp[8 + i4];
        let z3 = tmp[i4] + tmp[8 + i4];
        coeffs[o] = (((z0 + z3) * qmul + (1 << 5)) >> 6) as i16;
        coeffs[y_off[1] + o] = (((z1 + z2) * qmul + (1 << 5)) >> 6) as i16;
        coeffs[y_off[2] + o] = (((z1 - z2) * qmul + (1 << 5)) >> 6) as i16;
        coeffs[y_off[3] + o] = (((z0 - z3) * qmul + (1 << 5)) >> 6) as i16;
    }
}

/// `WelsChromaDcIdct` (inverse 2x2 Hadamard, samples at {0,16,32,48}).
fn chroma_dc_idct(block: &mut [i16; 64]) {
    let a = block[0] as i32;
    let b = block[16] as i32;
    let c = block[32] as i32;
    let d = block[48] as i32;
    let e = a - b;
    let a2 = a + b;
    let b2 = c - d;
    let c2 = c + d;
    block[0] = (a2 + c2) as i16;
    block[16] = (e + b2) as i16;
    block[32] = (a2 - c2) as i16;
    block[48] = (e - b2) as i16;
}
