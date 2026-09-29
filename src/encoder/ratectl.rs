//! Bitrate-mode rate control, ported from
//! `reference/codec/encoder/core/src/ratectl.cpp` (`RC_BITRATE_MODE`, the
//! `WelsRcPictureInitGom` / `WelsRcPictureInfoUpdateGom` pair) for a single
//! spatial + temporal layer with `bFixRCOverShoot` on and frame skipping off.
//!
//! Each frame gets a bit budget from a virtual GOP of [`VGOP_SIZE`] frames
//! (`bitrate / fps` per frame, an IDR [`IDR_BITRATE_RATIO`] times that; an
//! overshoot is carried into the next VGOP). The QP comes from the linear R-Q
//! model `bits * qstep ~ complexity`, scaled by the frame's complexity against
//! its running mean, and moves at most `-3/+5` per frame.
//!
//! PORT (simplifications, all at the frame level):
//! - No GOM / MB-level QP: the whole frame is coded at the picture QP. The C
//!   turns GOM off for IDRs and for multi-slice frames anyway.
//! - No adaptive quantisation offset (VAA's `iAverMotionTextureIndexToDeltaQp`).
//! - When the budget is exhausted (`BITS_EXCEEDED`) the QP rises instead of the
//!   frame being skipped; skip / max-bitrate / padding buffers are not ported.
//! - Frame complexity is computed by the encoder itself (see
//!   [`super::Encoder`]) rather than by the VAA preprocessing module.

/// `INT_MULTIPLY`: fixed-point scale of ratios and qsteps.
const INT_MULTIPLY: i64 = 100;
/// `WEIGHT_MULTIPLY`: fixed-point scale of temporal-layer weights.
const WEIGHT_MULTIPLY: i64 = 2000;
/// `VGOP_SIZE`: frames per virtual GOP.
const VGOP_SIZE: i64 = 8;
/// GOPs per VGOP with one temporal layer (`VGOP_SIZE / (1 << 0)`).
const GOP_NUMBER_IN_VGOP: i64 = VGOP_SIZE;
/// Weight of temporal layer 0 with no decomposition (`iWeightArray[0][0]`).
const TLAYER_WEIGHT: i64 = WEIGHT_MULTIPLY;
/// `IDR_BITRATE_RATIO`: an IDR's budget in average frames.
const IDR_BITRATE_RATIO: i64 = 4;
/// `REMAIN_BITS_TH`.
const REMAIN_BITS_TH: i64 = 1;
const MAX_BITS_VARY_PERCENTAGE: i64 = 100;
const MAX_BITS_VARY_PERCENTAGE_X3D2: i64 = 150;
/// `SWelsSvcCodingParam::iBitsVaryPercentage` default.
const BITS_VARY_PERCENTAGE: i64 = 10;
/// Weight (/100) of the old value in the complexity models' running averages.
const LINEAR_MODEL_DECAY_FACTOR: i64 = 80;
/// Clamp (/100) of the complexity ratio around 1.
const FRAME_CMPLX_RATIO_RANGE: i64 = 20;
/// Cap of the IDR / P frame counters.
const MODEL_FRAME_COUNT_MAX: i32 = 255;
/// `DELTA_QP_BGD_THD`: QP step when the budget is exceeded.
const DELTA_QP_BGD_THD: i32 = 3;
const LAST_FRAME_QP_RANGE_UPPER_MODE0: i64 = 3;
const LAST_FRAME_QP_RANGE_LOWER_MODE0: i64 = 2;
const LAST_FRAME_QP_RANGE_UPPER_MODE1: i64 = 5;
const LAST_FRAME_QP_RANGE_LOWER_MODE1: i64 = 3;
/// Largest QP rise between consecutive frames (`iFrameDeltaQpUpper`).
const FRAME_DELTA_QP_UPPER: i32 = (LAST_FRAME_QP_RANGE_UPPER_MODE1
    - (LAST_FRAME_QP_RANGE_UPPER_MODE1 - LAST_FRAME_QP_RANGE_UPPER_MODE0) * BITS_VARY_PERCENTAGE
        / MAX_BITS_VARY_PERCENTAGE) as i32;
/// Largest QP drop between consecutive frames (`iFrameDeltaQpLower`).
const FRAME_DELTA_QP_LOWER: i32 = (LAST_FRAME_QP_RANGE_LOWER_MODE1
    - (LAST_FRAME_QP_RANGE_LOWER_MODE1 - LAST_FRAME_QP_RANGE_LOWER_MODE0) * BITS_VARY_PERCENTAGE
        / MAX_BITS_VARY_PERCENTAGE) as i32;

/// Lowest QP the controller picks (`GOM_MIN_QP_MODE`, OpenH264's real-time floor).
pub const MIN_QP: u8 = 12;
/// Highest QP the controller picks (OpenH264's real-time cap is 42; a little
/// more headroom here since this encoder has no CABAC / sub-partitions).
pub const MAX_QP: u8 = 45;

/// `g_kiQpToQstepTable`: `round(100 * 2^((qp - 4) / 6))`.
const QP_TO_QSTEP: [i64; 52] = [
    63, 71, 79, 89, 100, 112, 126, 141, 159, 178, 200, 224, 252, 283, 317, 356, 400, 449, 504, 566,
    635, 713, 800, 898, 1008, 1131, 1270, 1425, 1600, 1796, 2016, 2263, 2540, 2851, 3200, 3592,
    4032, 4525, 5080, 5702, 6400, 7184, 8063, 9051, 10159, 11404, 12800, 14368, 16127, 18102,
    20319, 22807,
];

/// `ceil(100 * 2^((qp - 4.5) / 6))` for qp 1..=51: the smallest qstep that
/// `RcConvertQStep2Qp`'s `round(6 * log2(qstep / 100) + 4)` maps to `qp`.
/// Integer thresholds keep the conversion exact without `libm` (`no_std`).
const QSTEP_QP_THRESHOLDS: [i64; 51] = [
    67, 75, 85, 95, 106, 119, 134, 150, 169, 189, 212, 238, 267, 300, 337, 378, 424, 476, 534, 600,
    673, 756, 848, 952, 1068, 1199, 1346, 1511, 1696, 1903, 2136, 2398, 2691, 3021, 3391, 3806,
    4272, 4795, 5382, 6041, 6781, 7611, 8543, 9590, 10764, 12082, 13562, 15222, 17086, 19179,
    21527,
];

/// `RcCalculateIdrQp` tables: bits-per-pixel buckets per picture-area class,
/// the first IDR's QP, and the IDR QP range per bucket.
const BPP_BUCKETS: [[f64; 4]; 4] = [
    [0.25, 0.5, 0.75, 1.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.03, 0.05, 0.09, 0.13],
    [0.01, 0.03, 0.06, 0.1],
];
const INITIAL_IDR_QP: [[i32; 5]; 4] = [
    [34, 28, 26, 24, 22],
    [36, 30, 28, 26, 24],
    [36, 32, 30, 28, 26],
    [36, 34, 32, 30, 28],
];
const IDR_QP_RANGE: [[i32; 2]; 5] = [[40, 28], [37, 25], [36, 24], [35, 23], [34, 22]];
/// Upper picture areas (luma samples) of the 90p / 180p / 360p classes.
const AREA_CLASSES: [u64; 3] = [28_800, 115_200, 460_800];
/// `dBpp` fallback when the frame rate or size is unknown.
const DEFAULT_BPP: f64 = 0.1;

/// `WELS_DIV_ROUND` / `WELS_DIV_ROUND64` (including their `y == 0` form).
const fn div_round(x: i64, y: i64) -> i64 {
    if y == 0 { x / (y + 1) } else { (y / 2 + x) / y }
}

/// `RcConvertQStep2Qp`, saturated to 51.
fn qstep_to_qp(qstep: i64) -> i32 {
    QSTEP_QP_THRESHOLDS.iter().filter(|&&t| qstep >= t).count() as i32
}

const fn qp_to_qstep(qp: i32) -> i64 {
    QP_TO_QSTEP[qp as usize]
}

/// Per-encoder rate-control state (`SWelsSvcRc` + its one `SRCTemporal`).
pub(crate) struct RateControl {
    area: u64,
    min_qp: i32,
    max_qp: i32,
    /// Requested bitrate (bits/s) and frame rate.
    bitrate: i64,
    fps: u32,
    /// The values last applied (`iPreviousBitrate` / `dPreviousFps`).
    applied_bitrate: i64,
    applied_fps: u32,
    bits_per_frame: i64,
    min_bits: i64,
    max_bits: i64,
    remaining_bits: i64,
    last_allocated_bits: i64,
    remaining_weights: i64,
    gop_index_in_vgop: i64,
    target_bits: i64,
    bits_exceeded: bool,
    initial_qp: i32,
    /// `iLastCalculatedQScale`.
    last_qp: i32,
    idr_num: i32,
    intra_complexity: i64,
    intra_cmplx_mean: i64,
    p_frame_num: i32,
    linear_cmplx: i64,
    frame_cmplx_mean: i64,
}

impl RateControl {
    pub(crate) const fn new(width: u32, height: u32, bitrate: i64, fps: u32) -> Self {
        RateControl {
            area: width as u64 * height as u64,
            min_qp: MIN_QP as i32,
            max_qp: MAX_QP as i32,
            bitrate,
            fps,
            applied_bitrate: bitrate,
            applied_fps: fps,
            bits_per_frame: 0,
            min_bits: 0,
            max_bits: 0,
            remaining_bits: 0,
            last_allocated_bits: 0,
            remaining_weights: 0,
            gop_index_in_vgop: 0,
            target_bits: 0,
            bits_exceeded: false,
            initial_qp: 0,
            last_qp: 0,
            idr_num: 0,
            intra_complexity: 0,
            intra_cmplx_mean: 0,
            p_frame_num: 0,
            linear_cmplx: 0,
            frame_cmplx_mean: 0,
        }
    }

    /// New target bitrate (bits/s), applied from the next frame.
    pub(crate) const fn set_bitrate(&mut self, bitrate: i64) {
        self.bitrate = bitrate;
    }

    /// New frame rate, applied from the next frame.
    pub(crate) const fn set_fps(&mut self, fps: u32) {
        self.fps = fps;
    }

    /// `WelsRcPictureInitGom`: returns the QP to code the next frame at.
    /// `complexity` is the frame's intra (IDR) or inter (P) complexity.
    pub(crate) fn picture_init(&mut self, idr: bool, complexity: i64) -> u8 {
        if idr && self.idr_num == 0 {
            self.init_refresh();
        }
        if self.applied_bitrate != self.bitrate || self.applied_fps != self.fps {
            self.applied_bitrate = self.bitrate;
            self.applied_fps = self.fps;
            self.update_bitrate_fps();
        }
        // RcUpdateTemporalZero.
        if self.gop_index_in_vgop == GOP_NUMBER_IN_VGOP || idr {
            self.init_vgop();
        }
        self.gop_index_in_vgop += 1;
        self.decide_target_bits(idr);
        let qp = if idr {
            self.idr_qp(complexity)
        } else {
            self.picture_qp(complexity)
        };
        qp.clamp(0, 51) as u8
    }

    /// `WelsRcPictureInfoUpdateGom`: account a coded frame of `bits` at `qp`.
    pub(crate) fn picture_update(&mut self, idr: bool, qp: u8, bits: i64, complexity: i64) {
        // RcUpdatePictureQpBits: one QP per frame, so the average is the QP.
        let qp = qp as i32;
        self.last_qp = qp;
        let cmplx = bits * qp_to_qstep(qp);
        if idr {
            // RcUpdateIntraComplexity.
            if self.idr_num == 0 {
                self.intra_complexity = cmplx;
                self.intra_cmplx_mean = complexity;
            } else {
                self.intra_complexity = decay(self.intra_complexity, cmplx);
                self.intra_cmplx_mean = decay(self.intra_cmplx_mean, complexity);
            }
            self.idr_num = (self.idr_num + 1).min(MODEL_FRAME_COUNT_MAX);
        } else {
            // RcUpdateFrameComplexity.
            if self.p_frame_num == 0 {
                self.linear_cmplx = cmplx;
                self.frame_cmplx_mean = complexity;
            } else {
                self.linear_cmplx = decay(self.linear_cmplx, cmplx);
                self.frame_cmplx_mean = decay(self.frame_cmplx_mean, complexity);
            }
            self.p_frame_num = (self.p_frame_num + 1).min(MODEL_FRAME_COUNT_MAX);
        }
        self.remaining_bits -= bits;
    }

    /// `RcInitRefreshParameter`.
    fn init_refresh(&mut self) {
        self.intra_complexity = 0;
        self.intra_cmplx_mean = 0;
        self.p_frame_num = 0;
        self.linear_cmplx = 0;
        self.frame_cmplx_mean = 0;
        self.gop_index_in_vgop = 0;
        self.last_allocated_bits = 0;
        self.remaining_bits = 0;
        self.bits_per_frame = 0;
        self.applied_bitrate = self.bitrate;
        self.applied_fps = self.fps;
        self.update_bitrate_fps();
        self.init_vgop();
    }

    /// `RcUpdateBitrateFps`.
    fn update_bitrate_fps(&mut self) {
        let bits_per_frame = div_round(self.applied_bitrate, self.applied_fps.max(1) as i64);
        let target_vary_range = (MAX_BITS_VARY_PERCENTAGE - BITS_VARY_PERCENTAGE) >> 1;
        let min_ratio = MAX_BITS_VARY_PERCENTAGE - target_vary_range;
        let constraint = bits_per_frame * TLAYER_WEIGHT;
        let scale = MAX_BITS_VARY_PERCENTAGE * WEIGHT_MULTIPLY;
        self.min_bits = div_round(constraint * min_ratio, scale);
        self.max_bits = div_round(constraint * MAX_BITS_VARY_PERCENTAGE_X3D2, scale);
        if self.bits_per_frame > REMAIN_BITS_TH {
            self.remaining_bits =
                div_round(self.remaining_bits * bits_per_frame, self.bits_per_frame);
        }
        self.bits_per_frame = bits_per_frame;
    }

    /// `RcInitVGop` with `bFixRCOverShoot`.
    fn init_vgop(&mut self) {
        let left = GOP_NUMBER_IN_VGOP - self.gop_index_in_vgop;
        self.remaining_bits -= left * (self.last_allocated_bits / GOP_NUMBER_IN_VGOP);
        if self.remaining_bits < 0 {
            // Carry the deficit so the overshoot is paid back.
            self.remaining_bits += VGOP_SIZE * self.bits_per_frame;
        } else {
            self.remaining_bits = VGOP_SIZE * self.bits_per_frame;
        }
        self.last_allocated_bits = self.remaining_bits;
        self.remaining_weights = GOP_NUMBER_IN_VGOP * WEIGHT_MULTIPLY;
        self.gop_index_in_vgop = 0;
    }

    /// `RcDecideTargetBits`.
    fn decide_target_bits(&mut self, idr: bool) {
        self.bits_exceeded = false;
        if idr {
            self.target_bits = self.bits_per_frame * IDR_BITRATE_RATIO;
        } else {
            self.target_bits = if self.remaining_weights >= TLAYER_WEIGHT {
                div_round(
                    self.remaining_bits * TLAYER_WEIGHT,
                    self.remaining_weights,
                )
            } else {
                self.remaining_bits
            };
            if self.target_bits <= 0 {
                self.bits_exceeded = true;
            }
            self.target_bits = self.target_bits.clamp(self.min_bits, self.max_bits);
        }
        self.remaining_weights -= TLAYER_WEIGHT;
    }

    /// `RcCalculateIdrQp`.
    fn idr_qp(&mut self, complexity: i64) -> i32 {
        let bpp = if self.applied_fps > 0 && self.area > 0 {
            self.applied_bitrate as f64 / (self.applied_fps as f64 * self.area as f64)
        } else {
            DEFAULT_BPP
        };
        let class = AREA_CLASSES
            .iter()
            .position(|&a| self.area <= a)
            .unwrap_or(AREA_CLASSES.len());
        let bucket = BPP_BUCKETS[class]
            .iter()
            .position(|&b| bpp <= b)
            .unwrap_or(BPP_BUCKETS[class].len());
        let [range_max, range_min] = IDR_QP_RANGE[bucket];
        let min_qp = range_min.clamp(self.min_qp, self.max_qp);
        let max_qp = range_max.clamp(self.min_qp, self.max_qp);
        self.initial_qp = if self.idr_num == 0 {
            INITIAL_IDR_QP[class][bucket]
        } else {
            let ratio = cmplx_ratio(complexity, self.intra_cmplx_mean);
            let qstep = div_round(self.intra_complexity * ratio, self.target_bits * INT_MULTIPLY);
            qstep_to_qp(qstep)
        };
        self.initial_qp = self.initial_qp.clamp(min_qp, max_qp);
        self.last_qp = self.initial_qp;
        self.initial_qp
    }

    /// `RcCalculatePictureQp` (one temporal layer: no temporal delta).
    fn picture_qp(&mut self, complexity: i64) -> i32 {
        let qp = if self.p_frame_num == 0 {
            self.initial_qp
        } else if self.bits_exceeded {
            self.last_qp + DELTA_QP_BGD_THD
        } else {
            let ratio = cmplx_ratio(complexity, self.frame_cmplx_mean);
            let qstep = div_round(self.linear_cmplx * ratio, self.target_bits * INT_MULTIPLY);
            qstep_to_qp(qstep)
        };
        let lo = (self.last_qp - FRAME_DELTA_QP_LOWER).clamp(self.min_qp, self.max_qp);
        let hi = (self.last_qp + FRAME_DELTA_QP_UPPER).clamp(self.min_qp, self.max_qp);
        let qp = qp.clamp(lo, hi);
        self.last_qp = qp;
        qp
    }
}

/// Running average with [`LINEAR_MODEL_DECAY_FACTOR`].
const fn decay(old: i64, new: i64) -> i64 {
    div_round(
        LINEAR_MODEL_DECAY_FACTOR * old + (INT_MULTIPLY - LINEAR_MODEL_DECAY_FACTOR) * new,
        INT_MULTIPLY,
    )
}

/// `complexity / mean` in [`INT_MULTIPLY`] units, clamped to 1 +- 20 %.
fn cmplx_ratio(complexity: i64, mean: i64) -> i64 {
    div_round(complexity * INT_MULTIPLY, mean).clamp(
        INT_MULTIPLY - FRAME_CMPLX_RATIO_RANGE,
        INT_MULTIPLY + FRAME_CMPLX_RATIO_RANGE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qstep_qp_round_trip() {
        for qp in 0..52 {
            assert_eq!(qstep_to_qp(qp_to_qstep(qp)), qp, "qp {qp}");
        }
        assert_eq!(qstep_to_qp(0), 0);
        assert_eq!(qstep_to_qp(i64::MAX), 51);
    }

    #[test]
    fn frame_delta_limits_match_c() {
        assert_eq!(FRAME_DELTA_QP_UPPER, 5);
        assert_eq!(FRAME_DELTA_QP_LOWER, 3);
    }

    #[test]
    fn qp_rises_when_frames_overshoot() {
        let mut rc = RateControl::new(640, 360, 500_000, 30);
        let qp0 = rc.picture_init(true, 1000);
        rc.picture_update(true, qp0, 200_000, 1000);
        let mut qp = rc.picture_init(false, 1000);
        for _ in 0..20 {
            // Every P frame costs 4x its budget.
            rc.picture_update(false, qp, 4 * 500_000 / 30, 1000);
            let next = rc.picture_init(false, 1000);
            assert!(next >= qp, "qp fell from {qp} to {next} while overshooting");
            qp = next;
        }
        assert!(qp > qp0);
    }
}
