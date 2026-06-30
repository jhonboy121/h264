//! CABAC arithmetic decoding engine and context-model initialisation.
//!
//! Ported faithfully from Cisco OpenH264:
//! - engine state + init: `InitCabacDecEngineFromBS`, `Read32BitsCabac`,
//!   `DecodeBinCabac`, `DecodeBypassCabac`, `DecodeTerminateCabac`
//!   (`reference/codec/decoder/core/src/cabac_decoder.cpp`).
//! - context init: `WelsCabacGlobalInit` / `WelsCabacContextInit` (same file),
//!   spec 9.3.1.1.
//!
//! The reference keeps a 64-bit `uiOffset` cache together with an `iBitsLeft`
//! counter rather than the byte-at-a-time `codIOffset` of the spec text; this
//! port reproduces that exact arithmetic so bit consumption matches the C.
//!
//! Unlike the C, which slices back into a `SBitStringAux` read cache, the engine
//! here is constructed directly from the RBSP byte slice and the byte offset at
//! which the CABAC data begins (i.e. the first byte after
//! `cabac_alignment_one_bit`). [`crate::bits::BitReader::byte_aligned`] /
//! `bit_pos` give that offset for the caller.

use crate::error::DecodeError;

use super::cabac_tables as t;
use super::slice_header::SliceType;

type Result<T> = core::result::Result<T, DecodeError>;

/// `WELS_CABAC_HALF` — the initial range (510, spec 9.3.1.2).
const WELS_CABAC_HALF: u64 = 0x01FE;
/// `WELS_CABAC_QUARTER` (256).
const WELS_CABAC_QUARTER: u64 = 0x0100;

/// One CABAC context model (`SWelsCabacCtx`): probability-state index and the
/// value of the most-probable symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CabacCtx {
    /// `pStateIdx` (0..=62).
    pub state: u8,
    /// `valMps` (0 or 1).
    pub mps: u8,
}

/// The full set of `WELS_CONTEXT_COUNT` context models for one slice
/// (`pCabacCtx[WELS_CONTEXT_COUNT]`).
#[derive(Clone)]
pub struct CabacContexts {
    ctx: [CabacCtx; t::WELS_CONTEXT_COUNT],
}

impl CabacContexts {
    /// Initialise all contexts from `(slice_type, cabac_init_idc, qp)` per
    /// `WelsCabacContextInit` + the `{m, n}` formula of spec 9.3.1.1.
    ///
    /// `cabac_init_idc` is ignored for I/SI slices (model index 0); for other
    /// slice types the model index is `cabac_init_idc + 1`.
    pub fn init(slice_type: SliceType, cabac_init_idc: u32, qp: i32) -> Self {
        let model: usize = if slice_type.is_intra() {
            0
        } else {
            (cabac_init_idc as usize) + 1
        };
        let mut ctx = [CabacCtx::default(); t::WELS_CONTEXT_COUNT];
        for (i, c) in ctx.iter_mut().enumerate() {
            let m = t::CABAC_CONTEXT_INIT[i][model][0] as i32;
            let n = t::CABAC_CONTEXT_INIT[i][model][1] as i32;
            // WELS_CLIP3(((m * qp) >> 4) + n, 1, 126). The `>> 4` is an
            // arithmetic shift in the C (i32 >> in Rust matches).
            let pre = (((m * qp) >> 4) + n).clamp(1, 126);
            if pre <= 63 {
                c.state = (63 - pre) as u8;
                c.mps = 0;
            } else {
                c.state = (pre - 64) as u8;
                c.mps = 1;
            }
        }
        Self { ctx }
    }

    /// Borrow a context model by index.
    #[inline]
    pub fn ctx(&mut self, idx: usize) -> &mut CabacCtx {
        &mut self.ctx[idx]
    }

    /// Immutable read of one context (for tests / inspection).
    #[inline]
    pub fn get(&self, idx: usize) -> CabacCtx {
        self.ctx[idx]
    }
}

/// CABAC arithmetic decoding engine over an RBSP byte slice
/// (`SWelsCabacDecEngine`).
pub struct CabacDecoder<'a> {
    data: &'a [u8],
    /// Index of the next unread byte (`pBuffCurr`).
    curr: usize,
    /// `uiRange`.
    range: u64,
    /// `uiOffset` cache (up to 40 valid low bits).
    offset: u64,
    /// `iBitsLeft` — number of low bits of `offset` not yet "consumed".
    bits_left: i32,
}

impl<'a> CabacDecoder<'a> {
    /// Construct the engine from the RBSP bytes and the byte offset where CABAC
    /// data begins (`InitCabacDecEngineFromBS`, spec 9.3.1.2). Pre-reads 5
    /// bytes into the offset cache; `range` is set to 510.
    pub fn new(rbsp: &'a [u8], byte_offset: usize) -> Result<Self> {
        // The C requires `pCurr < pEndBuf - 1`; mirror that minimally by
        // demanding at least two readable bytes at the start position.
        if byte_offset + 1 >= rbsp.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let b = |i: usize| -> u64 { rbsp.get(byte_offset + i).copied().unwrap_or(0) as u64 };
        let mut offset = (b(0) << 16) | (b(1) << 8) | b(2);
        offset <<= 16;
        offset |= (b(3) << 8) | b(4);
        Ok(Self {
            data: rbsp,
            curr: byte_offset + 5,
            range: WELS_CABAC_HALF,
            offset,
            bits_left: 31,
        })
    }

    /// Current `uiRange` (test/inspection).
    #[inline]
    pub fn range(&self) -> u64 {
        self.range
    }

    /// Current `uiOffset` (test/inspection).
    #[inline]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// `iBitsLeft` (test/inspection).
    #[inline]
    pub fn bits_left(&self) -> i32 {
        self.bits_left
    }

    /// `Read32BitsCabac`: pull up to 4 more bytes from the stream into a value,
    /// returning `(value, num_bits_read)`. Returns `(0, 0)` at end of stream
    /// (the C surfaces an error here but still proceeds; valid bitstreams never
    /// read past the end before terminating).
    #[inline]
    fn read32(&mut self) -> (u64, i32) {
        let left = self.data.len() as isize - self.curr as isize;
        if left <= 0 {
            return (0, 0);
        }
        let d = self.data;
        match left {
            3 => {
                let v = ((d[self.curr] as u64) << 16)
                    | ((d[self.curr + 1] as u64) << 8)
                    | (d[self.curr + 2] as u64);
                self.curr += 3;
                (v, 24)
            }
            2 => {
                let v = ((d[self.curr] as u64) << 8) | (d[self.curr + 1] as u64);
                self.curr += 2;
                (v, 16)
            }
            1 => {
                let v = d[self.curr] as u64;
                self.curr += 1;
                (v, 8)
            }
            _ => {
                let v = ((d[self.curr] as u64) << 24)
                    | ((d[self.curr + 1] as u64) << 16)
                    | ((d[self.curr + 2] as u64) << 8)
                    | (d[self.curr + 3] as u64);
                self.curr += 4;
                (v, 32)
            }
        }
    }

    /// Read the 384 raw I_PCM sample bytes (256 luma + 64 Cb + 64 Cr, 4:2:0)
    /// that follow an I_PCM `mb_type`, then re-initialise the arithmetic engine
    /// just past them. This reproduces `RestoreCabacDecEngineToBS` +
    /// `InitReadBits(.., 1)` + `InitCabacDecEngineFromBS` (spec 9.3.1): the
    /// actual stream position is `pBuffCurr - (iBitsLeft >> 3)`, and re-init from
    /// that position + 384 is exactly a fresh engine constructed there.
    pub fn read_pcm_bytes(&mut self) -> Result<[u8; 384]> {
        let pcm_start = self.curr - ((self.bits_left >> 3) as usize);
        if pcm_start + 384 > self.data.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let mut out = [0u8; 384];
        out.copy_from_slice(&self.data[pcm_start..pcm_start + 384]);
        let data = self.data;
        *self = CabacDecoder::new(data, pcm_start + 384)?;
        Ok(out)
    }

    /// `DecodeBinCabac`: decode one bin using context `ctx`, updating its state
    /// and the engine range/offset (the `DecodeDecision` of spec 9.3.4.2).
    pub fn decode_decision(&mut self, ctx: &mut CabacCtx) -> u32 {
        let state = ctx.state as usize;
        let mut bin_val = ctx.mps as u32;
        let mut offset = self.offset;
        let mut range = self.range;

        let mut renorm: i32 = 1;
        let range_lps = t::RANGE_LPS[state][((range >> 6) & 0x03) as usize] as u64;
        range -= range_lps;
        if offset >= (range << self.bits_left) {
            // LPS
            offset -= range << self.bits_left;
            bin_val ^= 0x0001;
            if state == 0 {
                ctx.mps ^= 0x01;
            }
            ctx.state = t::STATE_TRANS[state][0];
            renorm = t::RENORM_TABLE[range_lps as usize] as i32;
            range = range_lps << renorm;
        } else {
            // MPS
            ctx.state = t::STATE_TRANS[state][1];
            if range >= WELS_CABAC_QUARTER {
                self.range = range;
                return bin_val;
            } else {
                range <<= 1;
            }
        }
        // Renorm
        self.range = range;
        self.bits_left -= renorm;
        if self.bits_left > 0 {
            self.offset = offset;
            return bin_val;
        }
        let (val, num_bits) = self.read32();
        self.offset = (offset << num_bits) | val;
        self.bits_left += num_bits;
        bin_val
    }

    /// `DecodeBypassCabac`: decode one equiprobable (bypass) bin
    /// (spec 9.3.4.3).
    pub fn decode_bypass(&mut self) -> u32 {
        let mut bits_left = self.bits_left;
        let mut offset = self.offset;
        if bits_left <= 0 {
            let (val, num_bits) = self.read32();
            offset = (offset << num_bits) | val;
            bits_left = num_bits;
        }
        bits_left -= 1;
        let range_value = self.range << bits_left;
        if offset >= range_value {
            self.bits_left = bits_left;
            self.offset = offset - range_value;
            1
        } else {
            self.bits_left = bits_left;
            self.offset = offset;
            0
        }
    }

    /// `DecodeExpBypassCabac`: decode an Exp-Golomb-order-`k` suffix entirely in
    /// bypass mode (`iCount == k`). Used by the UEGk binarisations of
    /// `coeff_abs_level_minus1` (k=0) and `mvd` (k=3).
    pub fn decode_exp_bypass(&mut self, mut count: i32) -> u32 {
        let mut sym: i32 = 0;
        let mut sym2: i32 = 0;
        loop {
            let code = self.decode_bypass();
            if code == 1 {
                sym += 1 << count;
                count += 1;
            }
            if code == 0 || count == 16 {
                break;
            }
        }
        while count > 0 {
            count -= 1;
            if self.decode_bypass() == 1 {
                sym2 |= 1 << count;
            }
        }
        (sym + sym2) as u32
    }

    /// `DecodeTerminateCabac`: decode the `end_of_slice_flag` / PCM terminator
    /// bin (spec 9.3.4.4). Returns 1 when the arithmetic decode terminates.
    pub fn decode_terminate(&mut self) -> u32 {
        let range = self.range - 2;
        let offset = self.offset;
        if offset >= (range << self.bits_left) {
            return 1;
        }
        // bin == 0; renormalise.
        if range < WELS_CABAC_QUARTER {
            let renorm = t::RENORM_TABLE[range as usize] as i32;
            self.range = range << renorm;
            self.bits_left -= renorm;
            if self.bits_left < 0 {
                let (val, num_bits) = self.read32();
                self.offset = (self.offset << num_bits) | val;
                self.bits_left += num_bits;
            }
        } else {
            self.range = range;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_have_expected_dims_and_entries() {
        assert_eq!(t::RANGE_LPS.len(), 64);
        assert_eq!(t::RANGE_LPS[0].len(), 4);
        // spec Table 9-44 corners.
        assert_eq!(t::RANGE_LPS[0], [128, 176, 208, 240]);
        assert_eq!(t::RANGE_LPS[63], [2, 2, 2, 2]);

        assert_eq!(t::STATE_TRANS.len(), 64);
        // spec Table 9-45: state 0 -> {0, 1}; state 62 -> {38, 62}; 63 -> {63,63}.
        assert_eq!(t::STATE_TRANS[0], [0, 1]);
        assert_eq!(t::STATE_TRANS[62], [38, 62]);
        assert_eq!(t::STATE_TRANS[63], [63, 63]);

        assert_eq!(t::RENORM_TABLE.len(), 256);
        assert_eq!(t::RENORM_TABLE[0], 6);
        assert_eq!(t::RENORM_TABLE[8], 5);
        assert_eq!(t::RENORM_TABLE[255], 1);

        assert_eq!(t::CABAC_CONTEXT_INIT.len(), 460);
        // ctx 0 is {20,-15} for every model (spec Table 9-12).
        assert_eq!(t::CABAC_CONTEXT_INIT[0][0], [20, -15]);
        assert_eq!(t::CABAC_CONTEXT_INIT[0][3], [20, -15]);
        // last ctx 459 (Table 9-24 tail).
        assert_eq!(t::CABAC_CONTEXT_INIT[459][0], [14, 67]);
        assert_eq!(t::CABAC_CONTEXT_INIT[459][3], [20, 64]);
    }

    // Reference computation of one context's (state, mps) from (m, n, qp),
    // mirroring WelsCabacGlobalInit exactly.
    fn ref_ctx(m: i32, n: i32, qp: i32) -> CabacCtx {
        let pre = (((m * qp) >> 4) + n).clamp(1, 126);
        if pre <= 63 {
            CabacCtx { state: (63 - pre) as u8, mps: 0 }
        } else {
            CabacCtx { state: (pre - 64) as u8, mps: 1 }
        }
    }

    #[test]
    fn context_init_matches_hand_computed() {
        // I slice -> model 0. ctx 0 is {20, -15}.
        let qp = 26;
        let c = CabacContexts::init(SliceType::I, 0, qp);
        // pre = ((20*26)>>4) + (-15) = (520>>4) - 15 = 32 - 15 = 17 -> <=63
        // state = 63-17 = 46, mps = 0.
        assert_eq!(c.get(0), CabacCtx { state: 46, mps: 0 });
        assert_eq!(c.get(0), ref_ctx(20, -15, qp));

        // ctx 2 = {3, 74}: pre = ((3*26)>>4)+74 = (78>>4)+74 = 4+74 = 78 -> >63
        // state = 78-64 = 14, mps = 1.
        assert_eq!(c.get(2), CabacCtx { state: 14, mps: 1 });
        assert_eq!(c.get(2), ref_ctx(3, 74, qp));

        // P slice, cabac_init_idc = 0 -> model 1. ctx 11 = {23, 33}.
        let qp2 = 30;
        let cp = CabacContexts::init(SliceType::P, 0, qp2);
        // pre = ((23*30)>>4)+33 = (690>>4)+33 = 43+33 = 76 -> state 12, mps 1.
        assert_eq!(cp.get(11), CabacCtx { state: 12, mps: 1 });
        assert_eq!(cp.get(11), ref_ctx(23, 33, qp2));

        // B slice, cabac_init_idc = 2 -> model 3. ctx 11 = {29, 16}.
        let cb = CabacContexts::init(SliceType::B, 2, qp2);
        // pre = ((29*30)>>4)+16 = (870>>4)+16 = 54+16 = 70 -> state 6, mps 1.
        assert_eq!(cb.get(11), CabacCtx { state: 6, mps: 1 });
        assert_eq!(cb.get(11), ref_ctx(29, 16, qp2));

        // Clip bound: a tiny qp with large negative pre clips to 1 -> state 62.
        // ctx 6 model 0 = {-28, 127}: at qp=0 pre = 0+127 = 127 -> clip 126 ->
        // state = 126-64 = 62, mps 1.
        let c0 = CabacContexts::init(SliceType::I, 0, 0);
        assert_eq!(c0.get(6), CabacCtx { state: 62, mps: 1 });
    }

    #[test]
    fn engine_init_invariants() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let dec = CabacDecoder::new(&data, 0).unwrap();
        assert_eq!(dec.range(), 510);
        // 0 <= offset < range << bits_left? The *effective* codIOffset is
        // offset >> bits_left and must be < range. Check that.
        let cod_offset = dec.offset() >> dec.bits_left();
        assert!(cod_offset < dec.range(), "codIOffset {cod_offset} < range");
    }

    #[test]
    fn new_rejects_too_short() {
        let data = [0xFFu8];
        assert!(CabacDecoder::new(&data, 0).is_err());
        // offset at the very end is rejected too.
        let data2 = [0xFFu8, 0x00];
        assert!(CabacDecoder::new(&data2, 1).is_err());
    }

    // Bypass decoding is purely `offset`-doubling vs `range << bits_left`
    // comparisons, so we can hand-construct a stream and predict each bin.
    //
    // After init: range = 510, offset = 40-bit pre-read, bits_left = 31.
    // decode_bypass: bits_left -= 1 (=30); range_value = 510 << 30. bin = 1 iff
    // offset >= range_value, then offset -= range_value.
    #[test]
    fn bypass_decodes_predictably() {
        let data = [0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut dec = CabacDecoder::new(&data, 0).unwrap();

        // Mirror the arithmetic exactly with a shadow model.
        let mut shadow_offset = dec.offset();
        let mut shadow_bits = dec.bits_left();
        let range = dec.range();
        let mut curr = 5usize; // bytes already consumed by new()

        for _ in 0..40 {
            // Predict using the shadow state (replicating decode_bypass).
            let mut off = shadow_offset;
            let mut bl = shadow_bits;
            if bl <= 0 {
                // Replicate read32 from `data`.
                let left = data.len() as isize - curr as isize;
                let (v, nb): (u64, i32) = if left <= 0 {
                    (0, 0)
                } else if left == 3 {
                    let v = ((data[curr] as u64) << 16)
                        | ((data[curr + 1] as u64) << 8)
                        | (data[curr + 2] as u64);
                    curr += 3;
                    (v, 24)
                } else if left == 2 {
                    let v = ((data[curr] as u64) << 8) | (data[curr + 1] as u64);
                    curr += 2;
                    (v, 16)
                } else if left == 1 {
                    let v = data[curr] as u64;
                    curr += 1;
                    (v, 8)
                } else {
                    let v = ((data[curr] as u64) << 24)
                        | ((data[curr + 1] as u64) << 16)
                        | ((data[curr + 2] as u64) << 8)
                        | (data[curr + 3] as u64);
                    curr += 4;
                    (v, 32)
                };
                off = (off << nb) | v;
                bl = nb;
            }
            bl -= 1;
            let rv = range << bl;
            let predicted = if off >= rv {
                shadow_offset = off - rv;
                1
            } else {
                shadow_offset = off;
                0
            };
            shadow_bits = bl;

            let got = dec.decode_bypass();
            assert_eq!(got, predicted);
        }
    }

    // decode_exp_bypass is pure bypass, so it is fully determined by the
    // decode_bypass sequence. On an all-zero stream every bypass bin is 0, so
    // the very first loop iteration (code == 0) ends the prefix with count
    // unchanged at the initial k, then the k-bit suffix reads k zero bins.
    #[test]
    fn exp_bypass_zero_stream() {
        let zeros = [0u8; 12];
        let mut dec = CabacDecoder::new(&zeros, 0).unwrap();
        // k = 0: prefix bin 0 -> sym 0, suffix 0 bits -> 0.
        assert_eq!(dec.decode_exp_bypass(0), 0);
        // k = 3: prefix bin 0 -> sym 0, then 3 suffix bins all 0 -> 0.
        let mut dec2 = CabacDecoder::new(&zeros, 0).unwrap();
        assert_eq!(dec2.decode_exp_bypass(3), 0);
    }

    // DecodeTerminate on a crafted stream: with offset's top bits below
    // range-2, the terminator returns 0 and renormalises; a stream whose
    // effective codIOffset is large returns 1.
    #[test]
    fn terminate_behaviour() {
        // Stream of 0x00 bytes -> offset == 0 -> never terminates (returns 0).
        let zeros = [0u8; 12];
        let mut dec = CabacDecoder::new(&zeros, 0).unwrap();
        assert_eq!(dec.decode_terminate(), 0);
        // range was 510, range-2 = 508 >= QUARTER so range becomes 508 with no
        // renorm shift.
        assert_eq!(dec.range(), 508);

        // Stream of 0xFF bytes -> offset has all high bits set -> codIOffset is
        // large, terminate returns 1.
        let ones = [0xFFu8; 12];
        let mut dec2 = CabacDecoder::new(&ones, 0).unwrap();
        assert_eq!(dec2.decode_terminate(), 1);
    }

    // A decision bin on an all-zero offset stream: offset == 0 always selects
    // the MPS path (offset < range-rangeLPS shifted), so the returned bin equals
    // the context MPS and the state advances along transIdxMPS.
    #[test]
    fn decision_mps_path_on_zero_offset() {
        let zeros = [0u8; 12];
        let mut dec = CabacDecoder::new(&zeros, 0).unwrap();
        let mut ctx = CabacCtx { state: 10, mps: 1 };
        let bin = dec.decode_decision(&mut ctx);
        assert_eq!(bin, 1); // MPS value
        // state 10 -> transIdxMPS = STATE_TRANS[10][1].
        assert_eq!(ctx.state, t::STATE_TRANS[10][1]);
        assert_eq!(ctx.mps, 1);
    }
}
