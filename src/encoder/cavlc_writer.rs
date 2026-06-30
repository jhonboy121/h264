//! CAVLC residual-block encoding, ported from Cisco OpenH264
//! `reference/codec/encoder/core/src/set_mb_syn_cavlc.cpp`
//! (`CavlcParamCal_c` + `WriteBlockResidualCavlc`) with the encode VLC tables
//! from `encoder_data_tables.cpp` (`g_kuiVlcCoeffToken`, `g_kuiVlcTotalZeros`,
//! `g_kuiVlcTotalZerosChromaDc`, `g_kuiVlcRunBefore`, `g_kuiZeroLeftMap`,
//! `g_kuiEncNcMapTable`).
//!
//! This is the exact inverse of the decoder's
//! [`crate::decoder::cavlc::residual_block_cavlc`]: the coefficient block is
//! supplied in zig-zag *scan order* (`coeff_level[0..=end_idx]`), run/level/
//! total_zeros/run_before are derived, and the codewords are emitted MSB-first.

use crate::bits::BitWriter;

/// `g_kuiVlcCoeffToken[nc_idx][total_coeff][trailing_ones] = (value, bits)`.
/// `nc_idx`: 0 -> 0<=nC<2, 1 -> 2<=nC<4, 2 -> 4<=nC<8, 3 -> 8<=nC, 4 -> nC==-1.
#[rustfmt::skip]
static COEFF_TOKEN: [[[(u8, u8); 4]; 17]; 5] = [
    [ // 0<=nC<2
        [(1,1),(0,0),(0,0),(0,0)], [(5,6),(1,2),(0,0),(0,0)], [(7,8),(4,6),(1,3),(0,0)],
        [(7,9),(6,8),(5,7),(3,5)], [(7,10),(6,9),(5,8),(3,6)], [(7,11),(6,10),(5,9),(4,7)],
        [(15,13),(6,11),(5,10),(4,8)], [(11,13),(14,13),(5,11),(4,9)], [(8,13),(10,13),(13,13),(4,10)],
        [(15,14),(14,14),(9,13),(4,11)], [(11,14),(10,14),(13,14),(12,13)], [(15,15),(14,15),(9,14),(12,14)],
        [(11,15),(10,15),(13,15),(8,14)], [(15,16),(1,15),(9,15),(12,15)], [(11,16),(14,16),(13,16),(8,15)],
        [(7,16),(10,16),(9,16),(12,16)], [(4,16),(6,16),(5,16),(8,16)],
    ],
    [ // 2<=nC<4
        [(3,2),(0,0),(0,0),(0,0)], [(11,6),(2,2),(0,0),(0,0)], [(7,6),(7,5),(3,3),(0,0)],
        [(7,7),(10,6),(9,6),(5,4)], [(7,8),(6,6),(5,6),(4,4)], [(4,8),(6,7),(5,7),(6,5)],
        [(7,9),(6,8),(5,8),(8,6)], [(15,11),(6,9),(5,9),(4,6)], [(11,11),(14,11),(13,11),(4,7)],
        [(15,12),(10,11),(9,11),(4,9)], [(11,12),(14,12),(13,12),(12,11)], [(8,12),(10,12),(9,12),(8,11)],
        [(15,13),(14,13),(13,13),(12,12)], [(11,13),(10,13),(9,13),(12,13)], [(7,13),(11,14),(6,13),(8,13)],
        [(9,14),(8,14),(10,14),(1,13)], [(7,14),(6,14),(5,14),(4,14)],
    ],
    [ // 4<=nC<8
        [(15,4),(0,0),(0,0),(0,0)], [(15,6),(14,4),(0,0),(0,0)], [(11,6),(15,5),(13,4),(0,0)],
        [(8,6),(12,5),(14,5),(12,4)], [(15,7),(10,5),(11,5),(11,4)], [(11,7),(8,5),(9,5),(10,4)],
        [(9,7),(14,6),(13,6),(9,4)], [(8,7),(10,6),(9,6),(8,4)], [(15,8),(14,7),(13,7),(13,5)],
        [(11,8),(14,8),(10,7),(12,6)], [(15,9),(10,8),(13,8),(12,7)], [(11,9),(14,9),(9,8),(12,8)],
        [(8,9),(10,9),(13,9),(8,8)], [(13,10),(7,9),(9,9),(12,9)], [(9,10),(12,10),(11,10),(10,10)],
        [(5,10),(8,10),(7,10),(6,10)], [(1,10),(4,10),(3,10),(2,10)],
    ],
    [ // 8<=nC
        [(3,6),(0,0),(0,0),(0,0)], [(0,6),(1,6),(0,0),(0,0)], [(4,6),(5,6),(6,6),(0,0)],
        [(8,6),(9,6),(10,6),(11,6)], [(12,6),(13,6),(14,6),(15,6)], [(16,6),(17,6),(18,6),(19,6)],
        [(20,6),(21,6),(22,6),(23,6)], [(24,6),(25,6),(26,6),(27,6)], [(28,6),(29,6),(30,6),(31,6)],
        [(32,6),(33,6),(34,6),(35,6)], [(36,6),(37,6),(38,6),(39,6)], [(40,6),(41,6),(42,6),(43,6)],
        [(44,6),(45,6),(46,6),(47,6)], [(48,6),(49,6),(50,6),(51,6)], [(52,6),(53,6),(54,6),(55,6)],
        [(56,6),(57,6),(58,6),(59,6)], [(60,6),(61,6),(62,6),(63,6)],
    ],
    [ // nC == -1 (chroma DC)
        [(1,2),(0,0),(0,0),(0,0)], [(7,6),(1,1),(0,0),(0,0)], [(4,6),(6,6),(1,3),(0,0)],
        [(3,6),(3,7),(2,7),(5,6)], [(2,6),(3,8),(2,8),(0,7)], [(0,0),(0,0),(0,0),(0,0)],
        [(0,0),(0,0),(0,0),(0,0)], [(0,0),(0,0),(0,0),(0,0)], [(0,0),(0,0),(0,0),(0,0)],
        [(0,0),(0,0),(0,0),(0,0)], [(0,0),(0,0),(0,0),(0,0)], [(0,0),(0,0),(0,0),(0,0)],
        [(0,0),(0,0),(0,0),(0,0)], [(0,0),(0,0),(0,0),(0,0)], [(0,0),(0,0),(0,0),(0,0)],
        [(0,0),(0,0),(0,0),(0,0)], [(0,0),(0,0),(0,0),(0,0)],
    ],
];

/// `g_kuiEncNcMapTable[18]`: maps `nC` (0..=16) to a coeff_token table index.
static ENC_NC_MAP: [u8; 18] = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4];

/// `g_kuiVlcTotalZeros[tz_vlc_index][total_zeros] = (value, bits)`.
#[rustfmt::skip]
static TOTAL_ZEROS: [[(u8, u8); 16]; 16] = [
    [(0,0);16],
    [(1,1),(3,3),(2,3),(3,4),(2,4),(3,5),(2,5),(3,6),(2,6),(3,7),(2,7),(3,8),(2,8),(3,9),(2,9),(1,9)],
    [(7,3),(6,3),(5,3),(4,3),(3,3),(5,4),(4,4),(3,4),(2,4),(3,5),(2,5),(3,6),(2,6),(1,6),(0,6),(0,0)],
    [(5,4),(7,3),(6,3),(5,3),(4,4),(3,4),(4,3),(3,3),(2,4),(3,5),(2,5),(1,6),(1,5),(0,6),(0,0),(0,0)],
    [(3,5),(7,3),(5,4),(4,4),(6,3),(5,3),(4,3),(3,4),(3,3),(2,4),(2,5),(1,5),(0,5),(0,0),(0,0),(0,0)],
    [(5,4),(4,4),(3,4),(7,3),(6,3),(5,3),(4,3),(3,3),(2,4),(1,5),(1,4),(0,5),(0,0),(0,0),(0,0),(0,0)],
    [(1,6),(1,5),(7,3),(6,3),(5,3),(4,3),(3,3),(2,3),(1,4),(1,3),(0,6),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(1,6),(1,5),(5,3),(4,3),(3,3),(3,2),(2,3),(1,4),(1,3),(0,6),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(1,6),(1,4),(1,5),(3,3),(3,2),(2,2),(2,3),(1,3),(0,6),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(1,6),(0,6),(1,4),(3,2),(2,2),(1,3),(1,2),(1,5),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(1,5),(0,5),(1,3),(3,2),(2,2),(1,2),(1,4),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(0,4),(1,4),(1,3),(2,3),(1,1),(3,3),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(0,4),(1,4),(1,2),(1,1),(1,3),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(0,3),(1,3),(1,1),(1,2),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(0,2),(1,2),(1,1),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(0,1),(1,1),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
];

/// `g_kuiVlcTotalZerosChromaDc[tz_vlc_index][total_zeros] = (value, bits)`.
#[rustfmt::skip]
static TOTAL_ZEROS_CHROMA_DC: [[(u8, u8); 4]; 4] = [
    [(0,0),(0,0),(0,0),(0,0)],
    [(1,1),(1,2),(1,3),(0,3)],
    [(1,1),(1,2),(0,2),(0,0)],
    [(1,1),(0,1),(0,0),(0,0)],
];

/// `g_kuiVlcRunBefore[zeros_left][run_before] = (value, bits)`.
#[rustfmt::skip]
static RUN_BEFORE: [[(u8, u8); 15]; 8] = [
    [(0,0);15],
    [(1,1),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(1,1),(1,2),(0,2),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(3,2),(2,2),(1,2),(0,2),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(3,2),(2,2),(1,2),(1,3),(0,3),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(3,2),(2,2),(3,3),(2,3),(1,3),(0,3),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(3,2),(0,3),(1,3),(3,3),(2,3),(5,3),(4,3),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
    [(7,3),(6,3),(5,3),(4,3),(3,3),(2,3),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(1,11)],
];

/// `g_kuiZeroLeftMap[16]`: clamp `zeros_left` to the run_before table row index.
static ZERO_LEFT_MAP: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7];

/// Run/level decomposition (`CavlcParamCal_c`). Walks `coeff_level[0..=last_idx]`
/// from the highest scan position down, producing `level[]`/`run[]` (highest
/// frequency first), the total coefficient count, and returning total_zeros.
fn cavlc_param_cal(
    coeff_level: &[i16],
    last_idx: i32,
    level: &mut [i16; 16],
    run: &mut [u8; 16],
    total_coeff: &mut usize,
) -> i32 {
    let mut total_zeros = 0i32;
    let mut total_coeffs = 0usize;
    let mut i = last_idx;
    while i >= 0 && coeff_level[i as usize] == 0 {
        i -= 1;
    }
    while i >= 0 {
        level[total_coeffs] = coeff_level[i as usize];
        i -= 1;
        let mut count_zero = 0i32;
        while i >= 0 && coeff_level[i as usize] == 0 {
            count_zero += 1;
            i -= 1;
        }
        total_zeros += count_zero;
        run[total_coeffs] = count_zero as u8;
        total_coeffs += 1;
    }
    *total_coeff = total_coeffs;
    total_zeros
}

/// Encode one residual block in CAVLC. `coeff_level` holds the block's
/// coefficients in zig-zag scan order; `end_idx` is the last valid scan index
/// (15 for a full 4x4 / luma-DC block, 14 for an AC block, 3 for chroma DC).
///
/// `nc` is the predicted non-zero count (context); pass any value for the
/// chroma-DC path and set `chroma_dc = true`, which selects the fixed nC==-1
/// coeff_token table and the chroma total_zeros tables.
///
/// Port of `WriteBlockResidualCavlc`.
pub fn write_residual_block(
    bw: &mut BitWriter,
    coeff_level: &[i16],
    end_idx: usize,
    chroma_dc: bool,
    nc: i32,
) {
    let mut level = [0i16; 16];
    let mut run = [0u8; 16];
    let mut total_coeffs = 0usize;
    let total_zeros = cavlc_param_cal(coeff_level, end_idx as i32, &mut level, &mut run, &mut total_coeffs);

    // Trailing ones (up to 3 leading +/-1 coefficients) and their sign bits.
    let mut trailing_ones = 0usize;
    let mut sign = 0u32;
    let count = total_coeffs.min(3);
    for i in 0..count {
        if level[i].unsigned_abs() == 1 {
            trailing_ones += 1;
            sign <<= 1;
            if level[i] < 0 {
                sign |= 1;
            }
        } else {
            break;
        }
    }

    // Step 3: coeff_token.
    let nc_idx = if chroma_dc { 4usize } else { ENC_NC_MAP[nc.clamp(0, 16) as usize] as usize };
    let (ct_val, ct_bits) = COEFF_TOKEN[nc_idx][total_coeffs][trailing_ones];
    if total_coeffs == 0 {
        bw.write_bits(ct_val as u32, ct_bits as u32);
        return;
    }

    // Step 4: coeff_token + trailing-one signs packed together.
    let n = ct_bits as u32 + trailing_ones as u32;
    let value = ((ct_val as u32) << trailing_ones) + sign;
    bw.write_bits(value, n);

    // Levels.
    let mut suffix_length: i32 = (total_coeffs > 10 && trailing_ones < 3) as i32;
    for i in trailing_ones..total_coeffs {
        let val = level[i] as i32;
        let mut level_code = (val - 1) * 2;
        let s = level_code >> 31;
        level_code = (level_code ^ s) + (s << 1);
        level_code -= (((i == trailing_ones) && (trailing_ones < 3)) as i32) << 1;

        let mut level_prefix = level_code >> suffix_length;
        let mut level_suffix_size = suffix_length;
        let mut level_suffix = level_code - (level_prefix << suffix_length);

        if level_prefix >= 14 && level_prefix < 30 && suffix_length == 0 {
            level_prefix = 14;
            level_suffix = level_code - level_prefix;
            level_suffix_size = 4;
        } else if level_prefix >= 15 {
            level_prefix = 15;
            level_suffix = level_code - (level_prefix << suffix_length);
            // Baseline profile overflow guard (matches ENC_RETURN_VLCOVERFLOWFOUND).
            debug_assert!((level_suffix >> 11) == 0, "cavlc level suffix overflow");
            if suffix_length == 0 {
                level_suffix -= 15;
            }
            level_suffix_size = 12;
        }

        let n = (level_prefix + 1 + level_suffix_size) as u32;
        let value = ((1i32 << level_suffix_size) | level_suffix) as u32;
        bw.write_bits(value, n);

        suffix_length += (suffix_length == 0) as i32;
        let threshold = 3 << (suffix_length - 1);
        suffix_length += ((val > threshold || val < -threshold) && suffix_length < 6) as i32;
    }

    // Step 5: total_zeros.
    if total_coeffs < end_idx + 1 {
        let (tz_val, tz_bits) = if chroma_dc {
            TOTAL_ZEROS_CHROMA_DC[total_coeffs][total_zeros as usize]
        } else {
            TOTAL_ZEROS[total_coeffs][total_zeros as usize]
        };
        bw.write_bits(tz_val as u32, tz_bits as u32);
    }

    // Step 6: run_before.
    let mut zeros_left = total_zeros;
    let mut i = 0usize;
    while i + 1 < total_coeffs && zeros_left > 0 {
        let r = run[i] as usize;
        let zl = ZERO_LEFT_MAP[zeros_left as usize] as usize;
        let (rb_val, rb_bits) = RUN_BEFORE[zl][r];
        bw.write_bits(rb_val as u32, rb_bits as u32);
        zeros_left -= run[i] as i32;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::decoder::cavlc::residual_block_cavlc;
    use alloc::vec;

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0
        }
    }

    /// Write a scan-order block, decode it back, assert identical coefficients
    /// and an exact bit-position match.
    fn roundtrip(coeff: &[i16], end_idx: usize, chroma_dc: bool, nc: i32) {
        let max = end_idx + 1;
        let mut bw = BitWriter::new();
        write_residual_block(&mut bw, coeff, end_idx, chroma_dc, nc);
        let written = bw.bit_len();
        // Pad so the decoder can peek past the payload (its read cache zero-fills).
        let mut bytes = bw.finish();
        bytes.extend_from_slice(&[0u8; 4]);

        let mut br = BitReader::new(&bytes);
        let dec_nc = if chroma_dc { -1 } else { nc };
        let mut out = vec![0i32; max];
        let total = residual_block_cavlc(&mut br, dec_nc, max, &mut out).unwrap();

        let expect_total = coeff[..max].iter().filter(|&&x| x != 0).count() as i32;
        assert_eq!(total, expect_total, "total_coeff mismatch");
        for k in 0..max {
            assert_eq!(out[k], coeff[k] as i32, "coeff[{k}] mismatch in {coeff:?}");
        }
        assert_eq!(br.bit_pos(), written, "bit position mismatch for {coeff:?}");
    }

    #[test]
    fn empty_block() {
        roundtrip(&[0i16; 16], 15, false, 0);
        roundtrip(&[0i16; 16], 14, false, 5);
        roundtrip(&[0i16; 4], 3, true, -1);
    }

    #[test]
    fn single_coeffs() {
        for &v in &[1i16, -1, 2, -2, 7, -8, 50, -50] {
            for pos in [0usize, 1, 7, 15] {
                let mut c = [0i16; 16];
                c[pos] = v;
                roundtrip(&c, 15, false, 0);
            }
        }
    }

    #[test]
    fn chroma_dc_blocks() {
        for &(a, b, c, d) in &[(1i16, 0, 0, 0), (1, -1, 1, -1), (3, 0, -2, 0), (0, 0, 0, 5)] {
            roundtrip(&[a, b, c, d], 3, true, -1);
        }
    }

    #[test]
    fn ac_blocks() {
        // 4x4 AC block: 15 coefficients (scan indices 0..=14).
        let mut c = [0i16; 16];
        c[0] = 3;
        c[1] = -1;
        c[2] = 1;
        c[5] = -4;
        roundtrip(&c, 14, false, 8);
    }

    #[test]
    fn random_blocks_all_nc() {
        let mut r = Lcg(0xC0FF_EE11);
        for _ in 0..20000 {
            // Sparse, small-ish coefficients spanning trailing-ones, runs, and
            // larger levels that exercise the suffix-length adaptation.
            let mut c = [0i16; 16];
            let density = (r.next() % 4) + 1;
            for x in c.iter_mut() {
                if r.next() % 5 < density {
                    let mag = (r.next() % 12) as i16 + 1;
                    *x = if r.next() & 1 == 0 { mag } else { -mag };
                }
            }
            // Occasionally inject one large coefficient.
            if r.next() & 7 == 0 {
                let p = (r.next() % 16) as usize;
                let mag = (r.next() % 600) as i16 + 1;
                c[p] = if r.next() & 1 == 0 { mag } else { -mag };
            }
            let nc = (r.next() % 17) as i32;
            roundtrip(&c, 15, false, nc);
        }
    }

    #[test]
    fn random_ac_and_chroma() {
        let mut r = Lcg(0x1234_ABCD);
        for _ in 0..10000 {
            let mut c = [0i16; 16];
            for x in c.iter_mut().take(15) {
                if r.next() % 3 == 0 {
                    let mag = (r.next() % 20) as i16 + 1;
                    *x = if r.next() & 1 == 0 { mag } else { -mag };
                }
            }
            roundtrip(&c, 14, false, (r.next() % 17) as i32);

            let mut cd = [0i16; 4];
            for x in cd.iter_mut() {
                if r.next() % 2 == 0 {
                    let mag = (r.next() % 10) as i16 + 1;
                    *x = if r.next() & 1 == 0 { mag } else { -mag };
                }
            }
            roundtrip(&cd, 3, true, -1);
        }
    }
}
