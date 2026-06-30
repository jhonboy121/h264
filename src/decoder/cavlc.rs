//! CAVLC residual-block decoding, ported from Cisco OpenH264
//! `reference/codec/decoder/core/src/parse_mb_syn_cavlc.cpp`
//! (`WelsResidualBlockCavlc` and its `CavlcGet*` helpers).
//!
//! The reference implementation drives a hand-rolled 32-bit MSB-aligned read
//! cache (`SReadBitsCache` + `POP_BUFFER`/`SHIFT_BUFFER`). This port expresses
//! the identical peek/consume sequence against [`BitReader`], which is also
//! MSB-first, so the bit consumption is bit-exact with the C.
//!
//! Conventions, matching the C:
//! * `n_c` is the predicted non-zero count (`nC`). `n_c < 0` selects the
//!   chroma-DC path (spec `nC == -1`), which uses the fixed 8-bit chroma
//!   coeff_token table and the chroma total_zeros tables.
//! * Decoded `level`/`run` are reassembled into `out_level` in coefficient
//!   scan order (the reference then remaps through a zig-zag table and applies
//!   dequantisation; that final mapping/scaling is left to the caller — it is
//!   wired up in P3 reconstruction). `out_level` must be at least
//!   `max_num_coeff` long and zero-initialised by the caller.

use crate::bits::BitReader;
use crate::error::DecodeError;

use super::cavlc_tables as t;

type Result<T> = core::result::Result<T, DecodeError>;

/// `MAX_LEVEL_PREFIX` from `parse_mb_syn_cavlc.cpp`.
const MAX_LEVEL_PREFIX: u32 = 15;

/// Peek `n` bits (MSB-first, zero-padded past EOF, matching the C read cache)
/// then consume them.
#[inline]
fn read(bs: &mut BitReader, n: u32) -> Result<u32> {
    let v = bs.peek_bits(n);
    bs.skip_bits(n as usize)?;
    Ok(v)
}

/// Number of leading zero bits before (and including) the next `1`, i.e. the
/// unary prefix length. Mirrors `WELS_GET_PREFIX_BITS` + `POP_BUFFER`.
#[inline]
fn read_prefix_len(bs: &mut BitReader) -> Result<u32> {
    Ok(bs.read_leading_zeros()? + 1)
}

/// Main coeff_token VLC table for `nc_map_idx` in 0..=3
/// (`kpCoeffTokenVlcTable[0][nc_map_idx]`).
#[inline]
fn coeff_token_main(nc_map_idx: usize) -> &'static [[u8; 2]] {
    match nc_map_idx {
        0 => &t::VLC_TABLE_0,
        1 => &t::VLC_TABLE_1,
        2 => &t::VLC_TABLE_2,
        _ => &t::VLC_TABLE_3,
    }
}

/// Secondary coeff_token VLC table selected by the leading 8-bit `value`
/// (`kpCoeffTokenVlcTable[nc_map_idx + 1][value]`).
#[inline]
fn coeff_token_sub(nc_map_idx: usize, value: usize) -> &'static [[u8; 2]] {
    match nc_map_idx {
        0 => match value {
            0 => &t::VLC_TABLE_0_0,
            1 => &t::VLC_TABLE_0_1,
            2 => &t::VLC_TABLE_0_2,
            _ => &t::VLC_TABLE_0_3,
        },
        1 => match value {
            0 => &t::VLC_TABLE_1_0,
            1 => &t::VLC_TABLE_1_1,
            2 => &t::VLC_TABLE_1_2,
            _ => &t::VLC_TABLE_1_3,
        },
        _ => match value {
            0 => &t::VLC_TABLE_2_0,
            1 => &t::VLC_TABLE_2_1,
            2 => &t::VLC_TABLE_2_2,
            3 => &t::VLC_TABLE_2_3,
            4 => &t::VLC_TABLE_2_4,
            5 => &t::VLC_TABLE_2_5,
            6 => &t::VLC_TABLE_2_6,
            _ => &t::VLC_TABLE_2_7,
        },
    }
}

/// `g_kuiVlcTableMoreBitsCount{0,1,2}` selector.
#[inline]
fn more_bits_count(nc_map_idx: usize) -> &'static [u8] {
    match nc_map_idx {
        0 => &t::VLC_TABLE_MORE_BITS_COUNT0,
        1 => &t::VLC_TABLE_MORE_BITS_COUNT1,
        _ => &t::VLC_TABLE_MORE_BITS_COUNT2,
    }
}

/// Decode `coeff_token`, returning `(total_coeff, trailing_ones)`.
///
/// Port of `CavlcGetTrailingOnesAndTotalCoeff`. `n_c < 0` selects the
/// chroma-DC fixed table (spec `nC == -1`).
pub fn read_coeff_token(bs: &mut BitReader, n_c: i32) -> Result<(u8, u8)> {
    let index_vlc: usize;
    if n_c < 0 {
        // chroma DC: 8-bit lookup into the chroma coeff_token table.
        let value = bs.peek_bits(8) as usize;
        index_vlc = t::VLC_CHROMA_TABLE[value][0] as usize;
        let count = t::VLC_CHROMA_TABLE[value][1] as usize;
        bs.skip_bits(count)?;
    } else {
        let nc_map_idx = t::NC_MAP_TABLE[n_c as usize] as usize;
        if nc_map_idx <= 2 {
            let value = bs.peek_bits(8) as usize;
            if (value as u8) < t::VLC_TABLE_NEED_MORE_BITS_THREAD[nc_map_idx] {
                bs.skip_bits(8)?;
                let more = more_bits_count(nc_map_idx)[value] as u32;
                let index_value = bs.peek_bits(more) as usize;
                let sub = coeff_token_sub(nc_map_idx, value);
                index_vlc = sub[index_value][0] as usize;
                let count = sub[index_value][1] as usize;
                bs.skip_bits(count)?;
            } else {
                let main = coeff_token_main(nc_map_idx);
                index_vlc = main[value][0] as usize;
                let count = main[value][1] as usize;
                bs.skip_bits(count)?;
            }
        } else {
            // nC >= 8: 6-bit FLC into g_kuiVlcTable_3.
            let value = bs.peek_bits(6) as usize;
            bs.skip_bits(6)?;
            index_vlc = t::VLC_TABLE_3[value][0] as usize;
        }
    }
    let trailing_ones = t::VLC_TRAILING_ONE_TOTAL_COEFF[index_vlc][0];
    let total_coeff = t::VLC_TRAILING_ONE_TOTAL_COEFF[index_vlc][1];
    Ok((total_coeff, trailing_ones))
}

/// Decode the `total_coeff` levels into `level[0..total_coeff]`. Port of
/// `CavlcGetLevelVal` including the suffix-length adaptation.
fn get_level_val(
    bs: &mut BitReader,
    total_coeff: u8,
    trailing_ones: u8,
    level: &mut [i32; 16],
) -> Result<()> {
    let total_coeff = total_coeff as usize;
    let trailing_ones = trailing_ones as usize;

    // Trailing-one signs: one bit each, 0 -> +1, 1 -> -1.
    for l in level.iter_mut().take(trailing_ones) {
        let sign = bs.read_bit()?;
        *l = 1 - ((sign as i32) << 1);
    }

    let mut suffix_length: i32 = (total_coeff > 10 && trailing_ones < 3) as i32;

    for i in trailing_ones..total_coeff {
        let prefix_bits = read_prefix_len(bs)?;
        if prefix_bits > MAX_LEVEL_PREFIX + 1 {
            return Err(DecodeError::InvalidSyntax("cavlc level_prefix"));
        }
        let level_prefix = (prefix_bits - 1) as i32;

        let mut level_code = level_prefix << suffix_length;
        let mut suffix_size = suffix_length;

        if level_prefix >= 14 {
            if level_prefix == 14 && suffix_length == 0 {
                suffix_size = 4;
            } else if level_prefix == 15 {
                suffix_size = 12;
                if suffix_length == 0 {
                    level_code += 15;
                }
            }
        }

        if suffix_size > 0 {
            level_code += read(bs, suffix_size as u32)? as i32;
        }

        level_code += (((i == trailing_ones) && (trailing_ones < 3)) as i32) << 1;

        let mut val = (level_code + 2) >> 1;
        if level_code & 1 != 0 {
            val = -val;
        }
        level[i] = val;

        suffix_length += (suffix_length == 0) as i32;
        let threshold = 3 << (suffix_length - 1);
        suffix_length += ((val > threshold || val < -threshold) && suffix_length < 6) as i32;
    }

    Ok(())
}

/// Luma/chroma-AC total_zeros table for `total_coeff` (`tzVlcIndex`) in 1..=15.
#[inline]
fn total_zeros_table(idx: usize) -> &'static [[u8; 2]] {
    match idx {
        0 => &t::TOTAL_ZEROS_TABLE0,
        1 => &t::TOTAL_ZEROS_TABLE1,
        2 => &t::TOTAL_ZEROS_TABLE2,
        3 => &t::TOTAL_ZEROS_TABLE3,
        4 => &t::TOTAL_ZEROS_TABLE4,
        5 => &t::TOTAL_ZEROS_TABLE5,
        6 => &t::TOTAL_ZEROS_TABLE6,
        7 => &t::TOTAL_ZEROS_TABLE7,
        8 => &t::TOTAL_ZEROS_TABLE8,
        9 => &t::TOTAL_ZEROS_TABLE9,
        10 => &t::TOTAL_ZEROS_TABLE10,
        11 => &t::TOTAL_ZEROS_TABLE11,
        12 => &t::TOTAL_ZEROS_TABLE12,
        13 => &t::TOTAL_ZEROS_TABLE13,
        _ => &t::TOTAL_ZEROS_TABLE14,
    }
}

/// Chroma-DC total_zeros table for `total_coeff` in 1..=3.
#[inline]
fn total_zeros_chroma_table(idx: usize) -> &'static [[u8; 2]] {
    match idx {
        0 => &t::TOTAL_ZEROS_CHROMA_TABLE0,
        1 => &t::TOTAL_ZEROS_CHROMA_TABLE1,
        _ => &t::TOTAL_ZEROS_CHROMA_TABLE2,
    }
}

/// Decode `total_zeros`. Port of `CavlcGetTotalZeros`.
fn get_total_zeros(bs: &mut BitReader, total_coeff: u8, chroma_dc: bool) -> Result<i32> {
    let vlc_idx = total_coeff as usize; // 1..=15 (or 1..=3 for chroma DC)
    let (bit_num, table) = if chroma_dc {
        (
            t::TOTAL_ZEROS_BIT_NUM_CHROMA_MAP[vlc_idx - 1] as u32,
            total_zeros_chroma_table(vlc_idx - 1),
        )
    } else {
        (
            t::TOTAL_ZEROS_BIT_NUM_MAP[vlc_idx - 1] as u32,
            total_zeros_table(vlc_idx - 1),
        )
    };
    let value = bs.peek_bits(bit_num) as usize;
    let count = table[value][1] as usize;
    bs.skip_bits(count)?;
    Ok(table[value][0] as i32)
}

/// `g_kuiZeroLeftTable{0..6}` selector for `zeros_left` in 1..=7.
#[inline]
fn zero_left_table(idx: usize) -> &'static [[u8; 2]] {
    match idx {
        0 => &t::ZERO_LEFT_TABLE0,
        1 => &t::ZERO_LEFT_TABLE1,
        2 => &t::ZERO_LEFT_TABLE2,
        3 => &t::ZERO_LEFT_TABLE3,
        4 => &t::ZERO_LEFT_TABLE4,
        5 => &t::ZERO_LEFT_TABLE5,
        _ => &t::ZERO_LEFT_TABLE6,
    }
}

/// Decode the `run_before` values into `run[0..total_coeff]`. Port of
/// `CavlcGetRunBefore`.
fn get_run_before(
    bs: &mut BitReader,
    total_coeff: u8,
    mut zeros_left: i32,
    run: &mut [i32; 16],
) -> Result<()> {
    let total_coeff = total_coeff as usize;

    for i in 0..total_coeff.saturating_sub(1) {
        if zeros_left > 0 {
            let count = t::ZERO_LEFT_BIT_NUM_MAP[zeros_left as usize] as u32;
            let value = bs.peek_bits(count) as usize;
            if zeros_left < 7 {
                let table = zero_left_table((zeros_left - 1) as usize);
                let consumed = table[value][1] as usize;
                bs.skip_bits(consumed)?;
                run[i] = table[value][0] as i32;
            } else {
                bs.skip_bits(count as usize)?;
                let table = &t::ZERO_LEFT_TABLE6;
                if (table[value][0] as i32) < 7 {
                    run[i] = table[value][0] as i32;
                } else {
                    let prefix_bits = read_prefix_len(bs)?;
                    run[i] = prefix_bits as i32 + 6;
                    if run[i] > zeros_left {
                        return Err(DecodeError::InvalidSyntax("cavlc run_before"));
                    }
                }
            }
        } else {
            for r in run.iter_mut().take(total_coeff).skip(i) {
                *r = 0;
            }
            return Ok(());
        }
        zeros_left -= run[i];
    }

    run[total_coeff - 1] = zeros_left;
    Ok(())
}

/// Decode one residual block in CAVLC. Returns `total_coeff`.
///
/// Port of `WelsResidualBlockCavlc` (the syntax-decode core, without the
/// in-loop dequant/IDCT which P3 layers on). `out_level` receives the decoded
/// levels at their scan-order positions; it must be at least `max_num_coeff`
/// entries and zero-initialised by the caller.
pub fn residual_block_cavlc(
    bs: &mut BitReader,
    n_c: i32,
    max_num_coeff: usize,
    out_level: &mut [i32],
) -> Result<i32> {
    debug_assert!(out_level.len() >= max_num_coeff);
    let chroma_dc = n_c < 0;

    let (total_coeff, trailing_ones) = read_coeff_token(bs, n_c)?;
    if total_coeff == 0 {
        return Ok(0);
    }
    if trailing_ones > 3 || total_coeff > 16 {
        return Err(DecodeError::InvalidSyntax("cavlc total_coeff/trailing_ones"));
    }

    let mut level = [0i32; 16];
    get_level_val(bs, total_coeff, trailing_ones, &mut level)?;

    let zeros_left = if (total_coeff as usize) < max_num_coeff {
        get_total_zeros(bs, total_coeff, chroma_dc)?
    } else {
        0
    };
    if zeros_left < 0 || (zeros_left as usize + total_coeff as usize) > max_num_coeff {
        return Err(DecodeError::InvalidSyntax("cavlc zeros_left"));
    }

    let mut run = [0i32; 16];
    get_run_before(bs, total_coeff, zeros_left, &mut run)?;

    // Reassemble in scan order, exactly as the C `iCoeffNum` walk.
    let mut coeff_num: i32 = -1;
    for i in (0..total_coeff as usize).rev() {
        coeff_num += run[i] + 1;
        out_level[coeff_num as usize] = level[i];
    }

    Ok(total_coeff as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Pack a list of `(value, num_bits)` codewords MSB-first into bytes, then
    /// append `pad_bytes` zero bytes so forward peeks past the payload read
    /// zero padding (mirroring the C read cache).
    fn pack(words: &[(u32, u32)], pad_bytes: usize) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for &(value, n) in words {
            for i in (0..n).rev() {
                bits.push(((value >> i) & 1) as u8);
            }
        }
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut out: Vec<u8> = Vec::new();
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            out.push(b);
        }
        out.extend(core::iter::repeat(0u8).take(pad_bytes));
        out
    }

    fn token(data: &[u8], n_c: i32) -> (u8, u8) {
        let mut bs = BitReader::new(data);
        read_coeff_token(&mut bs, n_c).unwrap()
    }

    // Cross-check coeff_token against spec Table 9-5 for several (nC, codeword)
    // pairs. Each tuple is (n_c, codeword_value, code_len, expect_total,
    // expect_trailing_ones).
    #[test]
    fn coeff_token_spec_table_9_5() {
        let cases: &[(i32, u32, u32, u8, u8)] = &[
            // 0 <= nC < 2
            (0, 0b1, 1, 0, 0),
            (0, 0b01, 2, 1, 1),
            (0, 0b001, 3, 2, 2),
            (0, 0b00011, 5, 3, 3),
            (0, 0b000101, 6, 1, 0),
            (0, 0b000100, 6, 2, 1),
            // 2 <= nC < 4
            (2, 0b11, 2, 0, 0),
            (2, 0b10, 2, 1, 1),
            // 4 <= nC < 8
            (4, 0b1111, 4, 0, 0),
            // nC >= 8: fixed 6-bit codes via g_kuiVlcTable_3
            (8, 0b000011, 6, 0, 0),
            (8, 0b000001, 6, 1, 1),
        ];
        for &(n_c, code, len, et, eo) in cases {
            let data = pack(&[(code, len)], 2);
            let (total, ones) = token(&data, n_c);
            assert_eq!(
                (total, ones),
                (et, eo),
                "n_c={n_c} code={code:b} len={len}"
            );
        }
    }

    // coeff_token == 0 -> empty block, nothing else read.
    #[test]
    fn all_zero_block() {
        // nC=0, codeword "1".
        let data = pack(&[(0b1, 1)], 2);
        let mut bs = BitReader::new(&data);
        let mut out = [0i32; 16];
        let total = residual_block_cavlc(&mut bs, 0, 16, &mut out).unwrap();
        assert_eq!(total, 0);
        assert_eq!(out, [0i32; 16]);
        assert_eq!(bs.bit_pos(), 1, "only the 1-bit coeff_token is consumed");
    }

    // Single trailing-one, total_zeros = 0: one coefficient = +1 at position 0.
    #[test]
    fn single_trailing_one() {
        // coeff_token "01" (total=1,t1=1) + sign "0" (+1) + total_zeros "1" (0).
        let data = pack(&[(0b01, 2), (0b0, 1), (0b1, 1)], 2);
        let mut bs = BitReader::new(&data);
        let mut out = [0i32; 16];
        let total = residual_block_cavlc(&mut bs, 0, 16, &mut out).unwrap();
        assert_eq!(total, 1);
        let mut expect = [0i32; 16];
        expect[0] = 1;
        assert_eq!(out, expect);
        assert_eq!(bs.bit_pos(), 4);
    }

    // Three coefficients with run_before, hand-traced against the spec tables.
    // Levels (scan order, reversed walk): level=[-1,2,-2], runs=[1,1,0],
    // total_zeros=2 -> out_level[0]=-2, out_level[2]=2, out_level[4]=-1.
    #[test]
    fn three_levels_with_runs() {
        let data = [0x06, 0xDE, 0x40];
        let mut bs = BitReader::new(&data);
        let mut out = [0i32; 16];
        let total = residual_block_cavlc(&mut bs, 0, 16, &mut out).unwrap();
        assert_eq!(total, 3);
        let mut expect = [0i32; 16];
        expect[0] = -2;
        expect[2] = 2;
        expect[4] = -1;
        assert_eq!(out, expect);
        // 8 (token) + 1 (sign) + 1 + 3 (levels) + 3 (tz) + 3 (runs) = 19 bits.
        assert_eq!(bs.bit_pos(), 19);
    }

    // Chroma-DC (n_c < 0) uses the fixed 8-bit chroma coeff_token table and the
    // chroma total_zeros tables, with max_num_coeff = 4.
    #[test]
    fn chroma_dc_flc() {
        // coeff_token "1" (total=1,t1=1) + sign "0" (+1) + total_zeros "1" (0).
        let data = pack(&[(0b1, 1), (0b0, 1), (0b1, 1)], 2);
        let mut bs = BitReader::new(&data);
        let mut out = [0i32; 4];
        let total = residual_block_cavlc(&mut bs, -1, 4, &mut out).unwrap();
        assert_eq!(total, 1);
        assert_eq!(out, [1, 0, 0, 0]);
        assert_eq!(bs.bit_pos(), 3);
    }

    // Chroma-DC coeff_token alone for (total=0): codeword "01" per Table 9-5.
    #[test]
    fn chroma_dc_coeff_token_zero() {
        let data = pack(&[(0b01, 2)], 2);
        assert_eq!(token(&data, -1), (0, 0));
    }
}
