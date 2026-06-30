//! MSB-first bitstream writer with Exp-Golomb encoding (encoder side).
//!
//! Mirrors the bit-packing in `set_mb_syn_cavlc.cpp` / `nal_encap.cpp`. Writes
//! into a caller-owned `Vec<u8>` (no global state).

use alloc::vec::Vec;

/// Accumulates bits MSB-first into a byte buffer.
pub struct BitWriter {
    out: Vec<u8>,
    /// Partial byte being filled (high bits first).
    cur: u8,
    /// Number of bits already placed in `cur` (0..=7).
    nbits: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            cur: 0,
            nbits: 0,
        }
    }

    /// Bits written so far.
    #[inline]
    pub fn bit_len(&self) -> usize {
        self.out.len() * 8 + self.nbits as usize
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: u32) {
        self.cur = (self.cur << 1) | (bit as u8 & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.out.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Write the low `n` bits of `value`, MSB-first. `n <= 32`.
    #[inline]
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        let mut i = n;
        while i > 0 {
            i -= 1;
            self.write_bit((value >> i) & 1);
        }
    }

    /// Write `flag` as one bit.
    #[inline]
    pub fn write_flag(&mut self, flag: bool) {
        self.write_bit(flag as u32);
    }

    /// Unsigned Exp-Golomb `ue(v)`.
    pub fn write_ue(&mut self, value: u32) {
        // codeNum = value; emit (leadingZeros) zeros, a 1, then the suffix.
        let code = value as u64 + 1;
        let n = 64 - code.leading_zeros(); // bit length of code
        let zeros = n - 1;
        // zeros leading zero bits
        for _ in 0..zeros {
            self.write_bit(0);
        }
        // then the n-bit value `code`
        self.write_bits_u64(code, n);
    }

    /// Signed Exp-Golomb `se(v)`.
    pub fn write_se(&mut self, value: i32) {
        let code = if value <= 0 {
            (-(value as i64) as u64) * 2
        } else {
            (value as u64) * 2 - 1
        };
        self.write_ue(code as u32);
    }

    fn write_bits_u64(&mut self, value: u64, n: u32) {
        let mut i = n;
        while i > 0 {
            i -= 1;
            self.write_bit(((value >> i) & 1) as u32);
        }
    }

    /// True if on a byte boundary.
    #[inline]
    pub fn byte_aligned(&self) -> bool {
        self.nbits == 0
    }

    /// Append `rbsp_trailing_bits`: a `1` then zero-pad to a byte boundary.
    pub fn write_trailing_bits(&mut self) {
        self.write_bit(1);
        while self.nbits != 0 {
            self.write_bit(0);
        }
    }

    /// Zero-pad to a byte boundary (without a stop bit).
    pub fn align_to_byte(&mut self) {
        while self.nbits != 0 {
            self.write_bit(0);
        }
    }

    /// Finish, flushing any partial byte (zero-padded), and return the buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits != 0 {
            self.cur <<= 8 - self.nbits;
            self.out.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
        self.out
    }

    /// Borrow the completed bytes (must be byte-aligned).
    pub fn as_bytes(&self) -> &[u8] {
        debug_assert!(self.byte_aligned());
        &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;

    #[test]
    fn write_then_read_bits() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010, 4);
        w.write_bits(0b110, 3);
        w.write_bit(1);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(3).unwrap(), 0b110);
        assert_eq!(r.read_bit().unwrap(), 1);
    }

    #[test]
    fn ue_roundtrip() {
        for v in [0u32, 1, 2, 3, 7, 16, 255, 1000, 65535] {
            let mut w = BitWriter::new();
            w.write_ue(v);
            w.write_trailing_bits();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_ue().unwrap(), v, "v={v}");
        }
    }

    #[test]
    fn se_roundtrip() {
        for v in [0i32, 1, -1, 2, -2, 50, -50, 1000, -1000] {
            let mut w = BitWriter::new();
            w.write_se(v);
            w.write_trailing_bits();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_se().unwrap(), v, "v={v}");
        }
    }
}
