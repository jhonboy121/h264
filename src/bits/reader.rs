//! MSB-first bitstream reader over an RBSP byte slice.

use crate::error::DecodeError;

type Result<T> = core::result::Result<T, DecodeError>;

/// Reads bits big-endian / MSB-first from a byte slice, the order used by H.264
/// RBSP syntax. Tracks an absolute bit position so `byte_aligned`,
/// `more_rbsp_data`, and trailing-bit handling are exact.
#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit offset of the next bit to read.
    pos: usize,
    /// Total readable bits (`data.len() * 8`).
    total: usize,
    /// Bit index of the final `1` bit (the rbsp_stop_one_bit), if any. Cached
    /// at construction so `more_rbsp_data` is O(1).
    stop_bit: Option<usize>,
}

impl<'a> BitReader<'a> {
    /// Create a reader over RBSP bytes (emulation-prevention already removed).
    pub fn new(data: &'a [u8]) -> Self {
        let total = data.len() * 8;
        // Find the last set bit (the rbsp_stop_one_bit) for more_rbsp_data().
        let mut stop_bit = None;
        'outer: for (i, &b) in data.iter().enumerate().rev() {
            if b != 0 {
                for bit in 0..8 {
                    // bit 0 is the MSB at this byte
                    if b & (1 << bit) != 0 {
                        stop_bit = Some(i * 8 + (7 - bit));
                        break 'outer;
                    }
                }
            }
        }
        Self { data, pos: 0, total, stop_bit }
    }

    /// Bits consumed so far.
    #[inline]
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Bits still available.
    #[inline]
    pub fn bits_left(&self) -> usize {
        self.total.saturating_sub(self.pos)
    }

    /// True if the next bit is on a byte boundary.
    #[inline]
    pub fn byte_aligned(&self) -> bool {
        self.pos.is_multiple_of(8)
    }

    /// Read a single bit without bounds-checking helpers (internal).
    #[inline]
    fn read_bit_raw(&mut self) -> Result<u32> {
        if self.pos >= self.total {
            return Err(DecodeError::UnexpectedEof);
        }
        let byte = self.data[self.pos >> 3];
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(bit as u32)
    }

    /// Read one bit (`u(1)`), returning 0/1.
    #[inline]
    pub fn read_bit(&mut self) -> Result<u32> {
        self.read_bit_raw()
    }

    /// Read one bit as a flag.
    #[inline]
    pub fn read_flag(&mut self) -> Result<bool> {
        Ok(self.read_bit_raw()? != 0)
    }

    /// Read `n` bits (`u(n)`), MSB first. `n` must be `<= 32`.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.pos + n as usize > self.total {
            return Err(DecodeError::UnexpectedEof);
        }
        let mut v: u32 = 0;
        let mut remaining = n as usize;
        while remaining > 0 {
            let byte = self.data[self.pos >> 3];
            let bit_in_byte = self.pos & 7;
            let avail = 8 - bit_in_byte;
            let take = avail.min(remaining);
            // Extract `take` bits starting at bit_in_byte (MSB-first).
            let shift = avail - take;
            let mask = ((1u32 << take) - 1) as u8;
            let chunk = ((byte >> shift) & mask) as u32;
            v = (v << take) | chunk;
            self.pos += take;
            remaining -= take;
        }
        Ok(v)
    }

    /// Read up to 64 bits.
    pub fn read_bits64(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        if n <= 32 {
            return Ok(self.read_bits(n)? as u64);
        }
        let hi = self.read_bits(n - 32)? as u64;
        let lo = self.read_bits(32)? as u64;
        Ok((hi << 32) | lo)
    }

    /// Peek `n` bits without consuming. Returns 0-padded value if near EOF.
    #[inline]
    pub fn peek_bits(&self, n: u32) -> u32 {
        let mut clone = self.clone();
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | clone.read_bit_raw().unwrap_or(0);
        }
        v
    }

    /// Skip `n` bits.
    #[inline]
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.total {
            self.pos = self.total;
            return Err(DecodeError::UnexpectedEof);
        }
        self.pos += n;
        Ok(())
    }

    /// Number of leading zero bits before the next `1` (consumes through the 1).
    /// Used by Exp-Golomb. Returns the count; positions past the `1`.
    #[inline]
    pub(crate) fn read_leading_zeros(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        loop {
            let b = self.read_bit_raw()?;
            if b == 1 {
                return Ok(zeros);
            }
            zeros += 1;
            if zeros > 31 {
                // Codes longer than 31 leading zeros are bitstream errors.
                return Err(DecodeError::InvalidSyntax("exp-golomb leading zeros overflow"));
            }
        }
    }

    /// True if there is RBSP payload remaining before the stop bit (7.2 spec
    /// `more_rbsp_data`).
    #[inline]
    pub fn more_rbsp_data(&self) -> bool {
        match self.stop_bit {
            Some(p) => self.pos < p,
            None => false,
        }
    }

    /// Consume `rbsp_trailing_bits` (a `1` then zero-padding to a byte boundary).
    pub fn read_trailing_bits(&mut self) -> Result<()> {
        let stop = self.read_bit_raw()?;
        if stop != 1 {
            return Err(DecodeError::InvalidSyntax("rbsp_stop_one_bit"));
        }
        while !self.byte_aligned() {
            if self.read_bit_raw()? != 0 {
                return Err(DecodeError::InvalidSyntax("rbsp_alignment_zero_bit"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bits_msb_first() {
        // 1010_0110  1100_0001
        let data = [0b1010_0110u8, 0b1100_0001];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(4).unwrap(), 0b0110);
        assert_eq!(r.read_bits(8).unwrap(), 0b1100_0001);
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn read_bits_crossing_bytes() {
        let data = [0b1111_0000u8, 0b1010_1010];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(6).unwrap(), 0b111100);
        assert_eq!(r.read_bits(6).unwrap(), 0b001010);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
    }

    #[test]
    fn single_bits() {
        let data = [0b1001_0110u8];
        let mut r = BitReader::new(&data);
        for expect in [1, 0, 0, 1, 0, 1, 1, 0] {
            assert_eq!(r.read_bit().unwrap(), expect);
        }
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn peek_does_not_consume() {
        let data = [0b1011_0000u8];
        let r_pos = {
            let mut r = BitReader::new(&data);
            assert_eq!(r.peek_bits(3), 0b101);
            assert_eq!(r.read_bits(3).unwrap(), 0b101);
            r.bit_pos()
        };
        assert_eq!(r_pos, 3);
    }

    #[test]
    fn byte_alignment_and_trailing() {
        // 1 bit of data 'b', then stop bit 1, then pad zeros: b=1, stop, 000000
        // byte = 1_1_000000
        let data = [0b1100_0000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert!(!r.byte_aligned());
        r.read_trailing_bits().unwrap();
        assert!(r.byte_aligned());
    }

    #[test]
    fn more_rbsp_data_detects_stop_bit() {
        // data bits: 101, stop bit 1, pad: 1011_0000 -> stop at index 3
        let data = [0b1011_0000u8];
        let r = BitReader::new(&data);
        // stop bit is the last set bit = index 3
        assert!(r.more_rbsp_data()); // pos 0 < 3
        let mut r2 = r.clone();
        r2.read_bits(3).unwrap();
        assert!(!r2.more_rbsp_data()); // pos 3 == stop
    }
}
