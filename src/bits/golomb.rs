//! Exp-Golomb decoding (`ue(v)`, `se(v)`, `te(v)`), ported from
//! `dec_golomb.h` (`BsGetUe`/`BsGetSe`/`BsGetTe0`).

use super::reader::BitReader;
use crate::error::DecodeError;

type Result<T> = core::result::Result<T, DecodeError>;

impl BitReader<'_> {
    /// Unsigned Exp-Golomb `ue(v)`.
    ///
    /// `codeNum = 2^leadingZeros - 1 + read_bits(leadingZeros)` (spec 9.1).
    #[inline]
    pub fn read_ue(&mut self) -> Result<u32> {
        let zeros = self.read_leading_zeros()?;
        if zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_bits(zeros)?;
        Ok((1u32 << zeros) - 1 + suffix)
    }

    /// Signed Exp-Golomb `se(v)` (spec 9.1.1).
    #[inline]
    pub fn read_se(&mut self) -> Result<i32> {
        let code = self.read_ue()?;
        // (-1)^(k+1) * ceil(k/2)
        Ok(if code & 1 != 0 {
            ((code + 1) >> 1) as i32
        } else {
            -((code >> 1) as i32)
        })
    }

    /// Truncated Exp-Golomb `te(v)` over the given inclusive range size
    /// (`BsGetTe0`). `range` is the count of possible values.
    #[inline]
    pub fn read_te(&mut self, range: u32) -> Result<u32> {
        match range {
            1 => Ok(0),
            2 => Ok(self.read_bit()? ^ 1),
            _ => self.read_ue(),
        }
    }

    /// `ue(v)` with an upper bound check; returns `InvalidSyntax(name)` if over.
    #[inline]
    pub fn read_ue_max(&mut self, max: u32, name: &'static str) -> Result<u32> {
        let v = self.read_ue()?;
        if v > max {
            return Err(DecodeError::InvalidSyntax(name));
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a reader from a bit string like "00100" (MSB-first), padded to bytes.
    fn reader_from_bits(bits: &str) -> alloc::vec::Vec<u8> {
        let mut bytes = alloc::vec::Vec::new();
        let mut cur = 0u8;
        let mut n = 0u32;
        for c in bits.chars() {
            cur = (cur << 1) | (if c == '1' { 1 } else { 0 });
            n += 1;
            if n == 8 {
                bytes.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            bytes.push(cur);
        }
        bytes
    }

    #[test]
    fn ue_examples() {
        // From H.264 Table 9-2: codeNum -> bit string
        let cases = [
            ("1", 0u32),
            ("010", 1),
            ("011", 2),
            ("00100", 3),
            ("00101", 4),
            ("00110", 5),
            ("00111", 6),
            ("0001000", 7),
            ("000010001", 16),
        ];
        for (bits, expect) in cases {
            let buf = reader_from_bits(bits);
            let mut r = BitReader::new(&buf);
            assert_eq!(r.read_ue().unwrap(), expect, "bits={bits}");
        }
    }

    #[test]
    fn se_examples() {
        // codeNum 0->0, 1->+1, 2->-1, 3->+2, 4->-2 ...
        let cases = [
            ("1", 0i32),
            ("010", 1),
            ("011", -1),
            ("00100", 2),
            ("00101", -2),
            ("00110", 3),
        ];
        for (bits, expect) in cases {
            let buf = reader_from_bits(bits);
            let mut r = BitReader::new(&buf);
            assert_eq!(r.read_se().unwrap(), expect, "bits={bits}");
        }
    }

    #[test]
    fn te_range_one_and_two() {
        let buf = reader_from_bits("0");
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_te(1).unwrap(), 0); // consumes nothing
        assert_eq!(r.read_te(2).unwrap(), 1); // reads '0' -> 0^1 = 1
    }
}
