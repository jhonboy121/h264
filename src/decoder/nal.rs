//! NAL unit framing and RBSP extraction.
//!
//! Ports the Annex-B start-code scanning and emulation-prevention-byte removal
//! used by `reference/codec/decoder/core/src/au_parser.cpp`
//! (`WelsParseNalHeader`, `ParseNalHeader`, and the EBSP→RBSP step).

use alloc::vec::Vec;

/// NAL unit type (Table 7-1), values from `wels_common_defs.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalUnitType {
    Unspecified0,
    CodedSlice,
    CodedSliceDpa,
    CodedSliceDpb,
    CodedSliceDpc,
    CodedSliceIdr,
    Sei,
    Sps,
    Pps,
    AuDelimiter,
    EndOfSeq,
    EndOfStream,
    FillerData,
    SpsExt,
    Prefix,
    SubsetSps,
    /// Any reserved/unspecified type, carrying its raw value.
    Other(u8),
}

impl NalUnitType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => NalUnitType::Unspecified0,
            1 => NalUnitType::CodedSlice,
            2 => NalUnitType::CodedSliceDpa,
            3 => NalUnitType::CodedSliceDpb,
            4 => NalUnitType::CodedSliceDpc,
            5 => NalUnitType::CodedSliceIdr,
            6 => NalUnitType::Sei,
            7 => NalUnitType::Sps,
            8 => NalUnitType::Pps,
            9 => NalUnitType::AuDelimiter,
            10 => NalUnitType::EndOfSeq,
            11 => NalUnitType::EndOfStream,
            12 => NalUnitType::FillerData,
            13 => NalUnitType::SpsExt,
            14 => NalUnitType::Prefix,
            15 => NalUnitType::SubsetSps,
            other => NalUnitType::Other(other),
        }
    }

    /// True for VCL slice types (1..=5).
    pub fn is_vcl(self) -> bool {
        matches!(
            self,
            NalUnitType::CodedSlice
                | NalUnitType::CodedSliceDpa
                | NalUnitType::CodedSliceDpb
                | NalUnitType::CodedSliceDpc
                | NalUnitType::CodedSliceIdr
        )
    }

    pub fn is_idr(self) -> bool {
        self == NalUnitType::CodedSliceIdr
    }
}

/// Parsed NAL unit header byte plus its RBSP payload.
#[derive(Debug, Clone)]
pub struct NalUnit {
    /// `nal_ref_idc` (bits 5..6).
    pub ref_idc: u8,
    /// `nal_unit_type` (bits 0..4).
    pub unit_type: NalUnitType,
    /// RBSP payload (emulation-prevention bytes removed), excluding the header.
    pub rbsp: Vec<u8>,
}

/// Remove emulation-prevention three-bytes: any `00 00 03` becomes `00 00`
/// (the `03` is dropped) — but only when the byte after `03` is `<= 03`
/// (spec 7.4.1). OpenH264 drops the `03` whenever it follows `00 00`; we apply
/// the spec rule which matches conformant streams.
pub fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0usize;
    let mut i = 0;
    while i < ebsp.len() {
        let b = ebsp[i];
        if zeros >= 2 && b == 0x03 {
            // Drop this emulation-prevention byte if a valid escape (next byte
            // is 00..03) or it is the last byte.
            let next_ok = i + 1 >= ebsp.len() || ebsp[i + 1] <= 0x03;
            if next_ok {
                zeros = 0;
                i += 1;
                continue;
            }
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        i += 1;
    }
    out
}

/// Iterate Annex-B NAL units in `data`, yielding the raw EBSP bytes of each NAL
/// (header byte included, start code excluded). Start codes are `00 00 01` or
/// `00 00 00 01`. Mirrors the framing in `WelsParseNalHeader`'s caller.
pub fn annexb_nal_units(data: &[u8]) -> NalIterator<'_> {
    NalIterator { data, pos: 0 }
}

/// Iterator over Annex-B framed NAL payloads (EBSP, header byte first).
pub struct NalIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for NalIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let data = self.data;
        // Find next start code from pos.
        let start = find_start_code(data, self.pos)?;
        // Payload begins after the start code (3 bytes; the optional leading
        // zero is absorbed by find_start_code reporting the 00 00 01 position).
        let payload_start = start + 3;
        // Find the following start code to bound this NAL.
        let next = find_start_code(data, payload_start);
        let mut payload_end = match next {
            Some(n) => {
                // The start code may be preceded by an extra 00 (4-byte code);
                // trim trailing zero bytes that belong to the next start code.
                let mut e = n;
                while e > payload_start && data[e - 1] == 0 {
                    e -= 1;
                }
                e
            }
            None => data.len(),
        };
        // Trim trailing zero padding at end of stream.
        if next.is_none() {
            while payload_end > payload_start && data[payload_end - 1] == 0 {
                payload_end -= 1;
            }
        }
        self.pos = next.unwrap_or(data.len());
        if payload_start >= payload_end {
            // Empty NAL; skip to keep iterating.
            return self.next();
        }
        Some(&data[payload_start..payload_end])
    }
}

/// Return the index of the byte where a `00 00 01` start code begins, at or
/// after `from`. The returned index points at the first `00` of the `00 00 01`.
fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Parse one EBSP NAL payload (header byte first) into a [`NalUnit`].
pub fn parse_nal(ebsp: &[u8]) -> Option<NalUnit> {
    let &hdr = ebsp.first()?;
    // forbidden_zero_bit must be 0; tolerate but note.
    let ref_idc = (hdr >> 5) & 0x03;
    let unit_type = NalUnitType::from_u8(hdr & 0x1f);
    let rbsp = ebsp_to_rbsp(&ebsp[1..]);
    Some(NalUnit { ref_idc, unit_type, rbsp })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn ebsp_removes_emulation_bytes() {
        // 00 00 03 00  ->  00 00 00
        let ebsp = [0x00, 0x00, 0x03, 0x00, 0x01];
        assert_eq!(ebsp_to_rbsp(&ebsp), vec![0x00, 0x00, 0x00, 0x01]);
        // 00 00 03 03 -> 00 00 03 (escape of a literal 03)
        let ebsp2 = [0x00, 0x00, 0x03, 0x03];
        assert_eq!(ebsp_to_rbsp(&ebsp2), vec![0x00, 0x00, 0x03]);
        // No emulation: passthrough
        let ebsp3 = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(ebsp_to_rbsp(&ebsp3), vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn ebsp_keeps_non_escape_03() {
        // 00 00 03 followed by 0x55 (>03) is NOT an emulation byte -> keep.
        let ebsp = [0x00, 0x00, 0x03, 0x55];
        assert_eq!(ebsp_to_rbsp(&ebsp), vec![0x00, 0x00, 0x03, 0x55]);
    }

    #[test]
    fn annexb_splits_three_and_four_byte_start_codes() {
        // NAL1: type 7 (SPS), payload AA BB ; NAL2: 4-byte start code, type 8.
        let stream = [
            0x00, 0x00, 0x01, 0x67, 0xAA, 0xBB, // 00 00 01 | 67 AA BB
            0x00, 0x00, 0x00, 0x01, 0x68, 0xCC, // 00 00 00 01 | 68 CC
        ];
        let nals: Vec<&[u8]> = annexb_nal_units(&stream).collect();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0x67, 0xAA, 0xBB]);
        assert_eq!(nals[1], &[0x68, 0xCC]);

        let n0 = parse_nal(nals[0]).unwrap();
        assert_eq!(n0.unit_type, NalUnitType::Sps);
        assert_eq!(n0.ref_idc, 0b011); // 0x67 = 0110_0111 -> ref_idc=3, type=7
        assert_eq!(n0.rbsp, vec![0xAA, 0xBB]);

        let n1 = parse_nal(nals[1]).unwrap();
        assert_eq!(n1.unit_type, NalUnitType::Pps);
    }

    #[test]
    fn nal_header_decoding() {
        // 0x65 = 0110_0101 -> ref_idc=3, type=5 (IDR slice)
        let n = parse_nal(&[0x65, 0x88]).unwrap();
        assert_eq!(n.ref_idc, 3);
        assert_eq!(n.unit_type, NalUnitType::CodedSliceIdr);
        assert!(n.unit_type.is_vcl());
        assert!(n.unit_type.is_idr());
    }
}
