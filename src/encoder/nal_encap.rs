//! Annex-B NAL encapsulation, ported from
//! `reference/codec/encoder/core/src/nal_encap.cpp` (`WelsEncodeNal`).
//!
//! Wraps an RBSP payload with a 4-byte start code, a NAL unit header byte, and
//! emulation-prevention (`0x03`) byte insertion (EBSP). The result is the exact
//! inverse of the decoder's NAL unescaping in [`crate::decoder::nal`].

use alloc::vec::Vec;

/// Append one Annex-B NAL unit (start code + header + escaped RBSP) to `out`.
///
/// `nal_ref_idc` is 0..=3, `nal_unit_type` 0..=31. `rbsp` is the raw byte
/// payload (already byte-aligned with rbsp_trailing_bits applied).
pub fn append_annexb_nal(out: &mut Vec<u8>, nal_ref_idc: u8, nal_unit_type: u8, rbsp: &[u8]) {
    // Start code prefix 0x00000001.
    out.extend_from_slice(&[0, 0, 0, 1]);
    // NAL unit header.
    out.push(((nal_ref_idc & 0x3) << 5) | (nal_unit_type & 0x1f));
    // RBSP with emulation-prevention bytes.
    let mut zero_count = 0;
    for &b in rbsp {
        if zero_count == 2 && b <= 3 {
            out.push(3);
            zero_count = 0;
        }
        if b == 0 {
            zero_count += 1;
        } else {
            zero_count = 0;
        }
        out.push(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::nal::{annexb_nal_units, parse_nal};

    #[test]
    fn emulation_prevention_inserted_and_stripped() {
        // Payload containing 0x000001 / 0x000002 / 0x000000 patterns that must
        // be escaped, then recovered exactly by the decoder's unescaper.
        let payload = [0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0xff];
        let mut out = Vec::new();
        append_annexb_nal(&mut out, 3, 7, &payload);

        // Decode back: the decoder iterates NAL units and returns unescaped RBSP.
        let mut found = false;
        for ebsp in annexb_nal_units(&out) {
            let nal = parse_nal(ebsp).expect("parse_nal");
            assert_eq!(nal.ref_idc, 3);
            assert_eq!(nal.rbsp, payload, "RBSP did not round-trip through EBSP");
            found = true;
        }
        assert!(found, "no NAL unit parsed back");
    }
}
