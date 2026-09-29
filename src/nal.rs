//! NAL unit type helpers (spec Table 7-1) for code that remuxes Annex-B.

/// Coded slice of an IDR picture.
pub const IDR: u8 = 5;
/// Sequence parameter set.
pub const SPS: u8 = 7;
/// Picture parameter set.
pub const PPS: u8 = 8;
/// Access unit delimiter.
pub const AUD: u8 = 9;

/// `nal_unit_type` of a NAL (header byte first, no start code).
pub const fn nal_type(nal: &[u8]) -> Option<u8> {
    const TYPE_MASK: u8 = 0x1f;
    match nal {
        [header, ..] => Some(*header & TYPE_MASK),
        [] => None,
    }
}
