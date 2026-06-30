//! H.264 decoder: NAL framing, parameter sets, slice parsing, reconstruction.
//!
//! See `docs/REFERENCE_MAP.md` for the C→Rust mapping. Built up across phases
//! P2 (parsing) and P3 (reconstruction).

pub mod nal;
pub mod params;

// Added per phase:
// pub mod slice_header;  // slice header parse
// pub mod cavlc;         // CAVLC MB syntax
// pub mod cabac;         // CABAC engine + MB syntax
// pub mod mv_pred;
// pub mod recon;
// pub mod decode_slice;
// pub mod dpb;
// pub mod ref_pic;
// pub mod fmo;
