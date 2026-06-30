//! H.264 decoder: NAL framing, parameter sets, slice parsing, reconstruction.
//!
//! See `docs/REFERENCE_MAP.md` for the C→Rust mapping. Built up across phases
//! P2 (parsing) and P3 (reconstruction).

pub mod cabac;
pub mod cabac_mb;
pub mod cabac_tables;
pub mod cavlc;
pub mod cavlc_tables;
pub mod context;
pub mod deblock;
pub mod dpb;
pub mod frame;
pub mod mb_parse_cabac;
pub mod mb_parse_cavlc;
pub mod mv_pred;
pub mod nal;
pub mod params;
pub mod picture;
pub mod recon_inter;
pub mod recon_intra;
pub mod slice_header;

pub use frame::{decode_intra_frame, decode_stream};

// Added per phase:
// pub mod mv_pred;
// pub mod recon;
// pub mod decode_slice;
// pub mod dpb;
// pub mod ref_pic;
// pub mod fmo;
