//! Decode the first intra frame of a BANM fixture and dump its visible I420
//! planes (no padding) to a file, for comparison against a C reference decoder.
//!
//! Usage: `cargo run --example dump_frame`
//! Writes the 38016-byte (176x144 Y + 88x72 U + 88x72 V) first frame to
//! `/tmp/banm_rs.yuv`.

use std::fs;

use h264::decoder::decode_intra_frame;

fn main() {
    let annexb = fs::read("tests/fixtures/BANM_MW_D.264").expect("read fixture");
    let pic = decode_intra_frame(&annexb).expect("decode first intra frame");

    let mut out = Vec::with_capacity(176 * 144 + 2 * 88 * 72);

    // Luma: 176x144 starting at luma_origin, walking luma_stride per row.
    let yo = pic.luma_origin();
    for row in 0..144 {
        let start = yo + row * pic.luma_stride;
        out.extend_from_slice(&pic.y[start..start + 176]);
    }

    // Chroma U then V: 88x72 each starting at chroma_origin.
    let co = pic.chroma_origin();
    for row in 0..72 {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.u[start..start + 88]);
    }
    for row in 0..72 {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.v[start..start + 88]);
    }

    assert_eq!(out.len(), 38016, "expected 38016-byte I420 first frame");
    fs::write("/tmp/banm_rs.yuv", &out).expect("write output");
    eprintln!("wrote {} bytes to /tmp/banm_rs.yuv", out.len());
}
