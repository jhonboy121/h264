//! Decode an entire baseline Annex-B stream (I + P) and dump every decoded
//! frame's visible I420 planes to a file for comparison against a C reference.
//!
//! Usage: `cargo run --example dump_frames`
//! Writes all frames (176x144 Y + 88x72 U + 88x72 V each) to
//! `/tmp/banm_rs_all.yuv`.

use std::fs;

use h264::decoder::decode_stream;
use h264::decoder::picture::Picture;

fn append_visible(out: &mut Vec<u8>, pic: &Picture) {
    let yo = pic.luma_origin();
    for row in 0..pic.height {
        let start = yo + row * pic.luma_stride;
        out.extend_from_slice(&pic.y[start..start + pic.width]);
    }
    let co = pic.chroma_origin();
    let (cw, ch) = (pic.width / 2, pic.height / 2);
    for row in 0..ch {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.u[start..start + cw]);
    }
    for row in 0..ch {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.v[start..start + cw]);
    }
}

fn main() {
    let annexb = fs::read("tests/fixtures/BANM_MW_D.264").expect("read fixture");
    let frames = decode_stream(&annexb).expect("decode stream");

    let mut out = Vec::new();
    for pic in &frames {
        append_visible(&mut out, pic);
    }
    fs::write("/tmp/banm_rs_all.yuv", &out).expect("write output");
    eprintln!("decoded {} frames, wrote {} bytes", frames.len(), out.len());
}
