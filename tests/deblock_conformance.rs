//! Conformance gate: the decoded + deblocked first IDR frame of BANM_MW_D must
//! be byte-identical to the C reference oracle's first frame.
//!
//! The golden `tests/fixtures/banm_frame0.yuv` (38016 bytes, planar I420
//! 176x144) is the C decoder's output; this test decodes the same stream with
//! the pure-Rust decoder and asserts bit-exact equality of the visible planes.

use h264::decoder::decode_intra_frame;

const BANM: &[u8] = include_bytes!("fixtures/BANM_MW_D.264");
const GOLDEN: &[u8] = include_bytes!("fixtures/banm_frame0.yuv");

/// Extract the visible I420 planes (no padding) into a contiguous buffer,
/// matching `examples/dump_frame.rs`.
fn visible_i420(annexb: &[u8]) -> Vec<u8> {
    let pic = decode_intra_frame(annexb).expect("decode first intra frame");
    assert_eq!(pic.width, 176);
    assert_eq!(pic.height, 144);

    let mut out = Vec::with_capacity(176 * 144 + 2 * 88 * 72);
    let yo = pic.luma_origin();
    for row in 0..144 {
        let start = yo + row * pic.luma_stride;
        out.extend_from_slice(&pic.y[start..start + 176]);
    }
    let co = pic.chroma_origin();
    for row in 0..72 {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.u[start..start + 88]);
    }
    for row in 0..72 {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.v[start..start + 88]);
    }
    out
}

#[test]
fn banm_first_frame_bit_exact_vs_c_oracle() {
    let got = visible_i420(BANM);
    assert_eq!(got.len(), GOLDEN.len(), "I420 frame size");

    // Locate the first differing byte (if any) for a useful failure message.
    let diffs = got
        .iter()
        .zip(GOLDEN.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .count();
    if diffs != 0 {
        let (i, (a, b)) = got
            .iter()
            .zip(GOLDEN.iter())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .unwrap();
        panic!("{diffs} differing bytes; first at offset {i}: rs={a} c={b}");
    }
}
