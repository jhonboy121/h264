//! Conformance gate for inter (P-frame) decoding: the full baseline CAVLC
//! stream BANM_MW_D must decode bit-exact against the C reference oracle.
//!
//! `tests/fixtures/banm_frame1.yuv` is the C decoder's second frame (the first
//! P frame); this test decodes the whole stream with the pure-Rust decoder and
//! asserts the visible I420 planes of frame 1 are byte-identical.

use h264::decoder::decode_stream;
use h264::decoder::picture::Picture;

const BANM: &[u8] = include_bytes!("fixtures/BANM_MW_D.264");
const GOLDEN_FRAME0: &[u8] = include_bytes!("fixtures/banm_frame0.yuv");
const GOLDEN_FRAME1: &[u8] = include_bytes!("fixtures/banm_frame1.yuv");

fn visible_i420(pic: &Picture) -> Vec<u8> {
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
fn banm_first_p_frame_bit_exact_vs_c_oracle() {
    let frames = decode_stream(BANM).expect("decode BANM stream");
    assert!(frames.len() >= 2, "expected at least 2 frames, got {}", frames.len());

    // Frame 0 (IDR) must still match, and frame 1 (first P frame) too.
    let f0 = visible_i420(&frames[0]);
    assert_eq!(f0.len(), GOLDEN_FRAME0.len());
    assert!(f0 == GOLDEN_FRAME0, "IDR frame diverged from oracle");

    let f1 = visible_i420(&frames[1]);
    assert_eq!(f1.len(), GOLDEN_FRAME1.len());
    if f1 != GOLDEN_FRAME1 {
        let first = (0..f1.len()).find(|&i| f1[i] != GOLDEN_FRAME1[i]).unwrap();
        panic!(
            "first P frame diverged at byte {first}: rust={} c={}",
            f1[first], GOLDEN_FRAME1[first]
        );
    }
}

#[test]
fn banm_full_stream_decodes_100_frames() {
    let frames = decode_stream(BANM).expect("decode BANM stream");
    assert_eq!(frames.len(), 100, "expected 100 decoded frames");
    for pic in &frames {
        assert_eq!(pic.width, 176);
        assert_eq!(pic.height, 144);
    }
}
