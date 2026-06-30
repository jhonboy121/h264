//! CABAC conformance: decode CABAC-coded streams and compare the visible I420
//! frames byte-for-byte against golden YUV produced by the Cisco OpenH264 C
//! reference decoder (`h264dec`).
//!
//! Golden files live in `tests/fixtures/*.golden.yuv` (raw I420, frames
//! concatenated in decode/display order).

use h264::decoder::decode_stream;
use h264::decoder::picture::Picture;

fn i420(pic: &Picture) -> Vec<u8> {
    let mut out = Vec::with_capacity(pic.width * pic.height * 3 / 2);
    let o = pic.luma_origin();
    for y in 0..pic.height {
        let s = o + y * pic.luma_stride;
        out.extend_from_slice(&pic.y[s..s + pic.width]);
    }
    let (cw, ch, co) = (pic.width / 2, pic.height / 2, pic.chroma_origin());
    for y in 0..ch {
        let s = co + y * pic.chroma_stride;
        out.extend_from_slice(&pic.u[s..s + cw]);
    }
    for y in 0..ch {
        let s = co + y * pic.chroma_stride;
        out.extend_from_slice(&pic.v[s..s + cw]);
    }
    out
}

const QCIF: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/test_qcif_cabac.264"
));
const QCIF_GOLDEN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/test_qcif_cabac.golden.yuv"
));
const CIF_I: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/test_cif_I_CABAC_slice.264"
));
const CIF_I_GOLDEN_F0: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/test_cif_I_CABAC_frame0.golden.yuv"
));

/// Main-profile CABAC, 1 I + 29 P (single reference), QCIF: every visible frame
/// must be bit-exact with the C reference.
#[test]
fn qcif_cabac_all_frames_bit_exact_vs_c() {
    let pics = decode_stream(QCIF).expect("decode test_qcif_cabac.264");
    assert_eq!(pics.len(), 30, "frame count");
    let fsize = 176 * 144 * 3 / 2;
    assert_eq!(QCIF_GOLDEN.len(), 30 * fsize);
    for (i, pic) in pics.iter().enumerate() {
        assert_eq!(pic.width, 176);
        assert_eq!(pic.height, 144);
        let got = i420(pic);
        let exp = &QCIF_GOLDEN[i * fsize..(i + 1) * fsize];
        let diff = got.iter().zip(exp).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "frame {i} differs from C oracle ({diff} bytes)");
    }
}

/// All-intra CABAC (Main profile, 14 slices/frame), CIF: the first IDR frame
/// must be bit-exact with the C reference.
#[test]
fn cif_intra_cabac_frame0_bit_exact_vs_c() {
    let pics = decode_stream(CIF_I).expect("decode test_cif_I_CABAC_slice.264");
    assert!(pics.len() >= 1);
    let pic = &pics[0];
    assert_eq!(pic.width, 352);
    assert_eq!(pic.height, 288);
    let got = i420(pic);
    assert_eq!(got.len(), CIF_I_GOLDEN_F0.len());
    let diff = got.iter().zip(CIF_I_GOLDEN_F0).filter(|(a, b)| a != b).count();
    assert_eq!(diff, 0, "CIF intra frame 0 differs from C oracle ({diff} bytes)");
}
