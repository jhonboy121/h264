//! Integration tests for the public [`h264::Decoder`] API: the ergonomic path
//! must equal the proven `decode_stream` path, and YUV→RGB conversion must hit
//! known values.

use h264::{Decoder, YUVSource, nal_units};

const BANM: &[u8] = include_bytes!("fixtures/BANM_MW_D.264");
const GOLDEN_FRAME0: &[u8] = include_bytes!("fixtures/banm_frame0.yuv");

/// Pack a `YUVSource`'s visible region as planar I420 (Y then U then V), the
/// same layout as the C-oracle golden fixtures.
fn visible_i420<S: YUVSource>(src: &S) -> Vec<u8> {
    let (w, h) = src.dimensions();
    let (ys, us, vs) = src.strides();
    let mut out = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
    let y = src.y();
    for row in 0..h {
        out.extend_from_slice(&y[row * ys..row * ys + w]);
    }
    let u = src.u();
    for row in 0..h / 2 {
        out.extend_from_slice(&u[row * us..row * us + w / 2]);
    }
    let v = src.v();
    for row in 0..h / 2 {
        out.extend_from_slice(&v[row * vs..row * vs + w / 2]);
    }
    out
}

#[test]
fn decode_all_matches_golden_and_count() {
    let mut dec = Decoder::new();
    let frames = dec.decode_all(BANM).expect("decode BANM via Decoder API");

    assert_eq!(frames.len(), 100, "expected 100 frames");
    for f in &frames {
        assert_eq!(f.dimensions(), (176, 144));
    }

    // First frame must equal the proven golden decode.
    let f0 = visible_i420(&frames[0]);
    assert_eq!(f0.len(), GOLDEN_FRAME0.len());
    assert!(
        f0 == GOLDEN_FRAME0,
        "Decoder API frame 0 diverged from golden"
    );
}

#[test]
fn incremental_decode_emits_all_frames() {
    // Feed the whole stream to `decode`, then `flush`, collecting every frame.
    // Each borrowed view is converted to an owned buffer before the next call,
    // since `decode`/`flush` invalidate the previous view.
    let mut dec = Decoder::new();
    let mut count = 0usize;
    let mut first: Option<Vec<u8>> = None;

    // First call carries the whole stream; later calls drain the ready queue.
    let mut packet: &[u8] = BANM;
    loop {
        let owned = {
            let got = dec.decode(packet).expect("decode");
            packet = &[];
            got.as_ref().map(visible_i420)
        };
        match owned {
            Some(buf) => {
                first.get_or_insert(buf);
                count += 1;
            }
            None => break,
        }
    }
    loop {
        let owned = dec.flush().as_ref().map(visible_i420);
        match owned {
            Some(buf) => {
                first.get_or_insert(buf);
                count += 1;
            }
            None => break,
        }
    }

    assert_eq!(count, 100, "incremental path must emit all frames");
    assert_eq!(first.expect("a first frame"), GOLDEN_FRAME0);
}

#[test]
fn nal_units_helper_splits_stream() {
    let n = nal_units(BANM).count();
    // SPS + PPS + 100 coded slices.
    assert!(n >= 100, "expected many NAL units, got {n}");
}

#[test]
fn write_rgb_lengths_and_alpha() {
    let frames = Decoder::new().decode_all(BANM).unwrap();
    let f = &frames[0];

    assert_eq!(f.rgb8_len(), 176 * 144 * 3);
    assert_eq!(f.rgba8_len(), 176 * 144 * 4);

    let mut rgb = vec![0u8; f.rgb8_len()];
    f.write_rgb8(&mut rgb);

    let mut rgba = vec![0u8; f.rgba8_len()];
    f.write_rgba8(&mut rgba);
    // RGB triples agree with the first three RGBA bytes, alpha is opaque.
    assert!(rgba.chunks_exact(4).all(|p| p[3] == 255));
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        assert_eq!(&px[..3], &rgb[i * 3..i * 3 + 3]);
    }
}
