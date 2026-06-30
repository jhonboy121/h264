//! Encoder round-trip validation: encode a real source frame (BANM frame 0, the
//! same 176x144 I420 golden used by the decoder conformance tests), decode the
//! result with our own (Cisco-conformant) decoder, and assert the reconstruction
//! PSNR is high and improves as QP drops. This proves the encoder emits a
//! spec-compliant, decodable bitstream of good quality.

use h264::{Decoder, YUVSource};

const W: usize = 176;
const H: usize = 144;
/// Planar I420 source: Y (W*H) then U then V (each W/2 * H/2).
const GOLDEN: &[u8] = include_bytes!("fixtures/banm_frame0.yuv");

fn source_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let ysize = W * H;
    let csize = (W / 2) * (H / 2);
    let y = GOLDEN[..ysize].to_vec();
    let u = GOLDEN[ysize..ysize + csize].to_vec();
    let v = GOLDEN[ysize + csize..ysize + 2 * csize].to_vec();
    (y, u, v)
}

fn psnr(orig: &[u8], rec: &[u8]) -> f64 {
    assert_eq!(orig.len(), rec.len());
    let mut sse = 0u64;
    for (&a, &b) in orig.iter().zip(rec.iter()) {
        let d = a as i64 - b as i64;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 99.0;
    }
    let mse = sse as f64 / orig.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Decode an access unit and return the reconstructed Y/U/V planes (tightly
/// packed at the visible resolution) plus dimensions.
fn decode_planes(au: &[u8]) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut dec = Decoder::new();
    let frames = dec.decode_all(au).expect("decode");
    assert_eq!(frames.len(), 1, "expected exactly one decoded frame");
    let f = &frames[0];
    let (w, h) = f.dimensions();
    let (ys, us, vs) = f.strides();
    let cw = w / 2;
    let ch = h / 2;

    let pack = |plane: &[u8], stride: usize, pw: usize, ph: usize| {
        let mut out = vec![0u8; pw * ph];
        for j in 0..ph {
            out[j * pw..j * pw + pw].copy_from_slice(&plane[j * stride..j * stride + pw]);
        }
        out
    };
    let y = pack(f.y(), ys, w, h);
    let u = pack(f.u(), us, cw, ch);
    let v = pack(f.v(), vs, cw, ch);
    (w, h, y, u, v)
}

#[test]
fn roundtrip_psnr_and_monotonicity() {
    let (sy, su, sv) = source_planes();

    let mut prev_psnr = 0.0f64;
    for (idx, &qp) in [32u8, 26].iter().enumerate() {
        let mut enc = h264::encoder::Encoder::new(W as u32, H as u32, qp).unwrap();
        let au = enc.encode_frame(&sy, W, &su, &sv, W / 2);

        let (dw, dh, ry, ru, rv) = decode_planes(&au);
        assert_eq!((dw, dh), (W, H), "decoded dimensions mismatch at qp={qp}");

        let py = psnr(&sy, &ry);
        let pu = psnr(&su, &ru);
        let pv = psnr(&sv, &rv);
        eprintln!("qp={qp}: PSNR Y={py:.2} U={pu:.2} V={pv:.2} dB, AU={} bytes", au.len());

        let bar = if qp == 26 { 36.0 } else { 32.0 };
        assert!(py >= bar, "luma PSNR {py:.2} dB below {bar} at qp={qp}");
        assert!(pu >= 38.0 && pv >= 38.0, "chroma PSNR too low at qp={qp}: U={pu:.2} V={pv:.2}");

        if idx > 0 {
            assert!(py >= prev_psnr - 0.01, "PSNR not monotonic with QP (qp={qp})");
        }
        prev_psnr = py;
    }
}

#[test]
fn roundtrip_dimensions_and_decodes() {
    // Also exercise a non-MB-aligned size (frame cropping path) end to end.
    let (w, h) = (170usize, 138usize);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (40 + (i + j) % 160) as u8;
        }
    }
    let u = vec![110u8; (w / 2) * (h / 2)];
    let v = vec![135u8; (w / 2) * (h / 2)];
    let mut enc = h264::encoder::Encoder::new(w as u32, h as u32, 28).unwrap();
    let au = enc.encode_frame(&y, w, &u, &v, w / 2);
    let (dw, dh, ry, _, _) = decode_planes(&au);
    assert_eq!((dw, dh), (w, h), "cropped dimensions must round-trip");
    assert!(psnr(&y, &ry) >= 34.0);
}
