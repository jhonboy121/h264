//! Multi-slice encode validation. The encoder partitions each frame into `S`
//! contiguous slices (MB-row bands), each emitted as its own slice NAL with an
//! independent CAVLC bitstream and prediction state reset at the boundary
//! (intra availability, mvd/MV prediction, and nC neighbour derivation all
//! treat the slice's first MB row as having no top neighbour — exactly as the
//! decoder gates on its per-MB `slice_idc`). Deblocking is signalled off
//! (`disable_deblocking_filter_idc = 1`), so slices are fully independent and
//! reconstruction == decoder output.
//!
//! These tests encode the same source single-slice and multi-slice (S = 4, 8),
//! decode both with this crate's own decoder, and assert: (a) the decoder
//! accepts the multi-slice stream and emits the right NAL count, (b) the
//! reconstruction PSNR matches the single-slice case within a small tolerance
//! (slice boundaries only cost a little prediction, never correctness).

use h264::{Decoder, YUVSource};

const W: usize = 176;
const H: usize = 144;

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

/// Synthesize a smooth, textured panning sequence (same shape as the IPPP
/// round-trip test): `n` frames of a window sliding `pan` luma px/frame.
fn panning_sequence(n: usize, pan: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let cw = W / 2;
    let ch = H / 2;
    let lum = |x: f64, y: f64| -> u8 {
        let v = 128.0 + 50.0 * (0.10 * x).sin() * (0.07 * y).cos() + 22.0 * (0.31 * (x + y)).sin();
        v.clamp(0.0, 255.0) as u8
    };
    let chr = |x: f64, y: f64, o: f64| -> u8 {
        let v = 128.0 + 28.0 * (0.18 * x + o).sin() * (0.12 * y).cos();
        v.clamp(0.0, 255.0) as u8
    };
    (0..n)
        .map(|k| {
            let dx = (k * pan) as f64;
            let cdx = (k * pan / 2) as f64;
            let mut y = vec![0u8; W * H];
            for j in 0..H {
                for i in 0..W {
                    y[j * W + i] = lum(i as f64 + dx, j as f64);
                }
            }
            let mut u = vec![0u8; cw * ch];
            let mut v = vec![0u8; cw * ch];
            for j in 0..ch {
                for i in 0..cw {
                    u[j * cw + i] = chr(i as f64 + cdx, j as f64, 0.0);
                    v[j * cw + i] = chr(i as f64 + cdx, j as f64, 1.7);
                }
            }
            (y, u, v)
        })
        .collect()
}

fn decode_all_planes(stream: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut dec = Decoder::new();
    let frames = dec.decode_all(stream).expect("decode multi-slice stream");
    frames
        .iter()
        .map(|f| {
            let (w, h) = f.dimensions();
            let (ys, us, vs) = f.strides();
            let pack = |plane: &[u8], stride: usize, pw: usize, ph: usize| {
                let mut out = vec![0u8; pw * ph];
                for j in 0..ph {
                    out[j * pw..j * pw + pw].copy_from_slice(&plane[j * stride..j * stride + pw]);
                }
                out
            };
            (pack(f.y(), ys, w, h), pack(f.u(), us, w / 2, h / 2), pack(f.v(), vs, w / 2, h / 2))
        })
        .collect()
}

/// Count Annex-B NAL units in a buffer.
fn nal_count(au: &[u8]) -> usize {
    h264::nal_units(au).count()
}

fn encode_seq(frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)], qp: u8, slices: u32, all_intra: bool) -> Vec<Vec<u8>> {
    let mut enc = h264::encoder::Encoder::new_with_slices(W as u32, H as u32, qp, slices).unwrap();
    frames
        .iter()
        .enumerate()
        .map(|(k, (y, u, v))| {
            if all_intra && k > 0 {
                enc.force_idr();
            }
            enc.encode_frame(y, W, u, v, W / 2)
        })
        .collect()
}

#[test]
fn multislice_intra_roundtrip_matches_single_slice() {
    let frames = panning_sequence(3, 3);
    let mb_h = H.div_ceil(16) as u32; // 9 MB rows
    for &qp in &[26u8, 32] {
        let single = encode_seq(&frames, qp, 1, true);
        for &s in &[4u32, 8] {
            let multi = encode_seq(&frames, qp, s, true);
            // Each IDR AU = SPS + PPS + S slice NALs.
            for au in &multi {
                assert_eq!(nal_count(au), 2 + s as usize, "qp={qp} S={s}: IDR NAL count");
            }
            // Decode both; PSNR per frame must match within a small tolerance.
            let dref = decode_all_planes(&single.concat());
            let dmul = decode_all_planes(&multi.concat());
            assert_eq!(dref.len(), frames.len());
            assert_eq!(dmul.len(), frames.len());
            for (k, (src, ((ry, _, _), (my, _, _)))) in frames.iter().zip(dref.iter().zip(dmul.iter())).enumerate() {
                let p1 = psnr(&src.0, ry);
                let pm = psnr(&src.0, my);
                assert!(pm >= 34.0, "qp={qp} S={s} frame {k}: multi-slice PSNR {pm:.2} too low");
                assert!((p1 - pm).abs() <= 1.5, "qp={qp} S={s} frame {k}: PSNR drift {p1:.2}->{pm:.2}");
            }
            assert!(s <= mb_h, "test S exceeds MB height");
        }
    }
}

#[test]
fn multislice_ippp_roundtrip_matches_single_slice() {
    let frames = panning_sequence(5, 2);
    for &qp in &[26u8, 32] {
        let single = encode_seq(&frames, qp, 1, false);
        for &s in &[4u32, 8] {
            let multi = encode_seq(&frames, qp, s, false);
            // IDR AU = SPS + PPS + S slices; each P AU = S slices.
            assert_eq!(nal_count(&multi[0]), 2 + s as usize, "qp={qp} S={s}: IDR NAL count");
            for au in &multi[1..] {
                assert_eq!(nal_count(au), s as usize, "qp={qp} S={s}: P NAL count");
            }
            let dref = decode_all_planes(&single.concat());
            let dmul = decode_all_planes(&multi.concat());
            assert_eq!(dmul.len(), frames.len());
            for (k, (src, ((ry, _, _), (my, _, _)))) in frames.iter().zip(dref.iter().zip(dmul.iter())).enumerate() {
                let p1 = psnr(&src.0, ry);
                let pm = psnr(&src.0, my);
                assert!(pm >= 32.0, "qp={qp} S={s} frame {k}: multi-slice PSNR {pm:.2} too low");
                assert!((p1 - pm).abs() <= 2.0, "qp={qp} S={s} frame {k}: PSNR drift {p1:.2}->{pm:.2}");
            }
        }
    }
}

#[test]
fn slices_clamped_to_mb_height() {
    // Requesting more slices than MB rows clamps to one slice per row.
    let enc = h264::encoder::Encoder::new_with_slices(W as u32, H as u32, 28, 1000).unwrap();
    assert_eq!(enc.slices(), H.div_ceil(16) as u32);
}
