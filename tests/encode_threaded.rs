//! Threaded-encode determinism (feature `threads`). The threaded paths must be
//! **byte-identical** to the serial paths — every slice is independent
//! (deblocking off, neighbours gated at the boundary), so concurrency cannot
//! change a single bit of the output stream or the reconstructed reference that
//! the next P frame is coded against.
#![cfg(feature = "threads")]

use h264::encoder::{Encoder, FrameInput};
use h264::Decoder;

const W: usize = 320;
const H: usize = 192; // 20x12 MBs — room for up to 12 slices

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

#[test]
fn slice_parallel_ippp_byte_identical_to_serial() {
    let frames = panning_sequence(6, 2);
    for &slices in &[1u32, 4, 8] {
        for &qp in &[26u8, 32] {
            let mut serial = Encoder::new_with_slices(W as u32, H as u32, qp, slices).unwrap();
            let mut threaded = Encoder::new_with_slices(W as u32, H as u32, qp, slices).unwrap();
            for (y, u, v) in &frames {
                let a = serial.encode_frame(y, W, u, v, W / 2);
                let b = threaded.encode_frame_parallel(y, W, u, v, W / 2);
                assert_eq!(a, b, "slice-parallel IPPP AU differs at slices={slices} qp={qp}");
            }
        }
    }
}

#[test]
fn frame_parallel_intra_byte_identical_to_serial() {
    let frames = panning_sequence(7, 3);
    for &slices in &[1u32, 4] {
        for &qp in &[26u8, 32] {
            // Serial all-intra reference (force IDR on every frame).
            let mut serial = Encoder::new_with_slices(W as u32, H as u32, qp, slices).unwrap();
            let want: Vec<Vec<u8>> = frames
                .iter()
                .enumerate()
                .map(|(k, (y, u, v))| {
                    if k > 0 {
                        serial.force_idr();
                    }
                    serial.encode_frame(y, W, u, v, W / 2)
                })
                .collect();

            // Frame-parallel path.
            let enc = Encoder::new_with_slices(W as u32, H as u32, qp, slices).unwrap();
            let inputs: Vec<FrameInput> = frames
                .iter()
                .map(|(y, u, v)| FrameInput { y, y_stride: W, u, v, c_stride: W / 2 })
                .collect();
            let got = enc.encode_frames_parallel(&inputs);

            assert_eq!(got.len(), want.len());
            for (k, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(a, b, "frame-parallel intra AU {k} differs at slices={slices} qp={qp}");
            }
        }
    }
}

#[test]
fn threaded_stream_decodes() {
    // Sanity: the threaded IPPP stream is a valid, decodable bitstream.
    let frames = panning_sequence(4, 2);
    let mut enc = Encoder::new_with_slices(W as u32, H as u32, 28, 6).unwrap();
    let mut stream = Vec::new();
    for (y, u, v) in &frames {
        stream.extend_from_slice(&enc.encode_frame_parallel(y, W, u, v, W / 2));
    }
    let mut dec = Decoder::new();
    let decoded = dec.decode_all(&stream).expect("threaded stream decodes");
    assert_eq!(decoded.len(), frames.len());
}
