//! Encode-throughput benchmark for the pure-Rust baseline encoder.
//!
//! Uses the real BANM source (decoded by our own decoder to recover 100 frames
//! of genuine 176x144 I420 content), then measures `Encoder::encode_frame`
//! throughput in two modes:
//!   - INTRA: every frame forced to an IDR (all-intra coding).
//!   - IPPP : one IDR then P frames (inter coding).
//!
//! Each is run at QP 26 and QP 32. We report ms/frame, frames/s, and MB/s
//! (source I420 throughput = frames * W * H * 3/2 bytes / time).
//!
//! Usage: `cargo run --release --example bench_encode`
//! (Add `--features simd` to exercise the SIMD SAD/MC kernels the encoder uses.)

use std::time::Instant;

use h264::decoder::decode_stream;
use h264::encoder::Encoder;

const BANM: &[u8] = include_bytes!("../tests/fixtures/BANM_MW_D.264");

const MIN_REPS: u32 = 3;
const MAX_REPS: u32 = 100;
const TARGET_SECS: f64 = 1.5;

/// A source frame: packed visible Y/U/V planes plus strides.
struct Src {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    ys: usize,
    cs: usize,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() {
    let pics = decode_stream(BANM).expect("decode BANM");
    let (w, h) = (pics[0].width, pics[0].height);
    let cw = w / 2;
    let ch = h / 2;

    // Pack each decoded picture into tight visible planes.
    let frames: Vec<Src> = pics
        .iter()
        .map(|p| {
            let lo = p.luma_stride * 32 + 32;
            let co = p.chroma_stride * 32 + 32;
            let pack = |plane: &[u8], origin: usize, stride: usize, pw: usize, ph: usize| {
                let mut out = vec![0u8; pw * ph];
                for j in 0..ph {
                    out[j * pw..j * pw + pw].copy_from_slice(&plane[origin + j * stride..origin + j * stride + pw]);
                }
                out
            };
            Src {
                y: pack(&p.y, lo, p.luma_stride, w, h),
                u: pack(&p.u, co, p.chroma_stride, cw, ch),
                v: pack(&p.v, co, p.chroma_stride, cw, ch),
                ys: w,
                cs: cw,
            }
        })
        .collect();

    let n = frames.len();
    let total_mb = n as f64 * (w * h * 3 / 2) as f64 / 1.0e6;

    println!("machine: arm64   build: --release (opt-level=3, lto=thin, codegen-units=1)");
    #[cfg(feature = "simd")]
    println!("encoder DSP: SIMD (SAD/MC NEON)");
    #[cfg(not(feature = "simd"))]
    println!("encoder DSP: scalar");
    println!(
        "source: BANM {w}x{h}, {n} real frames  |  MB/s = source I420 throughput\n"
    );
    println!("{:<14} {:>6} {:>10} {:>9} {:>9} {:>12}", "MODE", "QP", "ms/frame", "fps", "MB/s", "avg AU bytes");
    println!("{}", "-".repeat(66));

    for &qp in &[26u8, 32u8] {
        for intra in [true, false] {
            // The encoded size is deterministic, so measure it once (outside the
            // timing loop, which only re-runs the encode to gather timings).
            let au_total = {
                let mut enc = Encoder::new(w as u32, h as u32, qp).unwrap();
                let mut bytes = 0usize;
                for (k, f) in frames.iter().enumerate() {
                    if intra && k > 0 {
                        enc.force_idr();
                    }
                    bytes += enc.encode_frame(&f.y, f.ys, &f.u, &f.v, f.cs).len();
                }
                bytes
            };

            // Measure encode of the whole sequence, auto-scaled.
            let mut times = Vec::new();
            let start = Instant::now();
            let mut reps = 0u32;
            loop {
                let mut enc = Encoder::new(w as u32, h as u32, qp).unwrap();
                let t = Instant::now();
                for (k, f) in frames.iter().enumerate() {
                    if intra && k > 0 {
                        enc.force_idr();
                    }
                    let _ = enc.encode_frame(&f.y, f.ys, &f.u, &f.v, f.cs);
                }
                times.push(t.elapsed().as_secs_f64() * 1e3);
                reps += 1;
                if reps >= MAX_REPS {
                    break;
                }
                if reps >= MIN_REPS && start.elapsed().as_secs_f64() >= TARGET_SECS {
                    break;
                }
            }
            let med = median(&mut times);
            let ms_f = med / n as f64;
            let fps = n as f64 / (med / 1e3);
            let mbs = total_mb / (med / 1e3);
            println!(
                "{:<14} {:>6} {:>10.3} {:>9.1} {:>9.1} {:>12.1}",
                if intra { "INTRA (all-I)" } else { "IPPP" },
                qp,
                ms_f,
                fps,
                mbs,
                au_total as f64 / n as f64
            );
        }
    }
}
