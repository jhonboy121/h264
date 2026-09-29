//! Real-time throughput of the rate-controlled encoder (`Encoder::encode`),
//! as a live re-encode would drive it: 1920x1080@60 and 1280x720@30 at a
//! MediaCodec-like bitrate, slices encoded in parallel. Also times
//! `scale_i420` for the downscales a sender would do.
//!
//! The source is a synthetic camera-like sequence (a textured backdrop panned
//! under a moving object, plus sensor noise), generated before timing.
//!
//! Usage: `cargo run --release --features threads,simd --example bench_realtime`

use std::error::Error;
use std::time::Instant;

use h264::{Encoder, EncoderConfig, I420, scale_i420};

/// Distinct frames generated per case (the encoder loops over them).
const FRAMES: usize = 60;
/// Seconds of video encoded per case.
const SECONDS: usize = 5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Multi-octave value noise.
fn texture(w: usize, h: usize, octaves: &[(usize, i32)], rng: &mut Rng) -> Vec<u8> {
    let mut acc = vec![128i32; w * h];
    for &(cell, amp) in octaves {
        let gw = w / cell + 2;
        let grid: Vec<i32> = (0..gw * (h / cell + 2))
            .map(|_| (rng.next() % (amp as u32 + 1)) as i32 - amp / 2)
            .collect();
        let c = cell as i32;
        for y in 0..h {
            let (gy, fy) = (y / cell, (y % cell) as i32);
            for x in 0..w {
                let (gx, fx) = (x / cell, (x % cell) as i32);
                let top = grid[gy * gw + gx] * (c - fx) + grid[gy * gw + gx + 1] * fx;
                let bot = grid[(gy + 1) * gw + gx] * (c - fx) + grid[(gy + 1) * gw + gx + 1] * fx;
                acc[y * w + x] += (top * (c - fy) + bot * fy) / (c * c);
            }
        }
    }
    acc.into_iter().map(|v| v.clamp(0, 255) as u8).collect()
}

fn sequence(w: usize, h: usize) -> Vec<I420> {
    let mut rng = Rng(42);
    let (bw, bh) = (w + w / 2, h + h / 2);
    let bg = texture(bw, bh, &[(w / 8, 90), (w / 40, 40), (8, 24), (2, 10)], &mut rng);
    let bg_c = texture(bw / 2, bh / 2, &[(w / 16, 50), (w / 80, 16)], &mut rng);
    let (ow, oh) = (w / 4, h / 3);
    let obj = texture(ow, oh, &[(ow / 4, 120), (4, 40)], &mut rng);
    (0..FRAMES)
        .map(|k| {
            let (ox, oy) = (k * 3 % (bw - w), k % (bh - h));
            let mut f = I420::new(w as u32, h as u32);
            for y in 0..h {
                let s = (oy + y) * bw + ox;
                f.y[y * w..(y + 1) * w].copy_from_slice(&bg[s..s + w]);
            }
            for y in 0..h / 2 {
                let s = (oy / 2 + y) * (bw / 2) + ox / 2;
                f.u[y * w / 2..(y + 1) * w / 2].copy_from_slice(&bg_c[s..s + w / 2]);
                f.v[y * w / 2..(y + 1) * w / 2].copy_from_slice(&bg_c[s..s + w / 2]);
            }
            let (px, py) = (k * 9 % (w - ow), h / 3);
            for y in 0..oh {
                let d = (py + y) * w + px;
                f.y[d..d + ow].copy_from_slice(&obj[y * ow..(y + 1) * ow]);
            }
            for p in &mut f.y {
                *p = (*p as i32 + (rng.next() % 5) as i32 - 2).clamp(0, 255) as u8;
            }
            f
        })
        .collect()
}

fn bench(w: u32, h: u32, fps: u32, kbps: u32, slices: u32) -> Result<(), Box<dyn Error>> {
    let frames = sequence(w as usize, h as usize);
    let config = EncoderConfig {
        width: w,
        height: h,
        fps,
        bitrate_kbps: kbps,
        keyframe_interval: Some(2 * fps),
        slices,
    };
    let mut enc = Encoder::with_config(config)?;
    let n = SECONDS * fps as usize;
    let (mut bytes, mut qp_lo, mut qp_hi) = (0usize, u8::MAX, 0u8);
    let start = Instant::now();
    for k in 0..n {
        let out = enc.encode(&frames[k % FRAMES].as_ref())?;
        bytes += out.data.len();
        qp_lo = qp_lo.min(out.qp);
        qp_hi = qp_hi.max(out.qp);
    }
    let secs = start.elapsed().as_secs_f64();
    let got_fps = n as f64 / secs;
    println!(
        "{w}x{h}@{fps} {kbps} kbps, {slices} slices: {got_fps:.1} fps ({:.2} ms/frame, {:.2}x real time), \
         {:.0} kbps out, qp {qp_lo}..{qp_hi}",
        1000.0 * secs / n as f64,
        got_fps / fps as f64,
        bytes as f64 * 8.0 / SECONDS as f64 / 1000.0,
    );
    Ok(())
}

fn bench_scale(src: &I420, w: u32, h: u32) {
    const REPS: usize = 100;
    let mut dst = I420::new(w, h);
    let start = Instant::now();
    for _ in 0..REPS {
        scale_i420(&src.as_ref(), &mut dst);
    }
    let ms = 1000.0 * start.elapsed().as_secs_f64() / REPS as f64;
    println!("scale_i420 {}x{} -> {w}x{h}: {ms:.2} ms", src.width, src.height);
}

fn main() -> Result<(), Box<dyn Error>> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("threads available: {threads}");
    bench(1920, 1080, 60, 6000, 16)?;
    bench(1920, 1080, 30, 4000, 16)?;
    bench(1280, 720, 30, 2000, 16)?;
    bench(1280, 720, 30, 2000, 8)?;
    bench(640, 360, 15, 350, 4)?;
    let src = sequence(1920, 1080).swap_remove(0);
    for (w, h) in [(1280, 720), (960, 540), (640, 360)] {
        bench_scale(&src, w, h);
    }
    Ok(())
}
