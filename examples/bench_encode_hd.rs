//! 1080p encode-throughput benchmark: single-threaded vs multithreaded.
//!
//! Generates a self-contained synthetic 1920x1080 moving source (a panning,
//! textured gradient + pseudo-noise window sliding over a larger analytic field,
//! so there is real frame-to-frame motion for inter prediction to exploit), then
//! measures `Encoder` throughput in two modes x two QPs:
//!   - INTRA: every frame an IDR (all-intra coding).
//!   - IPPP : one IDR then P frames (inter coding).
//!
//! Single-threaded uses one slice per frame (the compression-optimal baseline).
//! With `--features threads` it also runs the threaded fan-out:
//!   - INTRA -> `encode_frames_parallel` (frame-parallel, N independent IDRs).
//!   - IPPP  -> `encode_frame_parallel`  (slice-parallel, S slices/frame),
//!     S = available_parallelism, frames still serial.
//!
//! Reports ms/frame, fps, MB/s (source I420 throughput) and x-realtime-30, plus
//! the average AU size so the multi-slice compression overhead is visible.
//!
//! Usage: `cargo run --release --features "simd threads" --example bench_encode_hd`

use std::time::Instant;

use h264::encoder::Encoder;

const W: usize = 1920;
const H: usize = 1080;

const MIN_REPS: u32 = 3;
const MAX_REPS: u32 = 50;
const TARGET_SECS: f64 = 2.0;
const N_FRAMES: usize = 16;
/// Realtime threshold (fps) we judge against.
const RT: f64 = 30.0;

struct Src {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

/// Synthesize one frame: a window panned by `(dx, dy)` over a smooth analytic
/// field with a little high-frequency texture and pseudo-noise. Smooth content
/// plus pure translation is exactly what inter prediction should exploit, while
/// the texture keeps the intra path honest.
fn gen_frame(k: usize) -> Src {
    let dx = (k * 3) as f64;
    let dy = (k * 2) as f64;
    let cw = W / 2;
    let ch = H / 2;
    let lum = |x: f64, y: f64| -> u8 {
        let base = 128.0 + 46.0 * (0.012 * x).sin() * (0.015 * y).cos();
        let tex = 26.0 * (0.21 * (x + y)).sin() + 12.0 * (0.37 * x - 0.11 * y).cos();
        let noise = (((x as i64 * 1103515245 + y as i64 * 12345) >> 8) & 7) as f64 - 3.5;
        (base + tex + noise).clamp(0.0, 255.0) as u8
    };
    let chr = |x: f64, y: f64, o: f64| -> u8 {
        (128.0 + 30.0 * (0.02 * x + o).sin() * (0.018 * y).cos()).clamp(0.0, 255.0) as u8
    };
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            y[j * W + i] = lum(i as f64 + dx, j as f64 + dy);
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = chr(i as f64 + dx / 2.0, j as f64 + dy / 2.0, 0.0);
            v[j * cw + i] = chr(i as f64 + dx / 2.0, j as f64 + dy / 2.0, 1.7);
        }
    }
    Src { y, u, v }
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

/// Auto-scaled timing of `run`, returning the median wall time in ms.
fn time_it(mut run: impl FnMut()) -> f64 {
    let mut times = Vec::new();
    let start = Instant::now();
    let mut reps = 0u32;
    loop {
        let t = Instant::now();
        run();
        times.push(t.elapsed().as_secs_f64() * 1e3);
        reps += 1;
        if reps >= MAX_REPS || (reps >= MIN_REPS && start.elapsed().as_secs_f64() >= TARGET_SECS) {
            break;
        }
    }
    median(&mut times)
}

struct Row {
    label: &'static str,
    qp: u8,
    ms_total: f64,
    au_bytes: f64,
}

impl Row {
    fn print(&self, n: usize, total_mb: f64) {
        let ms_f = self.ms_total / n as f64;
        let fps = n as f64 / (self.ms_total / 1e3);
        let mbs = total_mb / (self.ms_total / 1e3);
        println!(
            "{:<22} {:>3} {:>10.3} {:>8.1} {:>8.1} {:>9.2} {:>12.0}",
            self.label,
            self.qp,
            ms_f,
            fps,
            mbs,
            fps / RT,
            self.au_bytes,
        );
    }
}

fn main() {
    let frames: Vec<Src> = (0..N_FRAMES).map(gen_frame).collect();
    let n = frames.len();
    let total_mb = n as f64 * (W * H * 3 / 2) as f64 / 1.0e6;

    #[cfg(feature = "threads")]
    let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    #[cfg(feature = "threads")]
    let slices = cores as u32;

    println!("machine: arm64   build: --release (opt-level=3, lto=thin, codegen-units=1)");
    #[cfg(feature = "simd")]
    println!("encoder DSP: SIMD (SAD/MC NEON)");
    #[cfg(not(feature = "simd"))]
    println!("encoder DSP: scalar");
    #[cfg(feature = "threads")]
    println!("threads: ON   available_parallelism = {cores}   IPPP slices/frame = {slices}");
    #[cfg(not(feature = "threads"))]
    println!("threads: OFF (build with --features threads for the multithreaded rows)");
    println!("source: synthetic {W}x{H}, {n} frames  |  MB/s = source I420 throughput  |  xRT = fps / {RT:.0}\n");
    println!(
        "{:<22} {:>3} {:>10} {:>8} {:>8} {:>9} {:>12}",
        "MODE", "QP", "ms/frame", "fps", "MB/s", "xRT-30", "avg AU B"
    );
    println!("{}", "-".repeat(78));

    for &qp in &[26u8, 32u8] {
        // ---- All-intra, single-threaded (1 slice). ----
        let au = {
            let mut e = Encoder::new(W as u32, H as u32, qp).unwrap();
            let mut b = 0usize;
            for (k, f) in frames.iter().enumerate() {
                if k > 0 {
                    e.force_idr();
                }
                b += e.encode_frame(&f.y, W, &f.u, &f.v, W / 2).len();
            }
            b as f64 / n as f64
        };
        let ms = time_it(|| {
            let mut e = Encoder::new(W as u32, H as u32, qp).unwrap();
            for (k, f) in frames.iter().enumerate() {
                if k > 0 {
                    e.force_idr();
                }
                let _ = e.encode_frame(&f.y, W, &f.u, &f.v, W / 2);
            }
        });
        Row { label: "INTRA  1T/1slice", qp, ms_total: ms, au_bytes: au }.print(n, total_mb);

        // ---- IPPP, single-threaded (1 slice). ----
        let au = {
            let mut e = Encoder::new(W as u32, H as u32, qp).unwrap();
            let mut b = 0usize;
            for f in &frames {
                b += e.encode_frame(&f.y, W, &f.u, &f.v, W / 2).len();
            }
            b as f64 / n as f64
        };
        let ms = time_it(|| {
            let mut e = Encoder::new(W as u32, H as u32, qp).unwrap();
            for f in &frames {
                let _ = e.encode_frame(&f.y, W, &f.u, &f.v, W / 2);
            }
        });
        Row { label: "IPPP   1T/1slice", qp, ms_total: ms, au_bytes: au }.print(n, total_mb);

        #[cfg(feature = "threads")]
        {
            use h264::encoder::FrameInput;
            // ---- All-intra, frame-parallel. ----
            let inputs: Vec<FrameInput> =
                frames.iter().map(|f| FrameInput { y: &f.y, y_stride: W, u: &f.u, v: &f.v, c_stride: W / 2 }).collect();
            let au = {
                let e = Encoder::new(W as u32, H as u32, qp).unwrap();
                let aus = e.encode_frames_parallel(&inputs);
                aus.iter().map(|a| a.len()).sum::<usize>() as f64 / n as f64
            };
            let ms = time_it(|| {
                let e = Encoder::new(W as u32, H as u32, qp).unwrap();
                let _ = e.encode_frames_parallel(&inputs);
            });
            Row { label: "INTRA  NT frame-par", qp, ms_total: ms, au_bytes: au }.print(n, total_mb);

            // ---- IPPP, slice-parallel (S slices/frame). ----
            let au = {
                let mut e = Encoder::new_with_slices(W as u32, H as u32, qp, slices).unwrap();
                let mut b = 0usize;
                for f in &frames {
                    b += e.encode_frame_parallel(&f.y, W, &f.u, &f.v, W / 2).len();
                }
                b as f64 / n as f64
            };
            let ms = time_it(|| {
                let mut e = Encoder::new_with_slices(W as u32, H as u32, qp, slices).unwrap();
                for f in &frames {
                    let _ = e.encode_frame_parallel(&f.y, W, &f.u, &f.v, W / 2);
                }
            });
            Row { label: "IPPP   NT slice-par", qp, ms_total: ms, au_bytes: au }.print(n, total_mb);
        }
        println!();
    }
}
