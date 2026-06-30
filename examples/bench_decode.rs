//! Decode-throughput benchmark: our pure-Rust (scalar) decoder vs the C
//! `h264dec` (OpenH264, ARM NEON asm) on a few streams we decode bit-exact.
//!
//! For each stream we decode it repeatedly (auto-scaled so the total wall time
//! is at least ~1.5 s, with a floor on the repetition count) and report the
//! median time, then derive:
//!   - ms/frame
//!   - frames/s
//!   - MB/s  (decoded I420 output throughput = frames * w * h * 3/2 bytes)
//!
//! The C binary is timed the same way via std::process + Instant (its number
//! therefore includes process spawn and the YUV file write — a small handicap
//! that is called out in the report).
//!
//! Env overrides:
//!   H264_CORPUS_DIR  corpus directory (default: the scratch openh264/res dir)
//!   H264_ORACLE      path to the C h264dec binary
//!
//! Usage: `cargo run --release --example bench_decode`

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use h264::decoder::decode_stream;

const DEFAULT_CORPUS_DIR: &str = "/private/tmp/claude-501/-Users-alfredmathew-code-experiments-h264/204f99a2-1e9c-4cb7-9281-a5cdb434a08d/scratchpad/openh264/res";
const DEFAULT_ORACLE: &str = "/private/tmp/claude-501/-Users-alfredmathew-code-experiments-h264/204f99a2-1e9c-4cb7-9281-a5cdb434a08d/scratchpad/openh264/h264dec";

/// Streams to benchmark (all decode bit-exact vs the oracle).
const STREAMS: &[&str] = &[
    "BANM_MW_D.264",          // QCIF baseline CAVLC I+P
    "BA1_FT_C.264",           // QCIF baseline CAVLC, long
    "test_cif_I_CABAC_slice.264", // CIF Main CABAC, all-I
    "test_cif_P_CABAC_slice.264", // CIF Main CABAC, I+P
];

const MIN_REPS: u32 = 5;
const MAX_REPS: u32 = 200;
const TARGET_SECS: f64 = 1.5;

struct Stat {
    median_ms: f64,
    frames: usize,
    width: usize,
    height: usize,
    reps: u32,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn bench_ours(annexb: &[u8]) -> Stat {
    // Warm-up + dimensions.
    let frames0 = decode_stream(annexb).expect("decode");
    let frames = frames0.len();
    let (width, height) = frames0
        .first()
        .map(|p| (p.width, p.height))
        .unwrap_or((0, 0));
    drop(frames0);

    let mut times = Vec::new();
    let start = Instant::now();
    let mut reps = 0u32;
    loop {
        let t = Instant::now();
        let f = decode_stream(annexb).expect("decode");
        std::hint::black_box(&f);
        times.push(t.elapsed().as_secs_f64() * 1e3);
        reps += 1;
        if reps >= MAX_REPS {
            break;
        }
        if reps >= MIN_REPS && start.elapsed().as_secs_f64() >= TARGET_SECS {
            break;
        }
    }
    Stat {
        median_ms: median(&mut times),
        frames,
        width,
        height,
        reps,
    }
}

fn bench_c(oracle: &str, stream: &Path, out_yuv: &Path) -> (f64, u32) {
    let mut times = Vec::new();
    let start = Instant::now();
    let mut reps = 0u32;
    loop {
        let _ = fs::remove_file(out_yuv);
        let t = Instant::now();
        let status = Command::new(oracle)
            .arg(stream)
            .arg(out_yuv)
            .output()
            .expect("spawn oracle");
        assert!(status.status.success(), "oracle failed on {stream:?}");
        times.push(t.elapsed().as_secs_f64() * 1e3);
        reps += 1;
        if reps >= MAX_REPS {
            break;
        }
        if reps >= MIN_REPS && start.elapsed().as_secs_f64() >= TARGET_SECS {
            break;
        }
    }
    let _ = fs::remove_file(out_yuv);
    (median(&mut times), reps)
}

fn main() {
    let corpus_dir =
        std::env::var("H264_CORPUS_DIR").unwrap_or_else(|_| DEFAULT_CORPUS_DIR.to_string());
    let oracle = std::env::var("H264_ORACLE").unwrap_or_else(|_| DEFAULT_ORACLE.to_string());
    let corpus = Path::new(&corpus_dir);
    let tmp = std::env::temp_dir();

    let arch = Command::new("uname")
        .arg("-m")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let backend = if cfg!(feature = "simd") { "SIMD (NEON)" } else { "SCALAR" };
    println!("machine: {arch}   build: --release (opt-level=3, lto=thin, codegen-units=1)");
    println!("our decoder: pure-Rust {backend}   |   C h264dec: OpenH264 ARM NEON asm\n");

    println!(
        "{:<30} {:>6} {:>7} | {:>9} {:>8} {:>8} | {:>9} {:>8} {:>8} | {:>8}",
        "STREAM", "WxH", "FRAMES", "RUST ms/f", "fps", "MB/s", "C ms/f", "fps", "MB/s", "C×"
    );
    println!("{}", "-".repeat(124));

    for name in STREAMS {
        let stream = corpus.join(name);
        let annexb = match fs::read(&stream) {
            Ok(b) => b,
            Err(e) => {
                println!("{name:<30} SKIP (read: {e})");
                continue;
            }
        };

        let ours = bench_ours(&annexb);
        let out_yuv = tmp.join(format!("bench_{name}.yuv"));
        let (c_ms, c_reps) = bench_c(&oracle, &stream, &out_yuv);

        let frames = ours.frames as f64;
        let bytes_per_frame = (ours.width * ours.height * 3 / 2) as f64;
        let total_mb = frames * bytes_per_frame / 1.0e6;

        let our_ms_f = ours.median_ms / frames;
        let our_fps = frames / (ours.median_ms / 1e3);
        let our_mbs = total_mb / (ours.median_ms / 1e3);

        let c_ms_f = c_ms / frames;
        let c_fps = frames / (c_ms / 1e3);
        let c_mbs = total_mb / (c_ms / 1e3);

        let ratio = ours.median_ms / c_ms;

        println!(
            "{:<30} {:>6} {:>7} | {:>9.3} {:>8.1} {:>8.1} | {:>9.3} {:>8.1} {:>8.1} | {:>7.2}×",
            name,
            format!("{}x{}", ours.width, ours.height),
            ours.frames,
            our_ms_f,
            our_fps,
            our_mbs,
            c_ms_f,
            c_fps,
            c_mbs,
            ratio
        );
        eprintln!(
            "  [{name}] rust median {:.1} ms over {} reps; C median {:.1} ms over {} reps",
            ours.median_ms, ours.reps, c_ms, c_reps
        );
    }

    println!("\nMB/s = decoded I420 output throughput (frames * W * H * 3/2 bytes).");
    println!("C× = our_time / C_time (how many times faster the C NEON build is).");
    println!("Note: the C number includes process spawn + YUV file write; ours is in-process decode only.");
}
