//! Conformance harness: decode every stream in an H.264 corpus with our pure-Rust
//! decoder and compare, byte-for-byte and by MD5, against the C reference
//! (`h264dec` from OpenH264).
//!
//! For each stream we:
//!   a. run the C oracle to produce reference I420 YUV (and time it),
//!   b. decode with `h264::decoder::decode_stream` (timed, with frame count),
//!   c. classify the result:
//!        - `Unsupported(feature)` from our decoder -> UNSUPPORTED(feature)
//!        - any other `Err`                         -> ERROR
//!        - success + bytes identical to oracle      -> BITEXACT
//!        - success + bytes differ                   -> MISMATCH(frame, bytes)
//!
//! A per-stream table and a summary are printed.
//!
//! Env overrides:
//!   H264_CORPUS_DIR  corpus directory (default: the scratch openh264/res dir)
//!   H264_ORACLE      path to the C h264dec binary
//!
//! Usage: `cargo run --release --example conformance`

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use h264::decoder::decode_stream;
use h264::decoder::picture::Picture;
use h264::DecodeError;

const DEFAULT_CORPUS_DIR: &str = "/private/tmp/claude-501/-Users-alfredmathew-code-experiments-h264/204f99a2-1e9c-4cb7-9281-a5cdb434a08d/scratchpad/openh264/res";
const DEFAULT_ORACLE: &str = "/private/tmp/claude-501/-Users-alfredmathew-code-experiments-h264/204f99a2-1e9c-4cb7-9281-a5cdb434a08d/scratchpad/openh264/h264dec";

/// Extract the visible (coded-size) I420 of one picture into `out`.
fn append_visible(out: &mut Vec<u8>, pic: &Picture) {
    let yo = pic.luma_origin();
    for row in 0..pic.height {
        let start = yo + row * pic.luma_stride;
        out.extend_from_slice(&pic.y[start..start + pic.width]);
    }
    let co = pic.chroma_origin();
    let (cw, ch) = (pic.width / 2, pic.height / 2);
    for row in 0..ch {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.u[start..start + cw]);
    }
    for row in 0..ch {
        let start = co + row * pic.chroma_stride;
        out.extend_from_slice(&pic.v[start..start + cw]);
    }
}

enum Outcome {
    BitExact,
    Mismatch {
        first_diff_frame: i64,
        diff_bytes: usize,
        note: String,
    },
    Unsupported(String),
    Error(String),
    /// Our decoder succeeded but the oracle could not produce a reference.
    OracleError(String),
}

struct Row {
    name: String,
    outcome: Outcome,
    frames: usize,
    our_ms: f64,
    c_ms: f64,
}

fn main() {
    let corpus_dir =
        std::env::var("H264_CORPUS_DIR").unwrap_or_else(|_| DEFAULT_CORPUS_DIR.to_string());
    let oracle = std::env::var("H264_ORACLE").unwrap_or_else(|_| DEFAULT_ORACLE.to_string());

    let corpus = Path::new(&corpus_dir);
    if !corpus.is_dir() {
        eprintln!("corpus dir not found: {corpus_dir}");
        std::process::exit(1);
    }
    if !Path::new(&oracle).exists() {
        eprintln!("oracle binary not found: {oracle}");
        std::process::exit(1);
    }

    // Gather *.264 and *.jsv streams, sorted by name.
    let mut streams: Vec<PathBuf> = fs::read_dir(corpus)
        .expect("read corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(OsStr::to_str),
                Some("264") | Some("jsv")
            )
        })
        .collect();
    streams.sort();

    eprintln!(
        "corpus: {corpus_dir}\noracle: {oracle}\nstreams: {}\n",
        streams.len()
    );

    let tmp = std::env::temp_dir();
    let mut rows: Vec<Row> = Vec::new();

    for stream in &streams {
        let name = stream.file_name().unwrap().to_string_lossy().into_owned();
        eprintln!("[run] {name}");

        let annexb = match fs::read(stream) {
            Ok(b) => b,
            Err(e) => {
                rows.push(Row {
                    name,
                    outcome: Outcome::Error(format!("read input: {e}")),
                    frames: 0,
                    our_ms: 0.0,
                    c_ms: 0.0,
                });
                continue;
            }
        };

        // (a) Run the C oracle (timed wall-clock).
        let out_yuv = tmp.join(format!("conf_oracle_{name}.yuv"));
        let _ = fs::remove_file(&out_yuv);
        let t0 = Instant::now();
        let c_status = Command::new(&oracle).arg(stream).arg(&out_yuv).output();
        let c_ms = t0.elapsed().as_secs_f64() * 1e3;
        let oracle_ok = matches!(&c_status, Ok(o) if o.status.success()) && out_yuv.exists();

        // (b) Decode with our decoder (timed wall-clock + frame count).
        let t1 = Instant::now();
        let our_result = decode_stream(&annexb);
        let our_ms = t1.elapsed().as_secs_f64() * 1e3;

        let (outcome, frames) = match our_result {
            Err(DecodeError::Unsupported(feat)) => (Outcome::Unsupported(feat.to_string()), 0),
            Err(e) => (Outcome::Error(format!("{e}")), 0),
            Ok(frames) => {
                let n = frames.len();
                if !oracle_ok {
                    let msg = match &c_status {
                        Ok(o) => format!(
                            "exit {} {}",
                            o.status,
                            String::from_utf8_lossy(&o.stderr).trim()
                        ),
                        Err(e) => format!("spawn: {e}"),
                    };
                    (Outcome::OracleError(msg), n)
                } else {
                    let mut ours = Vec::new();
                    for pic in &frames {
                        append_visible(&mut ours, pic);
                    }
                    let reference = fs::read(&out_yuv).unwrap_or_default();
                    (classify(&ours, &reference, &frames), n)
                }
            }
        };

        rows.push(Row {
            name,
            outcome,
            frames,
            our_ms,
            c_ms,
        });
        let _ = fs::remove_file(&out_yuv);
    }

    print_report(&rows);
}

fn classify(ours: &[u8], reference: &[u8], frames: &[Picture]) -> Outcome {
    // Per-frame byte size of OUR output (coded visible I420). Assumes uniform
    // frame dimensions, which holds for the streams we currently decode.
    let frame_size = frames
        .first()
        .map(|p| p.width * p.height * 3 / 2)
        .unwrap_or(0);

    // Byte-for-byte (and, equivalently, MD5) identity check.
    if ours == reference {
        return Outcome::BitExact;
    }

    let mut note = String::new();
    if ours.len() != reference.len() {
        note = format!(
            "len ours={} ref={} (md5 ours={} ref={})",
            ours.len(),
            reference.len(),
            &md5_hex(ours)[..8],
            &md5_hex(reference)[..8],
        );
    }

    let n = ours.len().min(reference.len());
    let mut first_diff = -1i64;
    let mut diff_bytes = 0usize;
    for i in 0..n {
        if ours[i] != reference[i] {
            if first_diff < 0 {
                first_diff = i.checked_div(frame_size).unwrap_or(0) as i64;
            }
            diff_bytes += 1;
        }
    }
    if first_diff < 0 {
        // Common prefix matched; only the trailing length differs.
        first_diff = n.checked_div(frame_size).unwrap_or(0) as i64;
    }
    diff_bytes += ours.len().abs_diff(reference.len());

    Outcome::Mismatch {
        first_diff_frame: first_diff,
        diff_bytes,
        note,
    }
}

fn print_report(rows: &[Row]) {
    println!();
    println!(
        "{:<46} {:>7} {:>9} {:>9}  RESULT",
        "STREAM", "FRAMES", "OUR(ms)", "C(ms)"
    );
    println!("{}", "-".repeat(120));

    let mut n_bitexact = 0usize;
    let mut n_mismatch = 0usize;
    let mut n_error = 0usize;
    let mut n_oracle_err = 0usize;
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    let mut bitexact_frames = 0usize;
    let mut bitexact_names: Vec<String> = Vec::new();

    for r in rows {
        let result = match &r.outcome {
            Outcome::BitExact => {
                n_bitexact += 1;
                bitexact_frames += r.frames;
                bitexact_names.push(r.name.clone());
                "BITEXACT".to_string()
            }
            Outcome::Mismatch {
                first_diff_frame,
                diff_bytes,
                note,
            } => {
                n_mismatch += 1;
                let extra = if note.is_empty() {
                    String::new()
                } else {
                    format!(" [{note}]")
                };
                format!("MISMATCH(frame={first_diff_frame}, {diff_bytes} bytes){extra}")
            }
            Outcome::Unsupported(feat) => {
                *unsupported.entry(feat.clone()).or_insert(0) += 1;
                format!("UNSUPPORTED({feat})")
            }
            Outcome::Error(msg) => {
                n_error += 1;
                format!("ERROR({msg})")
            }
            Outcome::OracleError(msg) => {
                n_oracle_err += 1;
                format!("ORACLE-ERR({msg})")
            }
        };
        println!(
            "{:<46} {:>7} {:>9.1} {:>9.1}  {}",
            trunc(&r.name, 46),
            r.frames,
            r.our_ms,
            r.c_ms,
            result
        );
    }

    let n_unsupported: usize = unsupported.values().sum();
    println!();
    println!("================ SUMMARY ================");
    println!("total streams      : {}", rows.len());
    println!("BITEXACT           : {n_bitexact}");
    println!("UNSUPPORTED        : {n_unsupported}");
    for (feat, count) in &unsupported {
        println!("    - {feat:<34} {count}");
    }
    println!("MISMATCH           : {n_mismatch}");
    println!("ERROR              : {n_error}");
    if n_oracle_err > 0 {
        println!(
            "ORACLE-ERR         : {n_oracle_err} (our decode produced output but oracle failed)"
        );
    }
    println!("frames bit-exact   : {bitexact_frames}");
    println!();
    println!("bit-exact streams:");
    for name in &bitexact_names {
        println!("    {name}");
    }
}

fn trunc(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max - 1).collect();
        format!("{cut}…")
    }
}

// ----------------------------------------------------------------------------
// Compact MD5 (RFC 1321) — used to summarize differing outputs. Byte-for-byte
// equality is the actual conformance test; MD5 is only a human-readable digest.
// ----------------------------------------------------------------------------
fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f
                .wrapping_add(a)
                .wrapping_add(K[i])
                .wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = String::with_capacity(32);
    for v in [a0, b0, c0, d0] {
        for byte in v.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}
