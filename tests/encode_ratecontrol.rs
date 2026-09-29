//! Rate-controlled encoding ([`Encoder::with_config`] + [`Encoder::encode`]):
//! the output bitrate tracks the target (also across a mid-stream
//! `set_bitrate`, without an IDR), and every stream decodes through this
//! crate's decoder to exactly the encoder's own reconstruction, including the
//! frame-cropped 1080 / 540 / 360 heights.

use h264::nal::{self, nal_type};
use h264::{
    Decoder, EncodedFrame, Encoder, EncoderConfig, I420, YuvRef, nal_units, scale_i420,
    sps_dimensions,
};

/// Deterministic LCG (PCG constants), so the sequences are reproducible.
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

/// Value noise: random lattice values every `cell` samples, bilinearly
/// interpolated, `amp` peak-to-peak.
fn add_octave(plane: &mut [i32], w: usize, h: usize, cell: usize, amp: i32, rng: &mut Rng) {
    let gw = w / cell + 2;
    let gh = h / cell + 2;
    let grid: Vec<i32> = (0..gw * gh).map(|_| (rng.next() % (amp as u32 + 1)) as i32 - amp / 2).collect();
    for y in 0..h {
        let (gy, fy) = (y / cell, (y % cell) as i32);
        for x in 0..w {
            let (gx, fx) = (x / cell, (x % cell) as i32);
            let c = cell as i32;
            let a = grid[gy * gw + gx];
            let b = grid[gy * gw + gx + 1];
            let d = grid[(gy + 1) * gw + gx];
            let e = grid[(gy + 1) * gw + gx + 1];
            let top = a * (c - fx) + b * fx;
            let bot = d * (c - fx) + e * fx;
            plane[y * w + x] += (top * (c - fy) + bot * fy) / (c * c);
        }
    }
}

fn texture(w: usize, h: usize, octaves: &[(usize, i32)], base: i32, rng: &mut Rng) -> Vec<u8> {
    let mut acc = vec![base; w * h];
    for &(cell, amp) in octaves {
        add_octave(&mut acc, w, h, cell, amp, rng);
    }
    acc.into_iter().map(|v| v.clamp(0, 255) as u8).collect()
}

/// A camera-like sequence: a textured backdrop larger than the frame, panned
/// and shaken, with a textured object moving across it and a little sensor
/// noise on every frame.
struct Scene {
    w: usize,
    h: usize,
    bw: usize,
    bh: usize,
    bg_y: Vec<u8>,
    bg_u: Vec<u8>,
    bg_v: Vec<u8>,
    obj: Vec<u8>,
    ow: usize,
    oh: usize,
    rng: Rng,
}

impl Scene {
    fn new(w: usize, h: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let (bw, bh) = (w + w / 2, h + h / 2);
        let luma = [(w / 8, 90), (w / 40, 40), (8, 24), (2, 10)];
        let bg_y = texture(bw, bh, &luma, 128, &mut rng);
        let chroma = [(w / 16, 50), (w / 80, 16)];
        let bg_u = texture(bw / 2, bh / 2, &chroma, 128, &mut rng);
        let bg_v = texture(bw / 2, bh / 2, &chroma, 128, &mut rng);
        let (ow, oh) = (w / 4, h / 3);
        let obj = texture(ow, oh, &[(ow / 4, 120), (4, 40)], 160, &mut rng);
        Scene { w, h, bw, bh, bg_y, bg_u, bg_v, obj, ow, oh, rng }
    }

    fn frame(&mut self, k: usize) -> I420 {
        let (w, h) = (self.w, self.h);
        let span_x = self.bw - w;
        let span_y = self.bh - h;
        // Slow pan that bounces, plus a small shake.
        let tri = |t: usize, span: usize| {
            let p = t % (2 * span);
            if p < span { p } else { 2 * span - p }
        };
        let shake = (self.rng.next() % 3) as usize;
        let ox = (tri(k * 2, span_x - 2) + shake).min(span_x);
        let oy = (tri(k, span_y - 2) + shake).min(span_y);
        let mut f = I420::new(w as u32, h as u32);
        for y in 0..h {
            let src = &self.bg_y[(oy + y) * self.bw + ox..(oy + y) * self.bw + ox + w];
            f.y[y * w..y * w + w].copy_from_slice(src);
        }
        let (cw, ch) = (w / 2, h / 2);
        for y in 0..ch {
            let s = (oy / 2 + y) * (self.bw / 2) + ox / 2;
            f.u[y * cw..y * cw + cw].copy_from_slice(&self.bg_u[s..s + cw]);
            f.v[y * cw..y * cw + cw].copy_from_slice(&self.bg_v[s..s + cw]);
        }
        // The object crosses the frame left to right.
        let px = (k * 7) % (w - self.ow);
        let py = h / 3 + tri(k * 3, h / 3) / 2;
        for y in 0..self.oh.min(h - py) {
            let d = (py + y) * w + px;
            f.y[d..d + self.ow].copy_from_slice(&self.obj[y * self.ow..y * self.ow + self.ow]);
        }
        for p in &mut f.y {
            let n = (self.rng.next() % 5) as i32 - 2;
            *p = (*p as i32 + n).clamp(0, 255) as u8;
        }
        f
    }
}

fn config(width: u32, height: u32, fps: u32, kbps: u32, slices: u32) -> EncoderConfig {
    EncoderConfig {
        width,
        height,
        fps,
        bitrate_kbps: kbps,
        keyframe_interval: None,
        slices,
    }
}

/// Encode `seconds` of the scene; returns the encoded frames.
fn encode_seconds(enc: &mut Encoder, scene: &mut Scene, start: usize, frames: usize) -> Vec<EncodedFrame> {
    (start..start + frames)
        .map(|k| {
            let f = scene.frame(k);
            enc.encode(&f.as_ref()).expect("encode")
        })
        .collect()
}

/// kbit/s over `frames[lo..lo + len]`.
fn window_kbps(frames: &[EncodedFrame], lo: usize, len: usize, fps: u32) -> f64 {
    let bits: usize = frames[lo..lo + len].iter().map(|f| f.data.len() * 8).sum();
    bits as f64 * fps as f64 / len as f64 / 1000.0
}

#[test]
fn bitrate_tracks_target_over_2s_windows() {
    const SECONDS: usize = 5;
    const TOLERANCE: f64 = 0.15;
    let cases = [
        (640, 360, 15, 350),
        (960, 540, 24, 900),
        (1280, 720, 30, 2000),
        (1920, 1080, 30, 4000),
        (1920, 1080, 60, 6000),
    ];
    for (i, &(w, h, fps, kbps)) in cases.iter().enumerate() {
        let mut scene = Scene::new(w, h, 7 + i as u64);
        let mut enc = Encoder::with_config(config(w as u32, h as u32, fps, kbps, 8)).expect("config");
        let n = SECONDS * fps as usize;
        let out = encode_seconds(&mut enc, &mut scene, 0, n);
        assert!(out[0].keyframe);
        assert!(out[1..].iter().all(|f| !f.keyframe), "{w}x{h}: unexpected IDR");
        let win = 2 * fps as usize;
        let (mut lo_kbps, mut hi_kbps) = (f64::MAX, 0.0f64);
        for start in fps as usize..=n - win {
            let got = window_kbps(&out, start, win, fps);
            lo_kbps = lo_kbps.min(got);
            hi_kbps = hi_kbps.max(got);
        }
        let qps: Vec<u8> = out.iter().map(|f| f.qp).collect();
        eprintln!(
            "{w}x{h}@{fps} {kbps} kbps: 2 s windows {lo_kbps:.0}..{hi_kbps:.0} kbps, qp {}..{}",
            qps.iter().min().copied().unwrap_or(0),
            qps.iter().max().copied().unwrap_or(0)
        );
        let target = kbps as f64;
        assert!(
            lo_kbps >= target * (1.0 - TOLERANCE) && hi_kbps <= target * (1.0 + TOLERANCE),
            "{w}x{h}@{fps}: windows {lo_kbps:.0}..{hi_kbps:.0} kbps vs target {kbps}"
        );
    }
}

/// A re-encoded conformance clip (CIF, scaled up to 640x360 and looped) tracks
/// its target too.
#[test]
fn bitrate_tracks_target_on_decoded_clip() {
    const CLIP: &[u8] = include_bytes!("fixtures/BA1_FT_C.264");
    const FPS: u32 = 30;
    const KBPS: u32 = 600;
    const SECONDS: usize = 6;
    let frames = Decoder::new().decode_all(CLIP).expect("decode clip");
    assert!(!frames.is_empty());
    let mut scaled = I420::new(640, 360);
    let mut enc = Encoder::with_config(config(640, 360, FPS, KBPS, 4)).expect("config");
    let n = SECONDS * FPS as usize;
    let out: Vec<EncodedFrame> = (0..n)
        .map(|k| {
            // Loop back and forth so the clip has no hard cut.
            let len = frames.len();
            let p = k % (2 * len - 2).max(1);
            let idx = if p < len { p } else { 2 * len - 2 - p };
            scale_i420(&frames[idx].yuv(), &mut scaled);
            enc.encode(&scaled.as_ref()).expect("encode")
        })
        .collect();
    let win = 2 * FPS as usize;
    for start in FPS as usize..=n - win {
        let got = window_kbps(&out, start, win, FPS);
        assert!(
            (got - KBPS as f64).abs() <= KBPS as f64 * 0.15,
            "clip window at frame {start}: {got:.0} kbps vs {KBPS}"
        );
    }
}

#[test]
fn set_bitrate_converges_without_idr() {
    const FPS: u32 = 30;
    const TOLERANCE: f64 = 0.20;
    for &(from, to) in &[(4000u32, 1000u32), (1000, 3000)] {
        let mut scene = Scene::new(1280, 720, 99);
        let mut enc = Encoder::with_config(config(1280, 720, FPS, from, 8)).expect("config");
        let sec = FPS as usize;
        let mut out = encode_seconds(&mut enc, &mut scene, 0, 3 * sec);
        enc.set_bitrate(to);
        out.extend(encode_seconds(&mut enc, &mut scene, 3 * sec, 3 * sec));
        assert!(out[1..].iter().all(|f| !f.keyframe), "set_bitrate emitted an IDR");
        // The second second after the change.
        let got = window_kbps(&out, 4 * sec, sec, FPS);
        eprintln!("{from} -> {to} kbps: second 2 after the change at {got:.0} kbps");
        assert!(
            (got - to as f64).abs() <= to as f64 * TOLERANCE,
            "{from} -> {to}: {got:.0} kbps in the second second"
        );
    }
}

/// Copy the visible planes of a picture (for comparing reconstructions).
fn planes(p: &YuvRef<'_>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (p.width as usize, p.height as usize);
    let (cw, ch) = (p.chroma_width() as usize, p.chroma_height() as usize);
    let pack = |plane: &[u8], stride: usize, pw: usize, ph: usize| {
        (0..ph).flat_map(|r| plane[r * stride..r * stride + pw].iter().copied()).collect::<Vec<u8>>()
    };
    (pack(p.y, p.y_stride, w, h), pack(p.u, p.c_stride, cw, ch), pack(p.v, p.c_stride, cw, ch))
}

#[test]
fn rc_streams_decode_to_the_reconstruction() {
    const FRAMES: usize = 12;
    let cases = [(1920u32, 1080u32, 8u32), (640, 360, 1), (640, 360, 5), (960, 540, 3)];
    for &(w, h, slices) in &cases {
        let mut scene = Scene::new(w as usize, h as usize, 3);
        let mut cfg = config(w, h, 30, 1500, slices);
        cfg.keyframe_interval = Some(5);
        let mut enc = Encoder::with_config(cfg).expect("config");
        let mut stream = Vec::new();
        let mut recon = Vec::new();
        let mut since_idr = 0;
        for k in 0..FRAMES {
            // Back-to-back IDRs (7, 8) must alternate idr_pic_id.
            if k == 7 || k == 8 {
                enc.force_idr();
                since_idr = 0;
            }
            if k == 3 {
                enc.set_bitrate(400);
            }
            let f = scene.frame(k);
            let au = enc.encode(&f.as_ref()).expect("encode");
            // Keyframes (the interval and the forced one) lead with SPS + PPS.
            let types: Vec<u8> = nal_units(&au.data).filter_map(nal_type).collect();
            let want_key = since_idr % 5 == 0;
            since_idr += 1;
            assert_eq!(au.keyframe, want_key, "{w}x{h} frame {k}");
            if au.keyframe {
                assert_eq!(&types[..2], &[nal::SPS, nal::PPS], "{w}x{h} frame {k}");
                assert!(types[2..].iter().all(|&t| t == nal::IDR));
                let sps = nal_units(&au.data).next().expect("sps");
                assert_eq!(sps_dimensions(sps), Some((w, h)));
            } else {
                assert!(types.iter().all(|&t| t == 1), "{w}x{h} frame {k}: {types:?}");
            }
            assert_eq!(types.iter().filter(|&&t| t != nal::SPS && t != nal::PPS).count(), slices as usize);
            stream.extend_from_slice(&au.data);
            recon.push(planes(&enc.reconstruction().expect("recon")));
        }
        let decoded = Decoder::new().decode_all(&stream).expect("decode");
        assert_eq!(decoded.len(), FRAMES, "{w}x{h}");
        for (k, (f, want)) in decoded.iter().zip(&recon).enumerate() {
            let got = f.yuv();
            assert_eq!((got.width, got.height), (w, h));
            assert!(planes(&got) == *want, "{w}x{h} slices={slices}: frame {k} differs from the reconstruction");
        }
    }
}

#[test]
fn with_config_rejects_bad_input() {
    assert!(Encoder::with_config(config(641, 360, 30, 500, 1)).is_err());
    assert!(Encoder::with_config(config(640, 360, 0, 500, 1)).is_err());
    assert!(Encoder::with_config(config(640, 360, 30, 0, 1)).is_err());
    let mut enc = Encoder::with_config(config(640, 360, 30, 500, 1)).expect("config");
    let small = I420::new(320, 180);
    assert!(enc.encode(&small.as_ref()).is_err());
    let f = I420::new(640, 360);
    let mut short = f.as_ref();
    short.y = &f.y[..100];
    assert!(enc.encode(&short).is_err());
}
