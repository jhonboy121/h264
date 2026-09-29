# h264

A pure-Rust H.264 decoder and encoder, ported from [Cisco OpenH264](https://github.com/cisco/openh264).

> **Disclaimer.** This is an AI-assisted port of the OpenH264 C code. It is
> **not formally verified** and has had no independent review. It exists for
> my own personal projects; use it elsewhere at your own risk.

## Features

- **Decoder:** Baseline, Main and High profile (CAVLC + CABAC, I/P/B slices,
  8×8 transform, deblocking, multi-ref). 49/54 conformance streams decode
  bit-exact against the C reference.
- **Encoder:** Baseline IPPP with CAVLC, fixed QP or frame-level rate control,
  multi-slice, and a MediaCodec-style live API (`set_bitrate`, `force_idr`).
- **Extras:** I420 downscaler (`scale_i420`) and Annex-B/NAL helpers.
- `no_std` + `alloc` and no dependencies by default.

Not supported: SVC, FMO, error concealment, and CABAC/B-frame/multi-ref encoding.
See [`docs/STATUS.md`](docs/STATUS.md).

## Usage

```toml
[dependencies]
h264 = { git = "https://github.com/jhonboy121/h264" }
```

```rust
use h264::{Decoder, Encoder, EncoderConfig, I420, YUVSource};

// Decode
let frames = Decoder::new().decode_all(&annexb)?;
let mut rgb = vec![0u8; frames[0].rgb8_len()];
frames[0].write_rgb8(&mut rgb);

// Encode
let mut enc = Encoder::with_config(EncoderConfig {
    width: 1280, height: 720, fps: 30, bitrate_kbps: 2000,
    keyframe_interval: Some(60), slices: 4,
})?;
let au = enc.encode(&I420::new(1280, 720).as_ref())?; // au.data is Annex-B
```

## Feature flags

| Flag | Default | Effect |
|---|---|---|
| `std` | yes | std helpers |
| `decoder`, `encoder` | yes | enable each half |
| `threads` | no | slice/frame-parallel encoding |
| `simd` | no | NEON / wasm simd128 kernels |
| `yuv-convert` | no | colorimetry-aware YUV→RGB via the `yuv` crate |

## Testing

```sh
cargo test --release
cargo test --release --features threads,simd
```

## License

BSD-2-Clause, the same as OpenH264. See [LICENSE](LICENSE). H.264 is
patent-encumbered, and this license grants no patent rights.
