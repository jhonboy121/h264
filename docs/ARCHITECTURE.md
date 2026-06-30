# Architecture & Design Decisions

## Crate shape
- Single crate `h264` (lib). Rationale: simplest for shared DSP, unified tests,
  feature gating. Decoder/encoder split by module + cargo features, not crates.
- `#![no_std]` + `extern crate alloc`. `std` feature (default ON) enables file
  helpers and the test oracle harness. no_std+alloc keeps wasm/iOS/Android clean
  and forces explicit allocation (aligns with "avoid global allocations").

## Features
- `default = ["std", "decoder", "encoder"]`
- `std`      — std lib (file I/O test helpers, threading).
- `decoder`  — decode path.
- `encoder`  — encode path.
- `simd`     — opt-in SIMD (NEON / wasm128 / SSE-AVX) w/ runtime detect (P10).
- `threads`  — multithreaded slice decode/encode (P9; requires std).

## "Avoid global allocations" — concretely
- No `static mut`, no `lazy_static`/`OnceCell` holding mutable codec state.
- No per-frame/per-MB heap churn: scratch buffers live in context structs,
  allocated once at `Decoder::new`/`Encoder::new` (sized from SPS) and reused.
- Static read-only tables are `const`/`static` arrays (fine — not allocations).
- Public API lets callers supply output buffers where practical
  (`write_rgb8(&mut [u8])`), mirroring openh264-rs.

## Types & conventions
- Pixels: `u8` sample planes. Intermediate transform math: `i32`.
- Planes stored as `Plane { data: Vec<u8>, stride, width, height }` with a small
  border for MC/deblock (mirrors C `iStride` + padding).
- Errors: `enum DecodeError`/`EncodeError` + `Result`. No panics on bad bitstream
  (return Err); panics only for internal invariants.
- Endianness: bitstream is big-endian bit order (H.264 RBSP). Reader handles
  emulation-prevention-byte (0x000003) removal.
- Faithful port: keep C algorithm structure & variable intent; translate idioms
  to safe Rust. Document any deliberate divergence inline with `// PORT:`.

## DSP portability
- Every kernel has a scalar reference (always compiled). SIMD variants are
  `#[cfg(...)]` behind `simd` and selected at runtime via `dsp::cpu`. Scalar is
  the conformance baseline and the wasm default.

## Module layout (target)
```
src/
  lib.rs            // crate root, re-exports, feature wiring
  api.rs            // Rusty Decoder/Encoder facades
  error.rs          // error enums
  bits/{reader,writer,golomb}.rs
  dsp/{transform,intra_pred,mc,deblock,sad,copy,expand,tables,cpu}.rs
  decoder/{mod,nal,params,slice_header,cavlc,cavlc_tables,cabac,cabac_mb,
           mv_pred,recon,decode_slice,ref_pic,dpb,fmo,error_conceal,tables}.rs
  encoder/{mod,cavlc_enc,cabac_enc,mode_decision,motion_est,mv_pred,
           rate_control,ref_list,encode_slice,encode_mb,slice_segment,
           nal_encap,paraset,picture}.rs
  formats/{mod,yuv,rgb}.rs
  processing/...    // P8
  util/{mem,threading}.rs
```

## Testing
- Unit tests inline (`#[cfg(test)]`) next to kernels, porting `DecUT_*`/`EncUT_*`.
- Integration tests in `tests/` for conformance decode + encode round-trip.
- Oracle: optional, behind `H264_ORACLE` env / ignored tests, builds C `h264dec`
  via meson when available; otherwise self-referential math checks.
- Golden vectors copied from the gtest fixtures where they embed literal arrays.
