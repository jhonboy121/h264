# Project Status — pure-Rust H.264 port

Final status of the OpenH264 → Rust port. Honest scope: what is ported and
conformant, the public API, build/test matrix, and an explicit deferred list.
See `TRACKER.md` for the phase-by-phase detail and `PERF_REPORT.md` for numbers.

- **Crate:** `h264`, `#![no_std]` + `extern crate alloc` (the `std` feature, on by
  default, adds file/test helpers). No global mutable state; codec objects own
  their scratch buffers and reuse them per frame.
- **Tracks upstream:** OpenH264 2.x. Kernels are faithful ports of the C `_c`
  reference functions; bit-exactness is validated against a C `h264dec` oracle.

---

## What's ported & conformant

### Decoder — baseline CAVLC + Main-profile CABAC, I + P

- NAL/Annex-B framing, SPS/PPS (+VUI) parse, slice headers.
- **CAVLC** and **CABAC** residual + MB-syntax decode (I and P slices).
- Intra prediction: I_4×4 (9 modes), I_16×16 (4 modes), chroma (4 modes).
- Inter: P_16×16 / 16×8 / 8×16 / 8×8 (+sub-partitions), P_Skip, MV prediction,
  luma 6-tap + quarter-pel and chroma bilinear motion compensation.
- Dequant + 4×4 inverse transform; luma-DC / chroma-DC paths.
- In-loop **deblocking** filter (boundary-strength derivation, luma+chroma).
- **DPB** with sliding-window short-term refs and **multi-reference** P lists;
  border expansion for sub-pel MC.
- **Conformance: 49/54 corpus streams BIT-EXACT** vs the C oracle (6164 frames),
  covering baseline CAVLC (I+P), Main CABAC (I+P, multi-ref, multi-slice), B-slices
  (CAVLC + CABAC, spatial + temporal direct), High-profile 8×8 transform, QCIF →
  1920×1080, and a 1700-frame stream. Results are **identical with `--features
  simd`**. The other 5 are cleanly classified: 1 MISMATCH (SVC base-layer-only
  `sps_subsetsps_bothVUI`) and 4 ERROR (corrupted error-resilience streams).

### Encoder — baseline intra + IPPP, CAVLC

- Fixed-QP baseline: SPS/PPS generation, Annex-B NAL encapsulation.
- Intra mode decision (SATD/SAD), forward 4×4 DCT + quant + DC Hadamard + scan,
  CAVLC + Exp-Golomb bit-writing, in-place reconstruction.
- P-frame inter: integer + sub-pel motion estimation (diamond + half/quarter-pel),
  inter mode decision (P_Skip / P_16×16 / intra), encoder-side MV prediction
  bit-identical to the decoder.
- **Round-trip validated** (encode → our own conformant decoder): single-frame
  intra Y-PSNR 41.16 dB @ QP26 / 37.27 dB @ QP32; IPPP holds quality with no drift
  and P access units ~12–16× smaller than the IDR. PSNR is monotonic in QP.
- **Multi-slice + multithreaded (P9):** each frame partitions into `S`
  independent slices (correct `first_mb_in_slice`, per-slice CAVLC, prediction
  reset at the boundary; deblocking off so slices are fully independent). The
  `threads` feature (std-only, `std::thread::scope`, zero deps) adds
  `encode_frames_parallel` (frame-parallel all-intra) and `encode_frame_parallel`
  (slice-parallel IPPP), **byte-identical to serial**. **Real-time 30 fps 1080p
  reached:** threaded IPPP **127 fps** (4.2× realtime-30), all-intra **300+ fps**
  (10×), vs 13.7 / 25 fps single-threaded.
- **Rate control (P16):** frame-level port of OpenH264's `RC_BITRATE_MODE`
  (virtual-GOP bit budget, linear R-Q model, `-3/+5` QP step, QP 12..=45, no frame
  skipping). Every 2 s window after the first second lands within ±5 % of target
  at 640×360@15/350k … 1920×1080@60/6000k; `set_bitrate` settles within ±2 % by
  the second second, with no IDR. Level, Constrained Baseline flags and
  alternating `idr_pic_id` are signalled. P-slice intra is only tried in full when
  the I16×16 estimate beats inter (`WelsMdFirstIntraMode`), ~1.4× faster.
  Rate-controlled 1920×1080@60 encodes at **~107 fps** (16 slices, 16 cores),
  1280×720@30 at **~260 fps**.

### SIMD (P10)

- **NEON** (aarch64, baseline — no runtime detection): `idct4x4_add`, luma
  half-pel `mc_hor_ver20` / `mc_hor_ver02` (+ transitively the quarter-pel
  positions via `pixel_avg`), and `sad`. **wasm `simd128`**: `idct4x4_add` + `sad`.
- Every SIMD kernel is **bit-identical** to scalar (the conformance corpus, all
  unit tests, and the encode round-trip pass unchanged with `--features simd`;
  wasm cross-checked under Node). Dispatch falls back to scalar on any
  unsupported target / width / when the feature is off.
- Speedups (vs scalar): decode CAVLC up to **1.38×** (BANM reaches ~parity with
  C NEON), CABAC-P **1.25×**; encode IPPP **~1.74×**. See `PERF_REPORT.md`.
- **Kept scalar (by design, to stay provably bit-exact):** the deblock edge
  filters (data-dependent per-line branching), the centre half-pel
  `mc_hor_ver22` (two-pass wide intermediate), and `mc_copy` (memcpy).

---

## Public API

- `h264::Decoder` — incremental `decode` / `flush` / `decode_all`; `nal_units`
  iterator; `decoder::decode_stream` convenience.
- `h264::Frame` / `DecodedYuv` / `YUVSource` / `VisibleRegion` — planar I420
  output with visible-region accessors and BT.601 `write_rgb8` / `write_rgba8`.
- **Live re-encode API (P16):** `h264::{Encoder, EncoderConfig, EncodedFrame}` —
  `Encoder::with_config(EncoderConfig { width, height, fps, bitrate_kbps,
  keyframe_interval, slices })`, `encode(&YuvRef) -> EncodedFrame { data, keyframe,
  qp }` (rate-controlled, slice-parallel with `threads`), `set_bitrate(kbps)` (no
  IDR), `set_frame_rate`, `force_idr`, `reconstruction()`. `h264::{YuvRef, I420}`
  picture buffers (`DecodedYuv::yuv()` / `Frame::yuv()` give the cropped decoded
  picture), `h264::scale_i420` (OpenH264 downsampler), `h264::nal::{nal_type, SPS,
  PPS, AUD, IDR}`, `h264::sps_dimensions`. See `TRACKER.md` P16.
- `h264::encoder::Encoder` — `new(w, h, qp)`, `new_with_slices(w, h, qp, slices)`,
  `set_slices`, `encode_frame`, `force_idr`, `set_frame_rate`; with `--features
  threads`: `encode_frame_parallel` (slice-parallel) and `encode_frames_parallel`
  (frame-parallel all-intra, takes `FrameInput`s).
- `h264::{DecodeError, EncodeError}`.

(The decoder/encoder/api/formats modules are feature-gated; `dsp` and `bits` are
always available.)

---

## Build / test matrix

- **Targets (8, lib builds green):** aarch64-apple-darwin, aarch64-apple-ios,
  aarch64-apple-ios-sim, aarch64-linux-android, armv7-linux-androideabi,
  wasm32-unknown-unknown, x86_64-apple-darwin, x86_64-pc-windows-gnu.
- **`#![no_std]` + alloc** (`--no-default-features`) builds, with and without simd.
- **wasm** builds (default and `RUSTFLAGS="-C target-feature=+simd128" --features simd`).
- **`--features simd`** builds on every arch (NEON on aarch64, simd128 on wasm,
  scalar fallback elsewhere).
- **Tests: 166 passing** (134 unit + 31 integration + 1 doc), **identical** with
  and without `--features simd`. Clippy-clean on default and `threads,simd` builds
  (all targets) and on the `no_std` lib builds.
- `tests/fixtures/banm_frame{0,1}.yuv` were missing from the vendored tree (`*.yuv`
  is git-ignored upstream); they were regenerated from this crate's decoder
  (`examples/dump_frames`), which the corpus run had shown bit-exact on BANM.

---

## Deferred (not implemented — explicit)

**Decoder (implemented since the original scope):** B-slices (CAVLC + CABAC,
spatial + temporal direct), I_PCM, and High-profile 8×8 transform / I_8×8 + inter
8×8 + temporal-direct multi-ref + weighted prediction (CAVLC + CABAC) — see
`TRACKER.md` P12/P13/P14/P15. All 6 High-profile `VID_*_temporal_direct` streams
(CAVLC + CABAC, 1280×544 / 1280×720 / 1920×1080) are now bit-exact. **49/54
conformance streams are bit-exact.**

The 3 CABAC `VID_*_cabac_temporal_direct` streams are fully bit-exact as of P15: the
final fix seeds the MV-prediction reference cache of a *temporal*-direct 8×8
sub-partition with its real colocated-derived `ref_idx` (rather than -1), matching
the C `Update8x8RefIdx`/`UpdateP8x8RefCacheIdxCabac` so a later non-direct
sub-partition's `PredMv` single-match neighbour rule fires correctly.

**Decoder (still deferred):** FMO/ASO, error concealment, monochrome / non-4:2:0
/ >8-bit, SVC scalable extension. (Guarded behind clean `DecodeError::Unsupported`
where they would change the bit parse.)

**Encoder:** sub-16×16 inter partitions, multiple reference frames, B-frames,
CABAC encoding, in-loop deblocking, MB-level (GOM) rate control, adaptive
quantisation, frame skipping. (Baseline IPPP works and round-trips without drift
— these are encoder-freedom features, not correctness gaps.)

**Pipeline (whole phases):** P8 processing — the downsampler is ported (P16);
denoise / scene-change / VAA are not.
**P9 threading is done** — encoder slice/frame parallelism (std-only); a decoder
thread pool is not implemented (decode already meets target).

**SIMD (remaining):** deblock filters, `mc_hor_ver22`, chroma MC, and x86
SSE/AVX — see the SIMD note above for why deblock/`mc_hor_ver22` were kept scalar.

See `TRACKER.md` for the full phase log and `PORTING_PLAN.md` for the original scope.
