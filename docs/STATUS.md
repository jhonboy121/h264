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
- **Conformance: 28/54 corpus streams BIT-EXACT** vs the C oracle (3875 frames),
  covering baseline CAVLC (I+P), Main CABAC (I+P, multi-ref, multi-slice), QCIF →
  1280×720, and a 1700-frame stream. Results are **identical with `--features
  simd`**. The other 26 are cleanly classified: 11 UNSUPPORTED (guarded
  `DecodeError::Unsupported`), 11 MISMATCH (crop/interlace length or remaining
  decode gaps), 4 ERROR (corrupted error-resilience streams).

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
- `h264::encoder::Encoder` — `new(w, h, qp)`, `encode_frame`, `force_idr`.
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
- **Tests: 139 passing** (126 unit + 13 integration), **identical** with and
  without `--features simd`. Warning-free on default and simd builds.

---

## Deferred (not implemented — explicit)

**Decoder:** B-slices, High-profile 8×8 transform / I_8×8, I_PCM, FMO/ASO,
error concealment. (All guarded behind clean `DecodeError::Unsupported`; not
present in the baseline / Main-IP streams that are in scope.)

**Encoder:** sub-16×16 inter partitions, multiple reference frames, B-frames,
CABAC encoding, rate control. (Fixed-QP baseline IPPP works and round-trips
without drift — these are encoder-freedom features, not correctness gaps.)

**Pipeline (whole phases):** P8 processing (downsample/denoise/scene-change/VAA)
and P9 threading (slice-level multithreading + thread pool, std-only).

**SIMD (remaining):** deblock filters, `mc_hor_ver22`, chroma MC, and x86
SSE/AVX — see the SIMD note above for why deblock/`mc_hor_ver22` were kept scalar.

See `TRACKER.md` for the full phase log and `PORTING_PLAN.md` for the original scope.
