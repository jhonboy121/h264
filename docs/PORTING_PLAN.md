# OpenH264 → Pure Rust Port — Master Plan

**Goal:** Faithful, internal-use port of Cisco OpenH264 (C/C++) to pure Rust.
No FFI. Builds for macOS, iOS (native + sim), Android (arm64/armv7), and wasm32.
Rusty API (not 1:1 with C). Avoid global allocations (no global mutable state;
explicit buffer ownership / preallocation). Validate against the C test suite
and the H.264 conformance corpus.

This document is the **stable, compaction-proof reference**. If context is lost,
read this + `TRACKER.md` + `REFERENCE_MAP.md` + `ARCHITECTURE.md` to resume.

---

## Source of truth (C reference)

- Vendored in-repo: `reference/codec/` (OpenH264 codec sources) and
  `reference/test/` (the C/gtest unit tests). ~9.8 MB total.
- Upstream: `https://github.com/cisco/openh264` (re-clone `--depth 1` if needed).
- Conformance corpus: 59 streams, 60 MB. NOT vendored. A small subset is in
  `tests/fixtures/`. Full corpus lives in upstream `res/` — re-clone to get it,
  or set `H264_CONFORMANCE_DIR` env var to point tests at it.

## Scope of the C code (LOC, .c/.cpp/.h)

| Component   | LOC    | Notes                                                        |
|-------------|--------|-------------------------------------------------------------|
| decoder     | 28,592 | Constrained Baseline + CAVLC + CABAC decode (no B-slices)   |
| encoder     | 37,292 | Constrained Baseline encode, rate control, ME, mode decision|
| common      | 28,286 | Shared DSP (mostly SIMD variants); scalar refs are the port |
| processing  | 7,389  | Encoder pre-processing (downsample, denoise, scene change)  |
| api         | 1,643  | C ABI surface (we replace with a Rusty API)                 |
| asm         | 24 files | x86 SSE/AVX + ARM NEON — ported as scalar first, SIMD later |

The `common` SIMD count is inflated by N variants of each kernel (sse2/avx2/neon).
We port **one scalar reference** per kernel, then add SIMD behind features.

---

## Profile target

OpenH264's decoder handles up to High Profile **CAVLC and CABAC**, progressive,
no B-slices, no interlace. Encoder emits Constrained Baseline. We mirror that:
- Decoder: I/P slices, CAVLC + CABAC, 4:2:0 8-bit, intra 4x4/8x8/16x16, deblocking,
  FMO, multiple ref frames, weighted pred (as upstream supports).
- Encoder: Constrained Baseline (I/P, CAVLC), rate control, intra/inter mode
  decision, motion estimation, deblocking.

---

## Phases (testable, dependency-ordered)

Each phase has a matching section in `TRACKER.md` with per-item checkboxes.

- **P0 — Scaffolding**: crate, no_std+alloc, feature flags, docs, target matrix. ✅
- **P1 — Foundations / DSP**: bitstream reader, Exp-Golomb, core types/consts,
  shared DSP kernels (IDCT/itrans, intra-pred common, MC, deblock-common,
  copy/expand/sad). Test anchors: `DecUT_IdctResAddPred`, `DecUT_IntraPrediction`,
  `DecUT_Deblock`, `DecUT_DeblockCommon`, `EncUT_DecodeMbAux`,
  `EncUT_MotionCompensation`, `EncUT_MBCopy`.
- **P2 — Decoder parsing**: NAL framing, SPS/PPS, slice header, CAVLC MB syntax
  (+VLC tables), CABAC engine + MB syntax. Anchor: `DecUT_ParseSyntax`.
- **P3 — Decoder reconstruction**: intra pred, mv_pred, MC, residual add, deblock,
  ref pic mgmt, FMO, slice/MB decode loop, error concealment. Anchor: `DecUT_PredMv`,
  conformance streams.
- **P4 — Decoder API + conformance**: Rusty `Decoder`, decode whole streams,
  diff against C-decoder YUV (md5 golden). Run conformance corpus.
- **P5 — Encoder DSP**: fwd transform/quant, intra predictor (enc), SAD/SATD,
  ME kernels. Anchors: `EncUT_EncoderMbAux`, `EncUT_Sample`, `EncUT_GetIntraPredictor`.
- **P6 — Encoder core**: CAVLC/Exp-Golomb writer, mode decision, ME, mv_pred,
  ref-list mgr, rate control, slice/MB encode, NAL encap, paraset gen, CABAC enc.
  Anchors: `EncUT_Cavlc`, `EncUT_ExpGolomb`, `EncUT_EncoderMb`, `EncUT_MotionEstimate`,
  `EncUT_Reconstruct`, `EncUT_SVC_me`.
- **P7 — Encoder API + round-trip**: Rusty `Encoder`, encode→decode PSNR tests
  (mirror `test/api/encode_decode_*`).
- **P8 — Processing**: downsample/denoise/scene-change/etc (encoder quality path).
- **P9 — Threading**: slice multithreading + thread pool (feature-gated; wasm off).
- **P10 — SIMD**: NEON + wasm-simd128 + x86 SSE/AVX behind `simd` feature with
  runtime detection; scalar fallback always present.
- **P11 — Perf report + platform build verification** → `docs/PERF_REPORT.md`.

## Definition of done per phase
1. Code ported and `cargo build` clean (default features).
2. Tests ported/written and `cargo test` green.
3. `TRACKER.md` items checked with the commit/file refs.
4. Cross-target `cargo build --target <t>` smoke-checked for the 7 targets (at
   least core; std-only paths gated).

## Validation strategy
- **DSP kernels**: port the gtest unit tests + independent scalar reference math.
- **Parsing/recon**: conformance streams; build C `h264dec` oracle (needs
  `brew install meson ninja nasm`) to produce golden YUV + md5.
- **Encoder**: encode then decode with our own decoder; assert PSNR ≥ threshold;
  cross-check bit syntax against ported `EncUT_*`.
