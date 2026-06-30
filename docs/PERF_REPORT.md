# H.264 Decoder — Conformance & Performance Report

Measured results for the pure-Rust H.264 decoder, validated against the C
reference (`h264dec`, built from Cisco OpenH264 with ARM NEON assembly).

- **Date:** 2026-06-30
- **Machine:** Apple M5 Max (`uname -m` = `arm64`)
- **Build flags:** `cargo run --release` → `opt-level=3`, `lto=thin`,
  `codegen-units=1` (see `Cargo.toml` `[profile.release]`).
- **Our decoder:** pure-Rust, **scalar** (no SIMD yet).
- **Oracle:** OpenH264 `h264dec`, ARM **NEON** asm kernels.

Reproduce:

```sh
cargo run --release --example conformance   # per-stream table + summary
cargo run --release --example bench_decode   # Rust-vs-C throughput table
```

Both examples accept `H264_CORPUS_DIR` and `H264_ORACLE` env overrides.

---

## 1. Conformance

Corpus: 54 streams (50 `*.264` + 4 `*.jsv`) from the OpenH264 conformance set.
Each stream is decoded by the C oracle to reference I420 YUV and by our
`decode_stream`; our visible (coded-size) I420 output is then compared
**byte-for-byte** (and by MD5) against the oracle output.

### Summary

| Outcome | Count |
|---|---|
| **BITEXACT** | **28** |
| UNSUPPORTED (returns `DecodeError::Unsupported`) | 11 |
| MISMATCH | 11 |
| ERROR | 4 |
| **Total** | **54** |

**Frames decoded bit-exact: 3875** (sum over all BITEXACT streams).

### Unsupported-feature breakdown

These streams exercise features our decoder does not yet implement; it cleanly
returns `DecodeError::Unsupported(feature)` rather than producing wrong output.

| Feature reported | Streams |
|---|---|
| `transform_size_8x8 / I_8x8` (CAVLC) | 3 |
| `transform_size_8x8 / I_8x8 (CABAC)` | 3 |
| `B slice` (CAVLC) | 2 |
| `B slice (CABAC)` | 1 |
| `I_PCM` | 1 |
| `I_PCM (CABAC)` | 1 |
| **Total** | **11** |

(High-profile 8x8 transform, B-slices, and I_PCM — the advanced features called
out as out-of-scope for the current phase.)

### Bit-exact streams (28)

```
Adobe_PDF_sample_a_1024x768_50Frms.264   BA1_FT_C.264          BA1_Sony_D.jsv
BAMQ1_JVC_C.264                          BAMQ2_JVC_C.264       BANM_MW_D.264
BASQP1_Sony_C.jsv                        BA_MW_D.264           LS_SVA_D.264
MIDR_MW_D.264                            MPS_MW_A.264          NL1_Sony_D.jsv
NLMQ1_JVC_C.264                          NLMQ2_JVC_C.264       NRF_MW_E.264
SVA_BA1_B.264                            SVA_BA2_D.264         SVA_Base_B.264
SVA_CL1_E.264                            SVA_FM1_E.264         SVA_NL1_B.264
SVA_NL2_E.264                            SarVui.264            Zhling_1280x720.264
test_cif_I_CABAC_PCM.264                 test_cif_I_CABAC_slice.264
test_cif_P_CABAC_slice.264               test_qcif_cabac.264
```

These cover baseline CAVLC (I+P), Main-profile CABAC (I+P, multi-ref, multi-slice),
QCIF/CIF/up-to-1280×720 resolutions, and a 1700-frame stream (`LS_SVA_D`).

### MISMATCH and ERROR detail (caveats)

The 11 MISMATCH and 4 ERROR cases are honestly *not* bit-exact today. They fall
into a few buckets:

- **Length mismatch from cropping / interlace** — our output is coded size
  (`mb_width*16 × mb_height*16`); the oracle emits the cropped/display size.
  Where the two differ in total byte length, the harness reports the length in
  the note column. Examples: `Static.264` (ours 268800 vs ref 228000 bytes/frame
  set — frame cropping), `jm_1080p_allslice.264` (1088 vs 1080 coded rows),
  `sps_subsetsps_bothVUI.264` (our decode yields 0 frames — subset-SPS / SVC
  base), `CVFC1_Sony_C.jsv` (≈2× size — interlaced/field coding).
- **Content mismatch** on `CI1_FT_B`, `CI_MW_D`, `MR1_MW_A`, `MR2_MW_A`,
  `MR2_TANDBERG_E`, `test_vd_1d`, `test_vd_rc` — first divergence frame and the
  differing-byte count are printed per stream; these are genuine remaining
  decode gaps to chase in a later phase.
- **ERROR (4):** `BA_MW_D_IDR_LOST` and `BA_MW_D_P_LOST` are deliberately
  corrupted error-resilience streams (we reject the broken reference state);
  `Error_I_P` likewise; `test_scalinglist_jm` references a parameter set our
  parser does not retain (`missing parameter set`).

No MISMATCH/ERROR stream is silently passed off as correct — equality is the
literal byte (and MD5) test in `examples/conformance.rs`.

---

## 2. Decode speed — Rust scalar vs C NEON

Streams benchmarked are all **bit-exact** (so we are comparing identical output).
Each is decoded repeatedly (auto-scaled to ≥1.5 s wall time, ≥5 reps) and the
**median** time is reported. `MB/s` is decoded I420 output throughput
(`frames × W × H × 3/2` bytes ÷ time). `C×` = `our_time / C_time`.

| Stream | W×H | Frames | Rust ms/frame | Rust fps | Rust MB/s | C ms/frame | C fps | C MB/s | C× faster |
|---|---|---|---|---|---|---|---|---|---|
| BANM_MW_D (QCIF baseline CAVLC) | 176×144 | 100 | 0.133 | 7492 | 284.8 | 0.101 | 9878 | 375.5 | **1.32×** |
| BA1_FT_C (QCIF baseline CAVLC) | 352×288 | 299 | 0.429 | 2332 | 354.6 | 0.260 | 3841 | 584.1 | **1.65×** |
| test_cif_I_CABAC_slice (CIF Main CABAC, all-I) | 352×288 | 300 | 1.218 | 821 | 124.9 | 1.137 | 880 | 133.7 | **1.07×** |
| test_cif_P_CABAC_slice (CIF Main CABAC, I+P) | 352×288 | 300 | 0.762 | 1312 | 199.5 | 0.555 | 1802 | 274.0 | **1.37×** |

Raw repetition data (from the bench's stderr):

```
BANM_MW_D                rust median 13.3 ms / 113 reps   |  C median 10.1 ms / 145 reps
BA1_FT_C                 rust median 128.2 ms / 12 reps   |  C median  77.8 ms /  20 reps
test_cif_I_CABAC_slice   rust median 365.3 ms /  5 reps   |  C median 341.1 ms /   5 reps
test_cif_P_CABAC_slice   rust median 228.7 ms /  7 reps   |  C median 166.5 ms /   9 reps
```

---

## 3. Analysis

**Where we stand.** A scalar, pure-Rust decoder is within **1.07×–1.65×** of a
hand-tuned NEON-assembly C decoder on these streams, while producing
**byte-identical** output. The gap is smallest on the CABAC all-I stream
(`test_cif_I_CABAC_slice`, 1.07×), where decode time is dominated by serial CABAC
arithmetic decoding (not SIMD-friendly) and intra reconstruction; it is largest
on the CAVLC streams (up to 1.65×), where the C build's NEON kernels for the
inverse transform, motion compensation interpolation, and deblock filter pull
ahead of our scalar loops.

**Closing the gap (planned, P10).** The remaining difference is almost entirely
in the sample-processing kernels — IDCT/dequant, luma/chroma interpolation, and
the deblocking filter — which are exactly what the C build accelerates with NEON.
The planned path is to add a SIMD DSP backend (ARM NEON, plus wasm `simd128` for
the web target) behind the existing scalar `dsp` traits, which should recover
most of the CAVLC-side gap. CABAC-bound streams will stay close to parity since
their bottleneck is inherently serial.

**Correctness caveats.** Bit-exactness is established for baseline CAVLC (I+P)
and Main-profile CABAC (I+P, multi-ref, multi-slice). Not yet supported (clean
`Unsupported` errors): High-profile 8×8 transform / I_8×8, B-slices, and I_PCM.
A handful of streams MISMATCH today due to frame cropping/interlace handling or
remaining decode gaps (Section 1) — these are tracked, not hidden.

---

## 4. Methodology

- **Conformance** (`examples/conformance.rs`): for every `*.264`/`*.jsv` in the
  corpus, run the C oracle → reference YUV, decode with `decode_stream`, extract
  the visible coded-size I420 of every returned `Picture`, and compare the full
  concatenated buffer to the reference **byte-for-byte**. A compact in-tree MD5
  (RFC 1321) summarizes differing buffers; byte equality is the pass/fail test.
  `Unsupported(feature)` is classified separately from other errors. The
  first-differing-frame index and differing-byte count are reported for each
  MISMATCH. Wall times for both decoders are captured with `std::time::Instant`.
- **Benchmark** (`examples/bench_decode.rs`): four bit-exact streams. Our decoder
  is timed in-process (`decode_stream`, after a warm-up call), repeated until
  ≥1.5 s total / ≥5 reps (cap 200); **median** time reported. The C binary is
  timed the same way via `std::process::Command` + `Instant`. Throughput numbers
  use the actual decoded frame count and coded dimensions.
- **Fairness note.** The C timing includes process spawn and writing the output
  YUV to disk; ours is in-process decode only. This *handicaps* the C number
  slightly, so the true C-kernel advantage is marginally larger than the `C×`
  column shows — i.e. our reported gap is, if anything, optimistic toward us.
- Build: `--release` (`opt-level=3`, `lto=thin`, `codegen-units=1`). YUV
  artifacts are written to the system temp dir and deleted; nothing large is
  committed (`*.yuv` is git-ignored).
