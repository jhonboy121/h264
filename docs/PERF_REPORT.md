# H.264 Codec — Conformance & Performance Report

Measured results for the pure-Rust H.264 decoder and encoder, validated against
the C reference (`h264dec`, built from Cisco OpenH264 with ARM NEON assembly).

- **Date:** 2026-06-30
- **Machine:** Apple silicon (`uname -m` = `arm64`)
- **Build flags:** `cargo run --release` → `opt-level=3`, `lto=thin`,
  `codegen-units=1` (see `Cargo.toml` `[profile.release]`).
- **Our codec:** pure-Rust. Decode/encode kernels run **scalar** by default and
  **NEON SIMD** with `--features simd` (P10). The SIMD path is **bit-identical**
  to scalar — the entire conformance corpus and all unit/round-trip tests pass
  unchanged with `--features simd`.
- **Oracle:** OpenH264 `h264dec`, ARM **NEON** asm kernels.

Reproduce:

```sh
cargo run --release --example conformance                 # per-stream conformance
cargo run --release --example bench_decode                # decode: Rust scalar vs C
cargo run --release --features simd --example bench_decode # decode: Rust SIMD  vs C
cargo run --release --features simd --example bench_encode # encode throughput
```

Both decode examples accept `H264_CORPUS_DIR` and `H264_ORACLE` env overrides.

---

## 1. Conformance

Corpus: 54 streams (50 `*.264` + 4 `*.jsv`) from the OpenH264 conformance set.
Each stream is decoded by the C oracle to reference I420 YUV and by our
`decode_stream`; our visible (coded-size) I420 output is then compared
**byte-for-byte** (and by MD5) against the oracle output. **Results are identical
with and without `--features simd`** (verified: 46/54 BITEXACT either way).

### Summary

| Outcome | Count |
|---|---|
| **BITEXACT** | **46** |
| UNSUPPORTED (returns `DecodeError::Unsupported`) | 1 |
| MISMATCH | 1 |
| ERROR | 6 |
| **Total** | **54** |

**Frames decoded bit-exact: 5659** (sum over all BITEXACT streams).

The 46 bit-exact streams cover baseline CAVLC (I+P), Main-profile CABAC (I+P,
multi-ref, multi-slice), B-slices (CAVLC + CABAC, spatial + temporal direct),
High-profile I_8×8 + inter 8×8 transform + temporal/spatial-direct multi-ref
(the 3 `VID_*_cavlc_temporal_direct` streams at 1280×544 / 1280×720 / 1920×1080),
I_PCM, QCIF/CIF/up-to-1080p resolutions, and a 1700-frame stream (`LS_SVA_D`).

### Unsupported / mismatch

- **MISMATCH (1):** the SVC stream (`sps_subsetsps_bothVUI`, base layer only).
- **UNSUPPORTED (1) / ERROR (2):** the 3 CABAC VID temporal_direct streams —
  I_8×8 intra parse is in place but the B CABAC path (multi-ref `ref_idx` + CABAC
  temporal direct) is not yet wired.
- **ERROR (4):** deliberately corrupted error-resilience streams + a missing-PPS
  stream.

`examples/conformance.rs` prints the first-differing frame and differing-byte count
for each. Nothing is silently passed off as correct; equality is the literal byte
(and MD5) test.

---

## 2. Decode speed — Rust scalar vs Rust SIMD vs C NEON

Four **bit-exact** streams (identical output across all three). Each is decoded
repeatedly (auto-scaled to ≥1.5 s, ≥5 reps) and the **median** time reported.
`C×` = `our_time / C_time`. `SIMD speedup` = `scalar_time / SIMD_time`.

| Stream | W×H | Frames | Rust scalar ms/f | Rust **SIMD** ms/f | C ms/f | scalar C× | **SIMD C×** | **SIMD speedup** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BANM_MW_D (QCIF baseline CAVLC) | 176×144 | 100 | 0.134 | **0.097** | 0.095 | 1.36× | **1.02×** | **1.38×** |
| BA1_FT_C (QCIF baseline CAVLC) | 352×288 | 299 | 0.435 | **0.329** | 0.255 | 1.67× | **1.29×** | **1.32×** |
| test_cif_I_CABAC (CIF Main CABAC, all-I) | 352×288 | 300 | 1.224 | **1.188** | 1.133 | 1.08× | **1.05×** | **1.03×** |
| test_cif_P_CABAC (CIF Main CABAC, I+P) | 352×288 | 300 | 0.786 | **0.629** | 0.557 | 1.42× | **1.13×** | **1.25×** |

Throughput of the **SIMD** build (fps / MB/s decoded-I420), with the C oracle for
reference:

| Stream | Rust SIMD fps | Rust SIMD MB/s | C fps | C MB/s |
|---|---:|---:|---:|---:|
| BANM_MW_D | 10288 | 391.1 | 10476 | 398.3 |
| BA1_FT_C | 3036 | 461.6 | 3918 | 595.8 |
| test_cif_I_CABAC | 842 | 128.0 | 883 | 134.2 |
| test_cif_P_CABAC | 1590 | 241.8 | 1795 | 272.9 |

Raw repetition medians (bench stderr, SIMD build):

```
BANM_MW_D                rust 9.7 ms / 154 reps   |  C 9.5 ms / 155 reps
BA1_FT_C                 rust 98.5 ms / 16 reps   |  C 76.3 ms /  20 reps
test_cif_I_CABAC_slice   rust 356.3 ms /  5 reps  |  C 340.0 ms /   5 reps
test_cif_P_CABAC_slice   rust 188.7 ms /  8 reps  |  C 167.2 ms /   9 reps
```

### How SIMD closes the gap

The NEON DSP backend accelerates the sample-processing kernels the scalar loops
spent time in — the 4×4 inverse transform/add, the luma half-pel 6-tap
interpolation (and the quarter-pel positions built on it + rounding average), and
the SAD used by reconstruction/encoding:

- **CAVLC streams** (MC/IDCT/deblock-bound) gain the most. `BANM` goes from
  **1.36× → 1.02×** of C (essentially parity — 0.097 vs 0.095 ms/frame, a 1.38×
  self-speedup), and `BA1_FT_C` from **1.67× → 1.29×** (1.32× self-speedup).
- **CABAC I+P** (`test_cif_P_CABAC`) goes from **1.42× → 1.13×** (1.25×
  self-speedup): inter MBs spend real time in MC, which SIMD recovers.
- **CABAC all-I** (`test_cif_I_CABAC`) barely moves (**1.08× → 1.05×**): its time
  is dominated by serial CABAC arithmetic decoding and intra reconstruction,
  which are inherently not SIMD-friendly. It was already near parity.

The remaining gap to C on the CAVLC streams is the deblocking filter (kept scalar
— its per-line data-dependent branching does not map to a provably bit-exact mask)
and the centre half-pel `mc_hor_ver22` (kept scalar to stay bit-exact). These are
the honest next SIMD targets if the gap matters.

---

## 3. Encode throughput + round-trip quality

`Encoder::encode_frame` measured over 100 real BANM 176×144 frames (decoded by our
own decoder to recover genuine content), in two modes at QP 26 and QP 32.
`MB/s` is source-I420 throughput (`frames × W × H × 3/2 / time`).
`cargo run --release --features simd --example bench_encode`.

| Mode | QP | scalar fps | scalar MB/s | **SIMD fps** | **SIMD MB/s** | SIMD speedup | avg AU bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| INTRA (all-I) | 26 | 1947 | 74.0 | **1978** | **75.2** | 1.02× | 3193 |
| IPPP | 26 | 579 | 22.0 | **1001** | **38.1** | **1.73×** | 1064 |
| INTRA (all-I) | 32 | 2163 | 82.2 | **2185** | **83.1** | 1.01× | 2027 |
| IPPP | 32 | 592 | 22.5 | **1038** | **39.5** | **1.75×** | 601 |

The IPPP encode path gets a large SIMD win (**~1.74×**) because motion estimation
is dominated by SAD and sub-pel MC — exactly the NEON kernels added in P10. Intra
encode is essentially unchanged (its cost is the forward transform / quant / CAVLC
bit-writing, not the SIMD'd kernels).

### Round-trip quality (encode → our own conformant decoder)

From `tests/encode_roundtrip.rs`. **Single intra frame** (BANM frame 0):

| QP | Y PSNR | U PSNR | V PSNR | AU bytes |
|---:|---:|---:|---:|---:|
| 26 | 41.16 dB | 46.17 | 47.41 | 2990 |
| 32 | 37.27 dB | 44.26 | 44.67 | 2240 |

**IPPP** (5-frame smooth panning sequence — pure translation, which inter
prediction should exploit):

| QP | I-frame Y / bytes | P-frame Y (range) | P bytes (range) | I÷P bitrate |
|---:|---|---|---|---:|
| 26 | 40.42 dB / 3133 B | 40.97–41.66 dB | 197–333 B | **~12.5×** |
| 32 | 36.01 dB / 2034 B | 36.63–37.19 dB | 114–137 B | **~16.5×** |

PSNR is monotonic in QP, P-frame quality holds (no drift across the GOP), and
every P access unit is an order of magnitude smaller than the IDR — confirming the
inter path and the encoder↔decoder reconstruction agree.

---

## 4. Methodology

- **Conformance** (`examples/conformance.rs`): for every `*.264`/`*.jsv`, run the C
  oracle → reference YUV, decode with `decode_stream`, extract the visible
  coded-size I420 of every returned `Picture`, and compare the full concatenated
  buffer **byte-for-byte** (compact in-tree MD5 summarizes diffs). `Unsupported`
  is classified separately. First-differing-frame index and differing-byte count
  reported per MISMATCH.
- **Decode bench** (`examples/bench_decode.rs`): four bit-exact streams, in-process
  `decode_stream` after a warm-up, repeated until ≥1.5 s / ≥5 reps (cap 200),
  **median** reported. The C binary is timed via `std::process::Command` +
  `Instant`. The example prints whether it was built scalar or SIMD.
- **Encode bench** (`examples/bench_encode.rs`): decodes the in-tree BANM stream to
  recover 100 real frames, then times `encode_frame` over the whole sequence
  (all-intra via `force_idr`, and IPPP), auto-scaled, **median** reported.
- **SIMD bit-exactness**: the same SIMD kernels are validated against scalar by
  the existing kernel unit tests under `--features simd` (e.g. the 2000-case
  random IDCT test, the all-position MC anchor test, the random SAD test) and by
  the full conformance corpus + encode round-trip — all byte-identical. The wasm
  `simd128` ports of `idct4x4_add`/`sad` were additionally cross-checked vs scalar
  under Node (50k IDCT + 12k SAD cases, 0 mismatches).
- **Fairness note.** The C decode timing includes process spawn + writing the YUV
  to disk; ours is in-process only. This *handicaps* the C number slightly, so the
  true C-kernel advantage is marginally larger than the `C×` column shows.
- Build: `--release` (`opt-level=3`, `lto=thin`, `codegen-units=1`). YUV artifacts
  go to the system temp dir and are deleted; `*.yuv` is git-ignored.
