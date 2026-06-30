# Port Tracker — live status

Legend: ✅ done & tested · 🚧 in progress · ⬜ not started · ⏸ deferred

**HOW TO RESUME:** read `PORTING_PLAN.md` + this file. Find the first 🚧/⬜ item.
The C source for it is in `reference/` (see `REFERENCE_MAP.md` for the exact file).
Port it, add tests, run `cargo test`, then check it off here with a one-line note.

Last updated: **P1 COMPLETE** (40 tests, 0 warnings, all 7 targets + no_std green).
NEXT: P2 parsing — SPS/PPS/slice-header (au_parser.cpp), then CAVLC + vlc tables
(parse_mb_syn_cavlc.cpp), then CABAC (cabac_decoder.cpp). Also pull dequant/scan
tables → dsp/tables.rs, and the luma/chroma DC dequant-IDCT (decode_slice.cpp:246,359).

---

## P0 — Scaffolding ✅
- [x] Vendored C source → `reference/codec`, `reference/test`
- [x] Conformance fixtures subset → `tests/fixtures/` (4 streams)
- [x] Docs: PORTING_PLAN, REFERENCE_MAP, ARCHITECTURE, TRACKER
- [x] git init
- [x] Cargo crate `h264`, no_std+alloc, feature flags, lib skeleton
- [x] 7-target `cargo build` smoke check

## P1 — Foundations / DSP 🚧
- [x] `bits/reader.rs` — MSB-first bit reader + more_rbsp_data + trailing bits (6 tests)
- [x] `bits/golomb.rs` — Exp-Golomb ue/se/te (3 tests)
- [x] `bits/writer.rs` — bit writer + ue/se write (3 tests, roundtrip-verified)
- [x] crate scaffold builds for all 7 targets + no_std + wasm
- [ ] `dsp/tables.rs` + core consts/types (`wels_const.h`, `codec_def.h`)
- [ ] `dsp/transform.rs` — IDCT 4x4/8x8, dequant, DC transforms (port `decode_mb_aux.cpp`)
      · anchor test: `DecUT_IdctResAddPred.cpp`
- [x] `dsp/transform.rs` — idct4x4_add, idct8x8_add (5 tests, faithful i16 trunc)
- [x] `dsp/intra_pred.rs` — 44 kernels (4x4/8x8/16x16/chroma) + tests vs ref (4 test groups)
- [x] `dsp/mc.rs` — luma 6-tap + 16 quarter-pel + chroma bilinear (4 tests, bit-exact)
      · anchor: `EncUT_MotionCompensation.cpp` ✓
- [x] `dsp/deblock.rs` — luma/chroma Lt4 & Eq4 edge filters H&V (6 tests, bit-exact)
      · anchor: `DecUT_DeblockCommon.cpp` ✓
- [x] `dsp/copy.rs` — copy_block + named sizes (2 tests) [STAGED, unwired]
- [x] `dsp/sad.rs` — sad + named sizes (3 tests) [STAGED, unwired]
- [⏸] `dsp/expand.rs` — DEFERRED to P3 (coupled to padded Plane buffer)

### Staged-but-unwired files (wire into mod.rs after intra agent returns, then test):
- src/dsp/copy.rs, src/dsp/sad.rs  → add `pub mod copy; pub mod sad;` to src/dsp/mod.rs
- src/decoder/nal.rs (Annex-B framing, EBSP→RBSP, NAL types; 4 tests) → needs src/decoder/mod.rs + `pub mod decoder;` in lib.rs

## P2 — Decoder parsing 🚧
- [x] `decoder/nal.rs` — Annex-B framing, NAL header, EPB strip (4 tests)
- [x] `decoder/params.rs` — SPS/PPS+VUI parse, validated on real fixtures (5 tests, fc0985f)
      · BANM 176x144, BA1 352x288, SVA 176x144. HRD not stored (Unsupported). 
- [x] `decoder/slice_header.rs` — slice header parse (7.3.3), AVC path; SliceType
      enum + reorder/weight/marking sub-structs; validated on BANM/BA1 (2 tests)
- [ ] `dsp/tables.rs` — dequant (g_kuiDequantCoeff[52][8]+8x8), scan (g_kuiScan8,
      zigzag, luma/chroma DC scan) — pull from common_tables.cpp + decoder_data_tables.cpp
- [ ] `decoder/cavlc_tables.rs` + `decoder/cavlc.rs` — CAVLC MB parse (parse_mb_syn_cavlc.cpp)
- [ ] `decoder/cabac.rs` + `decoder/cabac_mb.rs` — CABAC engine + MB parse
- [ ] anchor test: `DecUT_ParseSyntax.cpp`

## P3 — Decoder reconstruction ⬜
- [ ] `decoder/mv_pred.rs` · anchor `DecUT_PredMv.cpp`
- [ ] `decoder/recon.rs` — MB reconstruct (intra+inter+residual)
- [ ] `decoder/dpb.rs`, `decoder/ref_pic.rs` — DPB + ref mgmt
- [ ] `decoder/fmo.rs` — FMO
- [ ] `decoder/decode_slice.rs` — slice/MB decode loop
- [ ] in-loop deblock integration
- [ ] `decoder/error_conceal.rs`

## P4 — Decoder API + conformance ⬜
- [ ] `src/api.rs` Decoder facade, `src/formats/` YUV→RGB
- [ ] conformance harness: decode `tests/fixtures` + full corpus
- [ ] C `h264dec` oracle golden md5 compare (needs meson/ninja/nasm)

## P5 — Encoder DSP ⬜
- [ ] fwd transform/quant (`encode_mb_aux.cpp`) · `EncUT_EncoderMbAux`, `EncUT_DecodeMbAux`
- [ ] enc intra predictor · `EncUT_GetIntraPredictor`
- [ ] SAD/SATD (`sample.cpp`) · `EncUT_Sample`

## P6 — Encoder core ⬜
- [ ] `bits/writer.rs` + Exp-Golomb write · `EncUT_ExpGolomb`
- [ ] CAVLC write (`set_mb_syn_cavlc.cpp`) · `EncUT_Cavlc`
- [ ] mode decision, motion est · `EncUT_MotionEstimate`, `EncUT_SVC_me`
- [ ] mv_pred(enc), ref list mgr, rate control
- [ ] slice/mb encode, reconstruct · `EncUT_EncoderMb`, `EncUT_Reconstruct`
- [ ] NAL encap, paraset gen · `EncUT_ParameterSetStrategy`
- [ ] CABAC encode

## P7 — Encoder API + round-trip ⬜
- [ ] `Encoder` facade; encode→decode PSNR tests (`test/api/encode_decode_*`)

## P8 — Processing ⏸  (downsample/denoise/scenechange/vaa)
## P9 — Threading ⏸  (slice MT + thread pool; std-only, wasm off)
## P10 — SIMD ⏸  (NEON / wasm128 / SSE-AVX behind `simd`, runtime detect)
## P11 — Perf report ⬜ → `docs/PERF_REPORT.md`

---

## Notes / decisions log
- (P0) Single crate + features chosen over workspace; no_std+alloc; see ARCHITECTURE.md.
- Oracle build deferred; `brew install meson ninja nasm` when reaching P4.
