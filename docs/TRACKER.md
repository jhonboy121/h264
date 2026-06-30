# Port Tracker — live status

Legend: ✅ done & tested · 🚧 in progress · ⬜ not started · ⏸ deferred

**HOW TO RESUME:** read `PORTING_PLAN.md` + this file. Find the first 🚧/⬜ item.
The C source for it is in `reference/` (see `REFERENCE_MAP.md` for the exact file).
Port it, add tests, run `cargo test`, then check it off here with a one-line note.

Last updated: P1 — bits done; DSP transform next.

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
- [ ] `dsp/intra_pred.rs` — intra 4x4/8x8/16x16 + chroma (port `get_intra_predictor.cpp`,
      `intra_pred_common.cpp`) · anchor: `DecUT_IntraPrediction.cpp`
- [ ] `dsp/mc.rs` — luma 6-tap + chroma bilinear MC (port `mc.cpp`)
      · anchor: `EncUT_MotionCompensation.cpp`
- [ ] `dsp/deblock.rs` — edge filters (port `deblocking_common.cpp`)
      · anchor: `DecUT_Deblock.cpp`, `DecUT_DeblockCommon.cpp`
- [ ] `dsp/copy.rs`, `dsp/expand.rs`, `dsp/sad.rs` · anchors: `EncUT_MBCopy`, `EncUT_DecodeMbAux`

## P2 — Decoder parsing ⬜
- [ ] `decoder/nal.rs` — Annex-B framing, NAL header, EPB strip
- [ ] `decoder/params.rs` — SPS/PPS parse (port `au_parser.cpp`)
- [ ] `decoder/slice_header.rs` — slice header parse
- [ ] `decoder/cavlc_tables.rs` + `decoder/cavlc.rs` — CAVLC MB parse
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
