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

## P1 — Foundations / DSP ✅
- [x] `bits/{reader,golomb,writer}.rs` — MSB-first reader/writer + Exp-Golomb (12 tests)
- [x] crate scaffold builds for all 7 targets + no_std + wasm
- [x] `dsp/transform.rs` — idct4x4_add, idct8x8_add (5 tests, faithful i16 trunc)
- [x] `dsp/intra_pred.rs` — 44 kernels (4x4/8x8/16x16/chroma) + tests vs ref (4 test groups)
- [x] `dsp/mc.rs` — luma 6-tap + 16 quarter-pel + chroma bilinear (4 tests, bit-exact)
      · anchor: `EncUT_MotionCompensation.cpp` ✓
- [x] `dsp/deblock.rs` — luma/chroma Lt4 & Eq4 edge filters H&V (6 tests, bit-exact)
      · anchor: `DecUT_DeblockCommon.cpp` ✓
- [x] `dsp/copy.rs` (2 tests), `dsp/sad.rs` (3 tests)
- [⏸] `dsp/expand.rs` — DEFERRED to P3 (coupled to padded Plane buffer)
- [ ] `dsp/tables.rs` — dequant/scan tables — moved to P2 (needed by CAVLC residual)

## P2 — Decoder parsing 🚧
- [x] `decoder/nal.rs` — Annex-B framing, NAL header, EPB strip (4 tests)
- [x] `decoder/params.rs` — SPS/PPS+VUI parse, validated on real fixtures (5 tests, fc0985f)
      · BANM 176x144, BA1 352x288, SVA 176x144. HRD not stored (Unsupported). 
- [x] `decoder/slice_header.rs` — slice header parse (7.3.3), AVC path; SliceType
      enum + reorder/weight/marking sub-structs; validated on BANM/BA1 (2 tests)
- [x] `dsp/tables.rs` — dequant 4x4/8x8 + scan/zigzag tables (5 tests, a28ee89)
- [x] `decoder/cavlc.rs` + `cavlc_tables.rs` — coeff_token/level/total_zeros/run_before
      residual decode; vs spec Table 9-5/9-7/9-9 (6 tests, f454e0d)
- [🚧] `decoder/cabac.rs` + `cabac_tables.rs` + `cabac_mb.rs` — engine + residual — DELEGATED
- [ ] anchor `DecUT_ParseSyntax.cpp`: full-decoder integration → exercised in P3/P4
- NOTE: MB-level syntax needing neighbor context (mb_type/mvd/cbp via CAVLC/CABAC)
  folds into P3 decode loop (needs mb_cache). residual_block_cavlc fills out_level in
  scan order; P3 maps zigzag→raster + dequant.

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
