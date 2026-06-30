# Port Tracker — live status

Legend: ✅ done & tested · 🚧 in progress · ⬜ not started · ⏸ deferred

**HOW TO RESUME:** read `PORTING_PLAN.md` + this file. Find the first 🚧/⬜ item.
The C source for it is in `reference/` (see `REFERENCE_MAP.md` for the exact file).
Port it, add tests, run `cargo test`, then check it off here with a one-line note.

Last updated: **PROJECT COMPLETE (P1-P11)**. Decoder (baseline CAVLC + Main CABAC, I+P,
multi-ref, deblock, DPB — 28/54 BIT-EXACT vs C, 3875 frames) + encoder (baseline intra +
IPPP CAVLC, round-trip PSNR, no drift) + **SIMD (P10: NEON + wasm simd128, bit-exact)** +
**refreshed perf report (P11)**. 139 tests, 0 warnings, 8 targets + no_std + wasm + simd.
See `STATUS.md` for the final summary. Decoder now also does frame_cropping output +
I_PCM (CAVLC+CABAC) → 40/54 BIT-EXACT. Deferred (explicit, not gaps): decoder B-slices,
transform8x8/High, FMO, error-conceal, SVC scalable-extension (NAL type 20); encoder
sub-16x16, multi-ref, B, CABAC-encode, rate control; P8 processing; P9 threading;
SIMD for deblock/mc_hor_ver22/x86.

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

## P2 — Decoder parsing ✅ (parsing primitives; MB-level syntax → P3)
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

## P3 — Decoder reconstruction 🚧  (integration phase — shared DecoderContext)
Strategy: build the shared backbone types first, then layer decode paths. Order:
- [x] **P3a baseline I-frame (CAVLC) MVP** ✅ — decodes BANM_MW_D IDR 176x144,
      deterministic, luma 19-251 (80 tests). Files: picture/context/mb_parse_cavlc/
      recon_intra/frame.rs. Implements I4x4(9 modes)+I16x16+chroma, nC, dequant,
      luma/chroma DC IDCT. Deferred: PCM, 8x8/High, CABAC, P/B, deblock, FMO.
      ⚠️ NOT yet bit-exact-validated (no oracle, no deblock) — P4 gate next.
- [~] **P3a (original spec below, now done):**
      `decoder/picture.rs` (Plane/Picture YUV420 + PADDING border, alloc-once),
      `decoder/context.rs` (DecoderContext: SPS/PPS maps, cur pic, per-MB arrays
      mb_type/intra_modes/cbp/qp/nnz, mb_cache+neighbor avail, sized from SPS),
      `decoder/mb_parse_cavlc.rs` (I-slice mb_type, intra16x16/4x4 pred modes w/
      neighbor pred, chroma mode, cbp, mb_qp_delta, residual→zigzag→raster+dequant),
      `decoder/recon_intra.rs` (neighbor sample setup + dsp::intra_pred + idct add;
      luma/chroma DC dequant-IDCT from decode_slice.cpp:246/359),
      `decoder/decode_slice.rs` (MB loop). Test: decode 1st IDR of BANM_MW_D.264.
- [x] **P4 oracle set up** — C h264dec built (Makefile `make h264dec`), examples/dump_frame.rs.
      **VALIDATION (no-deblock):** BANM intra frame vs C: Y-PSNR 43.18dB, U 48.5, V 49.3;
      diffs ±1-3 at 4x4 edges (73 luma px ≥8, 80.8% edge-adjacent) → intra recon CORRECT,
      gap == missing deblock. Oracle ref at /tmp/banm_c_frame0.yuv.
- [x] **P3b deblock integration** ✅ **BIT-EXACT** — BANM intra frame == C oracle
      (0/38016 bytes differ). decoder/deblock.rs (alpha/beta/tc0 tables, intra bS=4/3,
      V then H edges, luma+chroma, qp avg). Golden `tests/fixtures/banm_frame0.yuv` +
      `tests/deblock_conformance.rs`. 386e9d1. **Baseline INTRA decode is conformant.**
      Inter bS (mv/ref) extends in P3c.
- [x] **P3c P-slice** ✅ — mv_pred (bit-exact vs DecUT_PredMv), inter MB CAVLC parse
      (P_16x16/16x8/8x16/8x8/8x8ref0, sub_mb, ref_idx/mvd, P_Skip, intra-in-P), recon_inter
      (MC+residual), inter deblock bS. ca3c2a3.
- [x] **P3d DPB/ref-list** ✅ — dpb.rs sliding-window short-term, default P list-0,
      dsp/expand.rs border replication. (POC/MMCO/long-term deferred — not needed for IPPP.)
- [x] **★ BASELINE CAVLC DECODER CONFORMANT ★** — ALL 100 BANM frames (I+P) decode
      BIT-EXACT vs C oracle (0/3,801,600 bytes). 90 tests, 0 warnings, wasm ok.
- [x] **P3e CABAC MB syntax** ✅ — full CABAC MB decode (I+P, mb_type/skip/sub_mb/
      intra-mode/ref_idx/mvd/cbp/delta_qp/cbf). BIT-EXACT vs C on test_cif_I_CABAC (300/300),
      test_qcif_cabac I+P (30/30), test_cif_P_CABAC multi-ref (300/300). Fixed P_8x8 ref
      cache + multi-ref deblock bugs. fc8b402. **Main profile CAVLC+CABAC I+P conformant.**
- [x] **P3f I_PCM** ✅ — CAVLC + CABAC I_PCM macroblock decode (see P4 below).
- [⏸] **P3f** B-slices, transform8x8/High, FMO, error-conceal — deferred (guarded
      Unsupported; not in baseline/Main-IP streams). Add when a target stream needs them.
- [ ] **P3f FMO** (`fmo.rs`), **error concealment** (`error_conceal.rs`) — lower priority.
- [ ] `dsp/expand.rs` border padding (deferred from P1) — needed by MC ref reads.

## P4 — Decoder API + conformance ✅
- [x] C `h264dec` oracle built; conformance harness `examples/conformance.rs`
- [x] **Corpus run: 28/54 streams BIT-EXACT** (3875 frames); 11 unsupported
      (8x8-transform/B/PCM), 11 mismatch (mostly crop/interlace len), 4 corrupted. d5011f7
- [x] **Output cropping fix → 38/54 BIT-EXACT.** `Picture` now carries its visible
      (post-crop) region (`visible_x/y/width/height`), stamped at emit time from the
      SPS `frame_cropping` rectangle; the conformance harness emits the cropped I420
      (4:2:0 chroma crop = luma/2). Moved jm_1080p_allslice (1920x1088→1080),
      Static, CVFC1_Sony_C (was not field-coded after all) to BITEXACT.
- [x] **I_PCM (CAVLC) → 39/54.** `MbType::IPcm` (intra for deblock; QP=0, nnz=16),
      `parse_pcm_mb_cavlc` byte-aligns past `pcm_alignment_zero_bit` then reads
      256+64+64 raw samples into the picture (`recon_pcm_mb`). CVPCMNL1_SVA_C BITEXACT.
- [x] **I_PCM (CABAC) → 40/54.** `CabacDecoder::read_pcm_bytes` reproduces
      RestoreCabacDecEngineToBS + InitCabacDecEngineFromBS (stream pos =
      `pBuffCurr - (iBitsLeft>>3)`, re-init engine past the 384 raw bytes).
      PCM neighbour coded_block_flag = 1 via `cbf_dc=0xFFFF` (spec 9.3.3.1.1.9).
      Wired into both I- and P-slice CABAC paths. QCIF_2P_I_allIPCM BITEXACT.
- [⏸] **sps_subsetsps_bothVUI DEFERRED.** Its only coded picture is in a NAL type
      20 (coded slice *extension*, SVC scalable layer) — no base-layer type-1/5
      slice exists. Decoding it needs full SVC scalable-extension slice support,
      a large separate feature (same class as interlace/field). Not a cheap bug.
- [x] **Perf report v1** `docs/PERF_REPORT.md` — Rust scalar 1.07-1.65× of C NEON, byte-identical
- [x] `src/api.rs` Rusty `Decoder` facade (incremental `decode`/`flush`/`decode_all`,
      `nal_units`) + `src/formats/` (`YUVSource`, `DecodedYuv`/`Frame`, BT.601
      `write_rgb8`/`write_rgba8`) — mirrors `openh264` crate shape, no_std-clean

## P5 — Encoder DSP ✅
- [x] fwd DCT/quant/DC-Hadamard/scan (dsp/transform.rs) · EncUT_EncoderMbAux ✓ (d8efa26)
- [x] SATD + four-pos SAD (dsp/satd.rs, sad.rs) · EncUT_Sample ✓ (fc81e7d)
- [x] enc intra predictors + SATD/SAD mode-cost combined3 helpers (encoder/intra_pred.rs)
      · EncUT_GetIntraPredictor ✓ (47854ea). 122 tests, encoder-only no_std builds.

## P6 — Encoder core 🚧
- [x] **Intra encoder MVP** ✅ — Encoder::new(w,h,qp)/encode_frame → Annex-B IDR.
      CAVLC writer (round-trip-validated vs decoder, 30k blocks exact), SPS/PPS gen,
      NAL encap, intra mode decision (SATD), fwd DCT/quant, in-place reconstruct.
      **Round-trip PSNR (encode→OUR decoder): QP26 Y=41.16dB, QP32 Y=37.27dB**, monotonic.
      134 tests. ed82638. Files: encoder/{cavlc_writer,paraset,nal_encap,encode_mb,mod}.rs.
- [x] **P-frame encode** ✅ — motion_est.rs (diamond + sub-pel, SATD), inter mode decision
      (P_Skip/P_16x16/intra), enc mv_pred (bit-identical to decoder), inter MB encode.
      **IPPP round-trip: QP26 I=40.4dB P≈41dB, P frames 10-16× smaller, NO drift.** 887aae2.
- [⏸] rate control, sub-16x16 partitions, multi-ref, CABAC-encode, B-encode — deferred
      (fixed-QP baseline IPPP works; encoder freedom — not correctness gaps).

## P7 — Encoder API + round-trip ✅
- [x] `Encoder::new(w,h,qp)`/`encode_frame`/`force_idr`; tests/encode_roundtrip.rs
      (intra + IPPP, encode→OUR-decoder PSNR). 139 tests, 7 targets + no_std + wasm.

## P6 — Encoder core ⬜
- [ ] `bits/writer.rs` + Exp-Golomb write · `EncUT_ExpGolomb`
- [ ] CAVLC write (`set_mb_syn_cavlc.cpp`) · `EncUT_Cavlc`
- [ ] mode decision, motion est · `EncUT_MotionEstimate`, `EncUT_SVC_me`
- [ ] mv_pred(enc), ref list mgr, rate control
- [ ] slice/mb encode, reconstruct · `EncUT_EncoderMb`, `EncUT_Reconstruct`
- [ ] NAL encap, paraset gen · `EncUT_ParameterSetStrategy`
- [ ] CABAC encode


## P8 — Processing ⏸  (downsample/denoise/scenechange/vaa)
## P9 — Threading ⏸  (slice MT + thread pool; std-only, wasm off)

## P10 — SIMD ✅  (NEON + wasm128 behind `simd`, bit-exact)
- [x] **simd.1 NEON** ✅ — `dsp/simd/{mod,neon}.rs`. aarch64 baseline (no runtime
      detect); each public kernel keeps a `*_scalar` body + cfg dispatch so callers
      are unchanged. NEON: `idct4x4_add` (dual 4x4 transpose), luma half-pel
      `mc_hor_ver20`/`mc_hor_ver02` + `pixel_avg` (vqrshrun / vrhadd) → quarter-pel
      positions accelerated transitively, and `sad` (vabd+vpadal). 2e46e10.
- [x] **simd.2 wasm simd128** ✅ — `dsp/simd/wasm.rs`: v128 `idct4x4_add` + `sad`.
      Builds with `-C target-feature=+simd128`; scalar fallback otherwise. Bit-exact
      cross-checked vs scalar under Node (50k IDCT + 12k SAD cases, 0 mismatch). a68c343.
- [x] **Bit-exact verified** — full conformance corpus stays 28/54 BITEXACT and all
      139 tests pass unchanged with `--features simd`. Kept scalar (bit-exactness):
      deblock filters, centre half-pel `mc_hor_ver22`, `mc_copy`. x86 SSE/AVX deferred.

## P11 — Perf report ✅ → `docs/PERF_REPORT.md`
- [x] Decode scalar-vs-SIMD-vs-C table (SIMD self-speedup 1.03-1.38x; BANM reaches
      ~parity with C NEON). Encoder section via `examples/bench_encode.rs` (IPPP
      ~1.74x SIMD speedup) + round-trip PSNR / I-vs-P bitrate. 78152da.
- [x] **`docs/STATUS.md`** — final honest project summary (ported/conformant, API,
      build+test matrix, deferred list).

---

## Notes / decisions log
- (P0) Single crate + features chosen over workspace; no_std+alloc; see ARCHITECTURE.md.
- Oracle build deferred; `brew install meson ninja nasm` when reaching P4.
