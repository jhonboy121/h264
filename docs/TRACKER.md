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
I_PCM (CAVLC+CABAC) → 40/54 BIT-EXACT, then **B-slice (bi-predictive) decode → 43/54**,
then **High-profile CAVLC temporal-direct (P14) → 46/54 BIT-EXACT**, then **CABAC
temporal-direct (P15) → 49/54 BIT-EXACT**. Deferred (explicit, not gaps):
transform8x8/High, FMO, error-conceal, SVC scalable-extension (NAL type 20); encoder
sub-16x16, multi-ref, B, CABAC-encode, rate control; P8 processing; P9 threading;
SIMD for deblock/mc_hor_ver22/x86. **P16** since added encoder rate control, the
downsampler and a live re-encode API (see the end of this file).

## P12 — Decoder B-slices (bi-predictive) ✅ → 43/54 BIT-EXACT
- [x] **POC derivation** (`PocState`, spec 8.2.1 type 0 + type 2; type 1 monotonic
      fallback) + display-order output reordering (POC-sorted within each CVS;
      a no-op for monotonic-POC I/P streams).
- [x] **B reference lists** (`Dpb::b_ref_lists`, 8.2.4.2.3 list-0/list-1 by POC,
      reorder for both) + colocated motion retained per ref picture (`ColMotion`).
- [x] **B CAVLC parse** (`mb_parse_cavlc`: `g_ksInterBMbTypeInfo`/`g_ksInterBSubMbTypeInfo`
      tables, ref_idx/mvd both lists, B_Skip/B_Direct, B_8x8 sub_mb_type).
- [x] **B CABAC parse** (`mb_parse_cabac`: B skip/mb_type/sub_mb_type context models
      at offsets 27/36/+13, both-list ref/mvd, B_Skip/B_Direct).
- [x] **Spatial direct** (`bdirect.rs`: neighbour ref/MV derivation + colZeroFlag).
- [x] **Bi-predictive reconstruction** (`recon_inter::recon_b_mb`: uni L0/L1 +
      default `(p0+p1+1)>>1` bi-average) — **incl. the OpenH264 16x8/8x16 bi quirk**
      (destination-pointer over-advance discards the bi-average: part0→L1, part1→L0).
- [x] **B deblock bS** (cross-list reference/MV comparison, `IN_SMB_EDGE_MV`/`ON_MB_BS`).
- [x] Bit-exact on all three corpus B streams (CAVLC Adobe 1024x768, CAVLC + CABAC
      Men_whisper 640x320).
- [x] **Temporal direct** (POC MV scaling, 16x16 + 8x8): now fully validated and
      bit-exact on all six `VID_*_temporal_direct` streams (CAVLC P14 + CABAC P15),
      including High-profile `transform_8x8`, `MapColToList0` multi-ref, and B_8x8
      mixed-direct-sub temporal prediction.
- [⏸] Explicit weighted bipred (`weighted_bipred_idc != 0`): not needed (corpus B
      streams use idc 0 / default averaging).

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

## P9 — Threading ✅  (multi-slice encode + slice/frame parallelism; std-only, wasm off)
- [x] **P9.1 multi-slice encode** ✅ — `Encoder::new_with_slices` / `set_slices`
      partition each frame into `S` contiguous MB-row-band slices, each its own
      slice NAL (correct `first_mb_in_slice`), independent CAVLC bitstream, and
      prediction state reset at the boundary (intra availability, mvd/MV pred and
      nC neighbour derivation gated by `slice_top_y`, mirroring the decoder's
      per-MB `slice_idc`). Deblocking off (`disable_deblocking_filter_idc=1`,
      output==recon) ⇒ slices fully independent, no cross-slice filter. P-slice
      `mb_skip_run` flushed per slice. `FrameEnc` refactored to borrow src+ref
      planes. Validated S=4/8 round-trip vs single-slice in
      `tests/encode_multislice.rs` (PSNR matches, decoder accepts).
- [x] **P9.2 threaded fan-out** ✅ (`threads` feature, `std::thread::scope`,
      zero-dep) — `encode_frames_parallel` (all-intra frame-parallel, N
      independent IDRs, 0% overhead) and `encode_frame_parallel` (IPPP
      slice-parallel: frames serial for the P reference, S slices concurrent,
      then a serial reconstruction-stitch + border-extend). **Byte-identical** to
      serial (asserted, `tests/encode_threaded.rs`). wasm builds with threads off.
- [x] **P9.3 1080p benchmark** ✅ — `examples/bench_encode_hd.rs`. **Real-time 30
      fps 1080p reached:** threaded IPPP **127 fps (4.2×RT30)**, all-intra **300+
      fps (10×RT30)** vs single-thread 13.7 / 25 fps. ~7–9% bits for 18-slice
      IPPP, 0% for frame-parallel intra. See `PERF_REPORT.md` §3.

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

## P13 — High-profile 8×8 transform / I_8×8 + temporal-direct multi-ref
- [x] `transform_size_8x8_flag` + I_8×8 intra (CAVLC & CABAC): per-8×8 luma mode
      parse (prev/rem with neighbour median pred, 8×8 right-top rule), 8×8 luma
      residual (CAVLC interleaved 4×4→64-coeff zig-zag8×8; CABAC 64-pos sig map),
      flat 8×8 dequant, `idct8x8` recon, 8×8-transform deblock (skip internal
      4-sample luma edges, OR the 8×8-block nzc for inter bS). I_8×8 IDR frame
      bit-exact (CAVLC). 055828a.
- [x] Inter 8×8 transform (P/B) + temporal-direct multi-ref + weighted prediction
      (CAVLC): `MapColToList0` (retain colocated L0 ref id + resolved colocated
      partition mode in DPB), 3-mode (16×16/8×8/4×4) partition selection, B_8×8
      direct subs via temporal direct (cache ref_idx=-1 so explicit subs exclude
      them from MV prediction), per-8×8 direct MC, implicit weighted bi-pred
      (`weighted_bipred_idc==2`), explicit P weighted pred. 0246b3e.
- CABAC VID streams: I_8×8 intra parse in place; the B CABAC path (multi-ref
  ref_idx + temporal direct) is not yet wired → ERROR/UNSUPPORTED.

## P14 — CAVLC VID temporal-direct streams BIT-EXACT ✅ → 46/54 BIT-EXACT
All three `VID_*_cavlc_temporal_direct` streams (1280×544, 1280×720, 1920×1080) are
now bit-exact vs the OpenH264 oracle. Three root causes, all in the B-direct +
deblock machinery (verified against the OpenH264 source by instrumenting the oracle
to dump per-MB bS / per-block ref+MV and a per-decode-frame recon checksum):
- [x] **Skip-MB internal deblock edges** (`deblock.rs inter_bs_b`): `WelsDeblockingMb`
      forces every internal bS to 0 for `IS_SKIP` (P_Skip *and* B_Skip), even when
      temporal-direct 8×8 sub-blocks reference different pictures. The B-slice bS path
      lacked the skip early-return the P-slice path already had, so it filtered the
      x=8/y=8 internal edges (bS=1) the oracle leaves untouched (±1 on p1/q1). Added
      the `is_skip()` early-return after the MB-boundary (marginal) edges.
- [x] **Direct ref_pic_id overwrite** (`mb_parse_cavlc.rs apply_b_direct`, temporal):
      after `b_direct_temporal` correctly stored each block's L0 identity
      (`cur_ref0_ids[ref0]`, `ref0` = `MapColToList0`, not necessarily 0), a stale loop
      clobbered every block's L0 `ref_pic_id` to list-0 index 0. When `ref0 != 0` this
      made the deblock boundary-strength reference comparison see two different ids for
      one physical picture → wrong marginal bS. Removed the overwrite (L1 part was a
      no-op since `col_id == ref_pic_ids[1][0]`).
- [x] **Spatial-direct colZero granularity** (`apply_b_direct`, spatial): a whole-MB
      spatial-direct MB (B_Skip / B_Direct_16x16) whose colocated MB forces 8×8
      resolution must evaluate `colZeroFlag` **per 8×8 sub-block**, not once from
      block 0. Branching on `info.mb16x16` (syntax-only) zeroed mvL1 uniformly across
      the MB and lost the per-8×8 variation; now branch on `!direct_8x8` (the resolved
      mode, same predicate the temporal path uses), filling per-8×8 colZero.
- CABAC VID streams untouched (separate B-CABAC-multiref work) → still ERROR/UNSUPPORTED.

## P15 — CABAC VID temporal-direct streams BIT-EXACT ✅ → 49/54 BIT-EXACT
All three `VID_*_cabac_temporal_direct` streams (1280×544 / 1280×720 / 1920×1080) are now
fully bit-exact (151 / 300 / 54 frames). Work done, all verified against an instrumented
OpenH264 oracle (per-MB mb_type / ref_idx ctxInc / dequantised-coeff / MV traces):
- [x] **`mb_qp_delta` sign bug** (`mb_parse_cabac.rs parse_delta_qp`) — the dominant fix.
      The C maps `uiCode = unary+1`, magnitude `(uiCode+1)>>1`, sign from the parity of
      `uiCode` (negative when even). Our code did `code = unary+2` and used *that* for both
      magnitude and parity, flipping the sign of **every** non-zero `mb_qp_delta`. Latent
      because the previously bit-exact CABAC clips use constant QP; the first non-zero
      delta (I-frame MB 7 here) mis-set QP, mis-scaling all subsequent dequant. Fixing it
      made the I-frame bit-exact (entropy was already in sync — QP only affects dequant).
- [x] **CABAC inter `transform_size_8x8_flag`** (`parse_inter_t8_flag_cabac`, wired into
      the P and B inter paths) — was hard-coded `transform_8x8: false`, so High-profile
      inter MBs desynced the residual parse. Mirrors the CAVLC presence condition
      (16x16/16x8/8x16, or 8x8 with all-8x8 subparts, `cbp_l != 0`, PPS 8x8 mode).
- [x] **B multi-ref `ParseRefIdxCabac`** (`parse_ref_idx_b`) — implements the B-slice ctx
      derivation (neighbour `ref_idx > 0` AND not direct-coded), with a per-block `direct`
      neighbour cache (`WelsFillDirectCacheCabac`) and per-MB `pDirect` reset; removes the
      `UNSUPPORTED` guard. Per-partition ref now stored into the per-MB ref array between
      partitions so the next partition's ctx sees it.
- [x] **B temporal-direct 8x8 (CABAC)** — `parse_b_8x8_cabac` now branches spatial vs
      temporal (`b_direct_temporal_sub`) like the CAVLC path, instead of always spatial.
- [x] **Temporal-direct 8×8 sub-partition MV-predictor neighbour ref (the final fix)** —
      `parse_b_8x8_cabac` was seeding the MV-prediction ref cache of a *temporal*-direct
      8×8 sub-partition to `REF_NOT_IN_LIST` (-1). The C reference instead writes the
      colocated-derived `ref_idx` (`iRef[LIST_0]`=`MapColToList0`, `iRef[LIST_1]`=0) into
      both the layer and the MV-prediction ref cache (`Update8x8RefIdx` +
      `UpdateP8x8RefCacheIdxCabac`, temporal branch of `ParseInterBMotionInfoCabac`), and
      its later non-direct ref loop leaves it untouched. With our -1, a subsequent
      non-direct sub-partition whose only ref-matching neighbour was a direct one fell out
      of the single-match `PredMv` rule into the 3-way median, yielding a wrong predictor
      (e.g. 1280×544 B-poc4 MB38 partition-3 L0: predictor `(2,-3)`→`(0,0)`). Fixed by
      seeding `iref` from the real `ctx.ref_idx`/`ctx.ref_idx_l1` that `b_direct_temporal_sub`
      already stored. Isolated by a systematic per-MB field diff (mb_type / cbp / qp / t8 /
      ref0 / ref1 / mv0 / mv1 / mvd / nzc) of the first intrinsically-diverging decode-order
      picture (POC 4, a B-reference slice whose own references were all bit-exact).

## P16 — Live re-encode: rate control, downsampler, picture/NAL API ✅
For a host CLI that decodes a looping clip, downscales it and re-encodes it the way
a phone's MediaCodec encoder would (bitrate changed mid-stream without an IDR,
size / rate steps as a new encoder, IDRs on request).
- [x] **Rate control** (`encoder/ratectl.rs`): frame-level port of `RC_BITRATE_MODE`
      (`WelsRcPictureInitGom` / `WelsRcPictureInfoUpdateGom`, `bFixRCOverShoot`):
      8-frame virtual GOP, IDR budget 4 frames, P budget clamped to 0.55..1.5 of
      `bitrate / fps`, overshoot carried into the next VGOP, bitrate/fps changes
      rescale the remaining bits (`RcUpdateBitrateFps`). QP from the linear model
      `bits * qstep` scaled by frame complexity / its running mean (±20 %), step
      limited to −3/+5 per frame, range 12..=45; the first IDR QP from the bpp
      table (`RcCalculateIdrQp`). Simplified: no GOM/MB-level QP (the C turns it
      off for IDRs and multi-slice anyway), no adaptive quant, no frame skipping
      (an exhausted budget raises the QP by 3 instead), and complexity comes from
      the encoder (16×16 SAD vs the previous source for P; min of vertical /
      horizontal source-prediction SAD per MB for IDR) instead of VAA.
      `RcConvertQStep2Qp`'s `log` is replaced by an exact integer threshold table.
- [x] **Frame QP in the slice header** (`slice_qp_delta` against `pic_init_qp` 26);
      the fixed-QP API is unchanged (delta 0).
- [x] **SPS:** level from `WelsGetLevelIdc` (size, MB rate, DPB, bitrate) for
      `with_config` encoders; `constraint_set0/1` (Constrained Baseline) always.
      Back-to-back IDRs alternate `idr_pic_id` (rate-controlled encoders).
- [x] **P-slice intra early-out** (`WelsMdFirstIntraMode`): the full intra decision
      (I4x4 + chroma) only runs when the I16×16 SATD estimate beats inter.
      1080p rate-controlled encode 76 → 107 fps.
- [x] **API:** `EncoderConfig` / `EncodedFrame`, `Encoder::with_config`, `encode`
      (slice-parallel with `threads` when `slices > 1`), `set_bitrate`,
      `set_frame_rate` (feeds RC), `reconstruction()`; `EncodeError::InvalidFrame`.
      `YuvRef` / `I420` (`image.rs`), `DecodedYuv::yuv` / `Frame::yuv`,
      `nal::{nal_type, SPS, PPS, AUD, IDR}`, `sps_dimensions` (decoder's parser).
- [x] **Downsampler** (`processing/downsample.rs`, `scale_i420`): `CDownsampling::
      Process` — dyadic halving while the half is still larger, then an exact
      dyadic pass or the general bilinear kernels (fast for luma, accurate for
      chroma, as the `_c` table). PORT: other sizes (upscale / mixed) use the same
      general kernel with clamped neighbour reads; the halving chain is decided per
      plane.
- [x] **Tests:** `tests/encode_ratecontrol.rs` (2 s windows within ±15 % on the
      five target configs and on a re-encoded conformance clip; `set_bitrate` down
      and up within ±20 % in the second second, no IDR; RC streams incl. 1080 /
      540 / 360 cropping, keyframe interval, forced and back-to-back IDRs decode to
      exactly the encoder's reconstruction), `tests/scale_and_nal.rs` (dyadic
      paths vs an independent reference, gradients stay gradients, High-profile
      SPS with scaling lists + cropping + VUI), RC / level / slice-header unit tests.
- [x] **Bench:** `examples/bench_realtime.rs` (`--release --features threads,simd`,
      16-core aarch64): 1920×1080@60/6000k 16 slices **~107 fps**, 1080p30/4000k
      ~109 fps, 1280×720@30/2000k 16 slices **~263 fps** (8 slices ~149 fps),
      640×360@15/350k ~293 fps; `scale_i420` 1080p → 720p 3.9 ms, → 540p 1.1 ms,
      → 360p 2.1 ms (single thread).
