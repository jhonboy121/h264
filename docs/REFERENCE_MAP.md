# C → Rust Reference Map

Maps each OpenH264 C source unit to its Rust home. Paths under `reference/` are
the vendored C; paths under `src/` are the Rust port. Update as modules land.

## Decoder (`reference/codec/decoder/core/{src,inc}` → `src/decoder/`)

| C file                       | Purpose                              | Rust module                  |
|------------------------------|--------------------------------------|------------------------------|
| bit_stream.{cpp,h}           | bit reader plumbing                  | `src/bits/reader.rs`         |
| dec_golomb.h                 | Exp-Golomb ue/se/te + bit reads      | `src/bits/golomb.rs`         |
| au_parser.cpp                | NAL/AU parse, SPS/PPS/slice header   | `src/decoder/nal.rs`, `params.rs`, `slice_header.rs` |
| nalu.h, nal_prefix.h         | NAL unit structs                     | `src/decoder/nal.rs`         |
| parameter_sets.h             | SPS/PPS structs                      | `src/decoder/params.rs`      |
| slice.h                      | slice header/data structs            | `src/decoder/slice_header.rs`|
| vlc_decoder.h                | CAVLC VLC tables                     | `src/decoder/cavlc_tables.rs`|
| parse_mb_syn_cavlc.cpp       | CAVLC MB syntax parse                | `src/decoder/cavlc.rs`       |
| cabac_decoder.{cpp,h}        | CABAC arithmetic engine              | `src/decoder/cabac.rs`       |
| parse_mb_syn_cabac.cpp       | CABAC MB syntax parse                | `src/decoder/cabac_mb.rs`    |
| decode_mb_aux.cpp            | IDCT / inverse transform / dequant   | `src/dsp/transform.rs`       |
| get_intra_predictor.cpp      | intra prediction (decoder)           | `src/dsp/intra_pred.rs`      |
| mv_pred.cpp                  | motion vector prediction             | `src/decoder/mv_pred.rs`     |
| rec_mb.cpp                   | macroblock reconstruction            | `src/decoder/recon.rs`       |
| decode_slice.cpp             | slice decode loop                    | `src/decoder/decode_slice.rs`|
| deblocking.cpp               | in-loop deblocking filter            | `src/dsp/deblock.rs`         |
| mc.cpp (common)              | motion compensation / interpolation  | `src/dsp/mc.rs`              |
| manage_dec_ref.cpp           | reference picture mgmt               | `src/decoder/ref_pic.rs`     |
| pic_queue.cpp, picture.h     | decoded picture buffer               | `src/decoder/dpb.rs`         |
| memmgr_nal_unit.cpp          | NAL memory mgmt                      | folded into `nal.rs`         |
| fmo.cpp                      | flexible macroblock ordering         | `src/decoder/fmo.rs`         |
| error_concealment.cpp        | error concealment                    | `src/decoder/error_conceal.rs`|
| decoder_core.cpp, decoder.cpp| top-level decode orchestration       | `src/decoder/mod.rs`         |
| wels_decoder_thread.cpp      | threaded decode                      | `src/decoder/threading.rs` (P9)|
| plus/welsDecoderExt.cpp      | C API impl                           | replaced by `src/api.rs`     |
| decoder_data_tables.cpp      | static tables                        | `src/decoder/tables.rs`      |

## Common DSP (`reference/codec/common/src` → `src/dsp/`)

| C file                  | Purpose                       | Rust module             |
|-------------------------|-------------------------------|-------------------------|
| intra_pred_common.cpp   | shared intra pred             | `src/dsp/intra_pred.rs` |
| mc.cpp                  | motion comp (luma/chroma)     | `src/dsp/mc.rs`         |
| deblocking_common.cpp   | deblock edge filters          | `src/dsp/deblock.rs`    |
| copy_mb.cpp             | block copies                  | `src/dsp/copy.rs`       |
| expand_pic.cpp          | picture border expansion      | `src/dsp/expand.rs`     |
| sad_common.cpp          | SAD                           | `src/dsp/sad.rs`        |
| common_tables.cpp       | shared static tables          | `src/dsp/tables.rs`     |
| cpu.cpp                 | runtime CPU feature detect    | `src/dsp/cpu.rs` (P10)  |
| mem_align, crt_util     | aligned alloc helpers         | `src/util/mem.rs`       |
| Wels*Thread*.cpp        | threading primitives          | `src/util/threading.rs` (P9)|
| utils.cpp, welsCodecTrace | logging/util                | `src/util/mod.rs`       |

## Encoder (`reference/codec/encoder/core/{src,inc}` → `src/encoder/`)

| C file                    | Purpose                        | Rust module                  |
|---------------------------|--------------------------------|------------------------------|
| encode_mb_aux.cpp         | fwd transform/quant, zigzag    | `src/dsp/transform.rs` (fwd) |
| decode_mb_aux.cpp         | inverse transform (enc recon)  | `src/dsp/transform.rs`       |
| get_intra_predictor.cpp   | intra predictor (enc)          | `src/dsp/intra_pred.rs`      |
| sample.cpp                | SAD/SATD sample metrics        | `src/dsp/sad.rs`             |
| set_mb_syn_cavlc.cpp      | CAVLC MB syntax write          | `src/encoder/cavlc_enc.rs`   |
| set_mb_syn_cabac.cpp      | CABAC MB syntax write          | `src/encoder/cabac_enc.rs`   |
| md.cpp, svc_mode_decision | mode decision                  | `src/encoder/mode_decision.rs`|
| svc_motion_estimate.cpp   | motion estimation              | `src/encoder/motion_est.rs`  |
| mv_pred.cpp               | mv prediction (enc)            | `src/encoder/mv_pred.rs`     |
| ratectl.cpp               | rate control                   | `src/encoder/rate_control.rs`|
| ref_list_mgr_svc.cpp      | reference list mgmt            | `src/encoder/ref_list.rs`    |
| svc_encode_slice.cpp      | slice encode                   | `src/encoder/encode_slice.rs`|
| svc_encode_mb.cpp         | mb encode                      | `src/encoder/encode_mb.rs`   |
| svc_enc_slice_segment.cpp | slice segmentation             | `src/encoder/slice_segment.rs`|
| nal_encap.cpp             | NAL encapsulation (EPB)        | `src/encoder/nal_encap.rs`   |
| au_set.cpp, paraset_strategy | SPS/PPS generation          | `src/encoder/paraset.rs`     |
| picture_handle.cpp        | source picture handling        | `src/encoder/picture.rs`     |
| deblocking.cpp            | deblock (enc)                  | `src/dsp/deblock.rs`         |
| encoder_ext.cpp           | C API impl                     | replaced by `src/api.rs`     |
| slice_multi_threading.cpp, wels_task_* | threading         | `src/encoder/threading.rs` (P9)|

## Bit writer (encoder)
| C: bit writing in set_mb_syn_*, nal_encap | `src/bits/writer.rs` |

## Processing (`reference/codec/processing/src` → `src/processing/`) — P8
downsample, denoise, scenechangedetection, vaacalc, complexityanalysis, etc.

## API (`reference/codec/api/wels/*.h`) → `src/api.rs` (Rusty, not 1:1).
Mirror ergonomics of `openh264-rs` crate: `Decoder::decode(&[u8]) -> DecodedYuv`,
`Encoder::encode(&YuvSource) -> EncodedBitstream`. YUV/RGB conversion in `src/formats/`.

## Tests (`reference/test/{decoder,encoder,api}` → `tests/` + inline `#[cfg(test)]`)
Per-kernel gtests → Rust unit tests next to the kernel. API/e2e gtests → `tests/`.
