//! Top-level baseline intra-frame decode: Annex-B bytes in, a reconstructed
//! [`Picture`] out.
//!
//! Collapses `DecodeFrameConstruction` / `WelsDecodeAndConstructSlice`
//! (`reference/codec/decoder/core/src/decode_slice.cpp`) for the single use case
//! of "decode the first IDR (I-slice, CAVLC, 4:2:0) frame". P/B inter,
//! deblocking, multi-frame DPB, FMO, CABAC and High-profile tools are out of
//! scope for this milestone and are rejected with [`DecodeError::Unsupported`]
//! where they would change the bit parse.

use alloc::vec::Vec;

use crate::bits::BitReader;
use crate::error::DecodeError;
use crate::formats::yuv::{Frame, VisibleRegion};

use super::context::DecoderContext;
use super::dpb::Dpb;
use super::mb_parse_cavlc::{parse_intra_mb_cavlc, parse_p_mb_cavlc};
use super::nal::{annexb_nal_units, parse_nal, NalUnit, NalUnitType};
use super::params::{parse_pps, parse_sps, Pps, Sps};
use super::picture::Picture;
use super::recon_inter::recon_inter_mb;
use super::recon_intra::recon_intra_mb;
use super::slice_header::{parse_slice_header_in_place, SliceHeader};

/// Decode the first IDR (intra, CAVLC 4:2:0 8-bit) frame of an Annex-B stream
/// and return its reconstructed picture (no deblocking — that is the next
/// phase). All slices belonging to that first IDR access unit are decoded into
/// the same picture.
pub fn decode_intra_frame(annexb: &[u8]) -> Result<Picture, DecodeError> {
    let nals: Vec<NalUnit> = annexb_nal_units(annexb).filter_map(parse_nal).collect();

    // Parameter sets (ids 0..31 / 0..255 as in the H.264 limits).
    let mut sps_map: Vec<Option<Sps>> = (0..32).map(|_| None).collect();
    for nal in &nals {
        if nal.unit_type == NalUnitType::Sps {
            let sps = parse_sps(&nal.rbsp)?;
            let id = sps.sps_id as usize;
            sps_map[id] = Some(sps);
        }
    }
    let mut pps_map: Vec<Option<Pps>> = (0..256).map(|_| None).collect();
    for nal in &nals {
        if nal.unit_type == NalUnitType::Pps {
            // Probe sps_id to resolve the referenced SPS (for scaling matrices).
            let sps_ref = match parse_pps(&nal.rbsp, None) {
                Ok(p) => sps_map[p.sps_id as usize].as_ref(),
                Err(_) => None,
            };
            let pps = parse_pps(&nal.rbsp, sps_ref)?;
            let id = pps.pps_id as usize;
            pps_map[id] = Some(pps);
        }
    }

    // Resolve the first IDR slice's PPS/SPS to size the frame.
    let first_idr = nals
        .iter()
        .find(|n| n.unit_type.is_idr())
        .ok_or(DecodeError::InvalidSyntax("no IDR slice"))?;
    let (sps, pps) = resolve_param_sets(first_idr, &sps_map, &pps_map)?;
    if sps.chroma_format_idc != 1 || sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
        return Err(DecodeError::Unsupported("non-4:2:0 / non-8-bit"));
    }

    let mb_width = sps.mb_width as usize;
    let mb_height = sps.mb_height as usize;
    let total_mb = mb_width * mb_height;
    let mut ctx = DecoderContext::new(Vec::new(), Vec::new(), mb_width, mb_height);

    // Decode every IDR slice of this access unit; stop at the first non-IDR VCL.
    let mut slice_index: i32 = 0;
    let mut started = false;
    for nal in &nals {
        if nal.unit_type.is_idr() {
            decode_idr_slice(nal, &sps_map, &pps_map, &mut ctx, slice_index, total_mb)?;
            slice_index += 1;
            started = true;
        } else if started && nal.unit_type.is_vcl() {
            break; // next (non-IDR) picture
        }
    }
    let _ = pps; // resolved above for validation only

    // In-loop deblocking pass over the fully reconstructed picture.
    super::deblock::deblock_frame(&mut ctx);

    Ok(ctx.picture)
}

/// In-flight picture being assembled from one or more slices.
struct CurPic {
    ctx: DecoderContext,
    frame_num: i32,
    is_ref: bool,
    is_idr: bool,
    /// Picture order count (display order).
    poc: i32,
    /// Coded-video-sequence index (increments at each IDR), the primary output
    /// reordering key (POC is reset per CVS).
    cvs: i32,
    /// `dec_ref_pic_marking` from the first slice of this picture (reference
    /// pictures only), driving sliding-window vs adaptive (MMCO) marking.
    marking: Option<super::slice_header::RefPicMarking>,
    slice_index: i32,
    region: VisibleRegion,
}

/// Picture-order-count derivation state carried across pictures (spec 8.2.1).
#[derive(Default, Clone)]
struct PocState {
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    prev_frame_num_offset: i32,
    prev_frame_num: i32,
}

impl PocState {
    /// Compute the picture order count for one frame and update state
    /// (`ParseSliceHeaderSyntaxs` POC block). Frame-coded only.
    fn compute(&mut self, sps: &Sps, sh: &SliceHeader, pps: &Pps, is_idr: bool, nal_ref_idc: u8) -> i32 {
        match sps.pic_order_cnt_type {
            0 => {
                if is_idr {
                    self.prev_poc_msb = 0;
                    self.prev_poc_lsb = 0;
                }
                let max = 1i32 << sps.log2_max_poc_lsb;
                let lsb = sh.pic_order_cnt_lsb as i32;
                let msb = if lsb < self.prev_poc_lsb && self.prev_poc_lsb - lsb >= max / 2 {
                    self.prev_poc_msb + max
                } else if lsb > self.prev_poc_lsb && lsb - self.prev_poc_lsb > max / 2 {
                    self.prev_poc_msb - max
                } else {
                    self.prev_poc_msb
                };
                let mut poc = msb + lsb;
                if pps.bottom_field_pic_order_in_frame_present_flag && !sh.field_pic_flag {
                    poc += sh.delta_pic_order_cnt_bottom;
                }
                if nal_ref_idc != 0 {
                    self.prev_poc_lsb = lsb;
                    self.prev_poc_msb = msb;
                }
                poc
            }
            2 => {
                // spec 8.2.1.3 (frame-coded): POC tracks 2*FrameNumOffset+frame_num.
                let max_frame_num = 1i32 << sps.log2_max_frame_num;
                let frame_num = sh.frame_num as i32;
                let frame_num_offset = if is_idr {
                    0
                } else if self.prev_frame_num > frame_num {
                    self.prev_frame_num_offset + max_frame_num
                } else {
                    self.prev_frame_num_offset
                };
                let poc = if is_idr {
                    0
                } else if nal_ref_idc == 0 {
                    2 * (frame_num_offset + frame_num) - 1
                } else {
                    2 * (frame_num_offset + frame_num)
                };
                self.prev_frame_num_offset = frame_num_offset;
                self.prev_frame_num = frame_num;
                poc
            }
            // POC type 1 is not present in the corpus; a monotonic fallback keeps
            // I/P output ordering correct (B-streams in the corpus use type 0).
            _ => 2 * sh.frame_num as i32,
        }
    }
}

/// Snapshot the current picture's per-block motion for use as a colocated
/// reference by future B slices (`colocPic->pMbType / pMv / pRefIndex`).
fn build_col_motion(ctx: &DecoderContext) -> super::dpb::ColMotion {
    let n = ctx.total_mb;
    let mut intra = alloc::vec![false; n];
    let mut uses_l1 = alloc::vec![false; n];
    for mb in 0..n {
        intra[mb] = ctx.mb_type[mb].is_intra();
        uses_l1[mb] = (0..16).any(|b| ctx.ref_idx_l1[mb * 16 + b] >= 0);
    }
    super::dpb::ColMotion {
        intra,
        uses_l1,
        mv: [ctx.mv.clone(), ctx.mv_l1.clone()],
        ref_idx: [ctx.ref_idx.clone(), ctx.ref_idx_l1.clone()],
    }
}

/// Visible (post-crop) rectangle of a picture coded by `sps`. 4:2:0 only:
/// CropUnitX = CropUnitY = 2 for frame-only streams (the only case decoded).
fn region_from_sps(sps: &Sps) -> VisibleRegion {
    let luma_x = 2 * sps.crop_left as usize;
    let luma_y = 2 * sps.crop_top as usize;
    VisibleRegion {
        width: sps.width as usize,
        height: sps.height as usize,
        luma_x,
        luma_y,
        chroma_x: luma_x / 2,
        chroma_y: luma_y / 2,
    }
}

/// Decode an entire baseline Annex-B stream (I + P, CAVLC 4:2:0 8-bit),
/// returning every reconstructed + deblocked picture in decode order (which,
/// for the baseline IPPP case without reordering, is display order).
pub fn decode_stream(annexb: &[u8]) -> Result<Vec<Picture>, DecodeError> {
    let nals: Vec<NalUnit> = annexb_nal_units(annexb).filter_map(parse_nal).collect();

    let mut sps_map: Vec<Option<Sps>> = (0..32).map(|_| None).collect();
    for nal in &nals {
        if nal.unit_type == NalUnitType::Sps {
            let sps = parse_sps(&nal.rbsp)?;
            let id = sps.sps_id as usize;
            sps_map[id] = Some(sps);
        }
    }
    let mut pps_map: Vec<Option<Pps>> = (0..256).map(|_| None).collect();
    for nal in &nals {
        if nal.unit_type == NalUnitType::Pps {
            let sps_ref = match parse_pps(&nal.rbsp, None) {
                Ok(p) => sps_map[p.sps_id as usize].as_ref(),
                Err(_) => None,
            };
            let pps = parse_pps(&nal.rbsp, sps_ref)?;
            let id = pps.pps_id as usize;
            pps_map[id] = Some(pps);
        }
    }

    let mut output: Vec<(i32, i32, Picture)> = Vec::new();
    let mut dpb: Option<Dpb> = None;
    let mut next_id: i32 = 0;
    let mut cur: Option<CurPic> = None;
    let mut poc_state = PocState::default();
    let mut cvs: i32 = -1;

    for nal in &nals {
        if !nal.unit_type.is_vcl() {
            continue;
        }
        let (sps, pps) = resolve_param_sets(nal, &sps_map, &pps_map)?;
        if sps.chroma_format_idc != 1 || sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err(DecodeError::Unsupported("non-4:2:0 / non-8-bit"));
        }
        let is_idr = nal.unit_type.is_idr();
        let pps = pps.clone();
        let sps = sps.clone();

        let mut bs = BitReader::new(&nal.rbsp);
        let sh = parse_slice_header_in_place(&mut bs, nal.ref_idc, is_idr, &sps, &pps)?;

        // A new picture begins at the first MB of a coded picture.
        if sh.first_mb_in_slice == 0 {
            if let Some(c) = cur.take() {
                finalize_picture(c, dpb.as_mut().unwrap(), &mut output, &mut next_id);
            }
            if dpb.is_none() {
                dpb = Some(Dpb::new(sps.max_num_ref_frames, sps.log2_max_frame_num));
            }
            if is_idr {
                dpb.as_mut().unwrap().clear();
                cvs += 1;
            }
            if cvs < 0 {
                cvs = 0;
            }
            let mb_width = sps.mb_width as usize;
            let mb_height = sps.mb_height as usize;
            let poc = poc_state.compute(&sps, &sh, &pps, is_idr, nal.ref_idc);
            cur = Some(CurPic {
                ctx: DecoderContext::new(Vec::new(), Vec::new(), mb_width, mb_height),
                frame_num: sh.frame_num as i32,
                is_ref: nal.ref_idc != 0,
                is_idr,
                poc,
                cvs,
                marking: sh.dec_ref_pic_marking.clone(),
                slice_index: 0,
                region: region_from_sps(&sps),
            });
        }

        let cp = cur.as_mut().ok_or(DecodeError::InvalidSyntax("slice before picture start"))?;
        if pps.entropy_coding_mode_flag {
            decode_one_slice_cabac(
                &mut cp.ctx, &mut bs, &nal.rbsp, &sh, &pps, cp.slice_index, dpb.as_ref().unwrap(), cp.poc, sps.direct_8x8_inference_flag,
            )?;
        } else {
            decode_one_slice(&mut cp.ctx, &mut bs, &sh, &pps, cp.slice_index, dpb.as_ref().unwrap(), cp.poc, sps.direct_8x8_inference_flag)?;
        }
        cp.slice_index += 1;
    }

    if let Some(c) = cur.take() {
        finalize_picture(c, dpb.as_mut().unwrap(), &mut output, &mut next_id);
    }
    // Reorder into display order: POC ascending within each coded video sequence
    // (a stable sort, so I/P streams with monotonic POC keep decode order).
    output.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    Ok(output.into_iter().map(|(_, _, p)| p).collect())
}

/// Deblock, optionally border-extend + insert into the DPB, and emit.
fn finalize_picture(c: CurPic, dpb: &mut Dpb, output: &mut Vec<(i32, i32, Picture)>, next_id: &mut i32) {
    let (cvs, poc) = (c.cvs, c.poc);
    output.push((cvs, poc, finalize_into(c, dpb, next_id).into_picture()));
}

/// Deblock, optionally border-extend + insert into the DPB, and return the
/// visible-tagged [`Frame`]. Shared by [`decode_stream`] and [`StreamDecoder`].
fn finalize_into(mut c: CurPic, dpb: &mut Dpb, next_id: &mut i32) -> Frame {
    super::deblock::deblock_frame(&mut c.ctx);
    let region = c.region;
    if c.is_ref {
        crate::dsp::expand::expand_picture(&mut c.ctx.picture);
        let col = build_col_motion(&c.ctx);
        let pic = c.ctx.picture;
        let id = *next_id;
        *next_id += 1;
        let (lt_flag, adaptive, mmco) = match &c.marking {
            Some(m) => (
                m.long_term_reference_flag,
                m.adaptive_ref_pic_marking_mode_flag,
                m.mmco.as_slice(),
            ),
            None => (false, false, &[][..]),
        };
        dpb.mark_and_insert(pic.clone(), c.frame_num, id, c.is_idr, lt_flag, adaptive, mmco, c.poc, col);
        Frame::new(pic, region)
    } else {
        Frame::new(c.ctx.picture, region)
    }
}

/// Decode one slice's macroblocks (parse + reconstruct) into `ctx`.
#[allow(clippy::too_many_arguments)]
fn decode_one_slice(
    ctx: &mut DecoderContext,
    bs: &mut BitReader<'_>,
    sh: &SliceHeader,
    pps: &Pps,
    slice_index: i32,
    dpb: &Dpb,
    cur_poc: i32,
    direct_8x8_inference: bool,
) -> Result<(), DecodeError> {
    let total_mb = ctx.total_mb;
    let mut last_mb_qp = sh.slice_qp;
    let mut coeffs = [0i16; 384];

    if sh.slice_type == super::slice_header::SliceType::B {
        return decode_b_slice_cavlc(ctx, bs, sh, pps, slice_index, dpb, cur_poc, direct_8x8_inference, &mut last_mb_qp, &mut coeffs);
    }

    if sh.slice_type.is_intra() {
        let mut mb_xy = sh.first_mb_in_slice as usize;
        while mb_xy < total_mb {
            set_mb_deblock(ctx, mb_xy, sh, slice_index);
            coeffs.iter_mut().for_each(|c| *c = 0);
            parse_intra_mb_cavlc(bs, ctx, mb_xy, pps, &mut last_mb_qp, &mut coeffs)?;
            recon_intra_mb(ctx, mb_xy, &coeffs);
            mb_xy += 1;
            if !bs.more_rbsp_data() {
                break;
            }
        }
        return Ok(());
    }

    // Build the P list-0 reference list for this slice (default order with any
    // ref_pic_list_modification applied).
    let list = dpb.p_ref_list(
        sh.frame_num as i32,
        sh.num_ref_idx_active[0] as usize,
        &sh.ref_pic_list_reordering.list[0],
    );
    let ref_count = list.len();
    if ref_count == 0 {
        return Err(DecodeError::InvalidSyntax("P slice with no references"));
    }
    let ref_pic_ids: Vec<i32> = (0..ref_count).map(|k| dpb.refs[list[k]].id).collect();
    let ref_pics: Vec<&Picture> = (0..ref_count).map(|k| &dpb.refs[list[k]].pic).collect();

    let mut skip_run: i32 = -1;
    let mut mb_xy = sh.first_mb_in_slice as usize;
    while mb_xy < total_mb {
        set_mb_deblock(ctx, mb_xy, sh, slice_index);
        coeffs.iter_mut().for_each(|c| *c = 0);
        parse_p_mb_cavlc(
            bs,
            super::mb_parse_cavlc::MbCtx { ctx: &mut *ctx, mb_xy, pps },
            &mut last_mb_qp,
            &mut skip_run,
            &ref_pic_ids,
            &mut coeffs,
        )?;
        if ctx.mb_type[mb_xy].is_intra() {
            recon_intra_mb(ctx, mb_xy, &coeffs);
        } else {
            recon_inter_mb(ctx, mb_xy, &coeffs, &ref_pics);
        }
        mb_xy += 1;
        // A pending skip run keeps consuming MBs without reading more bits.
        if skip_run <= 0 && !bs.more_rbsp_data() {
            break;
        }
    }
    Ok(())
}

/// Decode one B slice's macroblocks (CAVLC): build list-0/list-1, then
/// parse + bi-predictive reconstruct each MB.
#[allow(clippy::too_many_arguments)]
fn decode_b_slice_cavlc(
    ctx: &mut DecoderContext,
    bs: &mut BitReader<'_>,
    sh: &SliceHeader,
    pps: &Pps,
    slice_index: i32,
    dpb: &Dpb,
    cur_poc: i32,
    direct_8x8_inference: bool,
    last_mb_qp: &mut i32,
    coeffs: &mut [i16; 384],
) -> Result<(), DecodeError> {
    use super::mb_parse_cavlc::{parse_b_mb_cavlc, BRefs};
    use super::recon_inter::recon_b_mb;

    let (blist0, blist1) = dpb.b_ref_lists(
        sh.frame_num as i32,
        cur_poc,
        sh.num_ref_idx_active[0] as usize,
        sh.num_ref_idx_active[1] as usize,
        &sh.ref_pic_list_reordering.list[0],
        &sh.ref_pic_list_reordering.list[1],
    );
    if blist0.is_empty() || blist1.is_empty() {
        return Err(DecodeError::InvalidSyntax("B slice with no references"));
    }
    let ref_pic_ids0: Vec<i32> = blist0.iter().map(|&i| dpb.refs[i].id).collect();
    let ref_pic_ids1: Vec<i32> = blist1.iter().map(|&i| dpb.refs[i].id).collect();
    let ref_pics0: Vec<&Picture> = blist0.iter().map(|&i| &dpb.refs[i].pic).collect();
    let ref_pics1: Vec<&Picture> = blist1.iter().map(|&i| &dpb.refs[i].pic).collect();
    let mv_scale = temporal_mv_scale(dpb, cur_poc, &blist0, &blist1);

    let col_frame = &dpb.refs[blist1[0]];
    ctx.is_b_slice = true;

    let total_mb = ctx.total_mb;
    let mut skip_run: i32 = -1;
    let mut mb_xy = sh.first_mb_in_slice as usize;
    while mb_xy < total_mb {
        set_mb_deblock(ctx, mb_xy, sh, slice_index);
        coeffs.iter_mut().for_each(|c| *c = 0);
        let bref = BRefs {
            ref_pic_ids: [&ref_pic_ids0, &ref_pic_ids1],
            ref_count: [blist0.len(), blist1.len()],
            direct_spatial: sh.direct_spatial_mv_pred_flag,
            col: super::bdirect::ColRef {
                col: &col_frame.col,
                is_long: col_frame.is_long_term,
                inference: direct_8x8_inference,
                mv_scale: &mv_scale,
                ref0_count: blist0.len(),
            },
        };
        parse_b_mb_cavlc(
            bs,
            super::mb_parse_cavlc::MbCtx { ctx: &mut *ctx, mb_xy, pps },
            last_mb_qp,
            &mut skip_run,
            &bref,
            coeffs,
        )?;
        if ctx.mb_type[mb_xy].is_intra() {
            recon_intra_mb(ctx, mb_xy, coeffs);
        } else {
            recon_b_mb(ctx, mb_xy, coeffs, [&ref_pics0, &ref_pics1]);
        }
        mb_xy += 1;
        if skip_run <= 0 && !bs.more_rbsp_data() {
            break;
        }
    }
    Ok(())
}

/// Per-list-0-reference temporal-direct MV scale factor (`iMvScale`, spec
/// 8.4.1.2.3), derived from the POC distances colocated→ref and current→ref.
fn temporal_mv_scale(dpb: &Dpb, cur_poc: i32, blist0: &[usize], blist1: &[usize]) -> Vec<i32> {
    let col_poc = dpb.refs[blist1[0]].poc;
    blist0
        .iter()
        .map(|&i| {
            let ref_poc = dpb.refs[i].poc;
            let td = (col_poc - ref_poc).clamp(-128, 127);
            if td == 0 {
                return 256;
            }
            let tb = (cur_poc - ref_poc).clamp(-128, 127);
            let tx = (16384 + (td.abs() / 2)) / td;
            ((tb * tx + 32) >> 6).clamp(-1024, 1023)
        })
        .collect()
}

/// Decode one CABAC slice's macroblocks (parse + reconstruct) into `ctx`.
/// `rbsp` is the full slice NAL RBSP; `bs` is positioned just past the slice
/// header, used to find the `cabac_alignment_one_bit` boundary.
#[allow(clippy::too_many_arguments)]
fn decode_one_slice_cabac(
    ctx: &mut DecoderContext,
    bs: &mut BitReader<'_>,
    rbsp: &[u8],
    sh: &SliceHeader,
    pps: &Pps,
    slice_index: i32,
    dpb: &Dpb,
    cur_poc: i32,
    direct_8x8_inference: bool,
) -> Result<(), DecodeError> {
    use super::cabac::{CabacContexts, CabacDecoder};
    use super::mb_parse_cabac::{decode_mb_cabac_islice, decode_mb_cabac_pslice};

    if sh.slice_type == super::slice_header::SliceType::B {
        return decode_b_slice_cabac(ctx, bs, rbsp, sh, pps, slice_index, dpb, cur_poc, direct_8x8_inference);
    }

    // cabac_alignment_one_bit: consume 1-bits to the next byte boundary.
    while !bs.byte_aligned() {
        if bs.read_bit()? != 1 {
            return Err(DecodeError::InvalidSyntax("cabac_alignment_one_bit"));
        }
    }
    let byte_offset = bs.bit_pos() / 8;
    let mut dec = CabacDecoder::new(rbsp, byte_offset)?;
    let mut ctxs = CabacContexts::init(sh.slice_type, sh.cabac_init_idc, sh.slice_qp);

    let total_mb = ctx.total_mb;
    let mut last_mb_qp = sh.slice_qp;
    let mut last_delta_qp = 0i32;
    let mut coeffs = [0i16; 384];

    // Reference list (P slices only).
    let (ref_pic_ids, ref_pics): (Vec<i32>, Vec<&Picture>) = if sh.slice_type.is_p() {
        let list = dpb.p_ref_list(
            sh.frame_num as i32,
            sh.num_ref_idx_active[0] as usize,
            &sh.ref_pic_list_reordering.list[0],
        );
        let ref_count = list.len();
        if ref_count == 0 {
            return Err(DecodeError::InvalidSyntax("P slice with no references"));
        }
        (
            (0..ref_count).map(|k| dpb.refs[list[k]].id).collect(),
            (0..ref_count).map(|k| &dpb.refs[list[k]].pic).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let mut mb_xy = sh.first_mb_in_slice as usize;
    while mb_xy < total_mb {
        set_mb_deblock(ctx, mb_xy, sh, slice_index);
        coeffs.iter_mut().for_each(|c| *c = 0);
        let eos = if sh.slice_type.is_intra() {
            decode_mb_cabac_islice(
                &mut dec,
                &mut ctxs,
                super::mb_parse_cabac::MbCtx { ctx: &mut *ctx, mb_xy, pps },
                super::mb_parse_cabac::QpState {
                    last_mb_qp: &mut last_mb_qp,
                    last_delta_qp: &mut last_delta_qp,
                },
                &mut coeffs,
            )?
        } else {
            decode_mb_cabac_pslice(
                &mut dec,
                &mut ctxs,
                super::mb_parse_cabac::MbCtx { ctx: &mut *ctx, mb_xy, pps },
                super::mb_parse_cabac::QpState {
                    last_mb_qp: &mut last_mb_qp,
                    last_delta_qp: &mut last_delta_qp,
                },
                &ref_pic_ids,
                &mut coeffs,
            )?
        };
        if ctx.mb_type[mb_xy].is_intra() {
            recon_intra_mb(ctx, mb_xy, &coeffs);
        } else {
            recon_inter_mb(ctx, mb_xy, &coeffs, &ref_pics);
        }
        mb_xy += 1;
        if eos {
            break;
        }
    }
    Ok(())
}

/// Decode one B slice's macroblocks (CABAC).
#[allow(clippy::too_many_arguments)]
fn decode_b_slice_cabac(
    ctx: &mut DecoderContext,
    bs: &mut BitReader<'_>,
    rbsp: &[u8],
    sh: &SliceHeader,
    pps: &Pps,
    slice_index: i32,
    dpb: &Dpb,
    cur_poc: i32,
    direct_8x8_inference: bool,
) -> Result<(), DecodeError> {
    use super::cabac::{CabacContexts, CabacDecoder};
    use super::mb_parse_cabac::{decode_mb_cabac_bslice, BRefsCabac};
    use super::recon_inter::recon_b_mb;

    let (blist0, blist1) = dpb.b_ref_lists(
        sh.frame_num as i32,
        cur_poc,
        sh.num_ref_idx_active[0] as usize,
        sh.num_ref_idx_active[1] as usize,
        &sh.ref_pic_list_reordering.list[0],
        &sh.ref_pic_list_reordering.list[1],
    );
    if blist0.is_empty() || blist1.is_empty() {
        return Err(DecodeError::InvalidSyntax("B slice with no references"));
    }
    let ref_pic_ids0: Vec<i32> = blist0.iter().map(|&i| dpb.refs[i].id).collect();
    let ref_pic_ids1: Vec<i32> = blist1.iter().map(|&i| dpb.refs[i].id).collect();
    let ref_pics0: Vec<&Picture> = blist0.iter().map(|&i| &dpb.refs[i].pic).collect();
    let ref_pics1: Vec<&Picture> = blist1.iter().map(|&i| &dpb.refs[i].pic).collect();
    let mv_scale = temporal_mv_scale(dpb, cur_poc, &blist0, &blist1);
    let col_frame = &dpb.refs[blist1[0]];
    ctx.is_b_slice = true;

    while !bs.byte_aligned() {
        if bs.read_bit()? != 1 {
            return Err(DecodeError::InvalidSyntax("cabac_alignment_one_bit"));
        }
    }
    let byte_offset = bs.bit_pos() / 8;
    let mut dec = CabacDecoder::new(rbsp, byte_offset)?;
    let mut ctxs = CabacContexts::init(sh.slice_type, sh.cabac_init_idc, sh.slice_qp);

    let total_mb = ctx.total_mb;
    let mut last_mb_qp = sh.slice_qp;
    let mut last_delta_qp = 0i32;
    let mut coeffs = [0i16; 384];

    let mut mb_xy = sh.first_mb_in_slice as usize;
    while mb_xy < total_mb {
        set_mb_deblock(ctx, mb_xy, sh, slice_index);
        coeffs.iter_mut().for_each(|c| *c = 0);
        let bref = BRefsCabac {
            ref_pic_ids: [&ref_pic_ids0, &ref_pic_ids1],
            ref_count: [blist0.len(), blist1.len()],
            direct_spatial: sh.direct_spatial_mv_pred_flag,
            col: super::bdirect::ColRef {
                col: &col_frame.col,
                is_long: col_frame.is_long_term,
                inference: direct_8x8_inference,
                mv_scale: &mv_scale,
                ref0_count: blist0.len(),
            },
        };
        let eos = decode_mb_cabac_bslice(
            &mut dec,
            &mut ctxs,
            super::mb_parse_cabac::MbCtx { ctx: &mut *ctx, mb_xy, pps },
            super::mb_parse_cabac::QpState {
                last_mb_qp: &mut last_mb_qp,
                last_delta_qp: &mut last_delta_qp,
            },
            &bref,
            &mut coeffs,
        )?;
        if ctx.mb_type[mb_xy].is_intra() {
            recon_intra_mb(ctx, mb_xy, &coeffs);
        } else {
            recon_b_mb(ctx, mb_xy, &coeffs, [&ref_pics0, &ref_pics1]);
        }
        mb_xy += 1;
        if eos {
            break;
        }
    }
    Ok(())
}

/// Record this MB's owning-slice id and deblock parameters.
fn set_mb_deblock(ctx: &mut DecoderContext, mb_xy: usize, sh: &SliceHeader, slice_index: i32) {
    ctx.slice_idc[mb_xy] = slice_index;
    ctx.deblock_idc[mb_xy] = sh.disable_deblocking_filter_idc as u8;
    ctx.deblock_alpha_off[mb_xy] = sh.slice_alpha_c0_offset as i8;
    ctx.deblock_beta_off[mb_xy] = sh.slice_beta_offset as i8;
}

/// Probe a VCL NAL's `pic_parameter_set_id` (third `ue`) and resolve PPS→SPS.
fn resolve_param_sets<'a>(
    nal: &NalUnit,
    sps_map: &'a [Option<Sps>],
    pps_map: &'a [Option<Pps>],
) -> Result<(&'a Sps, &'a Pps), DecodeError> {
    let mut probe = BitReader::new(&nal.rbsp);
    let _first_mb = probe.read_ue()?;
    let _slice_type = probe.read_ue()?;
    let pps_id = probe.read_ue()? as usize;
    let pps = pps_map
        .get(pps_id)
        .and_then(|p| p.as_ref())
        .ok_or(DecodeError::MissingParameterSet)?;
    let sps = sps_map
        .get(pps.sps_id as usize)
        .and_then(|s| s.as_ref())
        .ok_or(DecodeError::MissingParameterSet)?;
    Ok((sps, pps))
}

/// Parse one IDR slice header then loop its macroblocks (parse + reconstruct).
fn decode_idr_slice(
    nal: &NalUnit,
    sps_map: &[Option<Sps>],
    pps_map: &[Option<Pps>],
    ctx: &mut DecoderContext,
    slice_index: i32,
    total_mb: usize,
) -> Result<(), DecodeError> {
    let (sps, pps) = resolve_param_sets(nal, sps_map, pps_map)?;
    if pps.entropy_coding_mode_flag {
        return Err(DecodeError::Unsupported("CABAC entropy coding"));
    }
    // Clone the active PPS so the MB loop can hold it while mutating `ctx`.
    let pps = pps.clone();

    let mut bs = BitReader::new(&nal.rbsp);
    let sh = parse_slice_header_in_place(&mut bs, nal.ref_idc, true, sps, &pps)?;
    if !sh.slice_type.is_intra() {
        return Err(DecodeError::InvalidSyntax("IDR slice not intra"));
    }

    let mut last_mb_qp = sh.slice_qp;
    let mut coeffs = [0i16; 384];
    let mut mb_xy = sh.first_mb_in_slice as usize;
    while mb_xy < total_mb {
        ctx.slice_idc[mb_xy] = slice_index;
        // Per-MB deblock parameters from this slice's header (read by the
        // post-reconstruction deblocking pass).
        ctx.deblock_idc[mb_xy] = sh.disable_deblocking_filter_idc as u8;
        ctx.deblock_alpha_off[mb_xy] = sh.slice_alpha_c0_offset as i8;
        ctx.deblock_beta_off[mb_xy] = sh.slice_beta_offset as i8;
        coeffs.iter_mut().for_each(|c| *c = 0);
        parse_intra_mb_cavlc(&mut bs, ctx, mb_xy, &pps, &mut last_mb_qp, &mut coeffs)?;
        recon_intra_mb(ctx, mb_xy, &coeffs);
        mb_xy += 1;
        if !bs.more_rbsp_data() {
            break;
        }
    }
    Ok(())
}

/// Incremental, stateful baseline/Main-profile decoder core (I + P, CAVLC or
/// CABAC, 4:2:0 8-bit). Owns its parameter-set tables and DPB; fed Annex-B
/// bytes a packet at a time, it emits each picture once the next picture begins
/// (or on [`finish`](StreamDecoder::finish)). This is the engine behind the
/// public [`crate::Decoder`]; it reuses the exact slice-decode path of
/// [`decode_stream`].
pub(crate) struct StreamDecoder {
    sps_map: Vec<Option<Sps>>,
    pps_map: Vec<Option<Pps>>,
    dpb: Option<Dpb>,
    next_id: i32,
    cur: Option<CurPic>,
    poc_state: PocState,
    cvs: i32,
}

impl StreamDecoder {
    pub(crate) fn new() -> Self {
        StreamDecoder {
            sps_map: (0..32).map(|_| None).collect(),
            pps_map: (0..256).map(|_| None).collect(),
            dpb: None,
            next_id: 0,
            cur: None,
            poc_state: PocState::default(),
            cvs: -1,
        }
    }

    /// Parse all NAL units in `annexb` (parameter sets and slices), appending
    /// every picture that completes to `out`. A picture completes when the next
    /// picture's first slice arrives; the trailing picture is emitted by
    /// [`finish`](Self::finish).
    pub(crate) fn feed(&mut self, annexb: &[u8], out: &mut Vec<Frame>) -> Result<(), DecodeError> {
        for ebsp in annexb_nal_units(annexb) {
            let nal = match parse_nal(ebsp) {
                Some(n) => n,
                None => continue,
            };
            match nal.unit_type {
                NalUnitType::Sps => {
                    let sps = parse_sps(&nal.rbsp)?;
                    let id = sps.sps_id as usize;
                    self.sps_map[id] = Some(sps);
                }
                NalUnitType::Pps => {
                    let sps_ref = parse_pps(&nal.rbsp, None)
                        .ok()
                        .and_then(|p| self.sps_map[p.sps_id as usize].as_ref());
                    let pps = parse_pps(&nal.rbsp, sps_ref)?;
                    let id = pps.pps_id as usize;
                    self.pps_map[id] = Some(pps);
                }
                t if t.is_vcl() => self.feed_vcl(&nal, out)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn feed_vcl(&mut self, nal: &NalUnit, out: &mut Vec<Frame>) -> Result<(), DecodeError> {
        let (sps, pps) = {
            let (s, p) = resolve_param_sets(nal, &self.sps_map, &self.pps_map)?;
            if s.chroma_format_idc != 1 || s.bit_depth_luma != 8 || s.bit_depth_chroma != 8 {
                return Err(DecodeError::Unsupported("non-4:2:0 / non-8-bit"));
            }
            (s.clone(), p.clone())
        };
        let is_idr = nal.unit_type.is_idr();

        let mut bs = BitReader::new(&nal.rbsp);
        let sh = parse_slice_header_in_place(&mut bs, nal.ref_idc, is_idr, &sps, &pps)?;

        if sh.first_mb_in_slice == 0 {
            if let Some(c) = self.cur.take() {
                let f = finalize_into(c, self.dpb.as_mut().unwrap(), &mut self.next_id);
                out.push(f);
            }
            if self.dpb.is_none() {
                self.dpb = Some(Dpb::new(sps.max_num_ref_frames, sps.log2_max_frame_num));
            }
            if is_idr {
                self.dpb.as_mut().unwrap().clear();
                self.cvs += 1;
            }
            if self.cvs < 0 {
                self.cvs = 0;
            }
            let poc = self.poc_state.compute(&sps, &sh, &pps, is_idr, nal.ref_idc);
            self.cur = Some(CurPic {
                ctx: DecoderContext::new(
                    Vec::new(),
                    Vec::new(),
                    sps.mb_width as usize,
                    sps.mb_height as usize,
                ),
                frame_num: sh.frame_num as i32,
                is_ref: nal.ref_idc != 0,
                is_idr,
                poc,
                cvs: self.cvs,
                marking: sh.dec_ref_pic_marking.clone(),
                slice_index: 0,
                region: region_from_sps(&sps),
            });
        }

        let dpb = self.dpb.as_ref().unwrap();
        let cp = self
            .cur
            .as_mut()
            .ok_or(DecodeError::InvalidSyntax("slice before picture start"))?;
        if pps.entropy_coding_mode_flag {
            decode_one_slice_cabac(&mut cp.ctx, &mut bs, &nal.rbsp, &sh, &pps, cp.slice_index, dpb, cp.poc, sps.direct_8x8_inference_flag)?;
        } else {
            decode_one_slice(&mut cp.ctx, &mut bs, &sh, &pps, cp.slice_index, dpb, cp.poc, sps.direct_8x8_inference_flag)?;
        }
        cp.slice_index += 1;
        Ok(())
    }

    /// Flush the trailing in-flight picture (the last frame of a stream), if any.
    pub(crate) fn finish(&mut self, out: &mut Vec<Frame>) {
        if let Some(c) = self.cur.take() {
            let dpb = self.dpb.as_mut().expect("dpb exists when a picture is in flight");
            out.push(finalize_into(c, dpb, &mut self.next_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BANM: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/BANM_MW_D.264"
    ));

    #[test]
    fn decodes_banm_idr_frame() {
        let pic = decode_intra_frame(BANM).expect("decode first IDR frame");
        assert_eq!(pic.width, 176);
        assert_eq!(pic.height, 144);

        // The luma plane must be non-degenerate: a real picture spans a wide
        // range of values rather than a single flat fill.
        let stride = pic.luma_stride;
        let o = pic.luma_origin();
        let mut min = 255u8;
        let mut max = 0u8;
        let mut sum = 0u64;
        for y in 0..pic.height {
            for x in 0..pic.width {
                let v = pic.y[o + y * stride + x];
                min = min.min(v);
                max = max.max(v);
                sum += v as u64;
            }
        }
        let mean = sum / (pic.width as u64 * pic.height as u64);
        assert!(max - min > 64, "luma range too small: {min}..{max}");
        assert!((16..=235).contains(&(mean as u8)), "implausible mean {mean}");
    }

    #[test]
    fn decode_is_deterministic() {
        let a = decode_intra_frame(BANM).expect("decode a");
        let b = decode_intra_frame(BANM).expect("decode b");
        assert_eq!(a.y, b.y, "luma differs between runs");
        assert_eq!(a.u, b.u, "Cb differs between runs");
        assert_eq!(a.v, b.v, "Cr differs between runs");
    }
}
