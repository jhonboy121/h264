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
    slice_index: i32,
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

    let mut output: Vec<Picture> = Vec::new();
    let mut dpb: Option<Dpb> = None;
    let mut next_id: i32 = 0;
    let mut cur: Option<CurPic> = None;

    for nal in &nals {
        if !nal.unit_type.is_vcl() {
            continue;
        }
        let (sps, pps) = resolve_param_sets(nal, &sps_map, &pps_map)?;
        if sps.chroma_format_idc != 1 || sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err(DecodeError::Unsupported("non-4:2:0 / non-8-bit"));
        }
        if pps.entropy_coding_mode_flag {
            return Err(DecodeError::Unsupported("CABAC entropy coding"));
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
            }
            let mb_width = sps.mb_width as usize;
            let mb_height = sps.mb_height as usize;
            cur = Some(CurPic {
                ctx: DecoderContext::new(Vec::new(), Vec::new(), mb_width, mb_height),
                frame_num: sh.frame_num as i32,
                is_ref: nal.ref_idc != 0,
                slice_index: 0,
            });
        }

        let cp = cur.as_mut().ok_or(DecodeError::InvalidSyntax("slice before picture start"))?;
        decode_one_slice(&mut cp.ctx, &mut bs, &sh, &pps, cp.slice_index, dpb.as_ref().unwrap())?;
        cp.slice_index += 1;
    }

    if let Some(c) = cur.take() {
        finalize_picture(c, dpb.as_mut().unwrap(), &mut output, &mut next_id);
    }
    Ok(output)
}

/// Deblock, optionally border-extend + insert into the DPB, and emit.
fn finalize_picture(mut c: CurPic, dpb: &mut Dpb, output: &mut Vec<Picture>, next_id: &mut i32) {
    super::deblock::deblock_frame(&mut c.ctx);
    if c.is_ref {
        crate::dsp::expand::expand_picture(&mut c.ctx.picture);
        output.push(c.ctx.picture.clone());
        let id = *next_id;
        *next_id += 1;
        dpb.add_short_term(c.ctx.picture, c.frame_num, id);
    } else {
        output.push(c.ctx.picture);
    }
}

/// Decode one slice's macroblocks (parse + reconstruct) into `ctx`.
fn decode_one_slice(
    ctx: &mut DecoderContext,
    bs: &mut BitReader<'_>,
    sh: &SliceHeader,
    pps: &Pps,
    slice_index: i32,
    dpb: &Dpb,
) -> Result<(), DecodeError> {
    let total_mb = ctx.total_mb;
    let mut last_mb_qp = sh.slice_qp;
    let mut coeffs = [0i16; 384];

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

    if !sh.slice_type.is_p() {
        return Err(DecodeError::Unsupported("B slice"));
    }

    // Build the default P list-0 reference list for this slice.
    let list = dpb.p_ref_list(sh.frame_num as i32);
    let ref_count = (sh.num_ref_idx_active[0] as usize).min(list.len());
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
        parse_p_mb_cavlc(bs, ctx, mb_xy, pps, &mut last_mb_qp, &mut skip_run, &ref_pic_ids, &mut coeffs)?;
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
