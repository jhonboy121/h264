//! `scale_i420` (the OpenH264 downsampler port) and the NAL / SPS helpers.

use h264::bits::BitWriter;
use h264::encoder::nal_encap::append_annexb_nal;
use h264::nal::{self, nal_type};
use h264::{Decoder, I420, YuvRef, nal_units, scale_i420, sps_dimensions};

/// Luma = horizontal ramp, U = vertical ramp, V = diagonal ramp.
fn gradient(w: u32, h: u32) -> I420 {
    let mut f = I420::new(w, h);
    let (wu, hu) = (w as usize, h as usize);
    for y in 0..hu {
        for x in 0..wu {
            f.y[y * wu + x] = (x * 255 / (wu - 1)) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for y in 0..ch {
        for x in 0..cw {
            f.u[y * cw + x] = (y * 255 / (ch - 1)) as u8;
            f.v[y * cw + x] = ((x + y) * 255 / (cw + ch - 2)) as u8;
        }
    }
    f
}

/// Deterministic busy content (so a dyadic comparison is not trivially flat).
fn noise(w: u32, h: u32) -> I420 {
    let mut f = I420::new(w, h);
    let mut s = 0x1234_5678u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 24) as u8
    };
    for p in f.y.iter_mut().chain(f.u.iter_mut()).chain(f.v.iter_mut()) {
        *p = next();
    }
    f
}

/// `DyadicBilinearDownsampler_c`, written out independently.
fn reference_half(src: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((w / 2) * (h / 2));
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            let p = |x: usize, y: usize| src[y * stride + x] as u32;
            let r1 = (p(2 * i, 2 * j) + p(2 * i + 1, 2 * j) + 1) >> 1;
            let r2 = (p(2 * i, 2 * j + 1) + p(2 * i + 1, 2 * j + 1) + 1) >> 1;
            out.push(((r1 + r2 + 1) >> 1) as u8);
        }
    }
    out
}

#[test]
fn dyadic_half_matches_reference() {
    let src = noise(1920, 1080);
    let mut dst = I420::new(960, 540);
    scale_i420(&src.as_ref(), &mut dst);
    assert_eq!(dst.y, reference_half(&src.y, 1920, 1920, 1080));
    assert_eq!(dst.u, reference_half(&src.u, 960, 960, 540));
    assert_eq!(dst.v, reference_half(&src.v, 960, 960, 540));
}

#[test]
fn dyadic_quarter_is_two_halvings() {
    let src = noise(1280, 720);
    let mut dst = I420::new(320, 180);
    scale_i420(&src.as_ref(), &mut dst);
    let half = reference_half(&src.y, 1280, 1280, 720);
    assert_eq!(dst.y, reference_half(&half, 640, 640, 360));
    let half_u = reference_half(&src.u, 640, 640, 360);
    assert_eq!(dst.u, reference_half(&half_u, 320, 320, 180));
}

/// Every row of `plane` is non-decreasing and within `tol` of `want(x, y)`.
fn assert_ramp(plane: &[u8], w: usize, h: usize, tol: i32, want: impl Fn(usize, usize) -> i32) {
    for y in 0..h {
        for x in 0..w {
            let got = plane[y * w + x] as i32;
            assert!((got - want(x, y)).abs() <= tol, "({x},{y}): {got} vs {}", want(x, y));
        }
    }
}

#[test]
fn gradient_stays_a_gradient() {
    // 1080p -> 360p: one halving, then the general bilinear kernels.
    let src = gradient(1920, 1080);
    let mut dst = I420::new(640, 360);
    scale_i420(&src.as_ref(), &mut dst);
    assert_ramp(&dst.y, 640, 360, 3, |x, _| (x * 255 / 639) as i32);
    assert_ramp(&dst.u, 320, 180, 3, |_, y| (y * 255 / 179) as i32);
    for row in dst.y.as_chunks::<640>().0 {
        assert!(row.windows(2).all(|p| p[0] <= p[1]), "luma row not monotonic");
    }
    for col in 0..320 {
        assert!((1..180).all(|y| dst.u[(y - 1) * 320 + col] <= dst.u[y * 320 + col]));
    }

    // 1080p -> 720p (general only) and 360p -> 720p (upscale).
    let mut hd = I420::new(1280, 720);
    scale_i420(&src.as_ref(), &mut hd);
    assert_ramp(&hd.y, 1280, 720, 3, |x, _| (x * 255 / 1279) as i32);
    let small = gradient(640, 360);
    scale_i420(&small.as_ref(), &mut hd);
    assert_ramp(&hd.y, 1280, 720, 3, |x, _| (x * 255 / 1279) as i32);
    assert_ramp(&hd.u, 640, 360, 3, |_, y| (y * 255 / 359) as i32);
}

#[test]
fn same_size_copies_and_strided_source_works() {
    let src = noise(64, 32);
    // The same picture behind a wider stride.
    let stride = 80;
    let mut y = vec![0u8; stride * 32];
    for r in 0..32 {
        y[r * stride..r * stride + 64].copy_from_slice(&src.y[r * 64..r * 64 + 64]);
    }
    let strided = YuvRef {
        y: &y,
        y_stride: stride,
        ..src.as_ref()
    };
    let mut dst = I420::new(64, 32);
    scale_i420(&strided, &mut dst);
    assert_eq!(dst, src);
}

#[test]
fn nal_type_reads_the_header() {
    assert_eq!(nal_type(&[0x67, 0x42]), Some(nal::SPS));
    assert_eq!(nal_type(&[0x68]), Some(nal::PPS));
    assert_eq!(nal_type(&[0x65]), Some(nal::IDR));
    assert_eq!(nal_type(&[0x09, 0xf0]), Some(nal::AUD));
    assert_eq!(nal_type(&[]), None);
}

/// A phone-style High-profile 1080p SPS: scaling lists, cropping, full VUI.
fn high_profile_sps() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_bits(100, 8); // profile_idc (High)
    bw.write_bits(0, 8); // constraint flags + reserved
    bw.write_bits(40, 8); // level_idc
    bw.write_ue(0); // sps_id
    bw.write_ue(1); // chroma_format_idc 4:2:0
    bw.write_ue(0); // bit_depth_luma_minus8
    bw.write_ue(0); // bit_depth_chroma_minus8
    bw.write_flag(false); // qpprime_y_zero_transform_bypass
    bw.write_flag(true); // seq_scaling_matrix_present
    for list in 0..8 {
        match list {
            0 => {
                bw.write_flag(true);
                bw.write_se(8); // 8 -> 16, then flat
                for _ in 1..16 {
                    bw.write_se(0);
                }
            }
            6 => {
                bw.write_flag(true);
                bw.write_se(-8); // next scale 0: use the default 8x8 list
            }
            _ => bw.write_flag(false),
        }
    }
    bw.write_ue(0); // log2_max_frame_num_minus4
    bw.write_ue(0); // pic_order_cnt_type
    bw.write_ue(2); // log2_max_poc_lsb_minus4
    bw.write_ue(4); // max_num_ref_frames
    bw.write_flag(false); // gaps
    bw.write_ue(119); // 120 MBs wide
    bw.write_ue(67); // 68 MBs high
    bw.write_flag(true); // frame_mbs_only
    bw.write_flag(true); // direct_8x8_inference
    bw.write_flag(true); // frame_cropping
    for crop in [0, 0, 0, 4] {
        bw.write_ue(crop);
    }
    bw.write_flag(true); // vui
    bw.write_flag(false); // aspect_ratio_info
    bw.write_flag(false); // overscan_info
    bw.write_flag(true); // video_signal_type
    bw.write_bits(5, 3); // video_format
    bw.write_flag(false); // full_range
    bw.write_flag(true); // colour_description
    for c in [1, 1, 1] {
        bw.write_bits(c, 8);
    }
    bw.write_flag(false); // chroma_loc_info
    bw.write_flag(true); // timing_info
    bw.write_bits(1, 32);
    bw.write_bits(60, 32);
    bw.write_flag(false); // fixed_frame_rate
    bw.write_flag(false); // nal_hrd
    bw.write_flag(false); // vcl_hrd
    bw.write_flag(false); // pic_struct_present
    bw.write_flag(true); // bitstream_restriction
    bw.write_flag(true); // motion_vectors_over_pic_boundaries
    for v in [0, 0, 16, 16, 2, 4] {
        bw.write_ue(v);
    }
    bw.write_trailing_bits();
    let mut out = Vec::new();
    append_annexb_nal(&mut out, 3, nal::SPS, &bw.finish());
    out
}

#[test]
fn sps_dimensions_high_profile_cropped() {
    let annexb = high_profile_sps();
    let sps = nal_units(&annexb).next().expect("one NAL");
    assert_eq!(nal_type(sps), Some(nal::SPS));
    assert_eq!(sps_dimensions(sps), Some((1920, 1080)));
    // Not an SPS / truncated.
    assert_eq!(sps_dimensions(&[0x68, 0xce]), None);
    assert_eq!(sps_dimensions(&sps[..4]), None);
    assert_eq!(sps_dimensions(&[]), None);
}

#[test]
fn sps_dimensions_matches_decoder_on_fixtures() {
    for stream in [
        &include_bytes!("fixtures/BA1_FT_C.264")[..],
        &include_bytes!("fixtures/test_qcif_cabac.264")[..],
        &include_bytes!("fixtures/CVPCMNL1_SVA_C.264")[..],
    ] {
        let sps = nal_units(stream)
            .find(|n| nal_type(n) == Some(nal::SPS))
            .expect("stream has an SPS");
        let frames = Decoder::new().decode_all(stream).expect("decode");
        let f = frames[0].yuv();
        assert_eq!(sps_dimensions(sps), Some((f.width, f.height)));
    }
}
