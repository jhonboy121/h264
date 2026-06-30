//! Decode a CABAC stream and compare visible I420 frames byte-for-byte to an
//! oracle YUV.
use h264::decoder::decode_stream;

fn extract_i420(pic: &h264::decoder::picture::Picture) -> Vec<u8> {
    let mut out = Vec::with_capacity(pic.width * pic.height * 3 / 2);
    let o = pic.luma_origin();
    for y in 0..pic.height {
        out.extend_from_slice(&pic.y[o + y * pic.luma_stride..o + y * pic.luma_stride + pic.width]);
    }
    let cw = pic.width / 2;
    let ch = pic.height / 2;
    let co = pic.chroma_origin();
    for y in 0..ch {
        out.extend_from_slice(&pic.u[co + y * pic.chroma_stride..co + y * pic.chroma_stride + cw]);
    }
    for y in 0..ch {
        out.extend_from_slice(&pic.v[co + y * pic.chroma_stride..co + y * pic.chroma_stride + cw]);
    }
    out
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let oracle_path = std::env::args().nth(2).unwrap();
    let max_frames: usize = std::env::args().nth(3).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
    let data = std::fs::read(&path).unwrap();
    let oracle = std::fs::read(&oracle_path).unwrap();

    let pics = decode_stream(&data).expect("decode");
    let fsize = pics[0].width * pics[0].height * 3 / 2;
    println!("decoded {} frames, frame size {}", pics.len(), fsize);

    let mut first_bad = None;
    let n = pics.len().min(oracle.len() / fsize).min(max_frames);
    for i in 0..n {
        let got = extract_i420(&pics[i]);
        let exp = &oracle[i * fsize..(i + 1) * fsize];
        let diff = got.iter().zip(exp).filter(|(a, b)| a != b).count();
        if diff != 0 {
            // find first differing byte location
            let pos = got.iter().zip(exp).position(|(a, b)| a != b).unwrap();
            if first_bad.is_none() {
                first_bad = Some(i);
            }
            println!("frame {i}: {diff} differing bytes (first at byte {pos})");
            if i >= 3 {
                break;
            }
        } else {
            println!("frame {i}: BIT-EXACT");
        }
    }
    match first_bad {
        None => println!("ALL {n} frames bit-exact"),
        Some(f) => {
            println!("first divergence at frame {f}");
            // Per-MB luma diff map for frame f.
            let w = pics[f].width;
            let h = pics[f].height;
            let got = extract_i420(&pics[f]);
            let exp = &oracle[f * fsize..(f + 1) * fsize];
            let mbw = w / 16;
            let mbh = h / 16;
            println!("per-MB luma diff map ({mbw}x{mbh}), '.'=0 'X'=diff:");
            for my in 0..mbh {
                let mut row = String::new();
                for mx in 0..mbw {
                    let mut d = 0;
                    for yy in 0..16 {
                        for xx in 0..16 {
                            let p = (my * 16 + yy) * w + mx * 16 + xx;
                            if got[p] != exp[p] {
                                d += 1;
                            }
                        }
                    }
                    row.push(if d == 0 { '.' } else { 'X' });
                }
                println!("{my:2}: {row}");
            }
        }
    }
}
