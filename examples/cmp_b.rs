//! Dump the 16x16 luma diff (decoded vs oracle) for one MB of one decoded frame.
use std::fs;

fn append(out: &mut Vec<u8>, pic: &h264::decoder::picture::Picture) {
    let yo = pic.luma_origin();
    for r in 0..pic.height { let s = yo + r * pic.luma_stride; out.extend_from_slice(&pic.y[s..s + pic.width]); }
    let co = pic.chroma_origin();
    for r in 0..pic.height / 2 { let s = co + r * pic.chroma_stride; out.extend_from_slice(&pic.u[s..s + pic.width / 2]); }
    for r in 0..pic.height / 2 { let s = co + r * pic.chroma_stride; out.extend_from_slice(&pic.v[s..s + pic.width / 2]); }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let stream = fs::read(&a[0]).unwrap();
    let oracle = fs::read(&a[1]).unwrap();
    let w: usize = a[2].parse().unwrap();
    let h: usize = a[3].parse().unwrap();
    let want: usize = a[4].parse().unwrap();
    let mx: usize = a[5].parse().unwrap();
    let my: usize = a[6].parse().unwrap();
    let fsz = w * h * 3 / 2;
    let nframes = oracle.len() / fsz;
    let frames = h264::decoder::decode_stream(&stream).unwrap();
    let mut b = Vec::new(); append(&mut b, &frames[want]);
    let mut best = (0usize, usize::MAX);
    for f in 0..nframes { let o = &oracle[f*fsz..(f+1)*fsz]; let d = b.iter().zip(o).filter(|(x,y)| x!=y).count(); if d<best.1 {best=(f,d);} }
    let truth = &oracle[best.0*fsz..(best.0+1)*fsz];
    println!("frame#{want}->oracle{} diff{}; MB({mx},{my}):", best.0, best.1);
    for yy in 0..16 {
        let mut l=String::new();
        for xx in 0..16 {
            let i=(my*16+yy)*w + mx*16+xx;
            let d = b[i] as i32 - truth[i] as i32;
            l.push(if d==0 {'.'} else if d.abs()<4 {'-'} else if d.abs()<16 {'o'} else {'#'});
        }
        println!("{l}");
    }
}
