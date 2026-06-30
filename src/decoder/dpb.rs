//! Short-term reference picture buffer (sliding-window DPB) for baseline P.
//!
//! Collapses the short-term parts of `manage_dec_ref.cpp`
//! (`WelsMarkAsRef` sliding window + `WelsInitRefList` default P list) for the
//! baseline case: short-term references only, list-0 ordered by descending
//! `PicNum` (= `FrameNumWrap`), with `max_num_ref_frames` sliding-window
//! removal. Long-term references and MMCO are out of scope here (BANM does not
//! use them; see the module note).

use alloc::vec::Vec;

use super::picture::Picture;

/// One decoded short-term reference frame held in the DPB.
pub struct RefFrame {
    /// Reconstructed, border-extended picture.
    pub pic: Picture,
    /// `frame_num` from the slice header.
    pub frame_num: i32,
    /// Monotonic decode-order identity, used by the deblocker to tell whether
    /// two blocks reference the same picture.
    pub id: i32,
}

/// Sliding-window short-term DPB.
pub struct Dpb {
    pub refs: Vec<RefFrame>,
    max_num_ref_frames: usize,
    max_frame_num: i32,
}

impl Dpb {
    pub fn new(max_num_ref_frames: u32, log2_max_frame_num: u32) -> Self {
        Dpb {
            refs: Vec::new(),
            max_num_ref_frames: max_num_ref_frames.max(1) as usize,
            max_frame_num: 1i32 << log2_max_frame_num,
        }
    }

    /// Clear all references (IDR).
    pub fn clear(&mut self) {
        self.refs.clear();
    }

    /// `FrameNumWrap` / `PicNum` of a short-term reference relative to the
    /// current frame (spec 8.2.4.1).
    fn pic_num(&self, frame_num: i32, cur_frame_num: i32) -> i32 {
        if frame_num > cur_frame_num {
            frame_num - self.max_frame_num
        } else {
            frame_num
        }
    }

    /// Indices into [`Dpb::refs`] forming the default list-0 reference list for
    /// a P slice with the given `cur_frame_num`: short-term refs by descending
    /// `PicNum`.
    pub fn p_ref_list(&self, cur_frame_num: i32) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.refs.len()).collect();
        idx.sort_by(|&a, &b| {
            let pa = self.pic_num(self.refs[a].frame_num, cur_frame_num);
            let pb = self.pic_num(self.refs[b].frame_num, cur_frame_num);
            pb.cmp(&pa)
        });
        idx
    }

    /// Add a freshly decoded reference frame and apply sliding-window removal
    /// (drop the short-term ref with the smallest `PicNum` while over budget).
    pub fn add_short_term(&mut self, pic: Picture, frame_num: i32, id: i32) {
        self.refs.push(RefFrame { pic, frame_num, id });
        while self.refs.len() > self.max_num_ref_frames {
            // Remove the oldest by FrameNumWrap relative to the just-added frame.
            let cur = frame_num;
            let mut min_i = 0;
            let mut min_pn = i32::MAX;
            for (i, r) in self.refs.iter().enumerate() {
                let pn = self.pic_num(r.frame_num, cur);
                if pn < min_pn {
                    min_pn = pn;
                    min_i = i;
                }
            }
            self.refs.remove(min_i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rf(frame_num: i32, id: i32) -> RefFrame {
        RefFrame { pic: Picture::new(1, 1), frame_num, id }
    }

    #[test]
    fn p_list_descending_picnum() {
        let mut dpb = Dpb::new(4, 4); // max_frame_num = 16
        dpb.refs.push(rf(0, 0));
        dpb.refs.push(rf(1, 1));
        dpb.refs.push(rf(2, 2));
        // Current frame_num = 3: PicNums are 0,1,2 -> order 2,1,0.
        let list = dpb.p_ref_list(3);
        let ids: Vec<i32> = list.iter().map(|&i| dpb.refs[i].id).collect();
        assert_eq!(ids, alloc::vec![2, 1, 0]);
    }

    #[test]
    fn p_list_wraps_frame_num() {
        let mut dpb = Dpb::new(4, 4);
        dpb.refs.push(rf(14, 14));
        dpb.refs.push(rf(15, 15));
        dpb.refs.push(rf(0, 16)); // wrapped past 15
        // Current frame_num = 1: PicNum(0)=0, PicNum(14)=14-16=-2, PicNum(15)=-1.
        let list = dpb.p_ref_list(1);
        let ids: Vec<i32> = list.iter().map(|&i| dpb.refs[i].id).collect();
        assert_eq!(ids, alloc::vec![16, 15, 14]);
    }

    #[test]
    fn sliding_window_drops_oldest() {
        let mut dpb = Dpb::new(2, 4);
        dpb.add_short_term(Picture::new(1, 1), 0, 0);
        dpb.add_short_term(Picture::new(1, 1), 1, 1);
        dpb.add_short_term(Picture::new(1, 1), 2, 2);
        assert_eq!(dpb.refs.len(), 2);
        let ids: Vec<i32> = dpb.refs.iter().map(|r| r.id).collect();
        assert_eq!(ids, alloc::vec![1, 2]);
    }
}
