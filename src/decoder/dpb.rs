//! Decoded-picture reference buffer for baseline/Main P decoding.
//!
//! Ports the reference-picture management of `manage_dec_ref.cpp`:
//!   * sliding-window + adaptive (MMCO) `dec_ref_pic_marking` (`WelsMarkAsRef`,
//!     `SlidingWindow`, `MMCOProcess`),
//!   * short- and long-term reference lists (`AddShortTermToList` /
//!     `AddLongTermToList`),
//!   * default list-0 construction (`WelsInitRefList`: short refs by descending
//!     `PicNum`, then long refs by ascending `LongTermFrameIdx`),
//!   * `ref_pic_list_modification` reordering (`WelsReorderRefList`, spec
//!     8.2.4.3.1).
//!
//! Frame-coded, 4:2:0 only (no fields), which is the scope of this decoder.

use alloc::vec::Vec;

use super::picture::Picture;
use super::slice_header::{MmcoEntry, RefListReorder};

// memory_management_control_operation values (`wels_common_defs.h`).
const MMCO_SHORT2UNUSED: u32 = 1;
const MMCO_LONG2UNUSED: u32 = 2;
const MMCO_SHORT2LONG: u32 = 3;
const MMCO_SET_MAX_LONG: u32 = 4;
const MMCO_RESET: u32 = 5;
const MMCO_LONG: u32 = 6;

/// Colocated per-block motion of a reference picture, read by B-slice direct
/// prediction (`colocPic->pMbType / pMv / pRefIndex`).
#[derive(Clone, Default)]
pub struct ColMotion {
    /// Per-MB: the colocated macroblock is intra-coded.
    pub intra: Vec<bool>,
    /// Per-MB: the colocated macroblock uses list-1 (bi/backward prediction).
    pub uses_l1: Vec<bool>,
    /// Per-4x4-block list-0 / list-1 motion, 16 per MB raster order, `[x,y]`.
    pub mv: [Vec<i16>; 2],
    /// Per-4x4-block list-0 / list-1 reference index (slice-local), -1 = unused.
    pub ref_idx: [Vec<i8>; 2],
}

/// One decoded reference frame held in the DPB.
pub struct RefFrame {
    /// Reconstructed, border-extended picture.
    pub pic: Picture,
    /// `frame_num` from the slice header.
    pub frame_num: i32,
    /// Monotonic decode-order identity, used by the deblocker to tell whether
    /// two blocks reference the same picture.
    pub id: i32,
    /// Marked as a long-term reference.
    pub is_long_term: bool,
    /// `LongTermFrameIdx` (== `LongTermPicNum` for frame-coded pictures).
    pub long_term_frame_idx: i32,
    /// Picture order count (display order; B ref-list ordering & temporal direct).
    pub poc: i32,
    /// Colocated motion of this picture for B direct prediction.
    pub col: ColMotion,
}

/// Reference-picture buffer (sliding-window + MMCO; short- and long-term).
pub struct Dpb {
    pub refs: Vec<RefFrame>,
    max_num_ref_frames: usize,
    max_frame_num: i32,
    /// `MaxLongTermFrameIdx` (-1 == "no long-term frame indices").
    max_long_term_frame_idx: i32,
}

impl Dpb {
    pub fn new(max_num_ref_frames: u32, log2_max_frame_num: u32) -> Self {
        Dpb {
            refs: Vec::new(),
            max_num_ref_frames: max_num_ref_frames.max(1) as usize,
            max_frame_num: 1i32 << log2_max_frame_num,
            max_long_term_frame_idx: -1,
        }
    }

    /// Clear all references (IDR / MMCO 5).
    pub fn clear(&mut self) {
        self.refs.clear();
        self.max_long_term_frame_idx = -1;
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

    fn short_count(&self) -> usize {
        self.refs.iter().filter(|r| !r.is_long_term).count()
    }

    fn long_count(&self) -> usize {
        self.refs.iter().filter(|r| r.is_long_term).count()
    }

    /// Default list-0 (spec 8.2.4.2.1): short-term references by descending
    /// `PicNum`, followed by long-term references by ascending
    /// `LongTermFrameIdx`. Returns indices into [`Dpb::refs`].
    fn default_list0(&self, cur_frame_num: i32) -> Vec<usize> {
        let mut short: Vec<usize> = (0..self.refs.len())
            .filter(|&i| !self.refs[i].is_long_term)
            .collect();
        short.sort_by(|&a, &b| {
            let pa = self.pic_num(self.refs[a].frame_num, cur_frame_num);
            let pb = self.pic_num(self.refs[b].frame_num, cur_frame_num);
            pb.cmp(&pa)
        });
        let mut long: Vec<usize> = (0..self.refs.len())
            .filter(|&i| self.refs[i].is_long_term)
            .collect();
        long.sort_by_key(|&i| self.refs[i].long_term_frame_idx);
        short.extend(long);
        short
    }

    /// Build the list-0 reference list for a P slice: the default list with any
    /// `ref_pic_list_modification` applied (spec 8.2.4.3.1), truncated to
    /// `num_ref_active`. Returns indices into [`Dpb::refs`].
    pub fn p_ref_list(
        &self,
        cur_frame_num: i32,
        num_ref_active: usize,
        reorder: &RefListReorder,
    ) -> Vec<usize> {
        let default = self.default_list0(cur_frame_num);
        self.apply_ref_reorder(default, cur_frame_num, num_ref_active, reorder)
    }

    /// Build the list-0 and list-1 reference lists for a B slice (spec
    /// 8.2.4.2.3): list-0 = short refs `POC < cur` (POC desc), then `POC > cur`
    /// (POC asc), then long-term (LongTermFrameIdx asc); list-1 = short refs
    /// `POC > cur` (POC asc), then `POC < cur` (POC desc), then long-term. Any
    /// `ref_pic_list_modification` is then applied to each. Returns indices into
    /// [`Dpb::refs`].
    pub fn b_ref_lists(
        &self,
        cur_frame_num: i32,
        cur_poc: i32,
        num0: usize,
        num1: usize,
        reorder0: &RefListReorder,
        reorder1: &RefListReorder,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut less: Vec<usize> = (0..self.refs.len())
            .filter(|&i| !self.refs[i].is_long_term && self.refs[i].poc < cur_poc)
            .collect();
        less.sort_by(|&a, &b| self.refs[b].poc.cmp(&self.refs[a].poc)); // desc
        let mut greater: Vec<usize> = (0..self.refs.len())
            .filter(|&i| !self.refs[i].is_long_term && self.refs[i].poc >= cur_poc)
            .collect();
        greater.sort_by(|&a, &b| self.refs[a].poc.cmp(&self.refs[b].poc)); // asc
        let mut long: Vec<usize> = (0..self.refs.len())
            .filter(|&i| self.refs[i].is_long_term)
            .collect();
        long.sort_by_key(|&i| self.refs[i].long_term_frame_idx);

        let mut list0: Vec<usize> = Vec::new();
        list0.extend(less.iter().copied());
        list0.extend(greater.iter().copied());
        list0.extend(long.iter().copied());

        let mut list1: Vec<usize> = Vec::new();
        list1.extend(greater.iter().copied());
        list1.extend(less.iter().copied());
        list1.extend(long.iter().copied());

        // When list1 has more than one entry and is identical to list0, swap the
        // first two of list1 (spec 8.2.4.2.3). OpenH264 omits this; replicate its
        // behaviour for bit-exactness (and it is a no-op for our single-ref case).
        (
            self.apply_ref_reorder(list0, cur_frame_num, num0, reorder0),
            self.apply_ref_reorder(list1, cur_frame_num, num1, reorder1),
        )
    }

    /// Apply `ref_pic_list_modification` (spec 8.2.4.3.1) to `default`, truncate
    /// to `num`. Shared by P list-0 and both B lists.
    fn apply_ref_reorder(
        &self,
        default: Vec<usize>,
        cur_frame_num: i32,
        num: usize,
        reorder: &RefListReorder,
    ) -> Vec<usize> {
        if !reorder.flag {
            let mut list = default;
            list.truncate(num);
            return list;
        }

        // Working list of length `num` (clamped to availability), with one
        // scratch slot used during each insertion (spec 8.2.4.3.1).
        let mut work: Vec<usize> = default.iter().take(num).copied().collect();
        let cur_pic_num = cur_frame_num;
        let mut pred = cur_frame_num;
        let mut ref_idx = 0usize;

        for cmd in &reorder.entries {
            let idc = cmd.modification_of_pic_nums_idc;
            let target: Option<usize> = if idc == 0 || idc == 1 {
                let abs_diff = cmd.abs_diff_pic_num_minus1 as i32 + 1;
                let no_wrap = if idc == 0 {
                    if pred - abs_diff < 0 {
                        pred - abs_diff + self.max_frame_num
                    } else {
                        pred - abs_diff
                    }
                } else if pred + abs_diff >= self.max_frame_num {
                    pred + abs_diff - self.max_frame_num
                } else {
                    pred + abs_diff
                };
                pred = no_wrap;
                let mut pic_num = no_wrap;
                if pic_num > cur_pic_num {
                    pic_num -= self.max_frame_num;
                }
                // Short-term picture with PicNum == pic_num.
                (0..self.refs.len()).find(|&i| {
                    !self.refs[i].is_long_term
                        && self.pic_num(self.refs[i].frame_num, cur_frame_num) == pic_num
                })
            } else if idc == 2 {
                let lt = cmd.long_term_pic_num as i32;
                (0..self.refs.len())
                    .find(|&i| self.refs[i].is_long_term && self.refs[i].long_term_frame_idx == lt)
            } else {
                None
            };

            let Some(target) = target else { continue };
            if ref_idx > work.len() {
                break;
            }
            // Insert target at ref_idx (shifts the rest right), then drop the
            // later duplicate (PicNumF/LongTermPicNumF identity == same picture).
            work.insert(ref_idx, target);
            if let Some(dup) = (ref_idx + 1..work.len()).find(|&c| work[c] == target) {
                work.remove(dup);
            } else if work.len() > num {
                work.truncate(num);
            }
            ref_idx += 1;
        }
        work.truncate(num);
        work
    }

    /// Apply `dec_ref_pic_marking` for a freshly decoded reference picture and
    /// insert it (`WelsMarkAsRef`). `is_idr` selects the IDR path; otherwise
    /// `marking` carries the sliding-window/adaptive (MMCO) decision.
    #[allow(clippy::too_many_arguments)]
    pub fn mark_and_insert(
        &mut self,
        pic: Picture,
        frame_num: i32,
        id: i32,
        is_idr: bool,
        long_term_reference_flag: bool,
        adaptive: bool,
        mmco: &[MmcoEntry],
        poc: i32,
        col: ColMotion,
    ) {
        if is_idr {
            // The caller has already cleared the DPB for the IDR.
            if long_term_reference_flag {
                self.max_long_term_frame_idx = 0;
                self.refs.push(RefFrame {
                    pic,
                    frame_num,
                    id,
                    is_long_term: true,
                    long_term_frame_idx: 0,
                    poc,
                    col,
                });
            } else {
                self.max_long_term_frame_idx = -1;
                self.push_short(pic, frame_num, id, poc, col);
            }
            return;
        }

        let mut cur: Option<Picture> = Some(pic);
        let mut had_mmco5 = false;
        if adaptive {
            had_mmco5 = self.apply_mmco(mmco, frame_num, id, &mut cur, poc, &col);
        } else {
            self.sliding_window(frame_num);
        }
        // After MMCO 5 the current picture is stored with frame_num 0 (spec
        // 8.2.4.1 / WelsMarkAsRef bLastHasMmco5), affecting future PicNums.
        let store_frame_num = if had_mmco5 { 0 } else { frame_num };
        // If MMCO 6 did not consume the current picture as a long-term ref,
        // add it as a short-term reference.
        if let Some(pic) = cur {
            self.push_short(pic, store_frame_num, id, poc, col);
        }
    }

    /// `AddShortTermToList`: insert as a short-term reference, replacing any
    /// existing short-term reference with the same `frame_num`.
    fn push_short(&mut self, pic: Picture, frame_num: i32, id: i32, poc: i32, col: ColMotion) {
        if let Some(slot) = self
            .refs
            .iter_mut()
            .find(|r| !r.is_long_term && r.frame_num == frame_num)
        {
            slot.pic = pic;
            slot.id = id;
            slot.poc = poc;
            slot.col = col;
            return;
        }
        self.refs.push(RefFrame {
            pic,
            frame_num,
            id,
            is_long_term: false,
            long_term_frame_idx: -1,
            poc,
            col,
        });
    }

    /// `SlidingWindow`: when the buffer is full, drop the short-term reference
    /// with the smallest `PicNum` (the oldest) before adding the current one.
    fn sliding_window(&mut self, cur_frame_num: i32) {
        if self.short_count() + self.long_count() < self.max_num_ref_frames {
            return;
        }
        let mut victim: Option<usize> = None;
        let mut min_pn = i32::MAX;
        for (i, r) in self.refs.iter().enumerate() {
            if r.is_long_term {
                continue;
            }
            let pn = self.pic_num(r.frame_num, cur_frame_num);
            if pn < min_pn {
                min_pn = pn;
                victim = Some(i);
            }
        }
        if let Some(i) = victim {
            self.refs.remove(i);
        }
    }

    /// Apply the MMCO command list (`MMCOProcess`). `cur`/`frame_num`/`id`
    /// describe the current picture, which MMCO 6 may move into the long-term
    /// list (taking it out of `cur`).
    fn apply_mmco(
        &mut self,
        mmco: &[MmcoEntry],
        frame_num: i32,
        id: i32,
        cur: &mut Option<Picture>,
        poc: i32,
        col: &ColMotion,
    ) -> bool {
        let mut had_mmco5 = false;
        for e in mmco {
            match e.mmco_type {
                MMCO_SHORT2UNUSED => {
                    // Remove the short-term ref with this ShortTermFrameNum.
                    if let Some(i) = self
                        .refs
                        .iter()
                        .position(|r| !r.is_long_term && r.frame_num == e.short_frame_num)
                    {
                        self.refs.remove(i);
                    }
                }
                MMCO_LONG2UNUSED => {
                    let lt = e.long_term_pic_num as i32;
                    if let Some(i) = self
                        .refs
                        .iter()
                        .position(|r| r.is_long_term && r.long_term_frame_idx == lt)
                    {
                        self.refs.remove(i);
                    }
                }
                MMCO_SHORT2LONG => {
                    // Free any long-term ref already holding this idx, then
                    // promote the short-term ref with ShortTermFrameNum.
                    self.remove_long(e.long_term_frame_idx);
                    if let Some(r) = self
                        .refs
                        .iter_mut()
                        .find(|r| !r.is_long_term && r.frame_num == e.short_frame_num)
                    {
                        r.is_long_term = true;
                        r.long_term_frame_idx = e.long_term_frame_idx;
                    }
                }
                MMCO_SET_MAX_LONG => {
                    self.max_long_term_frame_idx = e.max_long_term_frame_idx;
                    self.refs
                        .retain(|r| !r.is_long_term || r.long_term_frame_idx <= e.max_long_term_frame_idx);
                }
                MMCO_RESET => {
                    self.refs.clear();
                    self.max_long_term_frame_idx = -1;
                    had_mmco5 = true;
                }
                MMCO_LONG => {
                    // Mark the current picture as a long-term reference.
                    self.remove_long(e.long_term_frame_idx);
                    if let Some(pic) = cur.take() {
                        self.refs.push(RefFrame {
                            pic,
                            frame_num,
                            id,
                            is_long_term: true,
                            long_term_frame_idx: e.long_term_frame_idx,
                            poc,
                            col: col.clone(),
                        });
                    }
                }
                _ => {}
            }
        }
        had_mmco5
    }

    fn remove_long(&mut self, long_term_frame_idx: i32) {
        if let Some(i) = self
            .refs
            .iter()
            .position(|r| r.is_long_term && r.long_term_frame_idx == long_term_frame_idx)
        {
            self.refs.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::slice_header::{ReorderEntry, RefListReorder};

    fn rf(frame_num: i32, id: i32) -> RefFrame {
        RefFrame {
            pic: Picture::new(1, 1),
            frame_num,
            id,
            is_long_term: false,
            long_term_frame_idx: -1,
            poc: 0,
            col: ColMotion::default(),
        }
    }

    fn no_reorder() -> RefListReorder {
        RefListReorder::default()
    }

    #[test]
    fn p_list_descending_picnum() {
        let mut dpb = Dpb::new(4, 4); // max_frame_num = 16
        dpb.refs.push(rf(0, 0));
        dpb.refs.push(rf(1, 1));
        dpb.refs.push(rf(2, 2));
        let list = dpb.p_ref_list(3, 3, &no_reorder());
        let ids: Vec<i32> = list.iter().map(|&i| dpb.refs[i].id).collect();
        assert_eq!(ids, alloc::vec![2, 1, 0]);
    }

    #[test]
    fn p_list_wraps_frame_num() {
        let mut dpb = Dpb::new(4, 4);
        dpb.refs.push(rf(14, 14));
        dpb.refs.push(rf(15, 15));
        dpb.refs.push(rf(0, 16)); // wrapped past 15
        let list = dpb.p_ref_list(1, 3, &no_reorder());
        let ids: Vec<i32> = list.iter().map(|&i| dpb.refs[i].id).collect();
        assert_eq!(ids, alloc::vec![16, 15, 14]);
    }

    #[test]
    fn sliding_window_drops_oldest() {
        let mut dpb = Dpb::new(2, 4);
        dpb.mark_and_insert(Picture::new(1, 1), 0, 0, false, false, false, &[], 0, ColMotion::default());
        dpb.mark_and_insert(Picture::new(1, 1), 1, 1, false, false, false, &[], 0, ColMotion::default());
        dpb.mark_and_insert(Picture::new(1, 1), 2, 2, false, false, false, &[], 0, ColMotion::default());
        assert_eq!(dpb.refs.len(), 2);
        let mut ids: Vec<i32> = dpb.refs.iter().map(|r| r.id).collect();
        ids.sort();
        assert_eq!(ids, alloc::vec![1, 2]);
    }

    #[test]
    fn reorder_reverses_default() {
        // frames 0,1,2 in DPB, cur=3: default [2,1,0]; reorder picks PicNum 0
        // then PicNum 1 -> [0,1,2].
        let mut dpb = Dpb::new(4, 4);
        dpb.refs.push(rf(0, 0));
        dpb.refs.push(rf(1, 1));
        dpb.refs.push(rf(2, 2));
        let reorder = RefListReorder {
            flag: true,
            entries: alloc::vec![
                ReorderEntry { modification_of_pic_nums_idc: 0, abs_diff_pic_num_minus1: 2, long_term_pic_num: 0 },
                ReorderEntry { modification_of_pic_nums_idc: 1, abs_diff_pic_num_minus1: 0, long_term_pic_num: 0 },
            ],
        };
        let list = dpb.p_ref_list(3, 3, &reorder);
        let ids: Vec<i32> = list.iter().map(|&i| dpb.refs[i].id).collect();
        assert_eq!(ids, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn long_term_selected_by_reorder() {
        let mut dpb = Dpb::new(4, 4);
        // IDR as long-term idx 0.
        dpb.mark_and_insert(Picture::new(1, 1), 0, 0, true, true, false, &[], 0, ColMotion::default());
        // A couple short-term refs.
        dpb.mark_and_insert(Picture::new(1, 1), 1, 1, false, false, false, &[], 0, ColMotion::default());
        dpb.mark_and_insert(Picture::new(1, 1), 2, 2, false, false, false, &[], 0, ColMotion::default());
        // reorder idc 2 -> long-term pic num 0 at front.
        let reorder = RefListReorder {
            flag: true,
            entries: alloc::vec![ReorderEntry {
                modification_of_pic_nums_idc: 2,
                abs_diff_pic_num_minus1: 0,
                long_term_pic_num: 0,
            }],
        };
        let list = dpb.p_ref_list(3, 2, &reorder);
        assert_eq!(dpb.refs[list[0]].id, 0); // the long-term IDR
    }
}
