use std::cmp::Ordering;
use std::sync::OnceLock;

use crate::FxHashMap;

const MIN_FLOAT: f64 = -3.14e100;
const NUM_POS: usize = 4; // B=0, M=1, E=2, S=3
const NUM_TAGS: usize = 64;
const NUM_STATES: usize = NUM_POS * NUM_TAGS;

#[inline]
fn state_idx(pos: usize, tag: usize) -> usize {
    pos * NUM_TAGS + tag
}

#[inline]
fn state_pos(idx: usize) -> usize {
    idx / NUM_TAGS
}

#[inline]
fn state_tag(idx: usize) -> usize {
    idx % NUM_TAGS
}

pub(crate) struct PossegData {
    tags: Vec<Box<str>>,
    start_prob: [f64; NUM_STATES],
    /// Dense 256×256 transition matrix. trans_prob[from][to] = log-prob.
    trans_prob: Box<[[f64; NUM_STATES]; NUM_STATES]>,
    emit_prob: Vec<FxHashMap<char, f64>>,
    char_state_tab: FxHashMap<char, Vec<u16>>,
}

fn parse_posseg_data(data: &str) -> PossegData {
    let mut tags: Vec<Box<str>> = Vec::new();
    let mut start_prob = [MIN_FLOAT; NUM_STATES];
    // Initialize dense matrix to MIN_FLOAT
    let mut trans_prob = vec![[MIN_FLOAT; NUM_STATES]; NUM_STATES].into_boxed_slice();
    let trans_prob: Box<[[f64; NUM_STATES]; NUM_STATES]> = unsafe {
        let ptr = Box::into_raw(trans_prob) as *mut [[f64; NUM_STATES]; NUM_STATES];
        Box::from_raw(ptr)
    };
    let mut trans_prob = trans_prob;
    let mut emit_prob: Vec<FxHashMap<char, f64>> = vec![FxHashMap::default(); NUM_STATES];
    let mut char_state_tab: FxHashMap<char, Vec<u16>> = FxHashMap::default();

    let mut section = "";

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('@') {
            section = match line {
                "@TAGS" => "tags",
                "@START" => "start",
                "@TRANS" => "trans",
                "@EMIT" => "emit",
                "@CHAR_STATE" => "char_state",
                _ => "",
            };
            continue;
        }

        match section {
            "tags" => {
                tags = line.split(',').map(|s| s.trim().into()).collect();
                assert_eq!(tags.len(), NUM_TAGS, "expected {NUM_TAGS} POS tags, got {}", tags.len());
            }
            "start" => {
                let mut parts = line.splitn(3, ',');
                let pos: usize = parts.next().unwrap().parse().unwrap();
                let tag: usize = parts.next().unwrap().parse().unwrap();
                let prob: f64 = parts.next().unwrap().parse().unwrap();
                start_prob[state_idx(pos, tag)] = prob;
            }
            "trans" => {
                let mut segments = line.split('|');
                let from_part = segments.next().unwrap();
                let mut from_iter = from_part.splitn(2, ',');
                let from_pos: usize = from_iter.next().unwrap().parse().unwrap();
                let from_tag: usize = from_iter.next().unwrap().parse().unwrap();
                let from = state_idx(from_pos, from_tag);

                for seg in segments {
                    let seg = seg.trim();
                    if seg.is_empty() {
                        continue;
                    }
                    let mut parts = seg.splitn(3, ',');
                    let to_pos: usize = parts.next().unwrap().parse().unwrap();
                    let to_tag: usize = parts.next().unwrap().parse().unwrap();
                    let prob: f64 = parts.next().unwrap().parse().unwrap();
                    trans_prob[from][state_idx(to_pos, to_tag)] = prob;
                }
            }
            "emit" => {
                let mut segments = line.split('|');
                let state_part = segments.next().unwrap();
                let mut state_iter = state_part.splitn(2, ',');
                let pos: usize = state_iter.next().unwrap().parse().unwrap();
                let tag: usize = state_iter.next().unwrap().parse().unwrap();
                let si = state_idx(pos, tag);

                for seg in segments {
                    let seg = seg.trim();
                    if seg.is_empty() {
                        continue;
                    }
                    let mut parts = seg.rsplitn(2, ',');
                    let prob: f64 = parts.next().unwrap().parse().unwrap();
                    let ch_str = parts.next().unwrap();
                    let ch = ch_str.chars().next().unwrap();
                    emit_prob[si].insert(ch, prob);
                }
            }
            "char_state" => {
                let mut segments = line.split('|');
                let ch_str = segments.next().unwrap();
                let ch = ch_str.chars().next().unwrap();
                let mut states = Vec::new();
                for seg in segments {
                    let seg = seg.trim();
                    if seg.is_empty() {
                        continue;
                    }
                    let mut parts = seg.splitn(2, ',');
                    let pos: usize = parts.next().unwrap().parse().unwrap();
                    let tag: usize = parts.next().unwrap().parse().unwrap();
                    states.push(state_idx(pos, tag) as u16);
                }
                char_state_tab.insert(ch, states);
            }
            _ => {}
        }
    }

    PossegData {
        tags,
        start_prob,
        trans_prob,
        emit_prob,
        char_state_tab,
    }
}

#[cfg(feature = "default-dict")]
include_flate::flate!(static POSSEG_DATA: str from "src/data/posseg.txt");

#[cfg(feature = "default-dict")]
static POSSEG: OnceLock<PossegData> = OnceLock::new();

#[cfg(feature = "default-dict")]
pub(crate) fn posseg_data() -> &'static PossegData {
    POSSEG.get_or_init(|| parse_posseg_data(&POSSEG_DATA))
}

/// All possible state indices (used as fallback when char is not in char_state_tab).
fn all_states() -> Vec<u16> {
    (0..NUM_STATES as u16).collect()
}

impl PossegData {
    fn get_char_states(&self, ch: char) -> &[u16] {
        self.char_state_tab.get(&ch).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn emit(&self, state: usize, ch: char) -> f64 {
        self.emit_prob[state].get(&ch).copied().unwrap_or(MIN_FLOAT)
    }

    fn tag_str(&self, tag_idx: usize) -> &str {
        &self.tags[tag_idx]
    }
}

fn viterbi_posseg<'a>(data: &'a PossegData, chars: &[(usize, char)]) -> Vec<(usize, usize, &'a str)> {
    let c_len = chars.len();
    if c_len == 0 {
        return Vec::new();
    }

    let str_end = chars[c_len - 1].0 + chars[c_len - 1].1.len_utf8();
    let fallback = all_states();

    // Single character: just pick the best S state
    if c_len == 1 {
        let ch = chars[0].1;
        let candidates = data.get_char_states(ch);
        let candidates = if candidates.is_empty() { &fallback } else { candidates };
        let best = candidates
            .iter()
            .filter(|&&s| state_pos(s as usize) == 3) // S states only
            .map(|&s| {
                let prob = data.start_prob[s as usize] + data.emit(s as usize, ch);
                (prob, s)
            })
            .filter(|(p, _)| *p > MIN_FLOAT)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let tag = match best {
            Some((_, s)) => data.tag_str(state_tag(s as usize)),
            None => "x",
        };
        return vec![(chars[0].0, str_end, tag)];
    }

    // Rolling score buffers: only need prev and current rows
    let mut prev_scores = [MIN_FLOAT; NUM_STATES];
    let mut cur_scores = [MIN_FLOAT; NUM_STATES];
    // Backpointer table: still need full c_len × NUM_STATES for traceback
    let mut prev = vec![u16::MAX; c_len * NUM_STATES];

    // Initialize t=0
    let first_ch = chars[0].1;
    let first_states = data.get_char_states(first_ch);
    let first_states = if first_states.is_empty() {
        &fallback
    } else {
        first_states
    };
    for &s in first_states {
        let si = s as usize;
        prev_scores[si] = data.start_prob[si] + data.emit(si, first_ch);
    }

    // Recurse
    for t in 1..c_len {
        let ch = chars[t].1;
        let cur_states = data.get_char_states(ch);
        let cur_states = if cur_states.is_empty() { &fallback } else { cur_states };

        // Hoist prev_states lookup outside the inner loop
        let prev_ch = chars[t - 1].1;
        let prev_states = data.get_char_states(prev_ch);
        let prev_states = if prev_states.is_empty() { &fallback } else { prev_states };

        cur_scores.fill(MIN_FLOAT);

        for &s in cur_states {
            let si = s as usize;
            let em = data.emit(si, ch);
            let mut best_prob = MIN_FLOAT;
            let mut best_prev = u16::MAX;

            for &ps in prev_states {
                let psi = ps as usize;
                let pv = prev_scores[psi];
                if pv <= MIN_FLOAT {
                    continue;
                }
                // O(1) dense matrix lookup
                let tp = data.trans_prob[psi][si];
                if tp <= MIN_FLOAT {
                    continue;
                }
                let prob = pv + tp + em;
                if prob > best_prob {
                    best_prob = prob;
                    best_prev = ps;
                }
            }

            cur_scores[si] = best_prob;
            prev[t * NUM_STATES + si] = best_prev;
        }

        prev_scores = cur_scores;
    }

    // Terminate: find best E or S state at the last timestep
    let last_t = c_len - 1;
    let mut best_prob = MIN_FLOAT;
    let mut best_state = u16::MAX;
    for s in 0..NUM_STATES {
        let pos = state_pos(s);
        if (pos == 2 || pos == 3) && prev_scores[s] > best_prob {
            best_prob = prev_scores[s];
            best_state = s as u16;
        }
    }

    // Fallback if no valid E/S state was reachable
    if best_state == u16::MAX || best_prob <= MIN_FLOAT {
        return vec![(chars[0].0, str_end, "x")];
    }

    // Traceback
    let mut path = vec![0u16; c_len];
    path[last_t] = best_state;
    for t in (1..c_len).rev() {
        let backptr = prev[t * NUM_STATES + path[t] as usize];
        if backptr == u16::MAX {
            // Unreachable path — return whole span as fallback
            return vec![(chars[0].0, str_end, "x")];
        }
        path[t - 1] = backptr;
    }

    // Decode word boundaries
    let mut result = Vec::new();
    let mut word_start = chars[0].0;

    for t in 0..c_len {
        let s = path[t] as usize;
        let pos = state_pos(s);
        match pos {
            0 => {
                // B: start of a new word
                word_start = chars[t].0;
            }
            1 => {
                // M: middle, do nothing
            }
            2 => {
                // E: end of word
                let byte_end = if t + 1 < c_len { chars[t + 1].0 } else { str_end };
                let tag = data.tag_str(state_tag(s));
                result.push((word_start, byte_end, tag));
            }
            3 => {
                // S: single char word
                let byte_end = if t + 1 < c_len { chars[t + 1].0 } else { str_end };
                let tag = data.tag_str(state_tag(s));
                result.push((chars[t].0, byte_end, tag));
            }
            _ => unreachable!(),
        }
    }

    // Fallback if decoding produced no words (e.g. all B/M with no E)
    if result.is_empty() {
        return vec![(chars[0].0, str_end, "x")];
    }

    // Handle incomplete B..M sequence at end
    if let Some(&(_, byte_end, _)) = result.last() {
        if byte_end < str_end {
            result.push((byte_end, str_end, "x"));
        }
    }

    result
}

/// Segment and POS-tag a Chinese character string using the compound HMM.
///
/// Returns `(word_slice, pos_tag_str)` pairs where `pos_tag_str` has `'static`
/// lifetime because it references the lazily-initialized static data.
#[cfg(feature = "default-dict")]
pub(crate) fn cut_with_pos(sentence: &str) -> Vec<(&str, &'static str)> {
    let data = posseg_data();
    let chars: Vec<(usize, char)> = sentence.char_indices().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    let spans = viterbi_posseg(data, &chars);
    spans
        .into_iter()
        .map(|(start, end, tag)| (&sentence[start..end], tag))
        .collect()
}

#[cfg(all(test, feature = "default-dict"))]
mod tests {
    use super::*;
    use expect_test::expect;

    #[test]
    fn test_posseg_basic() {
        let results = cut_with_pos("我来到北京清华大学");
        let formatted: Vec<String> = results.iter().map(|(w, t)| format!("{}/{}", w, t)).collect();
        expect![[r#"["我/r", "来/v", "到/v", "北京/ns", "清华大学/nt"]"#]].assert_eq(&format!("{:?}", formatted));
    }

    #[test]
    fn test_posseg_person_name() {
        let results = cut_with_pos("小明硕士毕业于中国科学院计算所");
        let formatted: Vec<String> = results.iter().map(|(w, t)| format!("{}/{}", w, t)).collect();
        expect![[r#"["小明/nr", "硕士/n", "毕业/n", "于/p", "中国科学院/nt", "计算/v", "所/u"]"#]]
            .assert_eq(&format!("{:?}", formatted));
    }

    #[test]
    fn test_posseg_single_char() {
        let results = cut_with_pos("我");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "我");
        assert_eq!(results[0].1, "r"); // pronoun
    }

    #[test]
    fn test_posseg_oov_name() {
        let results = cut_with_pos("张尧");
        let formatted: Vec<String> = results.iter().map(|(w, t)| format!("{}/{}", w, t)).collect();
        assert!(
            results.iter().any(|(_, t)| *t == "nr"),
            "Expected person name tag for 张尧, got: {:?}",
            formatted
        );
    }

    #[test]
    fn test_posseg_empty() {
        let results = cut_with_pos("");
        assert!(results.is_empty());
    }
}
