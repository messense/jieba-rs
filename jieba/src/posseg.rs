use std::cell::RefCell;
use std::sync::OnceLock;

use crate::FxHashMap;

const MIN_FLOAT: f64 = -3.14e100;
const NUM_POS: usize = 4; // B=0, M=1, E=2, S=3
const NUM_TAGS: usize = 64;
const NUM_STATES: usize = NUM_POS * NUM_TAGS;

/// Positions that may precede each position: B after {E, S}, M and E after
/// {B, M}, S after {E, S}. The transition table only ever contains these pairs,
/// so restricting the search to them is exact and halves the inner loop.
const PREV_POS: [[usize; 2]; NUM_POS] = [[2, 3], [0, 1], [0, 1], [2, 3]];

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

/// Candidate states of one character, with the emission log-prob of each.
///
/// States are sorted ascending, so they are grouped by position;
/// `bounds[p]..bounds[p + 1]` is the range of states with position `p`.
struct CharStates {
    states: Box<[(u16, f64)]>,
    bounds: [u16; NUM_POS + 1],
}

impl CharStates {
    fn new(mut states: Vec<(u16, f64)>) -> Self {
        states.sort_unstable_by_key(|&(s, _)| s);
        let mut bounds = [0u16; NUM_POS + 1];
        for pos in 0..NUM_POS {
            let end = states.partition_point(|&(s, _)| state_pos(s as usize) <= pos);
            bounds[pos + 1] = end as u16;
        }
        CharStates {
            states: states.into_boxed_slice(),
            bounds,
        }
    }

    #[inline]
    fn with_pos(&self, pos: usize) -> &[(u16, f64)] {
        &self.states[self.bounds[pos] as usize..self.bounds[pos + 1] as usize]
    }
}

pub(crate) struct PossegData {
    tags: Vec<Box<str>>,
    start_prob: [f64; NUM_STATES],
    /// Dense 256×256 transition matrix. trans_prob[from][to] = log-prob.
    trans_prob: Box<[[f64; NUM_STATES]; NUM_STATES]>,
    /// Per character: the states it can take, each paired with its emission
    /// log-prob, so the Viterbi pass does a single hash lookup per character.
    ///
    /// A character with an explicit state list keeps exactly that list (an
    /// emission missing for one of its states is `MIN_FLOAT`). A character
    /// with emissions but no state list may take any state in the original
    /// model; a state without an emission scores `MIN_FLOAT` and so can never
    /// be chosen, so listing only the states that have an emission is exact.
    char_states: FxHashMap<char, CharStates>,
    /// Returned for characters the model has never seen: no state can be taken.
    no_states: CharStates,
}

fn parse_posseg_data(data: &str) -> PossegData {
    let mut tags: Vec<Box<str>> = Vec::new();
    let mut start_prob = [MIN_FLOAT; NUM_STATES];
    let mut trans_prob: Box<[[f64; NUM_STATES]; NUM_STATES]> = vec![[MIN_FLOAT; NUM_STATES]; NUM_STATES]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    let mut emit_prob: FxHashMap<char, Vec<(u16, f64)>> = FxHashMap::default();
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
                let si = state_idx(pos, tag) as u16;

                for seg in segments {
                    let seg = seg.trim();
                    if seg.is_empty() {
                        continue;
                    }
                    let mut parts = seg.rsplitn(2, ',');
                    let prob: f64 = parts.next().unwrap().parse().unwrap();
                    let ch_str = parts.next().unwrap();
                    let ch = ch_str.chars().next().unwrap();
                    let entries = emit_prob.entry(ch).or_default();
                    match entries.iter_mut().find(|(s, _)| *s == si) {
                        Some(entry) => entry.1 = prob,
                        None => entries.push((si, prob)),
                    }
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
                // An empty list means "no restriction", the same as no list at all.
                if states.is_empty() {
                    char_state_tab.remove(&ch);
                } else {
                    char_state_tab.insert(ch, states);
                }
            }
            _ => {}
        }
    }

    let mut char_states: FxHashMap<char, CharStates> =
        FxHashMap::with_capacity_and_hasher(emit_prob.len().max(char_state_tab.len()), rustc_hash::FxBuildHasher);
    for (ch, states) in char_state_tab {
        let emits = emit_prob.remove(&ch).unwrap_or_default();
        let states = states
            .into_iter()
            .map(|s| {
                let prob = emits.iter().find(|(es, _)| *es == s).map_or(MIN_FLOAT, |&(_, p)| p);
                (s, prob)
            })
            .collect();
        char_states.insert(ch, CharStates::new(states));
    }
    for (ch, emits) in emit_prob {
        char_states.insert(ch, CharStates::new(emits));
    }

    PossegData {
        tags,
        start_prob,
        trans_prob,
        char_states,
        no_states: CharStates::new(Vec::new()),
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

impl PossegData {
    #[inline]
    fn char_states(&self, ch: char) -> &CharStates {
        self.char_states.get(&ch).unwrap_or(&self.no_states)
    }

    fn tag_str(&self, tag_idx: usize) -> &str {
        &self.tags[tag_idx]
    }
}

/// Reusable Viterbi buffers, so tagging a short OOV word allocates nothing.
#[derive(Default)]
struct Scratch {
    chars: Vec<(usize, char)>,
    /// Backpointers, `c_len × NUM_STATES`.
    prev: Vec<u16>,
    path: Vec<u16>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// A word span decoded from the best state path: byte range and tag.
type Span<'a> = (usize, usize, &'a str);

/// Runs Viterbi over `chars` and feeds each decoded span to `emit`, in order.
fn viterbi_posseg<'a>(data: &'a PossegData, scratch: &mut Scratch, mut emit: impl FnMut(Span<'a>)) {
    let chars = &scratch.chars;
    let c_len = chars.len();
    if c_len == 0 {
        return;
    }

    let str_end = chars[c_len - 1].0 + chars[c_len - 1].1.len_utf8();

    // Single character: just pick the best S state
    if c_len == 1 {
        let ch = chars[0].1;
        let mut best: Option<(f64, u16)> = None;
        for &(s, em) in data.char_states(ch).with_pos(3) {
            let prob = data.start_prob[s as usize] + em;
            if prob > MIN_FLOAT && best.is_none_or(|(bp, _)| prob >= bp) {
                best = Some((prob, s));
            }
        }
        let tag = match best {
            Some((_, s)) => data.tag_str(state_tag(s as usize)),
            None => "x",
        };
        emit((chars[0].0, str_end, tag));
        return;
    }

    // Rolling score buffers: only need prev and current rows
    let mut scores = [[MIN_FLOAT; NUM_STATES]; 2];
    // Backpointer table: still need full c_len × NUM_STATES for traceback
    let prev = &mut scratch.prev;
    prev.clear();
    prev.resize(c_len * NUM_STATES, u16::MAX);

    // Initialize t=0
    let first_ch = chars[0].1;
    let mut prev_states = data.char_states(first_ch);
    for &(s, em) in prev_states.states.iter() {
        let si = s as usize;
        scores[0][si] = data.start_prob[si] + em;
    }

    // Recurse
    for t in 1..c_len {
        let ch = chars[t].1;
        let cur_states = data.char_states(ch);

        let (prev_scores, cur_scores) = {
            let (a, b) = scores.split_at_mut(1);
            if t % 2 == 1 { (&a[0], &mut b[0]) } else { (&b[0], &mut a[0]) }
        };
        // Only candidate entries are ever written; reset the ones this buffer
        // received two steps ago instead of clearing all NUM_STATES slots.
        if t >= 3 {
            for &(s, _) in data.char_states(chars[t - 2].1).states.iter() {
                cur_scores[s as usize] = MIN_FLOAT;
            }
        }

        for &(s, em) in cur_states.states.iter() {
            let si = s as usize;
            let mut best_prob = MIN_FLOAT;
            let mut best_prev = u16::MAX;

            for &pp in &PREV_POS[state_pos(si)] {
                for &(ps, _) in prev_states.with_pos(pp) {
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
            }

            cur_scores[si] = best_prob;
            prev[t * NUM_STATES + si] = best_prev;
        }

        prev_states = cur_states;
    }

    // Terminate: find best E or S state at the last timestep
    let last_t = c_len - 1;
    let last_scores = &scores[last_t % 2];
    let mut best_prob = MIN_FLOAT;
    let mut best_state = u16::MAX;
    for pos in [2, 3] {
        for &(s, _) in prev_states.with_pos(pos) {
            if last_scores[s as usize] > best_prob {
                best_prob = last_scores[s as usize];
                best_state = s;
            }
        }
    }

    // Fallback if no valid E/S state was reachable
    if best_state == u16::MAX || best_prob <= MIN_FLOAT {
        emit((chars[0].0, str_end, "x"));
        return;
    }

    // Traceback
    let path = &mut scratch.path;
    path.clear();
    path.resize(c_len, 0);
    path[last_t] = best_state;
    for t in (1..c_len).rev() {
        let backptr = prev[t * NUM_STATES + path[t] as usize];
        if backptr == u16::MAX {
            // Unreachable path — return whole span as fallback
            emit((chars[0].0, str_end, "x"));
            return;
        }
        path[t - 1] = backptr;
    }

    // Decode word boundaries
    let mut word_start = chars[0].0;
    let mut last_end = None;

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
                emit((word_start, byte_end, tag));
                last_end = Some(byte_end);
            }
            3 => {
                // S: single char word
                let byte_end = if t + 1 < c_len { chars[t + 1].0 } else { str_end };
                let tag = data.tag_str(state_tag(s));
                emit((chars[t].0, byte_end, tag));
                last_end = Some(byte_end);
            }
            _ => unreachable!(),
        }
    }

    match last_end {
        // Fallback if decoding produced no words (e.g. all B/M with no E)
        None => emit((chars[0].0, str_end, "x")),
        // Handle incomplete B..M sequence at end
        Some(byte_end) if byte_end < str_end => emit((byte_end, str_end, "x")),
        Some(_) => {}
    }
}

/// Segment and POS-tag a Chinese character string using the compound HMM.
///
/// Returns `(word_slice, pos_tag_str)` pairs where `pos_tag_str` has `'static`
/// lifetime because it references the lazily-initialized static data.
#[cfg(all(test, feature = "default-dict"))]
fn cut_with_pos(sentence: &str) -> Vec<(&str, &'static str)> {
    let data = posseg_data();
    let mut spans = Vec::new();
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.chars.clear();
        scratch.chars.extend(sentence.char_indices());
        viterbi_posseg(data, &mut scratch, |(start, end, tag)| spans.push((&sentence[start..end], tag)));
    });
    spans
}

/// The tag for an OOV word: the tag of the longest span the compound HMM
/// finds in it (the last such span on ties), or `"x"` for an empty word.
#[cfg(feature = "default-dict")]
pub(crate) fn guess_tag(word: &str) -> &'static str {
    let data = posseg_data();
    let mut best: Option<(usize, &'static str)> = None;
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.chars.clear();
        scratch.chars.extend(word.char_indices());
        viterbi_posseg(data, &mut scratch, |(start, end, tag)| {
            let len = end - start;
            if best.is_none_or(|(best_len, _)| len >= best_len) {
                best = Some((len, tag));
            }
        });
    });
    best.map_or("x", |(_, tag)| tag)
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

    #[test]
    fn test_guess_tag_matches_longest_span() {
        for word in ["张尧", "小明硕士毕业于中国科学院计算所", "云计算", "我", "创新办", "龘齉"] {
            let spans = cut_with_pos(word);
            let expected = spans.iter().max_by_key(|(w, _)| w.len()).map_or("x", |(_, t)| t);
            assert_eq!(guess_tag(word), expected, "{word}");
        }
    }
}
