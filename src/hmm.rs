use std::cmp::Ordering;

use lazy_static::lazy_static;
use regex::Regex;

use crate::SplitMatches;

lazy_static! {
    static ref RE_HAN: Regex = Regex::new(r"([\u{4E00}-\u{9FD5}]+)").unwrap();
    static ref RE_SKIP: Regex = Regex::new(r"([a-zA-Z0-9]+(?:.\d+)?%?)").unwrap();
}

pub const NUM_STATES: usize = 4;

pub type StateSet = [f64; NUM_STATES];

/// Result of hmm is a labeling of each Unicode Scalar Value in the input
/// string with Begin, Middle, End, or Single. These denote the proposed
/// segments. A segment is one of the following two patterns.
///
///   Begin, [Middle...], End
///   Single
///
/// Each state in the enum is also assigned an index value from 0-3 that
/// can be used as an index into an array representing data pertaining
/// to that state.
///
/// WARNING: The data file format for hmm.model comments imply one can
/// reassign the index values of each state at the top but `build.rs`
/// currently ignores the mapping. Do not reassign these indicies without
/// verifying how it interacts with `build.rs`.  These indicies must also
/// match the order if ALLOWED_PREV_STATUS.
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub enum State {
    Begin = 0,
    End = 1,
    Middle = 2,
    Single = 3,
}

// Mapping representing the allow transitiongs into the given state.
//
// WARNING: Ordering must match the indicies in State.
static ALLOWED_PREV_STATUS: [[State; 2]; NUM_STATES] = [
    // Can preceed State::Begin
    [State::End, State::Single],
    // Can preceed State::End
    [State::Begin, State::Middle],
    // Can preceed State::Middle
    [State::Middle, State::Begin],
    // Can preceed State::Single
    [State::Single, State::End],
];

include!(concat!(env!("OUT_DIR"), "/hmm_prob.rs"));

const MIN_FLOAT: f64 = -3.14e100;

pub(crate) struct HmmContext {
    v: Vec<f64>,
    prev: Vec<Option<State>>,
    best_path: Vec<State>,
}

impl HmmContext {
    pub fn new(num_characters: usize) -> Self {
        HmmContext {
            v: vec![0.0; NUM_STATES * num_characters],
            prev: vec![None; NUM_STATES * num_characters],
            best_path: vec![State::Begin; num_characters],
        }
    }
}

#[allow(non_snake_case)]
fn viterbi(sentence: &str, hmm_context: &mut HmmContext) {
    let str_len = sentence.len();
    let states = [State::Begin, State::Middle, State::End, State::Single];
    #[allow(non_snake_case)]
    let R = states.len();
    let C = sentence.chars().count();
    assert!(C > 1);

    // TODO: Can code just do fill() with the default instead of clear() and resize?
    if hmm_context.prev.len() < R * C {
        hmm_context.prev.resize(R * C, None);
    }

    if hmm_context.v.len() < R * C {
        hmm_context.v.resize(R * C, 0.0);
    }

    if hmm_context.best_path.len() < C {
        hmm_context.best_path.resize(C, State::Begin);
    }

    let mut curr = sentence.char_indices().map(|x| x.0).peekable();
    let x1 = curr.next().unwrap();
    let x2 = *curr.peek().unwrap();
    for y in &states {
        let first_word = &sentence[x1..x2];
        let prob = INITIAL_PROBS[*y as usize] + EMIT_PROBS[*y as usize].get(first_word).cloned().unwrap_or(MIN_FLOAT);
        hmm_context.v[*y as usize] = prob;
    }

    let mut t = 1;
    while let Some(byte_start) = curr.next() {
        for y in &states {
            let byte_end = *curr.peek().unwrap_or(&str_len);
            let word = &sentence[byte_start..byte_end];
            let em_prob = EMIT_PROBS[*y as usize].get(word).cloned().unwrap_or(MIN_FLOAT);
            let (prob, state) = ALLOWED_PREV_STATUS[*y as usize]
                .iter()
                .map(|y0| {
                    (
                        hmm_context.v[(t - 1) * R + (*y0 as usize)]
                            + TRANS_PROBS[*y0 as usize].get(*y as usize).cloned().unwrap_or(MIN_FLOAT)
                            + em_prob,
                        *y0,
                    )
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap();
            let idx = (t * R) + (*y as usize);
            hmm_context.v[idx] = prob;
            hmm_context.prev[idx] = Some(state);
        }

        t += 1;
    }

    let (_prob, state) = [State::End, State::Single]
        .iter()
        .map(|y| (hmm_context.v[(C - 1) * R + (*y as usize)], y))
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();

    let mut t = C - 1;
    let mut curr = *state;

    hmm_context.best_path[t] = *state;
    while let Some(p) = hmm_context.prev[t * R + (curr as usize)] {
        assert!(t > 0);
        hmm_context.best_path[t - 1] = p;
        curr = p;
        t -= 1;
    }

    hmm_context.prev.clear();
    hmm_context.v.clear();
}

#[allow(non_snake_case)]
pub(crate) fn cut_internal<'a>(sentence: &'a str, words: &mut Vec<&'a str>, hmm_context: &mut HmmContext) {
    let str_len = sentence.len();
    viterbi(sentence, hmm_context);
    let mut begin = 0;
    let mut next_byte_offset = 0;
    let mut i = 0;

    let mut curr = sentence.char_indices().map(|x| x.0).peekable();
    while let Some(curr_byte_offset) = curr.next() {
        let state = hmm_context.best_path[i];
        match state {
            State::Begin => begin = curr_byte_offset,
            State::End => {
                let byte_start = begin;
                let byte_end = *curr.peek().unwrap_or(&str_len);
                words.push(&sentence[byte_start..byte_end]);
                next_byte_offset = byte_end;
            }
            State::Single => {
                let byte_start = curr_byte_offset;
                let byte_end = *curr.peek().unwrap_or(&str_len);
                words.push(&sentence[byte_start..byte_end]);
                next_byte_offset = byte_end;
            }
            State::Middle => { /* do nothing */ }
        }

        i += 1;
    }

    if next_byte_offset < str_len {
        let byte_start = next_byte_offset;
        words.push(&sentence[byte_start..]);
    }

    hmm_context.best_path.clear();
}

#[allow(non_snake_case)]
pub(crate) fn cut_with_allocated_memory<'a>(sentence: &'a str, words: &mut Vec<&'a str>, hmm_context: &mut HmmContext) {
    let splitter = SplitMatches::new(&RE_HAN, sentence);
    for state in splitter {
        let block = state.into_str();
        if block.is_empty() {
            continue;
        }
        if RE_HAN.is_match(block) {
            if block.chars().count() > 1 {
                cut_internal(block, words, hmm_context);
            } else {
                words.push(block);
            }
        } else {
            let skip_splitter = SplitMatches::new(&RE_SKIP, block);
            for skip_state in skip_splitter {
                let x = skip_state.into_str();
                if x.is_empty() {
                    continue;
                }
                words.push(x);
            }
        }
    }
}

#[allow(non_snake_case)]
pub fn cut<'a>(sentence: &'a str, words: &mut Vec<&'a str>) {
    let mut hmm_context = HmmContext::new(sentence.chars().count());

    cut_with_allocated_memory(sentence, words, &mut hmm_context)
}

#[cfg(test)]
mod tests {
    use super::{cut, viterbi, HmmContext};

    #[test]
    #[allow(non_snake_case)]
    fn test_viterbi() {
        use super::State::*;

        let sentence = "小明硕士毕业于中国科学院计算所";

        let mut hmm_context = HmmContext::new(sentence.chars().count());
        viterbi(sentence, &mut hmm_context);
        assert_eq!(
            hmm_context.best_path,
            vec![Begin, End, Begin, End, Begin, Middle, End, Begin, End, Begin, Middle, End, Begin, End, Single]
        );
    }

    #[test]
    fn test_hmm_cut() {
        let sentence = "小明硕士毕业于中国科学院计算所";
        let mut words = Vec::with_capacity(sentence.chars().count() / 2);
        cut(sentence, &mut words);
        assert_eq!(words, vec!["小明", "硕士", "毕业于", "中国", "科学院", "计算", "所"]);
    }
}
