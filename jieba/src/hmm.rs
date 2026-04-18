use std::cmp::Ordering;
use std::io::BufRead;

use regex::Regex;

use crate::FxHashMap;
use crate::SplitMatches;
use crate::errors::Error;
use jieba_macros::generate_hmm_data;

thread_local! {
    static RE_HAN: Regex = Regex::new(r"([\u{4E00}-\u{9FD5}]+)").unwrap();
    static RE_SKIP: Regex = Regex::new(r"([a-zA-Z0-9]+(?:.\d+)?%?)").unwrap();
}

pub const NUM_STATES: usize = 4;

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
/// reassign the index values of each state at the top but `jieba-macros`
/// currently ignores the mapping. Do not reassign these indices without
/// verifying how it interacts with `jieba-macros`.  These indices must also
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

generate_hmm_data!();

const MIN_FLOAT: f64 = -3.14e100;

pub(crate) trait HmmParams {
    fn initial_prob(&self, state: usize) -> f64;
    fn trans_prob(&self, from: usize, to: usize) -> f64;
    fn emit_prob(&self, state: usize, word: &str) -> f64;
}

/// The compile-time embedded HMM parameters.
pub(crate) struct BuiltinHmm;

impl HmmParams for BuiltinHmm {
    #[inline]
    fn initial_prob(&self, state: usize) -> f64 {
        INITIAL_PROBS[state]
    }

    #[inline]
    fn trans_prob(&self, from: usize, to: usize) -> f64 {
        TRANS_PROBS[from].get(to).cloned().unwrap_or(MIN_FLOAT)
    }

    #[inline]
    fn emit_prob(&self, state: usize, word: &str) -> f64 {
        EMIT_PROBS[state].get(word).cloned().unwrap_or(MIN_FLOAT)
    }
}

#[derive(Default)]
pub(crate) struct HmmContext {
    v: Vec<f64>,
    prev: Vec<Option<State>>,
    best_path: Vec<State>,
}

#[allow(non_snake_case)]
fn viterbi(sentence: &str, params: &impl HmmParams, hmm_context: &mut HmmContext) {
    let str_len = sentence.len();
    let states = [State::Begin, State::Middle, State::End, State::Single];
    #[allow(non_snake_case)]
    let R = states.len();

    // Collect char byte offsets once, derive C from the length
    let char_offsets: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
    let C = char_offsets.len();
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

    let mut curr = char_offsets.iter().copied().peekable();
    let x1 = curr.next().unwrap();
    let x2 = *curr.peek().unwrap();
    for y in &states {
        let first_word = &sentence[x1..x2];
        let prob = params.initial_prob(*y as usize) + params.emit_prob(*y as usize, first_word);
        hmm_context.v[*y as usize] = prob;
    }

    let mut t = 1;
    while let Some(byte_start) = curr.next() {
        for y in &states {
            let byte_end = *curr.peek().unwrap_or(&str_len);
            let word = &sentence[byte_start..byte_end];
            let em_prob = params.emit_prob(*y as usize, word);
            let (prob, state) = ALLOWED_PREV_STATUS[*y as usize]
                .iter()
                .map(|y0| {
                    (
                        hmm_context.v[(t - 1) * R + (*y0 as usize)]
                            + params.trans_prob(*y0 as usize, *y as usize)
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
fn cut_internal<'a>(
    sentence: &'a str,
    words: &mut Vec<&'a str>,
    params: &impl HmmParams,
    hmm_context: &mut HmmContext,
) {
    let str_len = sentence.len();
    viterbi(sentence, params, hmm_context);
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
pub(crate) fn cut_with_allocated_memory<'a>(
    sentence: &'a str,
    words: &mut Vec<&'a str>,
    params: &impl HmmParams,
    hmm_context: &mut HmmContext,
) {
    RE_HAN.with(|re_han| {
        RE_SKIP.with(|re_skip| {
            let splitter = SplitMatches::new(re_han, sentence);
            for state in splitter {
                let block = state.as_str();
                if block.is_empty() {
                    continue;
                }
                if state.is_matched() {
                    if block.chars().nth(1).is_some() {
                        cut_internal(block, words, params, hmm_context);
                    } else {
                        words.push(block);
                    }
                } else {
                    let skip_splitter = SplitMatches::new(re_skip, block);
                    for skip_state in skip_splitter {
                        let x = skip_state.as_str();
                        if x.is_empty() {
                            continue;
                        }
                        words.push(x);
                    }
                }
            }
        })
    })
}

/// A runtime-loadable HMM model for custom segmentation.
///
/// This allows loading HMM parameters trained with `scripts/train_hmm.py`
/// instead of using the compile-time embedded model.
#[derive(Debug, Clone)]
pub struct HmmModel {
    initial_probs: [f64; NUM_STATES],
    trans_probs: [[f64; NUM_STATES]; NUM_STATES],
    emit_probs: [FxHashMap<Box<str>, f64>; NUM_STATES],
}

impl HmmParams for HmmModel {
    #[inline]
    fn initial_prob(&self, state: usize) -> f64 {
        self.initial_probs[state]
    }

    #[inline]
    fn trans_prob(&self, from: usize, to: usize) -> f64 {
        self.trans_probs[from][to]
    }

    #[inline]
    fn emit_prob(&self, state: usize, word: &str) -> f64 {
        self.emit_probs[state].get(word).copied().unwrap_or(MIN_FLOAT)
    }
}

impl HmmModel {
    /// Load an HMM model from a reader in the `hmm.model` file format.
    ///
    /// The format is compatible with the output of `scripts/train_hmm.py`.
    pub fn load<R: BufRead>(reader: &mut R) -> Result<Self, Error> {
        let mut data_lines = Vec::new();
        let mut buf = String::new();
        while reader.read_line(&mut buf)? > 0 {
            {
                let line = buf.trim();
                if !line.is_empty() && !line.starts_with('#') {
                    data_lines.push(line.to_string());
                }
            }
            buf.clear();
        }

        // Line 0: start probs (4 values)
        if data_lines.len() < 9 {
            return Err(Error::InvalidHmmModel(format!(
                "expected at least 9 data lines, got {}",
                data_lines.len()
            )));
        }

        let initial_probs = Self::parse_prob_line(&data_lines[0], "initial")?;

        // Lines 1-4: transition matrix
        let mut trans_probs = [[0.0f64; NUM_STATES]; NUM_STATES];
        for i in 0..NUM_STATES {
            let vals = Self::parse_prob_line(&data_lines[1 + i], "transition")?;
            trans_probs[i] = vals;
        }

        // Lines 5-8: emission probs (comma-separated char:prob pairs)
        let mut emit_probs: [FxHashMap<Box<str>, f64>; NUM_STATES] = [
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        ];
        for i in 0..NUM_STATES {
            for pair in data_lines[5 + i].split(',') {
                let pair = pair.trim();
                if pair.is_empty() {
                    continue;
                }
                let colon_pos = pair
                    .rfind(':')
                    .ok_or_else(|| Error::InvalidHmmModel(format!("invalid emit pair (missing ':'): `{pair}`")))?;
                let ch = &pair[..colon_pos];
                let prob: f64 = pair[colon_pos + 1..]
                    .parse()
                    .map_err(|e| Error::InvalidHmmModel(format!("invalid emit prob: {e}")))?;
                emit_probs[i].insert(ch.into(), prob);
            }
        }

        Ok(HmmModel {
            initial_probs,
            trans_probs,
            emit_probs,
        })
    }

    fn parse_prob_line(line: &str, context: &str) -> Result<[f64; NUM_STATES], Error> {
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|e| Error::InvalidHmmModel(format!("invalid {context} prob `{v}`: {e}")))
            })
            .collect::<Result<_, _>>()?;
        if vals.len() != NUM_STATES {
            return Err(Error::InvalidHmmModel(format!(
                "expected {NUM_STATES} {context} values, got {}",
                vals.len()
            )));
        }
        Ok([vals[0], vals[1], vals[2], vals[3]])
    }
}

pub(crate) fn builtin_hmm() -> BuiltinHmm {
    BuiltinHmm
}

#[cfg(test)]
mod tests {
    use super::{BuiltinHmm, HmmContext, cut_with_allocated_memory, viterbi};

    fn cut<'a>(sentence: &'a str, words: &mut Vec<&'a str>) {
        let mut hmm_context = HmmContext::default();

        cut_with_allocated_memory(sentence, words, &BuiltinHmm, &mut hmm_context)
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_viterbi() {
        use super::State::*;

        let sentence = "小明硕士毕业于中国科学院计算所";

        let mut hmm_context = HmmContext::default();
        viterbi(sentence, &BuiltinHmm, &mut hmm_context);
        assert_eq!(
            hmm_context.best_path,
            vec![
                Begin, End, Begin, End, Begin, Middle, End, Begin, End, Begin, Middle, End, Begin, End, Single
            ]
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
