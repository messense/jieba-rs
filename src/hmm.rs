use lazy_static::lazy_static;
use once_cell::sync::OnceCell;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::SplitMatches;

lazy_static! {
    static ref RE_HAN: Regex = Regex::new(r"([\u{4E00}-\u{9FD5}]+)").unwrap();
    static ref RE_SKIP: Regex = Regex::new(r"([a-zA-Z0-9]+(?:.\d+)?%?)").unwrap();
}

pub type StatusSet = [f64; 4];

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub enum Status {
    B = 0,
    E = 1,
    M = 2,
    S = 3,
}

pub struct HmmModel {
    pub initial_probs: StatusSet,
    pub trans_probs: [StatusSet; 4],
    pub emit_probs: [HashMap<String, f64>; 4],
}

static CUSTOM_HMM_MODEL: OnceCell<HmmModel> = OnceCell::new();

pub fn get_custom_hmm_model() -> Option<&'static HmmModel> {
    CUSTOM_HMM_MODEL.get()
}

pub fn init_custom_hmm_model(model: HmmModel) -> Result<(), HmmModel> {
    CUSTOM_HMM_MODEL.set(model)
}

static PREV_STATUS: [[Status; 2]; 4] = [
    [Status::E, Status::S], // B
    [Status::B, Status::M], // E
    [Status::M, Status::B], // M
    [Status::S, Status::E], // S
];

const MIN_FLOAT: f64 = -3.14e100;

#[cfg(feature = "default-hmm-model")]
mod default_hmm {
    use super::*;

    include!(concat!(env!("OUT_DIR"), "/hmm_prob.rs"));
}

#[inline]
fn get_initial_prob(index: usize) -> f64 {
    debug_assert!(index < 4);
    if let Some(model) = get_custom_hmm_model() {
        model.initial_probs[index]
    } else {
        #[cfg(feature = "default-hmm-model")]
        {
            default_hmm::INITIAL_PROBS[index]
        }
        #[cfg(not(feature = "default-hmm-model"))]
        {
            debug_assert!(
                true,
                "No default hmm model is provided, please use `set_custom_hmm_model` to set a custom model."
            );
            MIN_FLOAT
        }
    }
}

#[inline]
fn get_emit_prob(index: usize, word: &str) -> f64 {
    debug_assert!(index < 4);
    if let Some(model) = get_custom_hmm_model() {
        model.emit_probs[index].get(word).cloned().unwrap_or(MIN_FLOAT)
    } else {
        #[cfg(feature = "default-hmm-model")]
        {
            default_hmm::EMIT_PROBS[index].get(word).cloned().unwrap_or(MIN_FLOAT)
        }
        #[cfg(not(feature = "default-hmm-model"))]
        {
            debug_assert!(
                true,
                "No default hmm model is provided, please use `set_custom_hmm_model` to set a custom model."
            );
            MIN_FLOAT
        }
    }
}

#[inline]
fn get_trans_prob(from_index: usize, to_index: usize) -> f64 {
    debug_assert!(from_index < 4);
    debug_assert!(to_index < 4);
    if let Some(model) = get_custom_hmm_model() {
        model.trans_probs[from_index]
            .get(to_index)
            .cloned()
            .unwrap_or(MIN_FLOAT)
    } else {
        #[cfg(feature = "default-hmm-model")]
        {
            default_hmm::TRANS_PROBS[from_index]
                .get(to_index)
                .cloned()
                .unwrap_or(MIN_FLOAT)
        }
        #[cfg(not(feature = "default-hmm-model"))]
        {
            debug_assert!(
                true,
                "No default hmm model is provided, please use `set_custom_hmm_model` to set a custom model."
            );
            MIN_FLOAT
        }
    }
}

#[allow(non_snake_case)]
fn viterbi(sentence: &str, V: &mut Vec<f64>, prev: &mut Vec<Option<Status>>, best_path: &mut Vec<Status>) {
    let str_len = sentence.len();
    let states = [Status::B, Status::M, Status::E, Status::S];
    #[allow(non_snake_case)]
    let R = states.len();
    let C = sentence.chars().count();
    assert!(C > 1);

    if prev.len() < R * C {
        prev.resize(R * C, None);
    }

    if V.len() < R * C {
        V.resize(R * C, 0.0);
    }

    if best_path.len() < C {
        best_path.resize(C, Status::B);
    }

    let mut curr = sentence.char_indices().map(|x| x.0).peekable();
    let x1 = curr.next().unwrap();
    let x2 = *curr.peek().unwrap();
    for y in &states {
        let first_word = &sentence[x1..x2];
        let index = *y as usize;
        let prob = get_initial_prob(index) + get_emit_prob(index, first_word);
        V[*y as usize] = prob;
    }

    let mut t = 1;
    while let Some(byte_start) = curr.next() {
        for y in &states {
            let byte_end = *curr.peek().unwrap_or(&str_len);
            let word = &sentence[byte_start..byte_end];
            let em_prob = get_emit_prob(*y as usize, word);
            let (prob, state) = PREV_STATUS[*y as usize]
                .iter()
                .map(|y0| {
                    (
                        V[(t - 1) * R + (*y0 as usize)] + get_trans_prob(*y0 as usize, *y as usize) + em_prob,
                        *y0,
                    )
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap();
            let idx = (t * R) + (*y as usize);
            V[idx] = prob;
            prev[idx] = Some(state);
        }

        t += 1;
    }

    let (_prob, state) = [Status::E, Status::S]
        .iter()
        .map(|y| (V[(C - 1) * R + (*y as usize)], y))
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();

    let mut t = C - 1;
    let mut curr = *state;

    best_path[t] = *state;
    while let Some(p) = prev[t * R + (curr as usize)] {
        assert!(t > 0);
        best_path[t - 1] = p;
        curr = p;
        t -= 1;
    }

    prev.clear();
    V.clear();
}

#[allow(non_snake_case)]
pub fn cut_internal<'a>(
    sentence: &'a str,
    words: &mut Vec<&'a str>,
    V: &mut Vec<f64>,
    prev: &mut Vec<Option<Status>>,
    path: &mut Vec<Status>,
) {
    let str_len = sentence.len();
    viterbi(sentence, V, prev, path);
    let mut begin = 0;
    let mut next_byte_offset = 0;
    let mut i = 0;

    let mut curr = sentence.char_indices().map(|x| x.0).peekable();
    while let Some(curr_byte_offset) = curr.next() {
        let state = path[i];
        match state {
            Status::B => begin = curr_byte_offset,
            Status::E => {
                let byte_start = begin;
                let byte_end = *curr.peek().unwrap_or(&str_len);
                words.push(&sentence[byte_start..byte_end]);
                next_byte_offset = byte_end;
            }
            Status::S => {
                let byte_start = curr_byte_offset;
                let byte_end = *curr.peek().unwrap_or(&str_len);
                words.push(&sentence[byte_start..byte_end]);
                next_byte_offset = byte_end;
            }
            Status::M => { /* do nothing */ }
        }

        i += 1;
    }

    if next_byte_offset < str_len {
        let byte_start = next_byte_offset;
        words.push(&sentence[byte_start..]);
    }

    path.clear();
}

#[allow(non_snake_case)]
pub(crate) fn cut_with_allocated_memory<'a>(
    sentence: &'a str,
    words: &mut Vec<&'a str>,
    V: &mut Vec<f64>,
    prev: &mut Vec<Option<Status>>,
    path: &mut Vec<Status>,
) {
    let splitter = SplitMatches::new(&RE_HAN, sentence);
    for state in splitter {
        let block = state.into_str();
        if block.is_empty() {
            continue;
        }
        if RE_HAN.is_match(block) {
            if block.chars().count() > 1 {
                cut_internal(block, words, V, prev, path);
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
    let R = 4;
    let C = sentence.chars().count();
    let mut V = vec![0.0; R * C];
    let mut prev: Vec<Option<Status>> = vec![None; R * C];
    let mut path: Vec<Status> = vec![Status::B; C];

    cut_with_allocated_memory(sentence, words, &mut V, &mut prev, &mut path);
}

#[cfg(all(test, not(feature = "default-hmm-model")))]
pub fn test_init_custom_hmm_model() {
    use std::convert::TryInto;

    mod hmm_prob {
        use super::*;
        include!(concat!(env!("OUT_DIR"), "/hmm_prob.rs"));
    }

    if get_custom_hmm_model().is_none() {
        let initial_probs = hmm_prob::INITIAL_PROBS;
        let trans_probs = hmm_prob::TRANS_PROBS;
        let emit_probs: [HashMap<_, _>; 4] = {
            let mut emit_probs = Vec::new();
            for prob in hmm_prob::EMIT_PROBS {
                let mut probs = HashMap::new();
                for (k, v) in prob {
                    probs.insert(k.to_string(), *v);
                }
                emit_probs.push(probs);
            }
            emit_probs.try_into().unwrap()
        };
        let _ = init_custom_hmm_model(HmmModel {
            initial_probs,
            trans_probs,
            emit_probs,
        });
    }
}

#[cfg(all(test, feature = "default-hmm-model"))]
pub fn test_init_custom_hmm_model() {
    // nothing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn test_viterbi() {
        use super::Status::*;

        test_init_custom_hmm_model();
        let sentence = "小明硕士毕业于中国科学院计算所";

        let R = 4;
        let C = sentence.chars().count();
        let mut V = vec![0.0; R * C];
        let mut prev: Vec<Option<Status>> = vec![None; R * C];
        let mut path: Vec<Status> = vec![Status::B; C];
        viterbi(sentence, &mut V, &mut prev, &mut path);
        assert_eq!(path, vec![B, E, B, E, B, M, E, B, E, B, M, E, B, E, S]);
    }

    #[test]
    fn test_hmm_cut() {
        test_init_custom_hmm_model();
        let sentence = "小明硕士毕业于中国科学院计算所";
        let mut words = Vec::with_capacity(sentence.chars().count() / 2);
        cut(sentence, &mut words);
        assert_eq!(words, vec!["小明", "硕士", "毕业于", "中国", "科学院", "计算", "所"]);
    }
}
