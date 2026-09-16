use std::io::BufRead;

use crate::FxHashMap;
use crate::errors::Error;
use jieba_macros::generate_hmm_data;

/// HMM-specific CJK range `[\u{4E00}-\u{9FD5}]`, the range the generated
/// emission table covers.
#[inline]
fn is_hmm_han(c: char) -> bool {
    matches!(c, HMM_HAN_MIN..=HMM_HAN_MAX)
}

/// Characters that join two alphanumeric runs into one token.
///
/// These are the separators that actually occur inside identifiers — version
/// numbers, part numbers, ticket ids, code symbols, hostnames. They are a
/// subset of the characters [`is_han_default`] already keeps inside a block.
#[inline]
pub(crate) fn is_skip_connector(b: u8) -> bool {
    matches!(b, b'.' | b'_' | b'-')
}

/// [`is_skip_connector`] over a `char`, for callers that iterate characters.
#[inline]
pub(crate) fn is_skip_connector_char(c: char) -> bool {
    matches!(c, '.' | '_' | '-')
}

/// Length of the alphanumeric token starting at `start`, which must be an ASCII
/// alphanumeric byte.
///
/// An alphanumeric run, then any number of connector-plus-run groups, then an
/// optional percent sign. A connector only continues the token when another
/// alphanumeric follows it, so a trailing separator is left out (`abc-` is
/// `abc`). Every character this can match is ASCII, so the input is walked as
/// bytes: a UTF-8 continuation byte is never alphanumeric, so every boundary
/// returned falls on a character boundary.
#[inline]
fn skip_match_len(bytes: &[u8], start: usize) -> usize {
    debug_assert!(bytes[start].is_ascii_alphanumeric());
    let mut end = start;
    loop {
        while end < bytes.len() && bytes[end].is_ascii_alphanumeric() {
            end += 1;
        }
        // Look past one connector; only commit to it if a run follows.
        if end + 1 < bytes.len() && is_skip_connector(bytes[end]) && bytes[end + 1].is_ascii_alphanumeric() {
            end += 1;
        } else {
            break;
        }
    }
    if end < bytes.len() && bytes[end] == b'%' {
        end += 1;
    }
    end - start
}

/// Splits a non-CJK block into the same alternating sequence of unmatched gaps
/// and matched runs that `Regex::find_iter` produced for `re_skip`, so the HMM
/// keeps seeing identical blocks.
struct HmmSkipSplitter<'t> {
    text: &'t str,
    pos: usize,
    pending: Option<&'t str>,
}

impl<'t> HmmSkipSplitter<'t> {
    #[inline]
    fn new(text: &'t str) -> Self {
        HmmSkipSplitter {
            text,
            pos: 0,
            pending: None,
        }
    }
}

impl<'t> Iterator for HmmSkipSplitter<'t> {
    type Item = &'t str;

    fn next(&mut self) -> Option<&'t str> {
        // A gap and the match that ended it are found together; the match is
        // held back so the gap is yielded first, as the regex version did.
        if let Some(matched) = self.pending.take() {
            return Some(matched);
        }
        if self.pos >= self.text.len() {
            return None;
        }
        let bytes = self.text.as_bytes();
        let gap_start = self.pos;
        // A match can only start on an ASCII alphanumeric byte, so skip to the
        // next one in a tight loop rather than trying to match at every offset.
        let mut cursor = self.pos;
        while cursor < bytes.len() && !bytes[cursor].is_ascii_alphanumeric() {
            cursor += 1;
        }
        if cursor == bytes.len() {
            self.pos = bytes.len();
            return Some(&self.text[gap_start..]);
        }
        let len = skip_match_len(bytes, cursor);
        let matched = &self.text[cursor..cursor + len];
        self.pos = cursor + len;
        if cursor == gap_start {
            return Some(matched);
        }
        self.pending = Some(matched);
        Some(&self.text[gap_start..cursor])
    }
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
//
// Within each row the states are in ascending index order. `viterbi` lets
// the second candidate win an equal score, so this order is what makes a tie
// go to the state with the larger index, as the original comparison of
// `(score, State)` pairs did.
static ALLOWED_PREV_STATUS: [[State; 2]; NUM_STATES] = [
    // Can preceed State::Begin
    [State::End, State::Single],
    // Can preceed State::End
    [State::Begin, State::Middle],
    // Can preceed State::Middle
    [State::Begin, State::Middle],
    // Can preceed State::Single
    [State::End, State::Single],
];

generate_hmm_data!();

const MIN_FLOAT: f64 = -3.14e100;

pub(crate) trait HmmParams {
    fn initial_prob(&self, state: usize) -> f64;
    fn trans_prob(&self, from: usize, to: usize) -> f64;
    fn emit_probs(&self, ch: char) -> [f64; NUM_STATES];
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
        TRANS_PROBS[from][to]
    }

    #[inline]
    fn emit_probs(&self, ch: char) -> [f64; NUM_STATES] {
        // Two dependent loads from static tables; no hashing.
        let offset = (ch as u32).wrapping_sub(EMIT_MIN_CHAR) as usize;
        match EMIT_INDEX.get(offset) {
            Some(&row) if row != EMIT_NONE => EMIT_PROBS[row as usize],
            _ => [MIN_FLOAT; NUM_STATES],
        }
    }
}

#[derive(Default)]
pub(crate) struct HmmContext {
    /// Chosen predecessor of each state at each step but the first,
    /// `NUM_STATES` per character.
    prev: Vec<State>,
    best_path: Vec<State>,
}

impl HmmContext {
    pub(crate) fn release_if_huge(&mut self) {
        crate::release_if_huge(&mut self.prev, crate::SCRATCH_BUDGET);
        crate::release_if_huge(&mut self.best_path, crate::SCRATCH_BUDGET);
    }
}

/// Label each of `chars`, the already decoded characters of a run, leaving
/// the best state sequence in `hmm_context.best_path`.
#[allow(non_snake_case)]
fn viterbi(chars: &[(u32, char)], params: &impl HmmParams, hmm_context: &mut HmmContext) {
    const R: usize = NUM_STATES;

    let C = chars.len();
    assert!(C > 1);

    // Every entry the traceback reads is written below first, so stale
    // contents from a previous run are never observed.
    if hmm_context.prev.len() < R * C {
        hmm_context.prev.resize(R * C, State::Begin);
    }

    if hmm_context.best_path.len() < C {
        hmm_context.best_path.resize(C, State::Begin);
    }

    let prev = &mut hmm_context.prev;

    // The recurrence only reads the previous step's scores, so they roll
    // through two small arrays; only the predecessor choices are kept.
    let emit_probs = params.emit_probs(chars[0].1);
    let mut prev_v = [0.0; R];
    for y in 0..R {
        prev_v[y] = params.initial_prob(y) + emit_probs[y];
    }

    for t in 1..C {
        let emit_probs = params.emit_probs(chars[t].1);
        let mut curr_v = [0.0; R];
        for y in 0..R {
            let em_prob = emit_probs[y];
            let [y0, y1] = ALLOWED_PREV_STATUS[y];
            let prob0 = prev_v[y0 as usize] + params.trans_prob(y0 as usize, y) + em_prob;
            let prob1 = prev_v[y1 as usize] + params.trans_prob(y1 as usize, y) + em_prob;
            // On a tie the second candidate wins; `ALLOWED_PREV_STATUS` lists
            // it as the state with the larger index.
            let (prob, state) = if prob0 > prob1 { (prob0, y0) } else { (prob1, y1) };
            curr_v[y] = prob;
            prev[t * R + y] = state;
        }
        prev_v = curr_v;
    }

    let mut curr = if prev_v[State::End as usize] > prev_v[State::Single as usize] {
        State::End
    } else {
        State::Single
    };

    hmm_context.best_path[C - 1] = curr;
    for t in (1..C).rev() {
        curr = prev[t * R + (curr as usize)];
        hmm_context.best_path[t - 1] = curr;
    }
    hmm_context.best_path.truncate(C);
}

/// Cut the run `chars` of `text`, which ends at byte `run_end`, along the
/// best state sequence.
#[allow(non_snake_case)]
fn cut_internal<'a>(
    text: &'a str,
    chars: &[(u32, char)],
    run_end: usize,
    words: &mut impl FnMut(&'a str),
    params: &impl HmmParams,
    hmm_context: &mut HmmContext,
) {
    viterbi(chars, params, hmm_context);
    let mut begin = chars[0].0 as usize;
    let mut next_byte_offset = begin;

    for (i, &(curr_byte_offset, _)) in chars.iter().enumerate() {
        let curr_byte_offset = curr_byte_offset as usize;
        let state = hmm_context.best_path[i];
        match state {
            State::Begin => begin = curr_byte_offset,
            State::End => {
                let byte_start = begin;
                let byte_end = chars.get(i + 1).map_or(run_end, |&(offset, _)| offset as usize);
                words(&text[byte_start..byte_end]);
                next_byte_offset = byte_end;
            }
            State::Single => {
                let byte_start = curr_byte_offset;
                let byte_end = chars.get(i + 1).map_or(run_end, |&(offset, _)| offset as usize);
                words(&text[byte_start..byte_end]);
                next_byte_offset = byte_end;
            }
            State::Middle => { /* do nothing */ }
        }
    }

    if next_byte_offset < run_end {
        let byte_start = next_byte_offset;
        words(&text[byte_start..run_end]);
    }
}

/// Cut the characters `chars` of `text` (byte offsets into `text` and
/// values, already decoded by the caller) with the HMM: runs of the
/// characters the model covers go through Viterbi, anything else is split
/// like `RE_SKIP` did.
#[allow(non_snake_case)]
pub(crate) fn cut_with_allocated_memory<'a>(
    text: &'a str,
    chars: &[(u32, char)],
    words: &mut impl FnMut(&'a str),
    params: &impl HmmParams,
    hmm_context: &mut HmmContext,
) {
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let matched = is_hmm_han(chars[i].1);
        let mut j = i + 1;
        while j < n && is_hmm_han(chars[j].1) == matched {
            j += 1;
        }
        let start = chars[i].0 as usize;
        let end = chars.get(j).map_or_else(
            || chars[j - 1].0 as usize + chars[j - 1].1.len_utf8(),
            |&(offset, _)| offset as usize,
        );
        if matched {
            if j - i > 1 {
                cut_internal(text, &chars[i..j], end, words, params, hmm_context);
            } else {
                words(&text[start..end]);
            }
        } else {
            for x in HmmSkipSplitter::new(&text[start..end]) {
                if x.is_empty() {
                    continue;
                }
                words(x);
            }
        }
        i = j;
    }
}

/// A runtime-loadable HMM model for custom segmentation.
///
/// This allows loading HMM parameters trained with `scripts/train_hmm.py`
/// instead of using the compile-time embedded model.
#[derive(Debug, Clone)]
pub struct HmmModel {
    initial_probs: [f64; NUM_STATES],
    trans_probs: [[f64; NUM_STATES]; NUM_STATES],
    emit_probs: FxHashMap<char, [f64; NUM_STATES]>,
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
    fn emit_probs(&self, ch: char) -> [f64; NUM_STATES] {
        self.emit_probs.get(&ch).copied().unwrap_or([MIN_FLOAT; NUM_STATES])
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
        let mut emit_probs: FxHashMap<char, [f64; NUM_STATES]> = FxHashMap::default();
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
                let mut chars = ch.chars();
                let ch = chars
                    .next()
                    .ok_or_else(|| Error::InvalidHmmModel(format!("invalid emit char: `{pair}`")))?;
                if chars.next().is_some() {
                    return Err(Error::InvalidHmmModel(format!("emit key must be one char: `{pair}`")));
                }
                let prob: f64 = pair[colon_pos + 1..]
                    .parse()
                    .map_err(|e| Error::InvalidHmmModel(format!("invalid emit prob: {e}")))?;
                emit_probs.entry(ch).or_insert([MIN_FLOAT; NUM_STATES])[i] = prob;
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
    use expect_test::expect;

    use super::{BuiltinHmm, HmmContext, cut_with_allocated_memory, viterbi};

    fn decode(sentence: &str) -> Vec<(u32, char)> {
        sentence.char_indices().map(|(i, c)| (i as u32, c)).collect()
    }

    fn cut<'a>(sentence: &'a str, words: &mut Vec<&'a str>) {
        let mut hmm_context = HmmContext::default();
        let chars = decode(sentence);
        cut_with_allocated_memory(
            sentence,
            &chars,
            &mut |word| words.push(word),
            &BuiltinHmm,
            &mut hmm_context,
        )
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_viterbi() {
        let sentence = "小明硕士毕业于中国科学院计算所";

        let mut hmm_context = HmmContext::default();
        viterbi(&decode(sentence), &BuiltinHmm, &mut hmm_context);
        expect![[
            r#"[Begin, End, Begin, End, Begin, Middle, End, Begin, End, Begin, Middle, End, Begin, End, Single]"#
        ]]
        .assert_eq(&format!("{:?}", hmm_context.best_path));
    }

    /// Characters without emission data score every state the same, so the
    /// path is decided by tie-breaking alone: an equal score goes to the
    /// state with the larger index (`Single` over `End`, `Middle` over
    /// `Begin`), which keeps such runs as single characters.
    #[test]
    fn test_hmm_cut_ties_go_to_the_higher_state() {
        let mut words = Vec::new();
        cut("龘龘龘", &mut words);
        expect![[r#"["龘", "龘", "龘"]"#]].assert_eq(&format!("{:?}", words));
        words.clear();
        cut("龘龘龘龘龘", &mut words);
        expect![[r#"["龘", "龘", "龘", "龘", "龘"]"#]].assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_hmm_cut() {
        let sentence = "小明硕士毕业于中国科学院计算所";
        let mut words = Vec::with_capacity(sentence.chars().count() / 2);
        cut(sentence, &mut words);
        expect![[r#"["小明", "硕士", "毕业于", "中国", "科学院", "计算", "所"]"#]].assert_eq(&format!("{:?}", words));
    }

    /// An alphanumeric run joined by `.`, `_` or `-` is one token.
    ///
    /// The separator may repeat, and either side of it may be letters or
    /// digits, so identifiers survive whatever shape they happen to have.
    /// Before this, the expression could carry at most one separator and
    /// required digits after it, which cut identifiers apart in a way that
    /// depended on the characters they happened to contain: `1.2.3` split at
    /// the second dot, `G260911-0711` survived while `ISU-CNS24093` did not.
    ///
    /// A trailing separator is not part of the token, and decimals and
    /// percentages — what this expression was originally for — are unaffected.
    #[test]
    fn test_hmm_cut_keeps_connected_alphanumerics_together() {
        let mut got = Vec::new();
        for sentence in [
            "1.0",
            "1.2.3",
            "WES-5.4.5",
            "G260911-0711",
            "ISU-CNS24093",
            "E4850B-I",
            "OPENSSL_1_1_1",
            "well-known",
            "3.14",
            "50%",
            "abc-",
            "-abc",
        ] {
            let mut words = Vec::new();
            cut(sentence, &mut words);
            got.push(format!("{} -> {:?}", sentence, words));
        }
        expect![[r#"
            1.0 -> ["1.0"]
            1.2.3 -> ["1.2.3"]
            WES-5.4.5 -> ["WES-5.4.5"]
            G260911-0711 -> ["G260911-0711"]
            ISU-CNS24093 -> ["ISU-CNS24093"]
            E4850B-I -> ["E4850B-I"]
            OPENSSL_1_1_1 -> ["OPENSSL_1_1_1"]
            well-known -> ["well-known"]
            3.14 -> ["3.14"]
            50% -> ["50%"]
            abc- -> ["abc", "-"]
            -abc -> ["-", "abc"]"#]]
        .assert_eq(&got.join("\n"));
    }
}
