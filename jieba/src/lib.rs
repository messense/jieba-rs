//! The Jieba Chinese Word Segmentation Implemented in Rust
//!
//! ## Installation
//!
//! Add it to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! jieba-rs = "0.9"
//! ```
//!
//! then you are good to go. If you are using Rust 2015 you have to ``extern crate jieba_rs`` to your crate root as well.
//!
//! ## Example
//!
//! ```rust
//! use jieba_rs::Jieba;
//!
//! let jieba = Jieba::new();
//! let words = jieba.cut("我们中出了一个叛徒", false);
//! let words: Vec<&str> = words.iter().map(|t| t.word).collect();
//! assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
//! ```
//!
//! ```rust
//! # #[cfg(feature = "tfidf")] {
//! use jieba_rs::Jieba;
//! use jieba_rs::{TfIdf, KeywordExtract};
//!
//! fn main() {
//!     let jieba = Jieba::new();
//!     let keyword_extractor = TfIdf::default();
//!     let top_k = keyword_extractor.extract_keywords(
//!         &jieba,
//!         "今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。后天纽约的天气不好，昨天纽约的天气也不好，北京烤鸭真好吃",
//!         3,
//!         vec![],
//!     );
//!     println!("{:?}", top_k);
//! }
//! # }
//! ```
//!
//! ```rust
//! # #[cfg(feature = "textrank")] {
//! use jieba_rs::Jieba;
//! use jieba_rs::{TextRank, KeywordExtract};
//!
//! fn main() {
//!     let jieba = Jieba::new();
//!     let keyword_extractor = TextRank::default();
//!     let top_k = keyword_extractor.extract_keywords(
//!         &jieba,
//!         "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
//!         6,
//!         vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
//!     );
//!     println!("{:?}", top_k);
//! }
//! # }
//! ```
//!
//! ## Enabling Additional Features
//!
//! * `default-dict` feature enables embedded dictionary, this features is enabled by default
//! * `tfidf` feature enables TF-IDF keywords extractor
//! * `textrank` feature enables TextRank keywords extractor
//!
//! ```toml
//! [dependencies]
//! jieba-rs = { version = "0.7", features = ["tfidf", "textrank"] }
//! ```
//!

use std::collections::HashMap;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
use std::collections::HashSet;
use std::fmt;
use std::io::BufRead;

#[cfg(feature = "ts-rs")]
pub use ts_rs;
#[cfg(target_family = "wasm")]
use {
    serde::{Deserialize, Serialize},
    wasm_bindgen::prelude::*,
};

pub(crate) type FxHashMap<K, V> = HashMap<K, V, rustc_hash::FxBuildHasher>;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
pub(crate) type FxHashSet<K> = HashSet<K, rustc_hash::FxBuildHasher>;

pub use crate::errors::Error;
pub use crate::hmm::HmmModel;
#[cfg(feature = "textrank")]
pub use crate::keywords::textrank::TextRank;
#[cfg(feature = "tfidf")]
pub use crate::keywords::tfidf::TfIdf;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
pub use crate::keywords::{DEFAULT_STOP_WORDS, Keyword, KeywordExtract, KeywordExtractConfig};

mod errors;
mod hmm;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
mod keywords;
mod posseg;
mod sparse_dag;
mod trie;

#[cfg(feature = "default-dict")]
include_flate::flate!(static DEFAULT_DICT: str from "src/data/dict.txt" with zstd);

use sparse_dag::{NO_MATCH, StaticSparseDAG};
use trie::CharTrie;

/// One step of the best segmentation of a block.
#[derive(Clone, Copy)]
struct RouteEntry {
    /// Log-probability of the rest of the block from this character.
    prob: f64,
    /// Index of the character after the chosen word.
    end: usize,
    /// The chosen word's dictionary id, `NO_MATCH` for a character not in
    /// the dictionary.
    word_id: i32,
}

/// Where a token sits in the dictionary block it was cut from, so a caller
/// can consult the block's dictionary matches instead of looking words up
/// again.
struct BlockRef<'b> {
    dag: &'b StaticSparseDAG,
    chars: &'b [(u32, char)],
    /// Index in `chars` of the token's first character.
    start: usize,
}

/// Initial capacity for a result vector over `sentence`: generous for a
/// short input, so that segmenting a sentence allocates once, but bounded,
/// so that a long input with few tokens does not hand back a mostly empty
/// allocation.
fn output_capacity(sentence: &str) -> usize {
    const MAX_INITIAL_CAPACITY: usize = 1 << 12;
    (sentence.len() / 2).min(MAX_INITIAL_CAPACITY)
}

/// Per-thread buffers reused across `cut` calls, so segmenting a sentence
/// allocates only its output.
#[derive(Default)]
struct Scratch {
    route: Vec<RouteEntry>,
    dag: StaticSparseDAG,
    /// The characters of the block being segmented: byte offset within the
    /// block and value. Decoded once by the splitter and used by every
    /// later pass.
    chars: Vec<(u32, char)>,
    hmm: hmm::HmmContext,
}

/// How much of a per-thread scratch buffer is worth keeping between calls.
pub(crate) const SCRATCH_BUDGET: usize = 4 << 20;

/// Drop `buf` if it grew past `max_bytes`, so a thread that once handled a
/// huge input does not pin that memory forever.
#[inline(always)]
pub(crate) fn release_if_huge<T>(buf: &mut Vec<T>, max_bytes: usize) {
    if buf.capacity() * std::mem::size_of::<T>() > max_bytes {
        *buf = Vec::new();
    }
}

impl Scratch {
    fn release_if_huge(&mut self) {
        release_if_huge(&mut self.route, SCRATCH_BUDGET);
        release_if_huge(&mut self.chars, SCRATCH_BUDGET);
        self.dag.release_if_huge();
        self.hmm.release_if_huge();
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// The CJK Unified Ideographs ranges other than the main block.
#[inline]
fn is_cjk_extension(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4DBF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B73F}'
        | '\u{2B740}'..='\u{2B81F}'
        | '\u{2B820}'..='\u{2CEAF}'
        | '\u{2CEB0}'..='\u{2EBEF}'
        | '\u{2F800}'..='\u{2FA1F}'
    )
}

/// The main CJK Unified Ideographs block, tested on its own before the
/// extensions: nearly every character of ordinary text falls in it, and a
/// single range check is much cheaper than the vectorised nine-range test
/// the compiler otherwise emits.
#[inline(always)]
fn is_cjk_main(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}')
}

/// Check if a character is in a CJK Unified Ideographs range.
#[inline]
fn is_cjk(c: char) -> bool {
    is_cjk_main(c) || is_cjk_extension(c)
}

/// RE_HAN_DEFAULT character class: CJK + ASCII alphanumeric + `+#&._%\-`
#[inline]
fn is_han_default(c: char) -> bool {
    if is_cjk_main(c) {
        true
    } else if c.is_ascii() {
        c.is_ascii_alphanumeric() || matches!(c, '+' | '#' | '&' | '.' | '_' | '%' | '-')
    } else {
        is_cjk_extension(c)
    }
}

/// RE_HAN_CUT_ALL character class: CJK only
#[inline]
fn is_han_cut_all(c: char) -> bool {
    is_cjk(c)
}

/// RE_SKIP_CUT_ALL: anything not in `[a-zA-Z0-9+#\n]`
#[inline]
fn is_skip_cut_all(c: char) -> bool {
    !c.is_ascii_alphanumeric() && c != '+' && c != '#' && c != '\n'
}

#[inline]
fn char_count(s: &str) -> usize {
    if s.len() >= 16 {
        bytecount::num_chars(s.as_bytes())
    } else {
        s.as_bytes().iter().filter(|&&b| (b as i8) >= -0x40).count()
    }
}

/// Iterator that splits text into matched/unmatched regions by a character classifier.
/// Matched = maximal runs where `classify(c)` is true.
/// Unmatched = everything between matched runs.
pub(crate) struct SplitByCharacterClass<'t, F> {
    text: &'t str,
    pos: usize,
    classify: F,
}

impl<'t, F: Fn(char) -> bool> SplitByCharacterClass<'t, F> {
    #[inline]
    fn new(text: &'t str, classify: F) -> Self {
        SplitByCharacterClass { text, pos: 0, classify }
    }

    /// The next run, with each of its characters recorded in `chars` as
    /// `(byte offset within the run, char)`, so later passes over the run
    /// need not decode it again.
    #[inline]
    fn next_recording(&mut self, chars: &mut Vec<(u32, char)>) -> Option<SplitState<'t>> {
        chars.clear();
        self.advance(|offset, c| chars.push((offset as u32, c)))
    }

    #[inline]
    fn advance(&mut self, mut record: impl FnMut(usize, char)) -> Option<SplitState<'t>> {
        let text = self.text;
        if self.pos >= text.len() {
            return None;
        }

        let start = self.pos;
        let mut iter = text[start..].char_indices();
        let (_, first) = iter.next().unwrap();
        let matched = (self.classify)(first);
        record(0, first);
        let mut end = start + first.len_utf8();
        for (i, c) in iter {
            if (self.classify)(c) != matched {
                break;
            }
            record(i, c);
            end = start + i + c.len_utf8();
        }
        self.pos = end;
        let run = &text[start..end];
        Some(if matched {
            SplitState::Matched(run)
        } else {
            SplitState::Unmatched(run)
        })
    }
}

impl<'t, F: Fn(char) -> bool> Iterator for SplitByCharacterClass<'t, F> {
    type Item = SplitState<'t>;

    fn next(&mut self) -> Option<SplitState<'t>> {
        self.advance(|_, _| {})
    }
}

#[derive(Debug)]
pub(crate) enum SplitState<'t> {
    Unmatched(&'t str),
    Matched(&'t str),
}

#[cfg(test)]
impl<'t> SplitState<'t> {
    fn as_str(&self) -> &'t str {
        match self {
            SplitState::Unmatched(t) => t,
            SplitState::Matched(t) => t,
        }
    }
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizeMode {
    /// Default mode
    Default,
    /// Search mode
    Search,
}

/// A Token
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS))]
#[cfg_attr(target_family = "wasm", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token<'a> {
    /// Word of the token
    pub word: &'a str,
    /// Unicode start position of the token
    pub start: usize,
    /// Unicode end position of the token
    pub end: usize,
    /// Byte start position of the token in the original input
    pub byte_start: usize,
    /// Byte end position of the token in the original input
    pub byte_end: usize,
}

/// A tagged word
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS))]
#[cfg_attr(target_family = "wasm", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag<'a> {
    /// Word
    pub word: &'a str,
    /// Word tag
    pub tag: &'a str,
    /// Unicode start position of the word in the original input
    pub start: usize,
    /// Unicode end position of the word in the original input
    pub end: usize,
    /// Byte start position of the word in the original input
    pub byte_start: usize,
    /// Byte end position of the word in the original input
    pub byte_end: usize,
}

#[derive(Debug, Clone, Copy)]
struct Record {
    freq: usize,
    /// Index into `Jieba::tags`.
    tag: u32,
}

/// Jieba segmentation
#[derive(Clone)]
pub struct Jieba {
    records: Vec<Record>,
    /// `ln(freq)` of each record, kept apart from `records` so the route
    /// calculation touches a dense `f64` array rather than whole records.
    log_freqs: Vec<f64>,
    /// Distinct POS tags; records refer to them by index. A dictionary has
    /// only a few dozen tags, so this replaces one heap string per word.
    tags: Vec<Box<str>>,
    tag_ids: FxHashMap<Box<str>, u32>,
    trie: CharTrie,
    total: usize,
    log_total: f64,
    hmm_model: Option<HmmModel>,
}

impl fmt::Debug for Jieba {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Jieba")
            .field("records_len", &self.records.len())
            .field("total_freq", &self.total)
            .finish()
    }
}

#[cfg(feature = "default-dict")]
impl Default for Jieba {
    fn default() -> Self {
        Jieba::new()
    }
}

impl Jieba {
    /// Create a new instance with empty dict
    pub fn empty() -> Self {
        Jieba {
            records: Vec::new(),
            log_freqs: Vec::new(),
            tags: Vec::new(),
            tag_ids: FxHashMap::default(),
            trie: CharTrie::new(),
            total: 0,
            log_total: 0.0f64.ln(),
            hmm_model: None,
        }
    }

    /// Create a new instance with embed dict
    ///
    /// Requires `default-dict` feature to be enabled.
    #[cfg(feature = "default-dict")]
    pub fn new() -> Self {
        let mut instance = Self::empty();
        instance.load_default_dict();
        instance
    }

    /// Create a new instance with dict
    pub fn with_dict<R: BufRead>(dict: &mut R) -> Result<Self, Error> {
        let mut instance = Self::empty();
        instance.load_dict(dict)?;
        Ok(instance)
    }

    /// Loads the default dictionary into the instance.
    ///
    /// This method reads the default dictionary from a predefined byte slice (`DEFAULT_DICT`)
    /// and loads it into the current instance using the `load_dict` method.
    ///
    /// # Arguments
    ///
    /// * `&mut self` - Mutable reference to the current instance.
    ///
    /// Requires `default-dict` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs::Jieba;
    ///
    /// let mut instance = Jieba::empty();
    /// instance.load_default_dict(); // Loads the default dictionary into the instance
    /// assert!(instance.has_word("我们"), "The word '我们' should be in the dictionary after loading the default dictionary");
    /// ```
    #[cfg(feature = "default-dict")]
    pub fn load_default_dict(&mut self) {
        // The embedded dictionary is a well-formed `word freq tag` list, so it
        // is parsed straight from the string, with no per-line copy or UTF-8
        // re-validation, and never checked for duplicates when it is the
        // first thing loaded.
        let dict: &str = &DEFAULT_DICT;
        let check_duplicates = !self.records.is_empty();
        if !check_duplicates {
            let lines = bytecount::count(dict.as_bytes(), b'\n') + 1;
            self.records.reserve(lines);
            self.log_freqs.reserve(lines);
            self.trie.reserve(lines);
        }
        for line in dict.lines() {
            let mut iter = line.split_ascii_whitespace();
            let Some(word) = iter.next() else { continue };
            let freq = iter
                .next()
                .map_or(0, |x| x.parse::<usize>().expect("invalid frequency in default dict"));
            let tag = iter.next().unwrap_or("");
            self.load_entry(word, freq, tag, check_duplicates);
        }
        self.finish_load();
    }

    /// Set a custom HMM model for segmentation.
    ///
    /// When set, the custom model is used instead of the compile-time embedded model
    /// for HMM-based segmentation of out-of-vocabulary words.
    ///
    /// The model can be trained using `scripts/train_hmm.py`.
    ///
    /// ## Example
    ///
    /// ```no_run
    /// use std::io::BufReader;
    /// use std::fs::File;
    /// use jieba_rs::{Jieba, HmmModel};
    ///
    /// let mut jieba = Jieba::new();
    /// let mut f = BufReader::new(File::open("my_hmm.model").unwrap());
    /// let model = HmmModel::load(&mut f).unwrap();
    /// jieba.set_hmm_model(model);
    /// ```
    pub fn set_hmm_model(&mut self, model: HmmModel) {
        self.hmm_model = Some(model);
    }

    /// Clears all data
    ///
    /// This method performs the following actions:
    /// 1. Clears the `records` list, removing all entries.
    /// 2. Resets the dictionary trie to an empty one.
    /// 3. Sets `total` to 0, resetting the count.
    ///
    /// # Arguments
    ///
    /// * `&mut self` - Mutable reference to the current instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs::Jieba;
    ///
    /// let mut instance = Jieba::new();
    /// assert!(instance.has_word("我们"), "The word '我们' should be in the dictionary after loading the default dictionary");
    /// instance.clear(); // clear all dict data
    /// assert!(!instance.has_word("我们"), "The word '我们' should not be in the dictionary after clearing the dictionary");
    /// ```
    pub fn clear(&mut self) {
        self.records.clear();
        self.log_freqs.clear();
        self.tags.clear();
        self.tag_ids.clear();
        self.trie = CharTrie::new();
        self.total = 0;
        self.update_log_total();
    }

    /// Add word to dict, return `freq`
    ///
    /// `freq`: if `None`, will be given by [suggest_freq](#method.suggest_freq)
    ///
    /// `tag`: if `None`, will be given `""`
    pub fn add_word(&mut self, word: &str, freq: Option<usize>, tag: Option<&str>) -> usize {
        if word.is_empty() {
            return 0;
        }
        let freq = freq.unwrap_or_else(|| self.suggest_freq(word));
        let tag = tag.unwrap_or("");

        match self.trie.get(word) {
            Some(word_id) => {
                let old_freq = self.records[word_id as usize].freq;
                self.set_freq(word_id as usize, freq);

                self.total += freq;
                self.total -= old_freq;
            }
            None => {
                self.push_record(word, freq, tag);
                self.total += freq;
            }
        };
        self.update_log_total();

        freq
    }

    #[inline]
    fn set_freq(&mut self, word_id: usize, freq: usize) {
        self.records[word_id].freq = freq;
        self.log_freqs[word_id] = (freq as f64).ln();
    }

    #[inline]
    fn intern_tag(&mut self, tag: &str) -> u32 {
        if let Some(&id) = self.tag_ids.get(tag) {
            return id;
        }
        let id = self.tags.len() as u32;
        self.tags.push(tag.into());
        self.tag_ids.insert(tag.into(), id);
        id
    }

    /// Append a word the dictionary does not contain yet.
    #[inline]
    fn push_record(&mut self, word: &str, freq: usize, tag: &str) {
        let word_id = self.records.len() as i32;
        let tag = self.intern_tag(tag);
        self.records.push(Record { freq, tag });
        self.log_freqs.push((freq as f64).ln());
        self.trie.insert(word, word_id);
    }

    /// Add one dictionary entry; with `check_duplicates` an existing word only
    /// has its frequency replaced.
    #[inline]
    fn load_entry(&mut self, word: &str, freq: usize, tag: &str, check_duplicates: bool) {
        if check_duplicates {
            match self.trie.get(word) {
                Some(word_id) => self.set_freq(word_id as usize, freq),
                None => self.push_record(word, freq, tag),
            }
        } else {
            self.push_record(word, freq, tag);
        }
    }

    fn finish_load(&mut self) {
        self.total = self.records.iter().map(|n| n.freq).sum();
        self.update_log_total();
    }

    /// Checks if a word exists in the dictionary.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to check.
    ///
    /// # Returns
    ///
    /// * `bool` - Whether the word exists in the dictionary.
    pub fn has_word(&self, word: &str) -> bool {
        self.trie.get(word).is_some()
    }

    /// Loads a dictionary by adding entries to the existing dictionary rather than resetting it.
    ///
    /// This function reads from a `BufRead` source, parsing each line as a dictionary entry. Each entry
    /// is expected to contain a word, its frequency, and optionally a tag.
    ///
    /// # Type Parameters
    ///
    /// * `R`: A type that implements the `BufRead` trait, used for reading lines from the dictionary.
    ///
    /// # Arguments
    ///
    /// * `dict` - A mutable reference to a `BufRead` source containing the dictionary entries.
    ///
    /// # Returns
    ///
    /// * `Result<(), Error>` - Returns `Ok(())` if the dictionary is successfully loaded; otherwise,
    ///   returns an error describing what went wrong.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * There is an issue reading from the provided `BufRead` source.
    /// * A line in the dictionary file contains invalid frequency data (not a valid integer).
    pub fn load_dict<R: BufRead>(&mut self, dict: &mut R) -> Result<(), Error> {
        let mut buf = String::new();
        self.total = 0;

        let mut line_no = 0;
        while dict.read_line(&mut buf)? > 0 {
            {
                line_no += 1;
                let mut iter = buf.split_whitespace();
                if let Some(word) = iter.next() {
                    let freq = iter
                        .next()
                        .map(|x| {
                            x.parse::<usize>().map_err(|e| {
                                Error::InvalidDictEntry(format!(
                                    "line {line_no} `{buf}` frequency {x} is not a valid integer: {e}"
                                ))
                            })
                        })
                        .unwrap_or(Ok(0))?;
                    let tag = iter.next().unwrap_or("");
                    self.load_entry(word, freq, tag, true);
                }
            }
            buf.clear();
        }
        self.finish_load();

        Ok(())
    }

    #[inline]
    fn update_log_total(&mut self) {
        self.log_total = (self.total as f64).ln();
    }

    fn get_word_freq(&self, word: &str, default: usize) -> usize {
        match self.trie.get(word) {
            Some(word_id) => self.records[word_id as usize].freq,
            None => default,
        }
    }

    /// Suggest word frequency to force the characters in a word to be joined or split.
    pub fn suggest_freq(&self, segment: &str) -> usize {
        let logtotal = self.log_total;
        let logfreq = self.cut(segment, false).iter().fold(0f64, |freq, token| {
            freq + (self.get_word_freq(token.word, 1) as f64).ln() - logtotal
        });
        std::cmp::max((logfreq + logtotal).exp() as usize + 1, self.get_word_freq(segment, 1))
    }

    /// Byte offset of character `idx` of `block`, or the block's length
    /// for the index past its last character.
    #[inline(always)]
    fn byte_at(block: &str, chars: &[(u32, char)], idx: usize) -> usize {
        chars.get(idx).map_or(block.len(), |&(offset, _)| offset as usize)
    }

    /// Find the best segmentation of a block along `dag`, leaving in `route`
    /// the step to take from every character.
    fn calc(&self, chars: &[(u32, char)], dag: &StaticSparseDAG, route: &mut Vec<RouteEntry>) {
        let n = chars.len();

        // Every entry read below is either written first (the loop runs from
        // the end) or is the sentinel at `n`, so stale contents from a
        // previous block are never observed and need not be cleared.
        let end_entry = |end| RouteEntry {
            prob: 0.0,
            end,
            word_id: NO_MATCH,
        };
        if n + 1 > route.len() {
            route.resize(n + 1, end_entry(0));
        }
        route[n] = end_entry(n);

        let logtotal = self.log_total;
        let log1 = 0.0f64 - logtotal; // ln(1) - logtotal, precomputed for freq=1 case
        for idx in (0..n).rev() {
            let mut best: Option<RouteEntry> = None;
            for (end, word_id) in dag.iter_edges(idx) {
                let log_freq = if word_id != NO_MATCH {
                    self.log_freqs[word_id as usize]
                } else {
                    0.0 // ln(1)
                };
                let prob = log_freq - logtotal + route[end].prob;

                if let Some(b) = best {
                    if prob > b.prob || (prob == b.prob && end > b.end) {
                        best = Some(RouteEntry { prob, end, word_id });
                    }
                } else {
                    best = Some(RouteEntry { prob, end, word_id });
                }
            }

            route[idx] = best.unwrap_or_else(|| RouteEntry {
                prob: log1 + route[idx + 1].prob,
                end: idx + 1,
                word_id: NO_MATCH,
            });
        }
    }

    fn dag(&self, chars: &[(u32, char)], dag: &mut StaticSparseDAG) {
        self.trie
            .for_each_prefix_at_every_char(chars, |char_idx, end, word_id| dag.push_edge(char_idx, end, word_id));
        dag.finish(chars.len());
    }

    /// Push every dictionary word of `block` as a token, in position order.
    fn cut_all_tokens<'a>(
        &self,
        block: &'a str,
        chars: &[(u32, char)],
        base: usize,
        block_unicode_start: usize,
        tokens: &mut Vec<Token<'a>>,
    ) {
        // The trie walk visits the words in output order, so the tokens
        // come straight from it; nothing else reads this block's matches,
        // so no DAG is built.
        let byte_offset_in_sentence = block.as_ptr() as usize - base;
        self.trie.for_each_prefix_at_every_char(chars, |idx, end, _| {
            let byte_start = chars[idx].0 as usize;
            let word = &block[byte_start..Self::byte_at(block, chars, end)];
            let bs = byte_offset_in_sentence + byte_start;
            tokens.push(Token {
                word,
                start: block_unicode_start + idx,
                end: block_unicode_start + end,
                byte_start: bs,
                byte_end: bs + word.len(),
            });
        });
    }

    /// Emit the word spanning characters `x..y` of `block` with `word_id`.
    #[inline(always)]
    fn emit_span<'a>(
        block: &'a str,
        chars: &[(u32, char)],
        x: usize,
        y: usize,
        word_id: i32,
        dag: &StaticSparseDAG,
        words: &mut impl FnMut(&'a str, i32, usize, usize, &StaticSparseDAG),
    ) {
        let word = &block[Self::byte_at(block, chars, x)..Self::byte_at(block, chars, y)];
        words(word, word_id, y - x, x, dag);
    }

    /// Segment `block` and emit its words. What the best route leaves as
    /// single characters is joined: a run of ASCII alphanumerics into one
    /// word, and with `HMM` any other run through `cut_unjoined_run`.
    fn cut_route<'a, const HMM: bool>(
        &self,
        block: &'a str,
        chars: &[(u32, char)],
        words: &mut impl FnMut(&'a str, i32, usize, usize, &StaticSparseDAG),
        route: &mut Vec<RouteEntry>,
        dag: &mut StaticSparseDAG,
        hmm_context: &mut hmm::HmmContext,
    ) {
        self.dag(chars, dag);
        self.calc(chars, dag, route);
        let n = chars.len();
        let mut x = 0;
        let mut left: Option<usize> = None;

        while x < n {
            let RouteEntry { end: y, word_id, .. } = route[x];

            if y - x == 1 && (HMM || chars[x].1.is_ascii_alphanumeric()) {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(start) = left {
                    self.cut_run::<HMM>(block, chars, route, start, x, dag, words, hmm_context);
                    left = None;
                }
                Self::emit_span(block, chars, x, y, word_id, dag, words);
            }
            x = y;
        }

        if let Some(start) = left {
            self.cut_run::<HMM>(block, chars, route, start, n, dag, words, hmm_context);
        }

        dag.clear();
    }

    /// Emit the run of single characters the route left unjoined at
    /// characters `x..y` of `block`: without `HMM` it is a run of ASCII
    /// alphanumerics and becomes one word, whose id is known only when it
    /// is a single character the route looked up.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn cut_run<'a, const HMM: bool>(
        &self,
        block: &'a str,
        chars: &[(u32, char)],
        route: &[RouteEntry],
        x: usize,
        y: usize,
        dag: &StaticSparseDAG,
        words: &mut impl FnMut(&'a str, i32, usize, usize, &StaticSparseDAG),
        hmm_context: &mut hmm::HmmContext,
    ) {
        if HMM {
            self.cut_unjoined_run(block, chars, route, x, y, dag, words, hmm_context);
        } else {
            let id = if y - x == 1 { route[x].word_id } else { NO_MATCH };
            Self::emit_span(block, chars, x, y, id, dag, words);
        }
    }

    /// Emit the run of single characters the route left unjoined at
    /// characters `x..y` of `block`: as one word if it is in the
    /// dictionary, through the HMM if not.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn cut_unjoined_run<'a>(
        &self,
        block: &'a str,
        chars: &[(u32, char)],
        route: &[RouteEntry],
        x: usize,
        y: usize,
        dag: &StaticSparseDAG,
        words: &mut impl FnMut(&'a str, i32, usize, usize, &StaticSparseDAG),
        hmm_context: &mut hmm::HmmContext,
    ) {
        if y - x == 1 {
            Self::emit_span(block, chars, x, y, route[x].word_id, dag, words);
            return;
        }
        if dag.word_at(x, y).is_none() {
            self.hmm_cut(block, chars, x, y, dag, words, hmm_context);
        } else {
            // Each character is a route step of its own, so its id is known.
            let mut x = x;
            while x < y {
                let RouteEntry { end: next, word_id, .. } = route[x];
                Self::emit_span(block, chars, x, next, word_id, dag, words);
                x = next;
            }
        }
    }

    /// Cut characters `x..y` of `block` with the HMM.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn hmm_cut<'a>(
        &self,
        block: &'a str,
        chars: &[(u32, char)],
        x: usize,
        y: usize,
        dag: &StaticSparseDAG,
        words: &mut impl FnMut(&'a str, i32, usize, usize, &StaticSparseDAG),
        hmm_context: &mut hmm::HmmContext,
    ) {
        let run = &chars[x..y];
        let mut x = x;
        let mut words = |word: &'a str| {
            let count = char_count(word);
            words(word, NO_MATCH, count, x, dag);
            x += count;
        };
        if let Some(ref model) = self.hmm_model {
            hmm::cut_with_allocated_memory(block, run, &mut words, model, hmm_context);
        } else {
            hmm::cut_with_allocated_memory(block, run, &mut words, &hmm::builtin_hmm(), hmm_context);
        }
    }

    /// Create a Token for a word of `char_count` characters, advancing the
    /// tracked unicode offset past it.
    #[inline]
    fn make_token_incremental<'a>(
        word: &'a str,
        base: usize,
        unicode_offset: &mut usize,
        char_count: usize,
    ) -> Token<'a> {
        let ptr = word.as_ptr() as usize;
        debug_assert!(ptr >= base, "word is not a subslice of sentence");
        let byte_start = ptr - base;
        let byte_end = byte_start + word.len();
        let start = *unicode_offset;
        *unicode_offset = start + char_count;
        Token {
            word,
            start,
            end: *unicode_offset,
            byte_start,
            byte_end,
        }
    }

    /// Segment `sentence` and hand each token to `emit` in order, along with
    /// its dictionary id when the segmentation looked the word up
    /// (`NO_MATCH` otherwise, which does not mean the word is unknown) and,
    /// for a token cut from a dictionary block, where it sits in that block.
    fn cut_each<'a>(&self, sentence: &'a str, hmm: bool, mut emit: impl FnMut(Token<'a>, i32, Option<BlockRef<'_>>)) {
        let base = sentence.as_ptr() as usize;
        let mut unicode_offset = 0;

        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let Scratch {
                route,
                dag,
                chars,
                hmm: hmm_context,
            } = &mut *scratch;
            let mut splitter = SplitByCharacterClass::new(sentence, is_han_default);

            while let Some(state) = splitter.next_recording(chars) {
                match state {
                    SplitState::Matched(block) => {
                        assert!(!block.is_empty());

                        let chars: &[(u32, char)] = chars;
                        let mut sink =
                            |word: &'a str, word_id: i32, char_count: usize, start: usize, dag: &StaticSparseDAG| {
                                emit(
                                    Self::make_token_incremental(word, base, &mut unicode_offset, char_count),
                                    word_id,
                                    Some(BlockRef { dag, chars, start }),
                                );
                            };
                        if hmm {
                            self.cut_route::<true>(block, chars, &mut sink, route, dag, hmm_context);
                        } else {
                            self.cut_route::<false>(block, chars, &mut sink, route, dag, hmm_context);
                        }
                    }
                    SplitState::Unmatched(block) => {
                        assert!(!block.is_empty());

                        let mut i = 0;
                        while let Some(&(offset, c)) = chars.get(i) {
                            let offset = offset as usize;
                            // Group \r\n as a single token, otherwise emit each char
                            let (word, count) = if c == '\r' && matches!(chars.get(i + 1), Some(&(_, '\n'))) {
                                (&block[offset..offset + 2], 2)
                            } else {
                                (&block[offset..offset + c.len_utf8()], 1)
                            };
                            i += count;
                            emit(
                                Self::make_token_incremental(word, base, &mut unicode_offset, count),
                                NO_MATCH,
                                None,
                            );
                        }
                    }
                }
            }
            scratch.release_if_huge();
        });
    }

    /// Cut the input text
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<Token<'a>> {
        let mut tokens = Vec::with_capacity(output_capacity(sentence));
        self.cut_each(sentence, hmm, |token, _, _| tokens.push(token));
        tokens
    }

    /// Cut the input text, return all possible words
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    pub fn cut_all<'a>(&self, sentence: &'a str) -> Vec<Token<'a>> {
        let base = sentence.as_ptr() as usize;
        let mut unicode_offset = 0;

        let mut tokens = Vec::with_capacity(output_capacity(sentence));

        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let Scratch { chars, .. } = &mut *scratch;
            let mut splitter = SplitByCharacterClass::new(sentence, is_han_cut_all);

            while let Some(state) = splitter.next_recording(chars) {
                match state {
                    SplitState::Matched(block) => {
                        assert!(!block.is_empty());
                        let block_unicode_start = unicode_offset;
                        unicode_offset += chars.len();
                        self.cut_all_tokens(block, chars, base, block_unicode_start, &mut tokens);
                    }
                    SplitState::Unmatched(block) => {
                        assert!(!block.is_empty());
                        // RE_SKIP_CUT_ALL matched single characters, so each
                        // is a token of its own; a run of what it did not
                        // match (alphanumerics, `+`, `#`, newlines) is one.
                        let mut i = 0;
                        while i < chars.len() {
                            let mut j = i + 1;
                            if !is_skip_cut_all(chars[i].1) {
                                while j < chars.len() && !is_skip_cut_all(chars[j].1) {
                                    j += 1;
                                }
                            }
                            let word = &block[Self::byte_at(block, chars, i)..Self::byte_at(block, chars, j)];
                            tokens.push(Self::make_token_incremental(word, base, &mut unicode_offset, j - i));
                            i = j;
                        }
                    }
                }
            }
            scratch.release_if_huge();
        });
        tokens
    }

    /// Cut the input text in search mode
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut_for_search<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<Token<'a>> {
        let mut new_words = Vec::with_capacity(output_capacity(sentence));
        self.cut_each(sentence, hmm, |token, _, block| {
            // An alphanumeric token joined by connectors is a compound in the
            // same sense as a multi-word Chinese term, so search mode offers
            // its parts too: `WES-5.4.5` is findable as `WES` and `5.4.5` as
            // well as whole. Without this, making `cut` keep such tokens whole
            // would cost the recall the parts used to provide.
            if token.word.as_bytes().iter().any(|&b| hmm::is_skip_connector(b)) {
                let mut offset = 0;
                for part in token.word.split(hmm::is_skip_connector_char) {
                    // Only parts that could be searched on their own: a purely
                    // numeric fragment of a version or part number (the `6` of
                    // `3.6.3`) is noise, and so is a single character.
                    if part.len() >= 2 && part.bytes().any(|b| b.is_ascii_alphabetic()) {
                        let byte_start = token.byte_start + offset;
                        new_words.push(Token {
                            word: part,
                            start: token.start + char_count(&token.word[..offset]),
                            end: token.start + char_count(&token.word[..offset + part.len()]),
                            byte_start,
                            byte_end: byte_start + part.len(),
                        });
                    }
                    offset += part.len() + 1;
                }
            }
            // The dictionary matches of the block the token came from
            // already say which of its 2- and 3-character substrings are
            // words, and the block's characters give their byte offsets, so
            // neither the words nor the offsets need computing again. A token
            // from outside a dictionary block is a single character (or
            // `\r\n`) and has no such substrings.
            let Some(BlockRef { dag, chars, start: x }) = block else {
                new_words.push(token);
                return;
            };
            let word = token.word;
            let char_count = token.end - token.start;
            // Byte offset within `word` of its `k`th character.
            let rel = |k: usize| {
                if k == char_count {
                    word.len()
                } else {
                    (chars[x + k].0 - chars[x].0) as usize
                }
            };
            let mut push_gram = |i: usize, len: usize| {
                if dag.word_at(x + i, x + i + len).is_some() {
                    let gram = &word[rel(i)..rel(i + len)];
                    let byte_start = token.byte_start + rel(i);
                    new_words.push(Token {
                        word: gram,
                        start: token.start + i,
                        end: token.start + i + len,
                        byte_start,
                        byte_end: byte_start + gram.len(),
                    });
                }
            };
            if char_count > 2 {
                for i in 0..char_count - 1 {
                    push_gram(i, 2);
                }
            }
            if char_count > 3 {
                for i in 0..char_count - 2 {
                    push_gram(i, 3);
                }
            }
            new_words.push(token);
        });
        new_words
    }

    /// Tokenize
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `mode`: tokenize mode
    ///
    /// `hmm`: enable HMM or not
    pub fn tokenize<'a>(&self, sentence: &'a str, mode: TokenizeMode, hmm: bool) -> Vec<Token<'a>> {
        match mode {
            TokenizeMode::Default => self.cut(sentence, hmm),
            TokenizeMode::Search => self.cut_for_search(sentence, hmm),
        }
    }

    /// Tag the input text
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn tag<'a>(&'a self, sentence: &'a str, hmm: bool) -> Vec<Tag<'a>> {
        let mut tags = Vec::with_capacity(output_capacity(sentence));
        self.tag_each(sentence, hmm, |tag| tags.push(tag));
        tags
    }

    /// Tag `sentence` and hand each tagged word to `emit` in order, for
    /// callers that consume the tags as they come. `emit` runs while the
    /// segmentation scratch is borrowed, so it must not segment.
    #[cfg_attr(not(any(feature = "tfidf", feature = "textrank")), allow(dead_code))]
    pub(crate) fn tag_each<'a>(&'a self, sentence: &'a str, hmm: bool, mut emit: impl FnMut(Tag<'a>)) {
        self.cut_each(sentence, hmm, |token, word_id, block| {
            let word = token.word;
            let word_id = if word_id != NO_MATCH {
                Some(word_id)
            } else if let Some(BlockRef { dag, start, .. }) = block {
                dag.word_at(start, start + (token.end - token.start))
            } else {
                self.trie.get(word)
            };
            let tag = match word_id {
                Some(word_id) => &self.tags[self.records[word_id as usize].tag as usize],
                None => self.guess_tag(word),
            };
            emit(Tag {
                word,
                tag,
                start: token.start,
                end: token.end,
                byte_start: token.byte_start,
                byte_end: token.byte_end,
            });
        });
    }

    /// Guess the POS tag for an OOV word.
    ///
    /// For CJK words, uses the posseg HMM model (when available) to predict the tag.
    /// For ASCII words, uses simple heuristics (digits → "m", alpha → "eng", else → "x").
    fn guess_tag(&self, word: &str) -> &'static str {
        let mut eng = 0;
        let mut m = 0;
        for chr in word.chars() {
            if chr.is_ascii_alphanumeric() {
                eng += 1;
                if chr.is_ascii_digit() {
                    m += 1;
                }
            }
        }
        if eng > 0 {
            return if eng == m { "m" } else { "eng" };
        }

        #[cfg(feature = "default-dict")]
        {
            // Only use posseg HMM for words containing CJK characters
            if word.chars().any(is_cjk) {
                return posseg::guess_tag(word);
            }
        }

        "x"
    }
}

#[cfg(test)]
mod tests {
    use super::{Jieba, SplitByCharacterClass, SplitState, TokenizeMode, is_han_default};
    use expect_test::expect;
    use std::io::BufReader;

    #[test]
    fn test_init_with_default_dict() {
        let _ = Jieba::new();
    }

    #[test]
    fn test_has_word() {
        let jieba = Jieba::new();
        assert!(jieba.has_word("中国"));
        assert!(jieba.has_word("开源"));
        assert!(!jieba.has_word("不存在的词"));
    }

    #[test]
    fn test_split_matches() {
        let splitter = SplitByCharacterClass::new(
            "👪 PS: 我觉得开源有一个好处，就是能够敦促自己不断改进 👪，避免敞帚自珍",
            is_han_default,
        );
        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.as_str();
                    assert!(!block.is_empty());
                }
                SplitState::Unmatched(_) => {
                    let block = state.as_str();
                    assert!(!block.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_split_matches_against_unicode_sip() {
        let splitter = SplitByCharacterClass::new("讥䶯䶰䶱䶲䶳䶴䶵𦡦", is_han_default);

        let result: Vec<&str> = splitter.map(|x| x.as_str()).collect();
        expect![[r#"["讥䶯䶰䶱䶲䶳䶴䶵𦡦"]"#]].assert_eq(&format!("{:?}", result));
    }

    #[test]
    fn test_cut_all_skip_single_char() {
        let jieba = Jieba::new();
        let words: Vec<&str> = jieba.cut_all("a！！b").iter().map(|t| t.word).collect();
        assert_eq!(words, vec!["a", "！", "！", "b"]);
    }

    #[test]
    fn test_cut_default_crlf_and_whitespace() {
        let jieba = Jieba::new();
        let words: Vec<&str> = jieba.cut("x\r\n\ty", false).iter().map(|t| t.word).collect();
        assert_eq!(words, vec!["x", "\r\n", "\t", "y"]);
    }

    #[test]
    fn test_cut_all() {
        let jieba = Jieba::new();
        let tokens = jieba.cut_all("abc网球拍卖会def");
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["abc", "网", "网球", "网球拍", "球", "球拍", "拍", "拍卖", "拍卖会", "卖", "会", "def"]"#]]
            .assert_eq(&format!("{:?}", words));

        // The cut_all from the python de-facto implementation is loosely defined,
        // And the answer "我, 来到, 北京, 清华, 清华大学, 华大, 大学" from the python implementation looks weird since it drops the single character word even though it is part of the DAG candidates.
        // For example, it includes "华大" but it doesn't include "清" and "学"
        let tokens = jieba.cut_all("我来到北京清华大学");
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["我", "来", "来到", "到", "北", "北京", "京", "清", "清华", "清华大学", "华", "华大", "大", "大学", "学"]"#]]
            .assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_cut_no_hmm() {
        let jieba = Jieba::new();
        let tokens = jieba.cut("abc网球拍卖会def", false);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["abc", "网球", "拍卖会", "def"]"#]].assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_cut_no_hmm1() {
        let jieba = Jieba::new();
        let tokens = jieba.cut("abc网球拍卖会def！！？\r\n\t", false);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["abc", "网球", "拍卖会", "def", "！", "！", "？", "\r\n", "\t"]"#]]
            .assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_cut_with_hmm() {
        let jieba = Jieba::new();
        let tokens = jieba.cut("我们中出了一个叛徒", false);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["我们", "中", "出", "了", "一个", "叛徒"]"#]].assert_eq(&format!("{:?}", words));
        let tokens = jieba.cut("我们中出了一个叛徒", true);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["我们", "中出", "了", "一个", "叛徒"]"#]].assert_eq(&format!("{:?}", words));
        let tokens = jieba.cut("我们中出了一个叛徒👪", true);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["我们", "中出", "了", "一个", "叛徒", "👪"]"#]].assert_eq(&format!("{:?}", words));

        let tokens = jieba.cut("我来到北京清华大学", true);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["我", "来到", "北京", "清华大学"]"#]].assert_eq(&format!("{:?}", words));

        let tokens = jieba.cut("他来到了网易杭研大厦", true);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["他", "来到", "了", "网易", "杭研", "大厦"]"#]].assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_cut_weicheng() {
        static WEICHENG_TXT: &str = include_str!("../../examples/weicheng/src/weicheng.txt");
        let jieba = Jieba::new();
        for line in WEICHENG_TXT.split('\n') {
            let _ = jieba.cut(line, true);
        }
    }

    #[test]
    fn test_cut_for_search() {
        let jieba = Jieba::new();
        let tokens = jieba.cut_for_search("南京市长江大桥", true);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["南京", "京市", "南京市", "长江", "大桥", "长江大桥"]"#]].assert_eq(&format!("{:?}", words));

        let tokens = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", true);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();

        // The python implementation silently filtered "，". but we include it here in the output
        // to let the library user to decide their own filtering strategy
        expect![[r#"["小明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所", "，", "后", "在", "日本", "京都", "大学", "日本京都大学", "深造"]"#]]
            .assert_eq(&format!("{:?}", words));
    }

    /// Search mode offers a compound's parts as well as the whole, and that now
    /// covers alphanumeric compounds too: keeping `well-known` together in
    /// `cut` would otherwise cost the recall its parts used to provide.
    ///
    /// Parts are only worth offering when they could be searched on their own,
    /// so a purely numeric fragment of a version number (the `6` of `3.6.3`)
    /// and a single character are left out.
    #[test]
    fn test_cut_for_search_offers_parts_of_alphanumeric_compounds() {
        let jieba = Jieba::new();
        let mut got = Vec::new();
        for sentence in [
            "well-known",
            "state-of-the-art",
            "www.example.com",
            "ISU-CNS24093",
            "E4850B-I",
            "OPENSSL_1_1_1",
            "3.6.3",
            "01.01.00070",
            "3.14",
            "50%",
        ] {
            let words: Vec<&str> = jieba.cut_for_search(sentence, true).iter().map(|t| t.word).collect();
            got.push(format!("{} -> {:?}", sentence, words));
        }
        expect![[r#"
            well-known -> ["well", "known", "well-known"]
            state-of-the-art -> ["state", "of", "the", "art", "state-of-the-art"]
            www.example.com -> ["www", "example", "com", "www.example.com"]
            ISU-CNS24093 -> ["ISU", "CNS24093", "ISU-CNS24093"]
            E4850B-I -> ["E4850B", "E4850B-I"]
            OPENSSL_1_1_1 -> ["OPENSSL", "OPENSSL_1_1_1"]
            3.6.3 -> ["3.6.3"]
            01.01.00070 -> ["01.01.00070"]
            3.14 -> ["3.14"]
            50% -> ["50%"]"#]]
        .assert_eq(&got.join("\n"));
    }

    /// An offered part must address itself in the original text, so a caller can
    /// highlight what it matched.
    #[test]
    fn test_cut_for_search_part_offsets() {
        let jieba = Jieba::new();
        let sentence = "版本WES-5.4.5发布";
        let part = jieba
            .cut_for_search(sentence, true)
            .into_iter()
            .find(|t| t.word == "WES")
            .expect("the leading part is offered");
        assert_eq!(&sentence[part.byte_start..part.byte_end], "WES");
        assert_eq!(
            sentence
                .chars()
                .skip(part.start)
                .take(part.end - part.start)
                .collect::<String>(),
            "WES"
        );
    }

    #[test]
    fn test_tag() {
        let jieba = Jieba::new();
        let tags = jieba.tag(
            "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            true,
        );
        expect![[r#"
            [
                Tag {
                    word: "我",
                    tag: "r",
                    start: 0,
                    end: 1,
                    byte_start: 0,
                    byte_end: 3,
                },
                Tag {
                    word: "是",
                    tag: "v",
                    start: 1,
                    end: 2,
                    byte_start: 3,
                    byte_end: 6,
                },
                Tag {
                    word: "拖拉机",
                    tag: "n",
                    start: 2,
                    end: 5,
                    byte_start: 6,
                    byte_end: 15,
                },
                Tag {
                    word: "学院",
                    tag: "n",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Tag {
                    word: "手扶拖拉机",
                    tag: "n",
                    start: 7,
                    end: 12,
                    byte_start: 21,
                    byte_end: 36,
                },
                Tag {
                    word: "专业",
                    tag: "n",
                    start: 12,
                    end: 14,
                    byte_start: 36,
                    byte_end: 42,
                },
                Tag {
                    word: "的",
                    tag: "uj",
                    start: 14,
                    end: 15,
                    byte_start: 42,
                    byte_end: 45,
                },
                Tag {
                    word: "。",
                    tag: "x",
                    start: 15,
                    end: 16,
                    byte_start: 45,
                    byte_end: 48,
                },
                Tag {
                    word: "不用",
                    tag: "v",
                    start: 16,
                    end: 18,
                    byte_start: 48,
                    byte_end: 54,
                },
                Tag {
                    word: "多久",
                    tag: "m",
                    start: 18,
                    end: 20,
                    byte_start: 54,
                    byte_end: 60,
                },
                Tag {
                    word: "，",
                    tag: "x",
                    start: 20,
                    end: 21,
                    byte_start: 60,
                    byte_end: 63,
                },
                Tag {
                    word: "我",
                    tag: "r",
                    start: 21,
                    end: 22,
                    byte_start: 63,
                    byte_end: 66,
                },
                Tag {
                    word: "就",
                    tag: "d",
                    start: 22,
                    end: 23,
                    byte_start: 66,
                    byte_end: 69,
                },
                Tag {
                    word: "会",
                    tag: "v",
                    start: 23,
                    end: 24,
                    byte_start: 69,
                    byte_end: 72,
                },
                Tag {
                    word: "升职",
                    tag: "v",
                    start: 24,
                    end: 26,
                    byte_start: 72,
                    byte_end: 78,
                },
                Tag {
                    word: "加薪",
                    tag: "nr",
                    start: 26,
                    end: 28,
                    byte_start: 78,
                    byte_end: 84,
                },
                Tag {
                    word: "，",
                    tag: "x",
                    start: 28,
                    end: 29,
                    byte_start: 84,
                    byte_end: 87,
                },
                Tag {
                    word: "当上",
                    tag: "t",
                    start: 29,
                    end: 31,
                    byte_start: 87,
                    byte_end: 93,
                },
                Tag {
                    word: "CEO",
                    tag: "eng",
                    start: 31,
                    end: 34,
                    byte_start: 93,
                    byte_end: 96,
                },
                Tag {
                    word: "，",
                    tag: "x",
                    start: 34,
                    end: 35,
                    byte_start: 96,
                    byte_end: 99,
                },
                Tag {
                    word: "走上",
                    tag: "v",
                    start: 35,
                    end: 37,
                    byte_start: 99,
                    byte_end: 105,
                },
                Tag {
                    word: "人生",
                    tag: "n",
                    start: 37,
                    end: 39,
                    byte_start: 105,
                    byte_end: 111,
                },
                Tag {
                    word: "巅峰",
                    tag: "n",
                    start: 39,
                    end: 41,
                    byte_start: 111,
                    byte_end: 117,
                },
                Tag {
                    word: "。",
                    tag: "x",
                    start: 41,
                    end: 42,
                    byte_start: 117,
                    byte_end: 120,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tags));

        let tags = jieba.tag("今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。", true);
        expect![[r#"
            [
                Tag {
                    word: "今天",
                    tag: "t",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Tag {
                    word: "纽约",
                    tag: "ns",
                    start: 2,
                    end: 4,
                    byte_start: 6,
                    byte_end: 12,
                },
                Tag {
                    word: "的",
                    tag: "uj",
                    start: 4,
                    end: 5,
                    byte_start: 12,
                    byte_end: 15,
                },
                Tag {
                    word: "天气",
                    tag: "n",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Tag {
                    word: "真好",
                    tag: "d",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
                Tag {
                    word: "啊",
                    tag: "zg",
                    start: 9,
                    end: 10,
                    byte_start: 27,
                    byte_end: 30,
                },
                Tag {
                    word: "，",
                    tag: "x",
                    start: 10,
                    end: 11,
                    byte_start: 30,
                    byte_end: 33,
                },
                Tag {
                    word: "京华",
                    tag: "nz",
                    start: 11,
                    end: 13,
                    byte_start: 33,
                    byte_end: 39,
                },
                Tag {
                    word: "大酒店",
                    tag: "n",
                    start: 13,
                    end: 16,
                    byte_start: 39,
                    byte_end: 48,
                },
                Tag {
                    word: "的",
                    tag: "uj",
                    start: 16,
                    end: 17,
                    byte_start: 48,
                    byte_end: 51,
                },
                Tag {
                    word: "张尧",
                    tag: "nr",
                    start: 17,
                    end: 19,
                    byte_start: 51,
                    byte_end: 57,
                },
                Tag {
                    word: "经理",
                    tag: "n",
                    start: 19,
                    end: 21,
                    byte_start: 57,
                    byte_end: 63,
                },
                Tag {
                    word: "吃",
                    tag: "v",
                    start: 21,
                    end: 22,
                    byte_start: 63,
                    byte_end: 66,
                },
                Tag {
                    word: "了",
                    tag: "ul",
                    start: 22,
                    end: 23,
                    byte_start: 66,
                    byte_end: 69,
                },
                Tag {
                    word: "一只",
                    tag: "m",
                    start: 23,
                    end: 25,
                    byte_start: 69,
                    byte_end: 75,
                },
                Tag {
                    word: "北京烤鸭",
                    tag: "n",
                    start: 25,
                    end: 29,
                    byte_start: 75,
                    byte_end: 87,
                },
                Tag {
                    word: "。",
                    tag: "x",
                    start: 29,
                    end: 30,
                    byte_start: 87,
                    byte_end: 90,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tags));
    }

    #[test]
    fn test_tokenize() {
        let jieba = Jieba::new();
        let tokens = jieba.tokenize("南京市长江大桥", TokenizeMode::Default, false);
        expect![[r#"
            [
                Token {
                    word: "南京市",
                    start: 0,
                    end: 3,
                    byte_start: 0,
                    byte_end: 9,
                },
                Token {
                    word: "长江大桥",
                    start: 3,
                    end: 7,
                    byte_start: 9,
                    byte_end: 21,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));

        let tokens = jieba.tokenize("南京市长江大桥", TokenizeMode::Search, false);
        expect![[r#"
            [
                Token {
                    word: "南京",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "京市",
                    start: 1,
                    end: 3,
                    byte_start: 3,
                    byte_end: 9,
                },
                Token {
                    word: "南京市",
                    start: 0,
                    end: 3,
                    byte_start: 0,
                    byte_end: 9,
                },
                Token {
                    word: "长江",
                    start: 3,
                    end: 5,
                    byte_start: 9,
                    byte_end: 15,
                },
                Token {
                    word: "大桥",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "长江大桥",
                    start: 3,
                    end: 7,
                    byte_start: 9,
                    byte_end: 21,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));

        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        expect![[r#"
            [
                Token {
                    word: "我们",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3,
                    byte_start: 6,
                    byte_end: 9,
                },
                Token {
                    word: "出",
                    start: 3,
                    end: 4,
                    byte_start: 9,
                    byte_end: 12,
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5,
                    byte_start: 12,
                    byte_end: 15,
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        expect![[r#"
            [
                Token {
                    word: "我们",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4,
                    byte_start: 6,
                    byte_end: 12,
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5,
                    byte_start: 12,
                    byte_end: 15,
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));

        let tokens = jieba.tokenize("永和服装饰品有限公司", TokenizeMode::Default, true);
        expect![[r#"
            [
                Token {
                    word: "永和",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "服装",
                    start: 2,
                    end: 4,
                    byte_start: 6,
                    byte_end: 12,
                },
                Token {
                    word: "饰品",
                    start: 4,
                    end: 6,
                    byte_start: 12,
                    byte_end: 18,
                },
                Token {
                    word: "有限公司",
                    start: 6,
                    end: 10,
                    byte_start: 18,
                    byte_end: 30,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));
    }

    #[test]
    fn test_userdict() {
        let mut jieba = Jieba::new();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        expect![[r#"
            [
                Token {
                    word: "我们",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3,
                    byte_start: 6,
                    byte_end: 9,
                },
                Token {
                    word: "出",
                    start: 3,
                    end: 4,
                    byte_start: 9,
                    byte_end: 12,
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5,
                    byte_start: 12,
                    byte_end: 15,
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));
        let userdict = "中出 10000";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        expect![[r#"
            [
                Token {
                    word: "我们",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4,
                    byte_start: 6,
                    byte_end: 12,
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5,
                    byte_start: 12,
                    byte_end: 15,
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));
    }

    #[test]
    fn test_userdict_hmm() {
        let mut jieba = Jieba::new();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        expect![[r#"
            [
                Token {
                    word: "我们",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4,
                    byte_start: 6,
                    byte_end: 12,
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5,
                    byte_start: 12,
                    byte_end: 15,
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));
        let userdict = "出了 10000";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        expect![[r#"
            [
                Token {
                    word: "我们",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3,
                    byte_start: 6,
                    byte_end: 9,
                },
                Token {
                    word: "出了",
                    start: 3,
                    end: 5,
                    byte_start: 9,
                    byte_end: 15,
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));
        expect![[r#"
            [
                Token {
                    word: "我们",
                    start: 0,
                    end: 2,
                    byte_start: 0,
                    byte_end: 6,
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3,
                    byte_start: 6,
                    byte_end: 9,
                },
                Token {
                    word: "出了",
                    start: 3,
                    end: 5,
                    byte_start: 9,
                    byte_end: 15,
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7,
                    byte_start: 15,
                    byte_end: 21,
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9,
                    byte_start: 21,
                    byte_end: 27,
                },
            ]"#]]
        .assert_eq(&format!("{:#?}", tokens));
    }

    #[test]
    fn test_userdict_error() {
        let mut jieba = Jieba::empty();
        let userdict = "出了 not_a_int";
        let ret = jieba.load_dict(&mut BufReader::new(userdict.as_bytes()));
        assert!(ret.is_err());
    }

    #[test]
    fn test_suggest_freq() {
        // NOTE: Following behaviors are aligned with original Jieba

        let mut jieba = Jieba::new();
        // These values were calculated by original Jieba
        assert_eq!(jieba.suggest_freq("中出"), 348);
        assert_eq!(jieba.suggest_freq("出了"), 1263);

        // Freq in dict.txt was 3, which became 300 after loading user dict
        let userdict = "中出 300";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        // But it's less than calculated freq 348
        assert_eq!(jieba.suggest_freq("中出"), 348);

        let userdict = "中出 500";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        // Now it's significant enough
        assert_eq!(jieba.suggest_freq("中出"), 500)
    }

    #[test]
    fn test_custom_lower_freq() {
        let mut jieba = Jieba::new();

        jieba.add_word("测试", Some(2445), None);
        jieba.add_word("测试", Some(10), None);
        let tokens = jieba.cut("测试", false);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["测试"]"#]].assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_cut_dag_no_hmm_against_string_with_sip() {
        let mut jieba = Jieba::empty();

        //add fake word into dictionary
        jieba.add_word("䶴䶵𦡦", Some(1000), None);
        jieba.add_word("讥䶯䶰䶱䶲䶳", Some(1000), None);

        let tokens = jieba.cut("讥䶯䶰䶱䶲䶳䶴䶵𦡦", false);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["讥䶯䶰䶱䶲䶳", "䶴䶵𦡦"]"#]].assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_add_custom_word_with_underscrore() {
        let mut jieba = Jieba::empty();
        jieba.add_word("田-女士", Some(42), Some("n"));
        let tokens = jieba.cut("市民田-女士急匆匆", false);
        let words: Vec<&str> = tokens.iter().map(|t| t.word).collect();
        expect![[r#"["市", "民", "田-女士", "急", "匆", "匆"]"#]].assert_eq(&format!("{:?}", words));
    }

    #[test]
    fn test_cut_with_custom_hmm_model() {
        use crate::hmm::HmmModel;

        // Load the builtin hmm.model at runtime
        let hmm_data = include_str!("../../jieba-macros/src/hmm.model");
        let mut reader = BufReader::new(hmm_data.as_bytes());
        let model = HmmModel::load(&mut reader).unwrap();

        let mut jieba_custom = Jieba::new();
        jieba_custom.set_hmm_model(model);
        let jieba_builtin = Jieba::new();

        // Runtime-loaded model should produce the same results as the builtin
        let sentences = [
            "我们中出了一个叛徒",
            "小明硕士毕业于中国科学院计算所后在日本京都大学深造",
            "他来到了网易杭研大厦",
            "我来到北京清华大学",
        ];
        for sentence in sentences {
            let builtin_words = jieba_builtin.cut(sentence, true);
            let custom_words = jieba_custom.cut(sentence, true);
            assert_eq!(custom_words, builtin_words, "mismatch for: {sentence}");
        }
    }
}
