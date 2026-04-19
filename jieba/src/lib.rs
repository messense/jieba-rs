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

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::io::BufRead;

use cedarwood::Cedar;

pub(crate) type FxHashMap<K, V> = HashMap<K, V, rustc_hash::FxBuildHasher>;

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

#[cfg(feature = "default-dict")]
include_flate::flate!(static DEFAULT_DICT: str from "src/data/dict.txt");

use sparse_dag::StaticSparseDAG;

thread_local! {
    static HMM_CONTEXT: std::cell::RefCell<hmm::HmmContext> = std::cell::RefCell::new(hmm::HmmContext::default());
}

/// Check if a character is in a CJK Unified Ideographs range.
#[inline]
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B73F}'
        | '\u{2B740}'..='\u{2B81F}'
        | '\u{2B820}'..='\u{2CEAF}'
        | '\u{2CEB0}'..='\u{2EBEF}'
        | '\u{2F800}'..='\u{2FA1F}'
    )
}

/// RE_HAN_DEFAULT character class: CJK + ASCII alphanumeric + `+#&._%\-`
#[inline]
fn is_han_default(c: char) -> bool {
    is_cjk(c) || c.is_ascii_alphanumeric() || matches!(c, '+' | '#' | '&' | '.' | '_' | '%' | '-')
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
}

impl<'t, F: Fn(char) -> bool> Iterator for SplitByCharacterClass<'t, F> {
    type Item = SplitState<'t>;

    fn next(&mut self) -> Option<SplitState<'t>> {
        if self.pos >= self.text.len() {
            return None;
        }

        let remaining = &self.text[self.pos..];
        let first_char = remaining.chars().next().unwrap();

        if (self.classify)(first_char) {
            // Matched run: consume while classify is true
            let start = self.pos;
            let mut end = self.pos + first_char.len_utf8();
            for c in remaining[first_char.len_utf8()..].chars() {
                if (self.classify)(c) {
                    end += c.len_utf8();
                } else {
                    break;
                }
            }
            self.pos = end;
            Some(SplitState::Matched(&self.text[start..end]))
        } else {
            // Unmatched run: consume while classify is false
            let start = self.pos;
            let mut end = self.pos + first_char.len_utf8();
            for c in remaining[first_char.len_utf8()..].chars() {
                if (self.classify)(c) {
                    break;
                }
                end += c.len_utf8();
            }
            self.pos = end;
            Some(SplitState::Unmatched(&self.text[start..end]))
        }
    }
}

#[derive(Debug)]
pub(crate) enum SplitState<'t> {
    Unmatched(&'t str),
    Matched(&'t str),
}

impl<'t> SplitState<'t> {
    #[inline]
    fn as_str(&self) -> &'t str {
        match self {
            SplitState::Unmatched(t) => t,
            SplitState::Matched(t) => t,
        }
    }

    #[inline]
    pub fn is_matched(&self) -> bool {
        matches!(self, SplitState::Matched(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizeMode {
    /// Default mode
    Default,
    /// Search mode
    Search,
}

/// A Token
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

#[derive(Debug, Clone)]
struct Record {
    freq: usize,
    log_freq: f64,
    tag: Box<str>,
}

impl Record {
    #[inline(always)]
    fn new(freq: usize, tag: Box<str>) -> Self {
        Self {
            freq,
            log_freq: (freq as f64).ln(),
            tag,
        }
    }

    #[inline]
    fn set_freq(&mut self, freq: usize) {
        self.freq = freq;
        self.log_freq = (freq as f64).ln();
    }
}

/// Jieba segmentation
#[derive(Clone)]
pub struct Jieba {
    records: Vec<Record>,
    cedar: Cedar,
    total: usize,
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
            cedar: Cedar::new(),
            total: 0,
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
        use std::io::BufReader;

        let mut default_dict = BufReader::new(DEFAULT_DICT.as_bytes());
        self.load_dict(&mut default_dict).unwrap();
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
    /// 2. Resets `cedar` to a new instance of `Cedar`.
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
        self.cedar = Cedar::new();
        self.total = 0;
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

        match self.cedar.exact_match_search(word) {
            Some((word_id, _, _)) => {
                let old_freq = self.records[word_id as usize].freq;
                self.records[word_id as usize].set_freq(freq);

                self.total += freq;
                self.total -= old_freq;
            }
            None => {
                let word_id = self.records.len() as i32;
                self.records.push(Record::new(freq, tag.into()));

                self.cedar.update(word, word_id);
                self.total += freq;
            }
        };

        freq
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
        self.cedar.exact_match_search(word).is_some()
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

                    match self.cedar.exact_match_search(word) {
                        Some((word_id, _, _)) => {
                            self.records[word_id as usize].set_freq(freq);
                        }
                        None => {
                            let word_id = self.records.len() as i32;
                            self.records.push(Record::new(freq, tag.into()));
                            self.cedar.update(word, word_id);
                        }
                    };
                }
            }
            buf.clear();
        }
        self.total = self.records.iter().map(|n| n.freq).sum();

        Ok(())
    }

    fn get_word_freq(&self, word: &str, default: usize) -> usize {
        match self.cedar.exact_match_search(word) {
            Some((word_id, _, _)) => self.records[word_id as usize].freq,
            _ => default,
        }
    }

    /// Suggest word frequency to force the characters in a word to be joined or split.
    pub fn suggest_freq(&self, segment: &str) -> usize {
        let logtotal = (self.total as f64).ln();
        let logfreq = self.cut(segment, false).iter().fold(0f64, |freq, token| {
            freq + (self.get_word_freq(token.word, 1) as f64).ln() - logtotal
        });
        std::cmp::max((logfreq + logtotal).exp() as usize + 1, self.get_word_freq(segment, 1))
    }

    #[allow(clippy::ptr_arg)]
    fn calc(&self, sentence: &str, dag: &StaticSparseDAG, route: &mut Vec<(f64, usize)>) {
        let str_len = sentence.len();

        if str_len + 1 > route.len() {
            route.resize(str_len + 1, (0.0, 0));
        }

        let logtotal = (self.total as f64).ln();
        let log1 = 0.0f64 - logtotal; // ln(1) - logtotal, precomputed for freq=1 case
        let mut prev_byte_start = str_len;
        let curr = sentence.char_indices().map(|x| x.0).rev();
        for byte_start in curr {
            let pair = dag
                .iter_edges(byte_start)
                .map(|(byte_end, word_id)| {
                    let log_freq = if word_id != sparse_dag::NO_MATCH {
                        self.records[word_id as usize].log_freq
                    } else {
                        0.0 // ln(1)
                    };

                    (log_freq - logtotal + route[byte_end].0, byte_end)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));

            if let Some(p) = pair {
                route[byte_start] = p;
            } else {
                let byte_end = prev_byte_start;
                route[byte_start] = (log1 + route[byte_end].0, byte_end);
            }

            prev_byte_start = byte_start;
        }
    }

    fn dag(&self, sentence: &str, dag: &mut StaticSparseDAG) {
        for (byte_start, _) in sentence.char_indices() {
            dag.start(byte_start);
            let haystack = &sentence[byte_start..];

            for (word_id, end_index) in self.cedar.common_prefix_iter(haystack) {
                dag.insert(end_index + byte_start + 1, word_id);
            }

            dag.commit();
        }
    }

    /// Emits `Token`s directly with unicode positions for cut_all,
    /// avoiding the need for a separate byte-to-unicode lookup table.
    fn cut_all_tokens<'a>(&self, block: &'a str, base: usize, block_unicode_start: usize, tokens: &mut Vec<Token<'a>>) {
        let str_len = block.len();
        let mut dag = StaticSparseDAG::with_size_hint(block.len());
        self.dag(block, &mut dag);

        let block_base = block.as_ptr() as usize;
        let byte_offset_in_sentence = block_base - base;

        for (unicode_idx, (byte_start, _)) in block.char_indices().enumerate() {
            let unicode_start = block_unicode_start + unicode_idx;
            for (byte_end, _) in dag.iter_edges(byte_start) {
                let word = if byte_end == str_len {
                    &block[byte_start..]
                } else {
                    &block[byte_start..byte_end]
                };
                let char_count = word.as_bytes().iter().filter(|&&b| (b as i8) >= -0x40).count();
                let bs = byte_offset_in_sentence + byte_start;
                tokens.push(Token {
                    word,
                    start: unicode_start,
                    end: unicode_start + char_count,
                    byte_start: bs,
                    byte_end: bs + word.len(),
                });
            }
        }
    }

    fn cut_dag_no_hmm<'a>(
        &self,
        sentence: &'a str,
        words: &mut Vec<&'a str>,
        route: &mut Vec<(f64, usize)>,
        dag: &mut StaticSparseDAG,
    ) {
        self.dag(sentence, dag);
        self.calc(sentence, dag, route);
        let mut x = 0;
        let mut left: Option<usize> = None;

        while x < sentence.len() {
            let y = route[x].1;
            let l_str = &sentence[x..y];

            if l_str.chars().nth(1).is_none() && l_str.as_bytes()[0].is_ascii_alphanumeric() {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(byte_start) = left {
                    let word = &sentence[byte_start..x];
                    words.push(word);
                    left = None;
                }

                words.push(l_str);
            }
            x = y;
        }

        if let Some(byte_start) = left {
            let word = &sentence[byte_start..];
            words.push(word);
        }

        dag.clear();
        route.clear();
    }

    #[inline]
    fn hmm_cut<'a>(&self, word: &'a str, words: &mut Vec<&'a str>, hmm_context: &mut hmm::HmmContext) {
        if let Some(ref model) = self.hmm_model {
            hmm::cut_with_allocated_memory(word, words, model, hmm_context);
        } else {
            hmm::cut_with_allocated_memory(word, words, &hmm::builtin_hmm(), hmm_context);
        }
    }

    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn cut_dag_hmm<'a>(
        &self,
        sentence: &'a str,
        words: &mut Vec<&'a str>,
        route: &mut Vec<(f64, usize)>,
        dag: &mut StaticSparseDAG,
        hmm_context: &mut hmm::HmmContext,
    ) {
        self.dag(sentence, dag);
        self.calc(sentence, dag, route);
        let mut x = 0;
        let mut left: Option<usize> = None;

        while x < sentence.len() {
            let y = route[x].1;

            if sentence[x..y].chars().nth(1).is_none() {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(byte_start) = left {
                    let byte_end = x;
                    let word = &sentence[byte_start..byte_end];
                    if word.chars().nth(1).is_none() {
                        words.push(word);
                    } else if self.cedar.exact_match_search(word).is_none() {
                        self.hmm_cut(word, words, hmm_context);
                    } else {
                        let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                        while let Some(byte_start) = word_indices.next() {
                            if let Some(byte_end) = word_indices.peek() {
                                words.push(&word[byte_start..*byte_end]);
                            } else {
                                words.push(&word[byte_start..]);
                            }
                        }
                    }
                    left = None;
                }
                let word = &sentence[x..y];
                words.push(word);
            }
            x = y;
        }

        if let Some(byte_start) = left {
            let word = &sentence[byte_start..];

            if word.chars().nth(1).is_none() {
                words.push(word);
            } else if self.cedar.exact_match_search(word).is_none() {
                self.hmm_cut(word, words, hmm_context);
            } else {
                let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                while let Some(byte_start) = word_indices.next() {
                    if let Some(byte_end) = word_indices.peek() {
                        words.push(&word[byte_start..*byte_end]);
                    } else {
                        words.push(&word[byte_start..]);
                    }
                }
            }
        }

        dag.clear();
        route.clear();
    }

    /// Create a Token with incrementally tracked unicode offset.
    /// Returns the updated unicode_offset (past the end of this token).
    #[inline]
    fn make_token_incremental<'a>(word: &'a str, base: usize, unicode_offset: &mut usize) -> Token<'a> {
        let ptr = word.as_ptr() as usize;
        debug_assert!(ptr >= base, "word is not a subslice of sentence");
        let byte_start = ptr - base;
        let byte_end = byte_start + word.len();
        let start = *unicode_offset;
        // Count UTF-8 leading bytes to get char count without allocating
        let char_count = word.as_bytes().iter().filter(|&&b| (b as i8) >= -0x40).count();
        *unicode_offset = start + char_count;
        Token {
            word,
            start,
            end: *unicode_offset,
            byte_start,
            byte_end,
        }
    }

    #[allow(non_snake_case)]
    fn cut_internal<'a>(&self, sentence: &'a str, cut_all: bool, hmm: bool) -> Vec<Token<'a>> {
        if cut_all {
            return self.cut_all_toplevel(sentence);
        }
        let base = sentence.as_ptr() as usize;
        let mut unicode_offset = 0;

        let heuristic_capacity = sentence.len() / 2;
        let mut str_words = Vec::with_capacity(heuristic_capacity);
        let mut tokens = Vec::with_capacity(heuristic_capacity);

        let splitter = SplitByCharacterClass::new(sentence, is_han_default);
        let mut route = Vec::with_capacity(heuristic_capacity);
        let mut dag = StaticSparseDAG::with_size_hint(heuristic_capacity);

        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.as_str();
                    assert!(!block.is_empty());

                    str_words.clear();
                    if hmm {
                        HMM_CONTEXT.with(|ctx| {
                            let mut hmm_context = ctx.borrow_mut();
                            self.cut_dag_hmm(block, &mut str_words, &mut route, &mut dag, &mut hmm_context);
                        });
                    } else {
                        self.cut_dag_no_hmm(block, &mut str_words, &mut route, &mut dag);
                    }
                    for &word in &str_words {
                        tokens.push(Self::make_token_incremental(word, base, &mut unicode_offset));
                    }
                }
                SplitState::Unmatched(_) => {
                    let block = state.as_str();
                    assert!(!block.is_empty());

                    let mut chars = block.char_indices().peekable();
                    while let Some((i, c)) = chars.next() {
                        // Group \r\n as a single token, otherwise emit each char
                        let word = if c == '\r' {
                            if let Some(&(_, '\n')) = chars.peek() {
                                let _ = chars.next();
                                let end = i + 2;
                                &block[i..end]
                            } else {
                                let end = i + c.len_utf8();
                                &block[i..end]
                            }
                        } else {
                            let end = i + c.len_utf8();
                            &block[i..end]
                        };
                        tokens.push(Self::make_token_incremental(word, base, &mut unicode_offset));
                    }
                }
            }
        }
        tokens
    }

    /// Dedicated top-level cut_all implementation that avoids allocating a byte-to-unicode table.
    fn cut_all_toplevel<'a>(&self, sentence: &'a str) -> Vec<Token<'a>> {
        let base = sentence.as_ptr() as usize;
        let mut unicode_offset = 0;

        let heuristic_capacity = sentence.len() / 2;
        let mut tokens = Vec::with_capacity(heuristic_capacity);

        let splitter = SplitByCharacterClass::new(sentence, is_han_cut_all);

        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.as_str();
                    assert!(!block.is_empty());
                    let block_unicode_start = unicode_offset;
                    // Advance unicode_offset past this block
                    unicode_offset += block.chars().count();
                    self.cut_all_tokens(block, base, block_unicode_start, &mut tokens);
                }
                SplitState::Unmatched(_) => {
                    let block = state.as_str();
                    assert!(!block.is_empty());

                    let skip_splitter = SplitByCharacterClass::new(block, is_skip_cut_all);
                    for skip_state in skip_splitter {
                        let word = skip_state.as_str();
                        if word.is_empty() {
                            continue;
                        }
                        if skip_state.is_matched() {
                            // Emit each char individually to match old RE_SKIP_CUT_ALL
                            // which matched single characters, not runs.
                            let mut indices = word.char_indices().peekable();
                            while let Some((i, _)) = indices.next() {
                                let end = indices.peek().map_or(word.len(), |&(j, _)| j);
                                tokens.push(Self::make_token_incremental(&word[i..end], base, &mut unicode_offset));
                            }
                        } else {
                            tokens.push(Self::make_token_incremental(word, base, &mut unicode_offset));
                        }
                    }
                }
            }
        }
        tokens
    }

    /// Cut the input text
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<Token<'a>> {
        self.cut_internal(sentence, false, hmm)
    }

    /// Cut the input text, return all possible words
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    pub fn cut_all<'a>(&self, sentence: &'a str) -> Vec<Token<'a>> {
        self.cut_internal(sentence, true, false)
    }

    /// Cut the input text in search mode
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut_for_search<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<Token<'a>> {
        let words = self.cut(sentence, hmm);
        let mut new_words = Vec::with_capacity(words.len());
        let base = sentence.as_ptr() as usize;
        for token in words {
            let word = token.word;
            let char_indices: Vec<usize> = word.char_indices().map(|x| x.0).collect();
            let char_count = char_indices.len();
            if char_count > 2 {
                for i in 0..char_count - 1 {
                    let local_byte_start = char_indices[i];
                    let gram2 = if i + 2 < char_count {
                        &word[local_byte_start..char_indices[i + 2]]
                    } else {
                        &word[local_byte_start..]
                    };
                    if self.cedar.exact_match_search(gram2).is_some() {
                        let byte_start = gram2.as_ptr() as usize - base;
                        let byte_end = byte_start + gram2.len();
                        new_words.push(Token {
                            word: gram2,
                            start: token.start + i,
                            end: token.start + i + 2,
                            byte_start,
                            byte_end,
                        });
                    }
                }
            }
            if char_count > 3 {
                for i in 0..char_count - 2 {
                    let local_byte_start = char_indices[i];
                    let gram3 = if i + 3 < char_count {
                        &word[local_byte_start..char_indices[i + 3]]
                    } else {
                        &word[local_byte_start..]
                    };
                    if self.cedar.exact_match_search(gram3).is_some() {
                        let byte_start = gram3.as_ptr() as usize - base;
                        let byte_end = byte_start + gram3.len();
                        new_words.push(Token {
                            word: gram3,
                            start: token.start + i,
                            end: token.start + i + 3,
                            byte_start,
                            byte_end,
                        });
                    }
                }
            }
            new_words.push(token);
        }
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
        let tokens = self.cut(sentence, hmm);
        tokens
            .into_iter()
            .map(|token| {
                let word = token.word;
                if let Some((word_id, _, _)) = self.cedar.exact_match_search(word) {
                    let t = &self.records[word_id as usize].tag;
                    return Tag {
                        word,
                        tag: t,
                        start: token.start,
                        end: token.end,
                        byte_start: token.byte_start,
                        byte_end: token.byte_end,
                    };
                }
                let tag = self.guess_tag(word);
                Tag {
                    word,
                    tag,
                    start: token.start,
                    end: token.end,
                    byte_start: token.byte_start,
                    byte_end: token.byte_end,
                }
            })
            .collect()
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
            if word.chars().any(|c| is_cjk(c)) {
                let results = posseg::cut_with_pos(word);
                if results.len() == 1 {
                    return results[0].1;
                }
                if let Some((_w, tag)) = results.iter().max_by_key(|(w, _)| w.len()) {
                    return tag;
                }
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
