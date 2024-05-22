//! The Jieba Chinese Word Segmentation Implemented in Rust
//!
//! ## Installation
//!
//! Add it to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! jieba-rs = "0.6"
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

use include_flate::flate;
use lazy_static::lazy_static;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::BufRead;

use cedarwood::Cedar;
use regex::{Match, Matches, Regex};

pub(crate) type FxHashMap<K, V> = HashMap<K, V, fxhash::FxBuildHasher>;

pub use crate::errors::Error;
#[cfg(feature = "textrank")]
pub use crate::keywords::textrank::TextRank;
#[cfg(feature = "tfidf")]
pub use crate::keywords::tfidf::TfIdf;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
pub use crate::keywords::{Keyword, KeywordExtract, KeywordExtractConfig, DEFAULT_STOP_WORDS};

mod errors;
mod hmm;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
mod keywords;
mod sparse_dag;

#[cfg(feature = "default-dict")]
flate!(static DEFAULT_DICT: str from "src/data/dict.txt");

use sparse_dag::StaticSparseDAG;

lazy_static! {
    static ref RE_HAN_DEFAULT: Regex = Regex::new(r"([\u{3400}-\u{4DBF}\u{4E00}-\u{9FFF}\u{F900}-\u{FAFF}\u{20000}-\u{2A6DF}\u{2A700}-\u{2B73F}\u{2B740}-\u{2B81F}\u{2B820}-\u{2CEAF}\u{2CEB0}-\u{2EBEF}\u{2F800}-\u{2FA1F}a-zA-Z0-9+#&\._%\-]+)").unwrap();
    static ref RE_SKIP_DEFAULT: Regex = Regex::new(r"(\r\n|\s)").unwrap();
    static ref RE_HAN_CUT_ALL: Regex = Regex::new(r"([\u{3400}-\u{4DBF}\u{4E00}-\u{9FFF}\u{F900}-\u{FAFF}\u{20000}-\u{2A6DF}\u{2A700}-\u{2B73F}\u{2B740}-\u{2B81F}\u{2B820}-\u{2CEAF}\u{2CEB0}-\u{2EBEF}\u{2F800}-\u{2FA1F}]+)").unwrap();
    static ref RE_SKIP_CUT_ALL: Regex = Regex::new(r"[^a-zA-Z0-9+#\n]").unwrap();
}

struct SplitMatches<'r, 't> {
    finder: Matches<'r, 't>,
    text: &'t str,
    last: usize,
    matched: Option<Match<'t>>,
}

impl<'r, 't> SplitMatches<'r, 't> {
    #[inline]
    fn new(re: &'r Regex, text: &'t str) -> SplitMatches<'r, 't> {
        SplitMatches {
            finder: re.find_iter(text),
            text,
            last: 0,
            matched: None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum SplitState<'t> {
    Unmatched(&'t str),
    Matched(Match<'t>),
}

impl<'t> SplitState<'t> {
    #[inline]
    fn into_str(self) -> &'t str {
        match self {
            SplitState::Unmatched(t) => t,
            SplitState::Matched(matched) => matched.as_str(),
        }
    }
}

impl<'r, 't> Iterator for SplitMatches<'r, 't> {
    type Item = SplitState<'t>;

    fn next(&mut self) -> Option<SplitState<'t>> {
        if let Some(matched) = self.matched.take() {
            return Some(SplitState::Matched(matched));
        }
        match self.finder.next() {
            None => {
                if self.last >= self.text.len() {
                    None
                } else {
                    let s = &self.text[self.last..];
                    self.last = self.text.len();
                    Some(SplitState::Unmatched(s))
                }
            }
            Some(m) => {
                if self.last == m.start() {
                    self.last = m.end();
                    Some(SplitState::Matched(m))
                } else {
                    let unmatched = &self.text[self.last..m.start()];
                    self.last = m.end();
                    self.matched = Some(m);
                    Some(SplitState::Unmatched(unmatched))
                }
            }
        }
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
}

/// A tagged word
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag<'a> {
    /// Word
    pub word: &'a str,
    /// Word tag
    pub tag: &'a str,
}

#[derive(Debug, Clone)]
struct Record {
    freq: usize,
    tag: String,
}

impl Record {
    #[inline(always)]
    fn new(freq: usize, tag: String) -> Self {
        Self { freq, tag }
    }
}

/// Jieba segmentation
#[derive(Debug, Clone)]
pub struct Jieba {
    records: Vec<Record>,
    cedar: Cedar,
    total: usize,
    longest_word_len: usize,
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
            longest_word_len: 0,
        }
    }

    /// Create a new instance with embed dict
    ///
    /// Requires `default-dict` feature to be enabled.
    #[cfg(feature = "default-dict")]
    pub fn new() -> Self {
        use std::io::BufReader;

        let mut instance = Self::empty();
        let mut default_dict = BufReader::new(DEFAULT_DICT.as_bytes());
        instance.load_dict(&mut default_dict).unwrap();
        instance
    }

    /// Create a new instance with dict
    pub fn with_dict<R: BufRead>(dict: &mut R) -> Result<Self, Error> {
        let mut instance = Self::empty();
        instance.load_dict(dict)?;
        Ok(instance)
    }
    /// Add word to dict, return `freq`
    ///
    /// `freq`: if `None`, will be given by [suggest_freq](#method.suggest_freq)
    ///
    /// `tag`: if `None`, will be given `""`
    pub fn add_word(&mut self, word: &str, freq: Option<usize>, tag: Option<&str>) -> usize {
        let freq = freq.unwrap_or_else(|| self.suggest_freq(word));
        let tag = tag.unwrap_or("");

        match self.cedar.exact_match_search(word) {
            Some((word_id, _, _)) => {
                let old_freq = self.records[word_id as usize].freq;
                self.records[word_id as usize].freq = freq;

                self.total += freq;
                self.total -= old_freq;
            }
            None => {
                self.records.push(Record::new(freq, String::from(tag)));
                let word_id = (self.records.len() - 1) as i32;

                self.cedar.update(word, word_id);
                self.total += freq;
            }
        };

        let curr_word_len = word.chars().count();
        if self.longest_word_len < curr_word_len {
            self.longest_word_len = curr_word_len;
        }

        freq
    }

    /// Load dictionary
    pub fn load_dict<R: BufRead>(&mut self, dict: &mut R) -> Result<(), Error> {
        let mut buf = String::new();
        self.total = 0;
        self.longest_word_len = 0;

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
                                    "line {} `{}` frequency {} is not a valid integer: {}",
                                    line_no, buf, x, e
                                ))
                            })
                        })
                        .unwrap_or(Ok(0))?;
                    let tag = iter.next().unwrap_or("");

                    let curr_word_len = word.chars().count();
                    if self.longest_word_len < curr_word_len {
                        self.longest_word_len = curr_word_len;
                    }

                    match self.cedar.exact_match_search(word) {
                        Some((word_id, _, _)) => {
                            self.records[word_id as usize].freq = freq;
                        }
                        None => {
                            self.records.push(Record::new(freq, String::from(tag)));
                            let word_id = (self.records.len() - 1) as i32;
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
        let logfreq = self.cut(segment, false).iter().fold(0f64, |freq, word| {
            freq + (self.get_word_freq(word, 1) as f64).ln() - logtotal
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
        let mut prev_byte_start = str_len;
        let curr = sentence.char_indices().map(|x| x.0).rev();
        for byte_start in curr {
            let pair = dag
                .iter_edges(byte_start)
                .map(|byte_end| {
                    let wfrag = if byte_end == str_len {
                        &sentence[byte_start..]
                    } else {
                        &sentence[byte_start..byte_end]
                    };

                    let freq = if let Some((word_id, _, _)) = self.cedar.exact_match_search(wfrag) {
                        self.records[word_id as usize].freq
                    } else {
                        1
                    };

                    ((freq as f64).ln() - logtotal + route[byte_end].0, byte_end)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));

            if let Some(p) = pair {
                route[byte_start] = p;
            } else {
                let byte_end = prev_byte_start;
                let freq = 1;
                route[byte_start] = ((freq as f64).ln() - logtotal + route[byte_end].0, byte_end);
            }

            prev_byte_start = byte_start;
        }
    }

    fn dag(&self, sentence: &str, dag: &mut StaticSparseDAG) {
        for (byte_start, _) in sentence.char_indices().peekable() {
            dag.start(byte_start);
            let haystack = &sentence[byte_start..];

            for (_, end_index) in self.cedar.common_prefix_iter(haystack) {
                dag.insert(end_index + byte_start + 1);
            }

            dag.commit();
        }
    }

    fn cut_all_internal<'a>(&self, sentence: &'a str, words: &mut Vec<&'a str>) {
        let str_len = sentence.len();
        let mut dag = StaticSparseDAG::with_size_hint(sentence.len());
        self.dag(sentence, &mut dag);

        let curr = sentence.char_indices().map(|x| x.0);
        for byte_start in curr {
            for byte_end in dag.iter_edges(byte_start) {
                let word = if byte_end == str_len {
                    &sentence[byte_start..]
                } else {
                    &sentence[byte_start..byte_end]
                };

                words.push(word)
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
            let l_str = if y < sentence.len() {
                &sentence[x..y]
            } else {
                &sentence[x..]
            };

            if l_str.chars().count() == 1 && l_str.chars().all(|ch| ch.is_ascii_alphanumeric()) {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(byte_start) = left {
                    let word = &sentence[byte_start..x];
                    words.push(word);
                    left = None;
                }

                let word = if y < sentence.len() {
                    &sentence[x..y]
                } else {
                    &sentence[x..]
                };

                words.push(word);
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

            if sentence[x..y].chars().count() == 1 {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(byte_start) = left {
                    let byte_end = x;
                    let word = if byte_end < sentence.len() {
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };

                    if word.chars().count() == 1 {
                        words.push(word);
                    } else if self.cedar.exact_match_search(word).is_none() {
                        hmm::cut_with_allocated_memory(word, words, hmm_context);
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
                let word = if y < sentence.len() {
                    &sentence[x..y]
                } else {
                    &sentence[x..]
                };
                words.push(word);
            }
            x = y;
        }

        if let Some(byte_start) = left {
            let word = &sentence[byte_start..];

            if word.chars().count() == 1 {
                words.push(word);
            } else if self.cedar.exact_match_search(word).is_none() {
                hmm::cut(word, words);
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

    #[allow(non_snake_case)]
    fn cut_internal<'a>(&self, sentence: &'a str, cut_all: bool, hmm: bool) -> Vec<&'a str> {
        let heuristic_capacity = sentence.len() / 2;
        let mut words = Vec::with_capacity(heuristic_capacity);
        let re_han: &Regex = if cut_all { &RE_HAN_CUT_ALL } else { &RE_HAN_DEFAULT };
        let re_skip: &Regex = if cut_all { &RE_SKIP_CUT_ALL } else { &RE_SKIP_DEFAULT };
        let splitter = SplitMatches::new(re_han, sentence);
        let mut route = Vec::with_capacity(heuristic_capacity);
        let mut dag = StaticSparseDAG::with_size_hint(heuristic_capacity);

        let mut hmm_context = hmm::HmmContext::new(sentence.chars().count());

        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());

                    if cut_all {
                        self.cut_all_internal(block, &mut words);
                    } else if hmm {
                        self.cut_dag_hmm(block, &mut words, &mut route, &mut dag, &mut hmm_context);
                    } else {
                        self.cut_dag_no_hmm(block, &mut words, &mut route, &mut dag);
                    }
                }
                SplitState::Unmatched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());

                    let skip_splitter = SplitMatches::new(re_skip, block);
                    for skip_state in skip_splitter {
                        let word = skip_state.into_str();
                        if word.is_empty() {
                            continue;
                        }
                        if cut_all || re_skip.is_match(word) {
                            words.push(word);
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
                }
            }
        }
        words
    }

    /// Cut the input text
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        self.cut_internal(sentence, false, hmm)
    }

    /// Cut the input text, return all possible words
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    pub fn cut_all<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        self.cut_internal(sentence, true, false)
    }

    /// Cut the input text in search mode
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn cut_for_search<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        let words = self.cut(sentence, hmm);
        let mut new_words = Vec::with_capacity(words.len());
        for word in words {
            let char_indices: Vec<usize> = word.char_indices().map(|x| x.0).collect();
            let char_count = char_indices.len();
            if char_count > 2 {
                for i in 0..char_count - 1 {
                    let byte_start = char_indices[i];
                    let gram2 = if i + 2 < char_count {
                        &word[byte_start..char_indices[i + 2]]
                    } else {
                        &word[byte_start..]
                    };
                    if self.cedar.exact_match_search(gram2).is_some() {
                        new_words.push(gram2);
                    }
                }
            }
            if char_count > 3 {
                for i in 0..char_count - 2 {
                    let byte_start = char_indices[i];
                    let gram3 = if i + 3 < char_count {
                        &word[byte_start..char_indices[i + 3]]
                    } else {
                        &word[byte_start..]
                    };
                    if self.cedar.exact_match_search(gram3).is_some() {
                        new_words.push(gram3);
                    }
                }
            }
            new_words.push(word);
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
        let words = self.cut(sentence, hmm);
        let mut tokens = Vec::with_capacity(words.len());
        let mut start = 0;
        match mode {
            TokenizeMode::Default => {
                for word in words {
                    let width = word.chars().count();
                    tokens.push(Token {
                        word,
                        start,
                        end: start + width,
                    });
                    start += width;
                }
            }
            TokenizeMode::Search => {
                for word in words {
                    let width = word.chars().count();
                    if width > 2 {
                        let char_indices: Vec<usize> = word.char_indices().map(|x| x.0).collect();
                        for i in 0..width - 1 {
                            let byte_start = char_indices[i];
                            let gram2 = if i + 2 < width {
                                &word[byte_start..char_indices[i + 2]]
                            } else {
                                &word[byte_start..]
                            };
                            if self.cedar.exact_match_search(gram2).is_some() {
                                tokens.push(Token {
                                    word: gram2,
                                    start: start + i,
                                    end: start + i + 2,
                                });
                            }
                        }
                        if width > 3 {
                            for i in 0..width - 2 {
                                let byte_start = char_indices[i];
                                let gram3 = if i + 3 < width {
                                    &word[byte_start..char_indices[i + 3]]
                                } else {
                                    &word[byte_start..]
                                };
                                if self.cedar.exact_match_search(gram3).is_some() {
                                    tokens.push(Token {
                                        word: gram3,
                                        start: start + i,
                                        end: start + i + 3,
                                    });
                                }
                            }
                        }
                    }
                    tokens.push(Token {
                        word,
                        start,
                        end: start + width,
                    });
                    start += width;
                }
            }
        }
        tokens
    }

    /// Tag the input text
    ///
    /// ## Params
    ///
    /// `sentence`: input text
    ///
    /// `hmm`: enable HMM or not
    pub fn tag<'a>(&'a self, sentence: &'a str, hmm: bool) -> Vec<Tag> {
        let words = self.cut(sentence, hmm);
        words
            .into_iter()
            .map(|word| {
                if let Some((word_id, _, _)) = self.cedar.exact_match_search(word) {
                    let t = &self.records[word_id as usize].tag;
                    return Tag { word, tag: t };
                }
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
                let tag = if eng == 0 {
                    "x"
                } else if eng == m {
                    "m"
                } else {
                    "eng"
                };
                Tag { word, tag }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Jieba, SplitMatches, SplitState, Tag, Token, TokenizeMode, RE_HAN_DEFAULT};
    use std::io::BufReader;

    #[test]
    fn test_init_with_default_dict() {
        let _ = Jieba::new();
    }

    #[test]
    fn test_split_matches() {
        let re_han = &*RE_HAN_DEFAULT;
        let splitter = SplitMatches::new(
            re_han,
            "👪 PS: 我觉得开源有一个好处，就是能够敦促自己不断改进 👪，避免敞帚自珍",
        );
        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());
                }
                SplitState::Unmatched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_split_matches_against_unicode_sip() {
        let re_han = &*RE_HAN_DEFAULT;
        let splitter = SplitMatches::new(re_han, "讥䶯䶰䶱䶲䶳䶴䶵𦡦");

        let result: Vec<&str> = splitter.map(|x| x.into_str()).collect();
        assert_eq!(result, vec!["讥䶯䶰䶱䶲䶳䶴䶵𦡦"]);
    }

    #[test]
    fn test_cut_all() {
        let jieba = Jieba::new();
        let words = jieba.cut_all("abc网球拍卖会def");
        assert_eq!(
            words,
            vec![
                "abc",
                "网",
                "网球",
                "网球拍",
                "球",
                "球拍",
                "拍",
                "拍卖",
                "拍卖会",
                "卖",
                "会",
                "def"
            ]
        );

        // The cut_all from the python de-facto implementation is loosely defined,
        // And the answer "我, 来到, 北京, 清华, 清华大学, 华大, 大学" from the python implementation looks weird since it drops the single character word even though it is part of the DAG candidates.
        // For example, it includes "华大" but it doesn't include "清" and "学"
        let words = jieba.cut_all("我来到北京清华大学");
        assert_eq!(
            words,
            vec![
                "我",
                "来",
                "来到",
                "到",
                "北",
                "北京",
                "京",
                "清",
                "清华",
                "清华大学",
                "华",
                "华大",
                "大",
                "大学",
                "学"
            ]
        );
    }

    #[test]
    fn test_cut_no_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut("abc网球拍卖会def", false);
        assert_eq!(words, vec!["abc", "网球", "拍卖会", "def"]);
    }

    #[test]
    fn test_cut_with_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut("我们中出了一个叛徒", false);
        assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
        let words = jieba.cut("我们中出了一个叛徒", true);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒"]);
        let words = jieba.cut("我们中出了一个叛徒👪", true);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒", "👪"]);

        let words = jieba.cut("我来到北京清华大学", true);
        assert_eq!(words, vec!["我", "来到", "北京", "清华大学"]);

        let words = jieba.cut("他来到了网易杭研大厦", true);
        assert_eq!(words, vec!["他", "来到", "了", "网易", "杭研", "大厦"]);
    }

    #[test]
    fn test_cut_weicheng() {
        static WEICHENG_TXT: &str = include_str!("../examples/weicheng/src/weicheng.txt");
        let jieba = Jieba::new();
        for line in WEICHENG_TXT.split('\n') {
            let _ = jieba.cut(line, true);
        }
    }

    #[test]
    fn test_cut_for_search() {
        let jieba = Jieba::new();
        let words = jieba.cut_for_search("南京市长江大桥", true);
        assert_eq!(words, vec!["南京", "京市", "南京市", "长江", "大桥", "长江大桥"]);

        let words = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", true);

        // The python implementation silently filtered "，". but we includes it here in the output
        // to let the library user to decide their own filtering strategy
        assert_eq!(
            words,
            vec![
                "小明",
                "硕士",
                "毕业",
                "于",
                "中国",
                "科学",
                "学院",
                "科学院",
                "中国科学院",
                "计算",
                "计算所",
                "，",
                "后",
                "在",
                "日本",
                "京都",
                "大学",
                "日本京都大学",
                "深造"
            ]
        );
    }

    #[test]
    fn test_tag() {
        let jieba = Jieba::new();
        let tags = jieba.tag(
            "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            true,
        );
        assert_eq!(
            tags,
            vec![
                Tag { word: "我", tag: "r" },
                Tag { word: "是", tag: "v" },
                Tag {
                    word: "拖拉机",
                    tag: "n"
                },
                Tag {
                    word: "学院", tag: "n"
                },
                Tag {
                    word: "手扶拖拉机",
                    tag: "n"
                },
                Tag {
                    word: "专业", tag: "n"
                },
                Tag { word: "的", tag: "uj" },
                Tag { word: "。", tag: "x" },
                Tag {
                    word: "不用", tag: "v"
                },
                Tag {
                    word: "多久", tag: "m"
                },
                Tag { word: "，", tag: "x" },
                Tag { word: "我", tag: "r" },
                Tag { word: "就", tag: "d" },
                Tag { word: "会", tag: "v" },
                Tag {
                    word: "升职", tag: "v"
                },
                Tag {
                    word: "加薪",
                    tag: "nr"
                },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "当上", tag: "t"
                },
                Tag {
                    word: "CEO",
                    tag: "eng"
                },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "走上", tag: "v"
                },
                Tag {
                    word: "人生", tag: "n"
                },
                Tag {
                    word: "巅峰", tag: "n"
                },
                Tag { word: "。", tag: "x" }
            ]
        );

        let tags = jieba.tag("今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。", true);
        assert_eq!(
            tags,
            vec![
                Tag {
                    word: "今天", tag: "t"
                },
                Tag {
                    word: "纽约",
                    tag: "ns"
                },
                Tag { word: "的", tag: "uj" },
                Tag {
                    word: "天气", tag: "n"
                },
                Tag {
                    word: "真好", tag: "d"
                },
                Tag { word: "啊", tag: "zg" },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "京华",
                    tag: "nz"
                },
                Tag {
                    word: "大酒店",
                    tag: "n"
                },
                Tag { word: "的", tag: "uj" },
                Tag {
                    word: "张尧", tag: "x"
                }, // XXX: missing in dict
                Tag {
                    word: "经理", tag: "n"
                },
                Tag { word: "吃", tag: "v" },
                Tag { word: "了", tag: "ul" },
                Tag {
                    word: "一只", tag: "m"
                },
                Tag {
                    word: "北京烤鸭",
                    tag: "n"
                },
                Tag { word: "。", tag: "x" }
            ]
        );
    }

    #[test]
    fn test_tokenize() {
        let jieba = Jieba::new();
        let tokens = jieba.tokenize("南京市长江大桥", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "南京市",
                    start: 0,
                    end: 3
                },
                Token {
                    word: "长江大桥",
                    start: 3,
                    end: 7
                }
            ]
        );

        let tokens = jieba.tokenize("南京市长江大桥", TokenizeMode::Search, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "南京",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "京市",
                    start: 1,
                    end: 3
                },
                Token {
                    word: "南京市",
                    start: 0,
                    end: 3
                },
                Token {
                    word: "长江",
                    start: 3,
                    end: 5
                },
                Token {
                    word: "大桥",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "长江大桥",
                    start: 3,
                    end: 7
                }
            ]
        );

        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3
                },
                Token {
                    word: "出",
                    start: 3,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );

        let tokens = jieba.tokenize("永和服装饰品有限公司", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "永和",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "服装",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "饰品",
                    start: 4,
                    end: 6
                },
                Token {
                    word: "有限公司",
                    start: 6,
                    end: 10
                }
            ]
        );
    }

    #[test]
    fn test_userdict() {
        let mut jieba = Jieba::new();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3
                },
                Token {
                    word: "出",
                    start: 3,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
        let userdict = "中出 10000";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, false);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
    }

    #[test]
    fn test_userdict_hmm() {
        let mut jieba = Jieba::new();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中出",
                    start: 2,
                    end: 4
                },
                Token {
                    word: "了",
                    start: 4,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
        let userdict = "出了 10000";
        jieba.load_dict(&mut BufReader::new(userdict.as_bytes())).unwrap();
        let tokens = jieba.tokenize("我们中出了一个叛徒", TokenizeMode::Default, true);
        assert_eq!(
            tokens,
            vec![
                Token {
                    word: "我们",
                    start: 0,
                    end: 2
                },
                Token {
                    word: "中",
                    start: 2,
                    end: 3
                },
                Token {
                    word: "出了",
                    start: 3,
                    end: 5
                },
                Token {
                    word: "一个",
                    start: 5,
                    end: 7
                },
                Token {
                    word: "叛徒",
                    start: 7,
                    end: 9
                }
            ]
        );
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
        let words = jieba.cut("测试", false);
        assert_eq!(words, vec!["测试"]);
    }

    #[test]
    fn test_cut_dag_no_hmm_against_string_with_sip() {
        let mut jieba = Jieba::empty();

        //add fake word into dictionary
        jieba.add_word("䶴䶵𦡦", Some(1000), None);
        jieba.add_word("讥䶯䶰䶱䶲䶳", Some(1000), None);

        let words = jieba.cut("讥䶯䶰䶱䶲䶳䶴䶵𦡦", false);
        assert_eq!(words, vec!["讥䶯䶰䶱䶲䶳", "䶴䶵𦡦"]);
    }

    #[test]
    fn test_add_custom_word_with_underscrore() {
        let mut jieba = Jieba::empty();
        jieba.add_word("田-女士", Some(42), Some("n"));
        let words = jieba.cut("市民田-女士急匆匆", false);
        assert_eq!(words, vec!["市", "民", "田-女士", "急", "匆", "匆"]);
    }
}
