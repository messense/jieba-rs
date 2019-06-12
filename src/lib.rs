//! The Jieba Chinese Word Segmentation Implemented in Rust
//!
//! ## Installation
//!
//! Add it to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! jieba-rs = "0.3"
//! ```
//!
//! then you are good to go. If you are using Rust 2015 you have to ``extern crate jieba_rs`` to your crate root as well. 
//!
//! ## Example
//!
//! ```rust
//! use jieba_rs::Jieba;
//!
//! fn main() {
//!     let jieba = Jieba::new();
//!     let words = jieba.cut("我们中出了一个叛徒", false);
//!     assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
//! }
//! ```
//!
//! ```rust
//! use jieba_rs::Jieba;
//! use jieba_rs::{TFIDF, KeywordExtract};
//!
//! fn main() {
//!     let jieba = Jieba::new();
//!     let keyword_extractor = TFIDF::new_with_jieba(&jieba);
//!     let top_k = keyword_extractor.extract_tags(
//!         "今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。后天纽约的天气不好，昨天纽约的天气也不好，北京烤鸭真好吃",
//!         3,
//!         vec![],
//!     );
//!     assert_eq!(top_k, vec!["北京烤鸭", "纽约", "天气"]);
//! }
//! ```
//!
//! ```rust
//! use jieba_rs::Jieba;
//! use jieba_rs::{TextRank, KeywordExtract};
//!
//! fn main() {
//!     let jieba = Jieba::new();
//!     let keyword_extractor = TextRank::new_with_jieba(&jieba);
//!     let top_k = keyword_extractor.extract_tags(
//!         "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
//!         6,
//!         vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
//!     );
//!     assert_eq!(top_k, vec!["吉林", "欧亚", "置业", "实现", "收入", "增资"]);
//! }
//! ```
//!
//! ## Enabling Additional Features
//!
//! * `tfidf` feature enables TF-IDF keywords extractor
//! * `textrank` feature enables TextRank keywords extractor
//!
//! ```toml
//! [dependencies]
//! jieba-rs = { version = "0.3", features = ["tfidf", "textrank"] }
//! ```
//!


use lazy_static::lazy_static;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader};

use regex::{Match, Matches, Regex};
use smallvec::SmallVec;

#[cfg(feature = "textrank")]
pub use crate::keywords::textrank::TextRank;
#[cfg(feature = "tfidf")]
pub use crate::keywords::tfidf::TFIDF;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
pub use crate::keywords::KeywordExtract;

mod hmm;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
mod keywords;

static DEFAULT_DICT: &str = include_str!("data/dict.txt");

type DAG = BTreeMap<usize, SmallVec<[usize; 5]>>;

lazy_static! {
    static ref RE_HAN_DEFAULT: Regex = Regex::new(r"([\u{4E00}-\u{9FD5}a-zA-Z0-9+#&\._%]+)").unwrap();
    static ref RE_SKIP_DEAFULT: Regex = Regex::new(r"(\r\n|\s)").unwrap();
    static ref RE_HAN_CUT_ALL: Regex = Regex::new("([\u{4E00}-\u{9FD5}]+)").unwrap();
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

/// Jieba segmentation
#[derive(Debug)]
pub struct Jieba {
    dict: hashbrown::HashMap<String, (usize, String)>,
    total: usize,
}

impl Default for Jieba {
    fn default() -> Self {
        Jieba::new()
    }
}

impl Jieba {
    /// Create a new instance with empty dict
    pub fn empty() -> Self {
        Jieba {
            dict: hashbrown::HashMap::new(),
            total: 0,
        }
    }

    /// Create a new instance with embed dict
    pub fn new() -> Self {
        let mut instance = Self::empty();
        let mut default_dict = BufReader::new(DEFAULT_DICT.as_bytes());
        instance.load_dict(&mut default_dict).unwrap();
        instance
    }

    /// Create a new instance with dict
    pub fn with_dict<R: BufRead>(dict: &mut R) -> io::Result<Self> {
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

        self.dict.insert(word.to_string(), (freq, tag.to_string()));
        let char_indices = word.char_indices().map(|x| x.0).collect::<Vec<_>>();
        for index in char_indices.into_iter().skip(1) {
            let wfrag = &word[0..index];
            self.dict.entry(wfrag.to_string()).or_insert((0, "".to_string()));
        }

        self.total += freq;

        freq
    }

    /// Load dictionary
    pub fn load_dict<R: BufRead>(&mut self, dict: &mut R) -> io::Result<()> {
        let mut buf = String::new();
        while dict.read_line(&mut buf)? > 0 {
            {
                let parts: Vec<&str> = buf.trim().split_whitespace().collect();
                if parts.is_empty() {
                    // Skip empty lines
                    continue;
                }

                let word = parts[0];
                let freq = parts.get(1).map(|x| x.parse::<usize>().unwrap());
                let tag = parts.get(2).cloned();

                self.add_word(word, freq, tag);
            }
            buf.clear();
        }
        Ok(())
    }

    fn get_word_freq(&self, word: &str, default: usize) -> usize {
        match self.dict.get(word) {
            Some(e) => match *e {
                (freq, _) => freq,
            },
            _ => default,
        }
    }

    /// Suggest word frequency to force the characters in a word to be joined or splitted.
    pub fn suggest_freq(&self, segment: &str) -> usize {
        let logtotal = (self.total as f64).ln();
        let logfreq = self.cut(segment, false).iter().fold(0f64, |freq, word| {
            freq + (self.get_word_freq(word, 1) as f64).ln() - logtotal
        });
        std::cmp::max((logfreq + logtotal).exp() as usize + 1, self.get_word_freq(segment, 1))
    }

    fn calc(&self, sentence: &str, char_indices: &[usize], dag: &DAG) -> Vec<(f64, usize)> {
        let word_count = char_indices.len();
        let mut route = Vec::with_capacity(word_count + 1);
        for _ in 0..=word_count {
            route.push((0.0, 0));
        }
        let logtotal = (self.total as f64).ln();
        for i in (0..word_count).rev() {
            let pair = dag[&i]
                .iter()
                .map(|x| {
                    let byte_start = char_indices[i];
                    let end_index = x + 1;
                    let byte_end = if end_index < char_indices.len() {
                        char_indices[end_index]
                    } else {
                        sentence.len()
                    };
                    let wfrag = &sentence[byte_start..byte_end];
                    let freq = self.dict.get(wfrag).map(|x| x.0).unwrap_or(1);
                    ((freq as f64).ln() - logtotal + route[x + 1].0, *x)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
            route[i] = pair.unwrap();
        }
        route
    }

    // FIXME: Use a proper DAG impl?
    fn dag(&self, sentence: &str, char_indices: &[usize]) -> DAG {
        let mut dag = BTreeMap::new();
        let word_count = char_indices.len();
        for (k, &byte_start) in char_indices.iter().enumerate() {
            let mut tmplist = SmallVec::new();
            let mut i = k;
            let mut wfrag = if k + 1 < char_indices.len() {
                &sentence[byte_start..char_indices[k + 1]]
            } else {
                &sentence[byte_start..]
            };
            while i < word_count {
                if let Some(freq) = self.dict.get(wfrag).map(|x| x.0) {
                    if freq > 0 {
                        tmplist.push(i);
                    }
                    i += 1;
                    wfrag = if i + 1 < word_count {
                        let byte_end = char_indices[i + 1];
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };
                } else {
                    break;
                }
            }
            if tmplist.is_empty() {
                tmplist.push(k);
            }
            dag.insert(k, tmplist);
        }
        dag
    }

    fn cut_all_internal<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = self.dag(sentence, &char_indices);
        let mut words = Vec::with_capacity(char_indices.len() / 2);
        let mut old_j = -1;
        for (k, list) in dag.into_iter() {
            if list.len() == 1 && k as isize > old_j {
                let byte_start = char_indices[k];
                let end_index = list[0] + 1;
                let byte_end = if end_index < char_indices.len() {
                    char_indices[end_index]
                } else {
                    sentence.len()
                };
                words.push(&sentence[byte_start..byte_end]);
                old_j = list[0] as isize;
            } else {
                for j in list.into_iter() {
                    if j > k {
                        let byte_start = char_indices[k];
                        let end_index = j + 1;
                        let byte_end = if end_index < char_indices.len() {
                            char_indices[end_index]
                        } else {
                            sentence.len()
                        };
                        words.push(&sentence[byte_start..byte_end]);
                        old_j = j as isize;
                    }
                }
            }
        }
        words
    }

    fn cut_dag_no_hmm<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = self.dag(sentence, &char_indices);
        let route = self.calc(sentence, &char_indices, &dag);
        let mut words = Vec::with_capacity(char_indices.len() / 2);
        let mut x = 0;
        let mut buf_indices = Vec::new();
        while x < char_indices.len() {
            let y = route[x].1 + 1;
            let l_indices = &char_indices[x..y];
            let l_str = if y < char_indices.len() {
                &sentence[char_indices[x]..char_indices[y]]
            } else {
                &sentence[char_indices[x]..]
            };
            if l_indices.len() == 1 && l_str.chars().all(|ch| ch.is_ascii_alphanumeric()) {
                buf_indices.push(x);
            } else {
                if !buf_indices.is_empty() {
                    let byte_start = char_indices[buf_indices[0]];
                    let end_index = buf_indices[buf_indices.len() - 1] + 1;
                    let word = if end_index < char_indices.len() {
                        let byte_end = char_indices[end_index];
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };
                    words.push(word);
                    buf_indices.clear();
                }
                let word = if y < char_indices.len() {
                    &sentence[char_indices[x]..char_indices[y]]
                } else {
                    &sentence[char_indices[x]..]
                };
                words.push(word);
            }
            x = y;
        }
        if !buf_indices.is_empty() {
            let byte_start = char_indices[buf_indices[0]];
            let end_index = buf_indices[buf_indices.len() - 1] + 1;
            let word = if end_index < char_indices.len() {
                let byte_end = char_indices[end_index];
                &sentence[byte_start..byte_end]
            } else {
                &sentence[byte_start..]
            };
            words.push(word);
            buf_indices.clear();
        }
        words
    }

    fn cut_dag_hmm<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = self.dag(sentence, &char_indices);
        let route = self.calc(sentence, &char_indices, &dag);
        let mut words = Vec::with_capacity(char_indices.len() / 2);
        let mut x = 0;
        let mut buf_indices = Vec::new();
        while x < char_indices.len() {
            let y = route[x].1 + 1;
            let l_indices = &char_indices[x..y];
            if l_indices.len() == 1 {
                buf_indices.push(x);
            } else {
                if !buf_indices.is_empty() {
                    let byte_start = char_indices[buf_indices[0]];
                    let end_index = buf_indices[buf_indices.len() - 1] + 1;
                    let word = if end_index < char_indices.len() {
                        let byte_end = char_indices[end_index];
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };
                    if buf_indices.len() == 1 {
                        words.push(word);
                    } else if !self.dict.get(word).map(|x| x.0 > 0).unwrap_or(false) {
                        words.extend(hmm::cut(word));
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
                    buf_indices.clear();
                }
                let word = if y < char_indices.len() {
                    &sentence[char_indices[x]..char_indices[y]]
                } else {
                    &sentence[char_indices[x]..]
                };
                words.push(word);
            }
            x = y;
        }
        if !buf_indices.is_empty() {
            let byte_start = char_indices[buf_indices[0]];
            let end_index = buf_indices[buf_indices.len() - 1] + 1;
            let word = if end_index < char_indices.len() {
                let byte_end = char_indices[end_index];
                &sentence[byte_start..byte_end]
            } else {
                &sentence[byte_start..]
            };
            if buf_indices.len() == 1 {
                words.push(word);
            } else if !self.dict.get(word).map(|x| x.0 > 0).unwrap_or(false) {
                words.extend(hmm::cut(word));
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
            buf_indices.clear();
        }
        words
    }

    fn cut_internal<'a>(&self, sentence: &'a str, cut_all: bool, hmm: bool) -> Vec<&'a str> {
        let mut words = Vec::new();
        let re_han: &Regex = if cut_all { &*RE_HAN_CUT_ALL } else { &*RE_HAN_DEFAULT };
        let re_skip: &Regex = if cut_all { &*RE_SKIP_CUT_ALL } else { &*RE_SKIP_DEAFULT };
        let splitter = SplitMatches::new(&re_han, sentence);
        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());

                    if cut_all {
                        words.extend(self.cut_all_internal(block));
                    } else if hmm {
                        words.extend(self.cut_dag_hmm(block));
                    } else {
                        words.extend(self.cut_dag_no_hmm(block));
                    }
                }
                SplitState::Unmatched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());

                    let skip_splitter = SplitMatches::new(&re_skip, block);
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
                    if self.dict.get(gram2).map(|x| x.0 > 0).unwrap_or(false) {
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
                    if self.dict.get(gram3).map(|x| x.0 > 0).unwrap_or(false) {
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
                            if self.dict.get(gram2).map(|x| x.0 > 0).unwrap_or(false) {
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
                                if self.dict.get(gram3).map(|x| x.0 > 0).unwrap_or(false) {
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
                if let Some(tag) = self.dict.get(word) {
                    if tag.0 != 0 {
                        return Tag { word, tag: &tag.1 };
                    }
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
    use smallvec::SmallVec;
    use std::io::BufReader;

    #[test]
    fn test_init_with_default_dict() {
        let _ = Jieba::new();
    }

    #[test]
    fn test_split_matches() {
        let re_han = &*RE_HAN_DEFAULT;
        let splitter = SplitMatches::new(
            &re_han,
            "👪 PS: 我觉得开源有一个好处，就是能够敦促自己不断改进 👪，避免敞帚自珍",
        );
        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.into_str();
                    assert_eq!(block.is_empty(), false);
                }
                SplitState::Unmatched(_) => {
                    let block = state.into_str();
                    assert_eq!(block.is_empty(), false);
                }
            }
        }
    }

    #[test]
    fn test_dag() {
        let jieba = Jieba::new();
        let sentence = "网球拍卖会";
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = jieba.dag(sentence, &char_indices);
        assert_eq!(dag[&0], SmallVec::from_buf([0, 1, 2]));
        assert_eq!(dag[&1], SmallVec::from_buf([1, 2]));
        assert_eq!(dag[&2], SmallVec::from_buf([2, 3, 4]));
        assert_eq!(dag[&3], SmallVec::from_buf([3]));
        assert_eq!(dag[&4], SmallVec::from_buf([4]));
    }

    #[test]
    fn test_cut_all() {
        let jieba = Jieba::new();
        let words = jieba.cut_all("abc网球拍卖会def");
        assert_eq!(
            words,
            vec!["abc", "网球", "网球拍", "球拍", "拍卖", "拍卖会", "def"]
        );
    }

    #[test]
    fn test_cut_dag_no_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut_dag_no_hmm("网球拍卖会");
        assert_eq!(words, vec!["网球", "拍卖会"]);
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
        assert_eq!(
            words,
            vec!["南京", "京市", "南京市", "长江", "大桥", "长江大桥"]
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
                    word: "学院",
                    tag: "n"
                },
                Tag {
                    word: "手扶拖拉机",
                    tag: "n"
                },
                Tag {
                    word: "专业",
                    tag: "n"
                },
                Tag { word: "的", tag: "uj" },
                Tag { word: "。", tag: "x" },
                Tag {
                    word: "不用",
                    tag: "v"
                },
                Tag {
                    word: "多久",
                    tag: "m"
                },
                Tag { word: "，", tag: "x" },
                Tag { word: "我", tag: "r" },
                Tag { word: "就", tag: "d" },
                Tag { word: "会", tag: "v" },
                Tag {
                    word: "升职",
                    tag: "v"
                },
                Tag {
                    word: "加薪",
                    tag: "nr"
                },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "当上",
                    tag: "t"
                },
                Tag {
                    word: "CEO",
                    tag: "eng"
                },
                Tag { word: "，", tag: "x" },
                Tag {
                    word: "走上",
                    tag: "v"
                },
                Tag {
                    word: "人生",
                    tag: "n"
                },
                Tag {
                    word: "巅峰",
                    tag: "n"
                },
                Tag { word: "。", tag: "x" }
            ]
        );

        let tags = jieba.tag(
            "今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。",
            true,
        );
        assert_eq!(
            tags,
            vec![
                Tag {
                    word: "今天",
                    tag: "t"
                },
                Tag {
                    word: "纽约",
                    tag: "ns"
                },
                Tag { word: "的", tag: "uj" },
                Tag {
                    word: "天气",
                    tag: "n"
                },
                Tag {
                    word: "真好",
                    tag: "d"
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
                    word: "张尧",
                    tag: "x"
                }, // XXX: missing in dict
                Tag {
                    word: "经理",
                    tag: "n"
                },
                Tag { word: "吃", tag: "v" },
                Tag { word: "了", tag: "ul" },
                Tag {
                    word: "一只",
                    tag: "m"
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
}
