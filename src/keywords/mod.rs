use lazy_static::lazy_static;
use std::collections::BTreeSet;

use crate::Jieba;

#[cfg(feature = "textrank")]
pub mod textrank;
#[cfg(feature = "tfidf")]
pub mod tfidf;

lazy_static! {
    pub static ref DEFAULT_STOP_WORDS: BTreeSet<String> = {
        let mut set = BTreeSet::new();
        let words = [
            "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are", "by", "be", "as", "on", "with",
            "can", "if", "from", "which", "you", "it", "this", "then", "at", "have", "all", "not", "one", "has", "or",
            "that",
        ];

        for &s in words.iter() {
            set.insert(String::from(s));
        }

        set
    };
}

/// Keyword with weight
#[derive(Debug, Clone, PartialEq)]
pub struct Keyword {
    pub keyword: String,
    pub weight: f64,
}

#[derive(Debug)]
pub struct KeywordExtractConfig {
    stop_words: BTreeSet<String>,
    min_keyword_length: usize,
    use_hmm: bool,
}

impl KeywordExtractConfig {
    /// Creates a KeywordExtractConfig state that contains filter criteria as
    /// well as segmentation configuration for use by keyword extraction
    /// implementations.
    pub fn new(stop_words: BTreeSet<String>, min_keyword_length: usize, use_hmm: bool) -> Self {
        KeywordExtractConfig {
            stop_words,
            min_keyword_length,
            use_hmm,
        }
    }

    /// Add a new stop word.
    pub fn add_stop_word(&mut self, word: String) -> bool {
        self.stop_words.insert(word)
    }

    /// Remove an existing stop word.
    pub fn remove_stop_word(&mut self, word: &str) -> bool {
        self.stop_words.remove(word)
    }

    /// Replace all stop words with new stop words set.
    pub fn set_stop_words(&mut self, stop_words: BTreeSet<String>) {
        self.stop_words = stop_words
    }

    /// Get current set of stop words.
    pub fn get_stop_words(&self) -> &BTreeSet<String> {
        &self.stop_words
    }

    /// True if hmm is used during segmentation in `extract_tags`.
    pub fn get_use_hmm(&self) -> bool {
        self.use_hmm
    }

    /// Sets whether or not to use hmm during segmentation in `extract_tags`.
    pub fn set_use_hmm(&mut self, use_hmm: bool) {
        self.use_hmm = use_hmm
    }

    /// Gets the minimum number of Unicode Scalar Values required per keyword.
    pub fn get_min_keyword_length(&self) -> usize {
        self.min_keyword_length
    }

    /// Sets the minimum number of Unicode Scalar Values required per keyword.
    ///
    /// The default is 2. There is likely not much reason to change this.
    pub fn set_min_keyword_length(&mut self, min_keyword_length: usize) {
        self.min_keyword_length = min_keyword_length
    }

    #[inline]
    pub fn filter(&self, s: &str) -> bool {
        s.chars().count() >= self.min_keyword_length && !self.stop_words.contains(&s.to_lowercase())
    }
}

impl Default for KeywordExtractConfig {
    fn default() -> Self {
        KeywordExtractConfig::new(DEFAULT_STOP_WORDS.clone(), 2, false)
    }
}

pub trait KeywordExtract {
    fn extract_tags(&self, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword>;
}

/// Version of KeywordExtract trait that requires a Jieba instance on invocation.
pub trait JiebaKeywordExtract {
    fn extract_tags(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword>;
}
