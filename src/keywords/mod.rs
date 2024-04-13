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

/// Creates a KeywordExtractConfig state that contains filter criteria as
/// well as segmentation configuration for use by keyword extraction
/// implementations.
///
/// Use KeywordExtractConfigBuilder to change the defaults.
///
/// # Examples
/// ```
///    use jieba_rs::KeywordExtractConfig;
///
///    let mut config = KeywordExtractConfig::default();
///    assert!(config.stop_words().contains("the"));
///    assert!(!config.stop_words().contains("FakeWord"));
///    assert!(!config.use_hmm());
///    assert_eq!(2, config.min_keyword_length());
///
///    let built_default = KeywordExtractConfig::builder().build().unwrap();
///    assert_eq!(config, built_default);
///
///    let changed = KeywordExtractConfig::builder()
///        .add_stop_word("FakeWord".to_string())
///        .remove_stop_word("the")
///        .use_hmm(true)
///        .min_keyword_length(10)
///        .build().unwrap();
///
///    assert!(!changed.stop_words().contains("the"));
///    assert!(changed.stop_words().contains("FakeWord"));
///    assert!(changed.use_hmm());
///    assert_eq!(10, changed.min_keyword_length());
/// ```
#[derive(Builder, Debug, Clone, PartialEq)]
pub struct KeywordExtractConfig {
    #[builder(default = "self.default_stop_words()?", setter(custom))]
    stop_words: BTreeSet<String>,

    #[builder(default = "2")]
    #[doc = r"Any segments less than this length will not be considered a Keyword"]
    min_keyword_length: usize,

    #[builder(default = "false")]
    #[doc = r"If true, fall back to hmm model if segment cannot be found in the dictionary"]
    use_hmm: bool,
}

impl KeywordExtractConfig {
    pub fn builder() -> KeywordExtractConfigBuilder {
        KeywordExtractConfigBuilder::default()
    }

    /// Get current set of stop words.
    pub fn stop_words(&self) -> &BTreeSet<String> {
        &self.stop_words
    }

    /// True if hmm is used during segmentation in `extract_tags`.
    pub fn use_hmm(&self) -> bool {
        self.use_hmm
    }

    /// Gets the minimum number of Unicode Scalar Values required per keyword.
    pub fn min_keyword_length(&self) -> usize {
        self.min_keyword_length
    }

    #[inline]
    pub(crate) fn filter(&self, s: &str) -> bool {
        s.chars().count() >= self.min_keyword_length() && !self.stop_words.contains(&s.to_lowercase())
    }
}

impl KeywordExtractConfigBuilder {
    fn default_stop_words(&self) -> Result<BTreeSet<String>, KeywordExtractConfigBuilderError> {
        Ok(DEFAULT_STOP_WORDS.clone())
    }

    /// Add a new stop word.
    pub fn add_stop_word(&mut self, word: String) -> &mut Self {
        if self.stop_words.is_none() {
            self.stop_words = Some(self.default_stop_words().unwrap());
        }
        self.stop_words.as_mut().unwrap().insert(word);
        self
    }

    /// Remove an existing stop word.
    pub fn remove_stop_word(&mut self, word: &str) -> &mut Self {
        if self.stop_words.is_none() {
            self.stop_words = Some(self.default_stop_words().unwrap());
        }
        self.stop_words.as_mut().unwrap().remove(word);
        self
    }

    /// Replace all stop words with new stop words set.
    pub fn set_stop_words(&mut self, stop_words: BTreeSet<String>) -> &mut Self {
        self.stop_words = Some(stop_words);
        self
    }
}

impl Default for KeywordExtractConfig {
    fn default() -> KeywordExtractConfig {
        KeywordExtractConfigBuilder::default().build().unwrap()
    }
}

/// Extracts keywords from a given sentence with the Jieba instance.
pub trait KeywordExtract {
    fn extract_keywords(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword>;
}
