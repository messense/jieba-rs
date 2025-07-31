use crate::Jieba;

use std::collections::BTreeSet;
use std::sync::LazyLock;

#[cfg(feature = "textrank")]
pub mod textrank;
#[cfg(feature = "tfidf")]
pub mod tfidf;

pub static DEFAULT_STOP_WORDS: LazyLock<BTreeSet<String>> = LazyLock::new(|| {
    BTreeSet::from_iter(
        [
            "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are", "by", "be", "as", "on", "with",
            "can", "if", "from", "which", "you", "it", "this", "then", "at", "have", "all", "not", "one", "has", "or",
        ]
        .into_iter()
        .map(ToString::to_string),
    )
});

/// Keyword with weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyword {
    pub keyword: String,
    pub weight: f64,
}

/// Creates a KeywordExtractConfig state that contains filter criteria as well as segmentation
/// configuration for use by keyword extraction implementations.
///
/// Use [`KeywordExtractConfigBuilder`] to change the defaults.
///
/// # Examples
///
/// ```
/// use jieba_rs::KeywordExtractConfig;
///
/// let mut config = KeywordExtractConfig::default();
/// assert!(config.stop_words().contains("the"));
/// assert!(!config.stop_words().contains("FakeWord"));
/// assert!(!config.use_hmm());
/// assert_eq!(2, config.min_keyword_length());
///
/// let built_default = KeywordExtractConfig::builder().build();
/// assert_eq!(config, built_default);
///
/// let changed = KeywordExtractConfig::builder()
///     .add_stop_word("FakeWord".to_string())
///     .remove_stop_word("the")
///     .use_hmm(true)
///     .min_keyword_length(10)
///     .build();
///
/// assert!(!changed.stop_words().contains("the"));
/// assert!(changed.stop_words().contains("FakeWord"));
/// assert!(changed.use_hmm());
/// assert_eq!(10, changed.min_keyword_length());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordExtractConfig {
    stop_words: BTreeSet<String>,
    min_keyword_length: usize,
    use_hmm: bool,
}

impl Default for KeywordExtractConfig {
    fn default() -> KeywordExtractConfig {
        KeywordExtractConfig::builder().build()
    }
}

impl KeywordExtractConfig {
    /// Creates a new [`KeywordExtractConfigBuilder`] with default values.
    pub fn builder() -> KeywordExtractConfigBuilder {
        KeywordExtractConfigBuilder::default()
    }

    /// Gets the current set of stop words.
    pub fn stop_words(&self) -> &BTreeSet<String> {
        &self.stop_words
    }

    /// Returns whether HMM is used during segmentation in `extract_tags`.
    pub fn use_hmm(&self) -> bool {
        self.use_hmm
    }

    /// Gets the minimum number of Unicode Scalar Values required per keyword.
    pub fn min_keyword_length(&self) -> usize {
        self.min_keyword_length
    }

    #[inline]
    pub(crate) fn is_keyword(&self, s: &str) -> bool {
        s.chars().count() >= self.min_keyword_length() && !self.stop_words.contains(&s.to_lowercase())
    }
}

#[derive(Debug, Clone)]
pub struct KeywordExtractConfigBuilder {
    stop_words: BTreeSet<String>,
    min_keyword_length: usize,
    use_hmm: bool,
}

impl Default for KeywordExtractConfigBuilder {
    fn default() -> Self {
        KeywordExtractConfigBuilder {
            stop_words: DEFAULT_STOP_WORDS.clone(),
            min_keyword_length: 2,
            use_hmm: false,
        }
    }
}

impl KeywordExtractConfigBuilder {
    /// Builds the [`KeywordExtractConfig`] with the current configuration.
    pub fn build(self) -> KeywordExtractConfig {
        KeywordExtractConfig {
            stop_words: self.stop_words,
            min_keyword_length: self.min_keyword_length,
            use_hmm: self.use_hmm,
        }
    }

    /// If set, when segment cannot be found in the dictionary, fall back to HMM model.
    pub fn use_hmm(mut self, yes: bool) -> Self {
        self.use_hmm = yes;
        self
    }

    /// Sets the length that any segments less than it will not be considered as a keyword.
    pub fn min_keyword_length(mut self, length: usize) -> Self {
        self.min_keyword_length = length;
        self
    }

    /// Add a new stop word.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs::KeywordExtractConfig;
    /// use std::collections::BTreeSet;
    ///
    /// let populates_default = KeywordExtractConfig::builder()
    ///     .add_stop_word("FakeWord")
    ///     .build();
    ///
    /// assert!(populates_default.stop_words().contains("the"));
    /// assert!(populates_default.stop_words().contains("FakeWord"));
    ///
    /// let multiple_adds_stack = KeywordExtractConfig::builder()
    ///     .add_stop_word("FakeWord")
    ///     .add_stop_word("MoarFakeWord")
    ///     .build();
    ///
    /// assert!(multiple_adds_stack.stop_words().contains("the"));
    /// assert!(multiple_adds_stack.stop_words().contains("FakeWord"));
    /// assert!(multiple_adds_stack.stop_words().contains("MoarFakeWord"));
    ///
    /// let no_default_if_set = KeywordExtractConfig::builder()
    ///     .set_stop_words(BTreeSet::from(["boo".to_string()]))
    ///     .add_stop_word("FakeWord".to_string())
    ///     .build();
    ///
    /// assert!(!no_default_if_set.stop_words().contains("the"));
    /// assert!(no_default_if_set.stop_words().contains("boo"));
    /// assert!(no_default_if_set.stop_words().contains("FakeWord"));
    /// ```
    pub fn add_stop_word(mut self, word: impl Into<String>) -> Self {
        self.stop_words.insert(word.into());
        self
    }

    /// Remove a stop word.
    ///
    /// If the word is not in the set, this is no op.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs::KeywordExtractConfig;
    /// use std::collections::BTreeSet;
    ///
    /// let populates_default = KeywordExtractConfig::builder()
    ///     .remove_stop_word("the")
    ///     .build();
    ///
    /// assert!(!populates_default.stop_words().contains("the"));
    /// assert!(populates_default.stop_words().contains("of"));
    ///
    /// let no_default_if_set = KeywordExtractConfig::builder()
    ///     .set_stop_words(BTreeSet::from(["boo".to_string()]))
    ///     // removing non-existent word is okay
    ///     .remove_stop_word("the")
    ///     .build();
    ///
    /// assert!(!no_default_if_set.stop_words().contains("the"));
    /// assert!(!no_default_if_set.stop_words().contains("of"));
    /// assert!(no_default_if_set.stop_words().contains("boo"));
    /// ```
    pub fn remove_stop_word(mut self, word: impl AsRef<str>) -> Self {
        self.stop_words.remove(word.as_ref());
        self
    }

    /// Replace all stop words with new stop words set.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs::KeywordExtractConfig;
    /// use std::collections::BTreeSet;
    ///
    /// let no_default_if_set = KeywordExtractConfig::builder()
    ///     .set_stop_words(BTreeSet::from(["boo".to_string()]))
    ///     .build();
    ///
    /// assert!(!no_default_if_set.stop_words().contains("the"));
    /// assert!(no_default_if_set.stop_words().contains("boo"));
    ///
    /// let overwrites = KeywordExtractConfig::builder()
    ///     .add_stop_word("FakeWord".to_string())
    ///     .set_stop_words(BTreeSet::from(["boo".to_string()]))
    ///     .build();
    ///
    /// assert!(!no_default_if_set.stop_words().contains("FakeWord"));
    /// assert!(no_default_if_set.stop_words().contains("boo"));
    /// ```
    pub fn set_stop_words(mut self, stop_words: BTreeSet<String>) -> Self {
        self.stop_words = stop_words;
        self
    }
}

/// Extracts keywords from a given sentence with the Jieba instance.
pub trait KeywordExtract {
    fn extract_keywords(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword>;
}
