use lazy_static::lazy_static;
use std::collections::BTreeSet;

#[cfg(feature = "textrank")]
pub mod textrank;
#[cfg(feature = "tfidf")]
pub mod tfidf;

lazy_static! {
    pub static ref STOP_WORDS: BTreeSet<String> = {
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

pub trait KeywordExtract {
    fn extract_tags<'a>(&'a self, _: &'a str, _: usize, _: Vec<String>) -> Vec<String>;
}
