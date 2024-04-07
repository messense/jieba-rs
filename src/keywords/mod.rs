use lazy_static::lazy_static;
use std::collections::BTreeSet;

use crate::Jieba;

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

/// Keyword with weight
#[derive(Debug, Clone)]
pub struct Keyword {
    pub keyword: String,
    pub weight: f64,
}

pub trait KeywordExtract {
    fn extract_tags(&self, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword>;
}

/// Version of KeywordExtract trait that requires a Jieba instance on invocation.
pub trait JiebaKeywordExtract {
    fn extract_tags(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword>;
}
