use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};
use std::io::{self, BufRead, BufReader};

use hashbrown::HashMap;
use ordered_float::OrderedFloat;

use super::{Keyword, KeywordExtract, STOP_WORDS};
use crate::Jieba;

static DEFAULT_IDF: &str = include_str!("../data/idf.txt");

#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapNode<'a> {
    tfidf: OrderedFloat<f64>,
    word: &'a str,
}

impl<'a> Ord for HeapNode<'a> {
    fn cmp(&self, other: &HeapNode) -> Ordering {
        other.tfidf.cmp(&self.tfidf).then_with(|| self.word.cmp(&other.word))
    }
}

impl<'a> PartialOrd for HeapNode<'a> {
    fn partial_cmp(&self, other: &HeapNode) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// TF-IDF keywords extraction
///
/// Require `tfidf` feature to be enabled
#[derive(Debug)]
pub struct TFIDF<'a> {
    jieba: &'a Jieba,
    idf_dict: HashMap<String, f64>,
    median_idf: f64,
    stop_words: BTreeSet<String>,
}

impl<'a> TFIDF<'a> {
    pub fn new_with_jieba(jieba: &'a Jieba) -> Self {
        let mut instance = TFIDF {
            jieba,
            idf_dict: HashMap::new(),
            median_idf: 0.0,
            stop_words: STOP_WORDS.clone(),
        };

        let mut default_dict = BufReader::new(DEFAULT_IDF.as_bytes());
        instance.load_dict(&mut default_dict).unwrap();
        instance
    }

    pub fn load_dict<R: BufRead>(&mut self, dict: &mut R) -> io::Result<()> {
        let mut buf = String::new();
        let mut idf_heap = BinaryHeap::new();
        while dict.read_line(&mut buf)? > 0 {
            let parts: Vec<&str> = buf.trim().split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let word = parts[0];
            if let Some(idf) = parts.get(1).and_then(|x| x.parse::<f64>().ok()) {
                self.idf_dict.insert(word.to_string(), idf);
                idf_heap.push(OrderedFloat(idf));
            }

            buf.clear();
        }

        let m = idf_heap.len() / 2;
        for _ in 0..m {
            idf_heap.pop();
        }

        self.median_idf = idf_heap.pop().unwrap().into_inner();

        Ok(())
    }

    /// Add a new stop word
    pub fn add_stop_word(&mut self, word: String) -> bool {
        self.stop_words.insert(word)
    }

    /// Remove an existing stop word
    pub fn remove_stop_word(&mut self, word: &str) -> bool {
        self.stop_words.remove(word)
    }

    /// Replace all stop words with new stop words set
    pub fn set_stop_words(&mut self, stop_words: BTreeSet<String>) {
        self.stop_words = stop_words
    }

    #[inline]
    fn filter(&self, s: &str) -> bool {
        if s.chars().count() < 2 {
            return false;
        }

        if self.stop_words.contains(&s.to_lowercase()) {
            return false;
        }

        true
    }
}

impl<'a> KeywordExtract for TFIDF<'a> {
    fn extract_tags(&self, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        let tags = self.jieba.tag(sentence, false);
        let mut allowed_pos_set = BTreeSet::new();

        for s in allowed_pos {
            allowed_pos_set.insert(s);
        }

        let mut term_freq: HashMap<String, u64> = HashMap::new();
        for t in &tags {
            if !allowed_pos_set.is_empty() && !allowed_pos_set.contains(t.tag) {
                continue;
            }

            if !self.filter(t.word) {
                continue;
            }

            let entry = term_freq.entry(String::from(t.word)).or_insert(0);
            *entry += 1;
        }

        let total: u64 = term_freq.values().sum();
        let mut heap = BinaryHeap::new();
        for (cnt, (k, tf)) in term_freq.iter().enumerate() {
            let idf = self.idf_dict.get(k).unwrap_or(&self.median_idf);
            let node = HeapNode {
                tfidf: OrderedFloat(*tf as f64 * idf / total as f64),
                word: k,
            };
            heap.push(node);
            if cnt >= top_k {
                heap.pop();
            }
        }

        let mut res = Vec::new();
        for _ in 0..top_k {
            if let Some(w) = heap.pop() {
                res.push(Keyword {
                    keyword: String::from(w.word),
                    weight: w.tfidf.into_inner(),
                });
            }
        }

        res.reverse();
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_with_default_idf_dict() {
        let jieba = super::Jieba::new();
        let _ = TFIDF::new_with_jieba(&jieba);
    }

    #[test]
    fn test_extract_tags() {
        let jieba = super::Jieba::new();
        let keyword_extractor = TFIDF::new_with_jieba(&jieba);
        let mut top_k = keyword_extractor.extract_tags(
            "今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。后天纽约的天气不好，昨天纽约的天气也不好，北京烤鸭真好吃",
            3,
            vec![],
        );
        assert_eq!(
            top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
            vec!["北京烤鸭", "纽约", "天气"]
        );

        top_k = keyword_extractor.extract_tags(
            "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
            5,
            vec![],
        );
        assert_eq!(
            top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
            vec!["欧亚", "吉林", "置业", "万元", "增资"]
        );

        top_k = keyword_extractor.extract_tags(
            "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
            5,
            vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
        );
        assert_eq!(
            top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
            vec!["欧亚", "吉林", "置业", "增资", "实现"]
        );
    }
}
