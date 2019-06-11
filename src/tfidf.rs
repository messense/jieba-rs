use super::Jieba;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use std::collections::{BTreeSet, BinaryHeap};
use std::io::{self, BufRead, BufReader};

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

static DEFAULT_IDF: &str = include_str!("data/idf.txt");

pub trait KeywordExtract {
    fn extract_tags<'a>(&'a self, _: &'a str, _: usize, _: Vec<String>) -> Vec<String>;
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct HeapNode<'a> {
    tfidf: u64, //using u64 but not f64 so that it conforms to Ord
    word: &'a str,
}

#[derive(Debug)]
pub struct TFIDF<'a> {
    jieba: &'a Jieba,
    idf_dict: HashMap<String, u64>,
    median_idf: u64,
}

impl<'a> TFIDF<'a> {
    pub fn new_with_jieba(jieba: &'a Jieba) -> Self {
        let mut instance = TFIDF {
            jieba: jieba,
            idf_dict: HashMap::new(),
            median_idf: 0,
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
                //using fix-point integer but not f64
                let u64idf: u64 = (idf * 1e10) as u64;

                self.idf_dict.insert(word.to_string(), u64idf);
                idf_heap.push(u64idf);
            }

            buf.clear();
        }

        let m = idf_heap.len() / 2;
        for _ in 0..m {
            idf_heap.pop();
        }

        self.median_idf = idf_heap.pop().unwrap();

        Ok(())
    }
}

impl<'a> KeywordExtract for TFIDF<'a> {
    fn extract_tags(&self, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<String> {
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

            if !filter(t.word) {
                continue;
            }

            let entry = term_freq.entry(String::from(t.word)).or_insert(0);
            *entry += 1;
        }

        let mut heap = BinaryHeap::new();
        for (k, tf) in term_freq.iter() {
            if let Some(idf) = self.idf_dict.get(k) {
                //we don't care about the total in tf since it doesn't change the ranking
                let node = HeapNode {
                    tfidf: tf * idf,
                    word: k,
                };
                heap.push(node);
            } else {
                let node = HeapNode {
                    tfidf: tf * self.median_idf,
                    word: k,
                };
                heap.push(node);
            }
        }

        let mut res: Vec<String> = Vec::new();
        for _ in 0..top_k {
            if let Some(w) = heap.pop() {
                res.push(String::from(w.word));
            }
        }

        res
    }
}

#[inline]
fn filter(s: &str) -> bool {
    if s.chars().count() < 2 {
        return false;
    }

    if STOP_WORDS.contains(&s.to_lowercase()) {
        return false;
    }

    true
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
        assert_eq!(top_k, vec!["北京烤鸭", "纽约", "天气"]);

        top_k = keyword_extractor.extract_tags(
            "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
            5,
            vec![],
        );
        assert_eq!(top_k, vec!["欧亚", "吉林", "置业", "万元", "增资"]);

        top_k = keyword_extractor.extract_tags(
            "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
            5,
            vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
        );
        assert_eq!(top_k, vec!["欧亚", "吉林", "置业", "增资", "实现"]);
    }
}
