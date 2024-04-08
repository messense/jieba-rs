use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};
use std::io::{self, BufRead, BufReader};

use ordered_float::OrderedFloat;

use super::{JiebaKeywordExtract, Keyword, KeywordExtract, DEFAULT_STOP_WORDS};
use crate::FxHashMap as HashMap;
use crate::Jieba;

static DEFAULT_IDF: &str = include_str!("../data/idf.txt");

#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapNode<'a> {
    tfidf: OrderedFloat<f64>,
    word: &'a str,
}

impl<'a> Ord for HeapNode<'a> {
    fn cmp(&self, other: &HeapNode) -> Ordering {
        other.tfidf.cmp(&self.tfidf).then_with(|| self.word.cmp(other.word))
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
pub struct UnboundTfidf {
    idf_dict: HashMap<String, f64>,
    median_idf: f64,
    stop_words: BTreeSet<String>,
    min_keyword_length: usize,
    use_hmm: bool,
}

/// Implementation of JiebaKeywordExtract using a TFIDF dictionary.
///
/// This takes the segments produced by Jieba and attempts to extract keywords.
/// Segments are filtered for stopwords and short terms. They are then matched
/// against a loaded dictionary to calculate TFIDF scores.
impl UnboundTfidf {
    /// Creates an UnboundTfidf.
    ///
    /// # Examples
    ///
    /// New instance with custom stop words and idf dictionary. Also uses hmm
    /// for unknown words during segmentation and allows keywords of length 1.
    /// ```
    ///    use std::collections::BTreeSet;
    ///
    ///    let stop_words : BTreeSet<String> =
    ///        BTreeSet::from(["a", "the", "of"].map(|s| s.to_string()));
    ///    let mut sample_idf = "劳动防护 13.900677652\n\
    ///        生化学 13.900677652\n";
    ///    jieba_rs::UnboundTfidf::new(
    ///        Some(&mut sample_idf.as_bytes()),
    ///        stop_words,
    ///        1,
    ///        true);
    /// ```
    ///
    /// New instance with module default stop words and no initial IDF
    /// dictionary. Dictionary should be loaded later with `load_dict()` calls.
    /// No hmm and more standard minimal of length 2 keywords.
    /// ```
    ///    jieba_rs::UnboundTfidf::new(
    ///        None::<&mut std::io::Empty>,
    ///        jieba_rs::DEFAULT_STOP_WORDS.clone(),
    ///        2,
    ///        false);
    /// ```
    pub fn new(
        opt_dict: Option<&mut impl BufRead>,
        stop_words: BTreeSet<String>,
        min_keyword_length: usize,
        use_hmm: bool,
    ) -> Self {
        let mut instance = UnboundTfidf {
            idf_dict: HashMap::default(),
            median_idf: 0.0,
            stop_words,
            min_keyword_length,
            use_hmm,
        };
        if let Some(dict) = opt_dict {
            instance.load_dict(dict).unwrap();
        }
        instance
    }

    /// Merges entires from `dict` into the `idf_dict`.
    ///
    /// ```
    ///    use jieba_rs::{Jieba, JiebaKeywordExtract, Keyword,
    ///        UnboundTfidf, DEFAULT_STOP_WORDS};
    ///
    ///    let jieba = Jieba::default();
    ///    let mut init_idf = "生化学 13.900677652\n";
    ///
    ///    let mut tfidf = UnboundTfidf::new(
    ///        Some(&mut init_idf.as_bytes()),
    ///        DEFAULT_STOP_WORDS.clone(),
    ///        true);
    ///    let top_k = tfidf.extract_tags(&jieba, "生化学很難", 3, vec![]);
    ///    assert_eq!(
    ///        top_k,
    ///        vec![
    ///            Keyword { keyword: "很難".to_string(), weight: 6.950338826 },
    ///            Keyword { keyword: "生化学".to_string(), weight: 6.950338826 }
    ///        ]
    ///    );
    ///
    ///    let mut init_idf = "很難 99.123456789\n";
    ///    tfidf.load_dict(&mut init_idf.as_bytes());
    ///    let top_k = tfidf.extract_tags(&jieba, "生化学很難", 3, vec![]);
    ///    assert_eq!(
    ///        top_k,
    ///        vec![
    ///            Keyword { keyword: "很難".to_string(), weight: 49.5617283945 },
    ///            Keyword { keyword: "生化学".to_string(), weight: 6.950338826 }
    ///        ]
    ///    );
    /// ```
    pub fn load_dict(&mut self, dict: &mut impl BufRead) -> io::Result<()> {
        let mut buf = String::new();
        let mut idf_heap = BinaryHeap::new();
        while dict.read_line(&mut buf)? > 0 {
            let parts: Vec<&str> = buf.split_whitespace().collect();
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
    fn filter(&self, s: &str) -> bool {
        s.chars().count() >= self.min_keyword_length && !self.stop_words.contains(&s.to_lowercase())
    }
}

impl Default for UnboundTfidf {
    /// Creates UnboundTfidf with DEFAULT_STOP_WORDS, the default TFIDF dictionary,
    /// 2 Unicode Scalar Value minimum for keywords, and no hmm in segmentation.
    fn default() -> Self {
        let mut default_dict = BufReader::new(DEFAULT_IDF.as_bytes());
        UnboundTfidf::new(Some(&mut default_dict), DEFAULT_STOP_WORDS.clone(), 2, false)
    }
}

impl JiebaKeywordExtract for UnboundTfidf {
    fn extract_tags(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        let tags = jieba.tag(sentence, self.use_hmm);
        let mut allowed_pos_set = BTreeSet::new();

        for s in allowed_pos {
            allowed_pos_set.insert(s);
        }

        let mut term_freq: HashMap<String, u64> = HashMap::default();
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

/// TF-IDF keywords extraction
///
/// Require `tfidf` feature to be enabled
#[derive(Debug)]
pub struct TFIDF<'a> {
    jieba: &'a Jieba,
    unbound_tfidf: UnboundTfidf,
}

impl<'a> TFIDF<'a> {
    pub fn new_with_jieba(jieba: &'a Jieba) -> Self {
        TFIDF {
            jieba,
            unbound_tfidf: Default::default(),
        }
    }

    pub fn load_dict<R: BufRead>(&mut self, dict: &mut R) -> io::Result<()> {
        self.unbound_tfidf.load_dict(dict)
    }

    /// Add a new stop word
    pub fn add_stop_word(&mut self, word: String) -> bool {
        self.unbound_tfidf.add_stop_word(word)
    }

    /// Remove an existing stop word
    pub fn remove_stop_word(&mut self, word: &str) -> bool {
        self.unbound_tfidf.remove_stop_word(word)
    }

    /// Replace all stop words with new stop words set
    pub fn set_stop_words(&mut self, stop_words: BTreeSet<String>) {
        self.unbound_tfidf.set_stop_words(stop_words)
    }
}

impl<'a> KeywordExtract for TFIDF<'a> {
    fn extract_tags(&self, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        self.unbound_tfidf
            .extract_tags(self.jieba, sentence, top_k, allowed_pos)
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
