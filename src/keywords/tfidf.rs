use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};
use std::io::{self, BufRead, BufReader};

use include_flate::flate;
use ordered_float::OrderedFloat;

use super::{Keyword, KeywordExtract, KeywordExtractConfig, KeywordExtractConfigBuilder};
use crate::FxHashMap as HashMap;
use crate::Jieba;

flate!(static DEFAULT_IDF: str from "src/data/idf.txt");

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
pub struct TfIdf {
    idf_dict: HashMap<String, f64>,
    median_idf: f64,
    config: KeywordExtractConfig,
}

/// Implementation of JiebaKeywordExtract using a TF-IDF dictionary.
///
/// This takes the segments produced by Jieba and attempts to extract keywords.
/// Segments are filtered for stopwords and short terms. They are then matched
/// against a loaded dictionary to calculate TF-IDF scores.
impl TfIdf {
    /// Creates an TfIdf.
    ///
    /// # Examples
    ///
    /// New instance with custom idf dictionary.
    /// ```
    ///    use jieba_rs::{TfIdf, KeywordExtractConfig};
    ///
    ///    let mut sample_idf = "劳动防护 13.900677652\n\
    ///        生化学 13.900677652\n";
    ///    TfIdf::new(
    ///        Some(&mut sample_idf.as_bytes()),
    ///        KeywordExtractConfig::default());
    /// ```
    ///
    /// New instance with module default stop words and no initial IDF
    /// dictionary. Dictionary should be loaded later with `load_dict()` calls.
    /// ```
    ///    use jieba_rs::{TfIdf, KeywordExtractConfig};
    ///
    ///    TfIdf::new(
    ///        None::<&mut std::io::Empty>,
    ///        KeywordExtractConfig::default());
    /// ```
    pub fn new(opt_dict: Option<&mut impl BufRead>, config: KeywordExtractConfig) -> Self {
        let mut instance = TfIdf {
            idf_dict: HashMap::default(),
            median_idf: 0.0,
            config,
        };
        if let Some(dict) = opt_dict {
            instance.load_dict(dict).unwrap();
        }
        instance
    }

    /// Merges entires from `dict` into the `idf_dict`.
    ///
    /// ```
    ///    use jieba_rs::{Jieba, KeywordExtract, Keyword, KeywordExtractConfig,
    ///        TfIdf};
    ///
    ///    let jieba = Jieba::default();
    ///    let mut init_idf = "生化学 13.900677652\n";
    ///
    ///    let mut tfidf = TfIdf::new(
    ///        Some(&mut init_idf.as_bytes()),
    ///        KeywordExtractConfig::default());
    ///    let top_k = tfidf.extract_keywords(&jieba, "生化学不是光化学的,", 3, vec![]);
    ///    assert_eq!(
    ///        top_k,
    ///        vec![
    ///            Keyword { keyword: "不是".to_string(), weight: 4.6335592173333335 },
    ///            Keyword { keyword: "光化学".to_string(), weight: 4.6335592173333335 },
    ///            Keyword { keyword: "生化学".to_string(), weight: 4.6335592173333335 }
    ///        ]
    ///    );
    ///
    ///    let mut init_idf = "光化学 99.123456789\n";
    ///    tfidf.load_dict(&mut init_idf.as_bytes());
    ///    let new_top_k = tfidf.extract_keywords(&jieba, "生化学不是光化学的,", 3, vec![]);
    ///    assert_eq!(
    ///        new_top_k,
    ///        vec![
    ///            Keyword { keyword: "不是".to_string(), weight: 33.041152263 },
    ///            Keyword { keyword: "光化学".to_string(), weight: 33.041152263 },
    ///            Keyword { keyword: "生化学".to_string(), weight: 4.6335592173333335 }
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

    pub fn config(&self) -> &KeywordExtractConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut KeywordExtractConfig {
        &mut self.config
    }
}

/// TF-IDF keywords extraction.
///
/// Require `tfidf` feature to be enabled.
impl Default for TfIdf {
    /// Creates TfIdf with DEFAULT_STOP_WORDS, the default TfIdf dictionary,
    /// 2 Unicode Scalar Value minimum for keywords, and no hmm in segmentation.
    fn default() -> Self {
        let mut default_dict = BufReader::new(DEFAULT_IDF.as_bytes());
        TfIdf::new(
            Some(&mut default_dict),
            KeywordExtractConfigBuilder::default().build().unwrap(),
        )
    }
}

impl KeywordExtract for TfIdf {
    /// Uses TF-IDF algorithm to extract the `top_k` keywords from `sentence`.
    ///
    /// If `allowed_pos` is not empty, then only terms matching those parts if
    /// speech are considered.
    ///
    /// # Examples
    /// ```
    ///    use jieba_rs::{Jieba, KeywordExtract, TfIdf};
    ///
    ///    let jieba = Jieba::new();
    ///    let keyword_extractor = TfIdf::default();
    ///    let mut top_k = keyword_extractor.extract_keywords(
    ///        &jieba,
    ///        "今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。后天纽约的天气不好，昨天纽约的天气也不好，北京烤鸭真好吃",
    ///        3,
    ///        vec![],
    ///    );
    ///    assert_eq!(
    ///        top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///        vec!["北京烤鸭", "纽约", "天气"]
    ///    );
    ///
    ///    top_k = keyword_extractor.extract_keywords(
    ///        &jieba,
    ///        "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
    ///        5,
    ///        vec![],
    ///    );
    ///    assert_eq!(
    ///        top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///        vec!["欧亚", "吉林", "置业", "万元", "增资"]
    ///    );
    ///
    ///    top_k = keyword_extractor.extract_keywords(
    ///        &jieba,
    ///        "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
    ///        5,
    ///        vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
    ///    );
    ///    assert_eq!(
    ///        top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///        vec!["欧亚", "吉林", "置业", "增资", "实现"]
    ///    );
    /// ```
    fn extract_keywords(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        let tags = jieba.tag(sentence, self.config.use_hmm());
        let mut allowed_pos_set = BTreeSet::new();

        for s in allowed_pos {
            allowed_pos_set.insert(s);
        }

        let mut term_freq: HashMap<String, u64> = HashMap::default();
        for t in &tags {
            if !allowed_pos_set.is_empty() && !allowed_pos_set.contains(t.tag) {
                continue;
            }

            if !self.config.filter(t.word) {
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
