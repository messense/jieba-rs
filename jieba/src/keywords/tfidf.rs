use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{self, BufRead};

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

impl Ord for HeapNode<'_> {
    fn cmp(&self, other: &HeapNode) -> Ordering {
        other.tfidf.cmp(&self.tfidf).then_with(|| self.word.cmp(other.word))
    }
}

impl PartialOrd for HeapNode<'_> {
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

    /// Merges entries from `dict` into the `idf_dict`.
    ///
    /// ```
    /// use jieba_rs::{Jieba, KeywordExtract, Keyword, KeywordExtractConfig, TfIdf};
    ///
    /// let jieba = Jieba::default();
    /// let mut init_idf = "生化学 13.900677652\n";
    ///
    /// let mut tfidf = TfIdf::new(Some(&mut init_idf.as_bytes()), KeywordExtractConfig::default());
    /// let top_k = tfidf.extract_keywords(&jieba, "生化学不是光化学的,", 3, vec![]);
    /// assert_eq!(
    ///     top_k,
    ///     vec![
    ///         Keyword { keyword: "不是".to_string(), weight: 4.6335592173333335, tag: "c".to_string() },
    ///         Keyword { keyword: "光化学".to_string(), weight: 4.6335592173333335, tag: "n".to_string() },
    ///         Keyword { keyword: "生化学".to_string(), weight: 4.6335592173333335, tag: "n".to_string() }
    ///     ],
    /// );
    ///
    /// let mut init_idf = "光化学 99.123456789\n";
    /// tfidf.load_dict(&mut init_idf.as_bytes()).unwrap();
    /// let new_top_k = tfidf.extract_keywords(&jieba, "生化学不是光化学的,", 3, vec![]);
    /// assert_eq!(
    ///     new_top_k,
    ///     vec![
    ///         Keyword { keyword: "不是".to_string(), weight: 33.041152263, tag: "c".to_string() },
    ///         Keyword { keyword: "光化学".to_string(), weight: 33.041152263, tag: "n".to_string() },
    ///         Keyword { keyword: "生化学".to_string(), weight: 4.6335592173333335, tag: "n".to_string() }
    ///     ]
    /// );
    /// ```
    pub fn load_dict(&mut self, dict: &mut impl BufRead) -> io::Result<()> {
        let mut buf = String::new();
        let mut idfs = Vec::new();
        while dict.read_line(&mut buf)? > 0 {
            self.load_line(&buf, &mut idfs);
            buf.clear();
        }
        self.set_median(idfs);
        Ok(())
    }

    /// Load a dictionary held in memory, line by line and without copying.
    fn load_str(&mut self, dict: &str) {
        let mut idfs = Vec::with_capacity(bytecount::count(dict.as_bytes(), b'\n') + 1);
        self.idf_dict.reserve(idfs.capacity());
        for line in dict.lines() {
            self.load_line(line, &mut idfs);
        }
        self.set_median(idfs);
    }

    /// Add the `word idf` entry on `line`, if it is one, recording the
    /// value in `idfs` as well.
    #[inline]
    fn load_line(&mut self, line: &str, idfs: &mut Vec<OrderedFloat<f64>>) {
        let mut parts = line.split_whitespace();
        if let Some(word) = parts.next()
            && let Some(idf) = parts.next().and_then(|x| x.parse::<f64>().ok())
        {
            self.idf_dict.insert(word.to_string(), idf);
            idfs.push(OrderedFloat(idf));
        }
    }

    /// The median of the IDFs of one load, duplicates included, so that it
    /// is over what was read rather than over the merged dictionary. It is
    /// the lower median: the value with `len / 2` values above it, which is
    /// ascending index `(len - 1) / 2`. Selection is linear, where the heap
    /// this replaced popped half the values one by one.
    fn set_median(&mut self, mut idfs: Vec<OrderedFloat<f64>>) {
        let mid = idfs.len().saturating_sub(1) / 2;
        let (_, median, _) = idfs.select_nth_unstable(mid);
        self.median_idf = median.into_inner();
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
        let mut instance = TfIdf::new(None::<&mut io::Empty>, KeywordExtractConfigBuilder::default().build());
        instance.load_str(&DEFAULT_IDF);
        instance
    }
}

impl KeywordExtract for TfIdf {
    /// Uses TF-IDF algorithm to extract the `top_k` keywords from `sentence`.
    ///
    /// If `allowed_pos` is not empty, then only terms matching those parts if
    /// speech are considered.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs::{Jieba, KeywordExtract, TfIdf};
    ///
    /// let jieba = Jieba::new();
    /// let keyword_extractor = TfIdf::default();
    /// let mut top_k = keyword_extractor.extract_keywords(
    ///     &jieba,
    ///     "今天纽约的天气真好啊，京华大酒店的张尧经理吃了一只北京烤鸭。后天纽约的天气不好，昨天纽约的天气也不好，北京烤鸭真好吃",
    ///     3,
    ///     vec![],
    /// );
    /// assert_eq!(
    ///     top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///     vec!["北京烤鸭", "纽约", "天气"],
    /// );
    ///
    /// top_k = keyword_extractor.extract_keywords(
    ///     &jieba,
    ///     "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
    ///     5,
    ///     vec![],
    /// );
    /// assert_eq!(
    ///     top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///     vec!["欧亚", "吉林", "置业", "万元", "增资"],
    /// );
    ///
    /// top_k = keyword_extractor.extract_keywords(
    ///     &jieba,
    ///     "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
    ///     5,
    ///     vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
    /// );
    /// assert_eq!(
    ///     top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///     vec!["欧亚", "吉林", "置业", "增资", "实现"]
    /// );
    /// ```
    fn extract_keywords(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        let tags = jieba.tag(sentence, self.config.use_hmm());
        // A handful of tags at most, so a scan beats building a set.
        let allowed = |tag: &str| allowed_pos.is_empty() || allowed_pos.iter().any(|p| p == tag);

        // Per word: its frequency and the tag of its first occurrence.
        let mut term_freq: HashMap<&str, (u64, &str)> = HashMap::default();
        for t in &tags {
            if !allowed(t.tag) {
                continue;
            }

            if !self.config.is_keyword(t.word) {
                continue;
            }

            term_freq.entry(t.word).or_insert((0, t.tag)).0 += 1;
        }

        if top_k == 0 {
            return Vec::new();
        }

        let total: u64 = term_freq.values().map(|(tf, _)| tf).sum();
        // The `top_k` best so far, the worst of them at the root.
        let mut heap = BinaryHeap::with_capacity(top_k.min(term_freq.len()));
        for (k, (tf, _)) in term_freq.iter() {
            let idf = self.idf_dict.get(*k).unwrap_or(&self.median_idf);
            let node = HeapNode {
                tfidf: OrderedFloat(*tf as f64 * idf / total as f64),
                word: k,
            };
            if heap.len() < top_k {
                heap.push(node);
            } else if let Some(mut worst) = heap.peek_mut()
                && node < *worst
            {
                *worst = node;
            }
        }

        let mut res = Vec::with_capacity(heap.len());
        while let Some(w) = heap.pop() {
            res.push(Keyword {
                keyword: String::from(w.word),
                weight: w.tfidf.into_inner(),
                tag: String::from(term_freq.get(w.word).map_or("", |(_, tag)| tag)),
            });
        }

        res.reverse();
        res
    }
}
