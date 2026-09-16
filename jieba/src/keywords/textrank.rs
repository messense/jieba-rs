use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};

use ordered_float::OrderedFloat;

use super::{Keyword, KeywordExtract, KeywordExtractConfig, KeywordExtractConfigBuilder};
use crate::FxHashMap as HashMap;
use crate::Jieba;

type Weight = f64;

/// The co-occurrence graph, in compressed sparse row form: the neighbours
/// of vertex `v` are `dst[offsets[v]..offsets[v + 1]]`, and each carries the
/// share of `dst`'s rank it hands to `v`, `edge weight / total weight out of
/// dst`, which is constant across the ranking sweeps.
struct StateDiagram {
    damping_factor: Weight,
    offsets: Vec<usize>,
    dst: Vec<usize>,
    share: Vec<Weight>,
}

impl StateDiagram {
    /// Build the graph of `size` vertices from undirected weighted edges.
    /// Each vertex's neighbours keep the order the edges were given in.
    fn new(size: usize, edges: &[(usize, usize, Weight)]) -> Self {
        // Counting sort by source: each undirected edge is an entry in both
        // endpoints' lists.
        let mut offsets = vec![0usize; size + 1];
        for &(u, v, _) in edges {
            offsets[u + 1] += 1;
            offsets[v + 1] += 1;
        }
        for i in 0..size {
            offsets[i + 1] += offsets[i];
        }
        let m = offsets[size];
        let mut dst = vec![0usize; m];
        let mut weight = vec![0.0; m];
        let mut next = offsets[..size].to_vec();
        for &(u, v, w) in edges {
            dst[next[u]] = v;
            weight[next[u]] = w;
            next[u] += 1;
            dst[next[v]] = u;
            weight[next[v]] = w;
            next[v] += 1;
        }

        let mut outflow = vec![0.0; size];
        for v in 0..size {
            outflow[v] = weight[offsets[v]..offsets[v + 1]].iter().sum();
        }
        let share = dst.iter().zip(&weight).map(|(&d, &w)| w / outflow[d]).collect();

        StateDiagram {
            damping_factor: 0.85,
            offsets,
            dst,
            share,
        }
    }

    fn rank(&self) -> Vec<Weight> {
        let n = self.offsets.len() - 1;
        let default_weight = 1.0 / (n as f64);

        let mut ranking_vector = vec![default_weight; n];

        for _ in 0..20 {
            for i in 0..n {
                let range = self.offsets[i]..self.offsets[i + 1];
                let s: f64 = self.dst[range.clone()]
                    .iter()
                    .zip(&self.share[range])
                    .map(|(&d, &share)| share * ranking_vector[d])
                    .sum();

                ranking_vector[i] = (1.0 - self.damping_factor) + self.damping_factor * s;
            }
        }

        ranking_vector
    }
}

/// Text rank keywords' extraction.
///
/// Requires `textrank` feature to be enabled.
#[derive(Debug)]
pub struct TextRank {
    span: usize,
    config: KeywordExtractConfig,
}

impl TextRank {
    /// Creates an TextRank.
    ///
    /// # Examples
    ///
    /// New instance with custom stop words. Also uses hmm for unknown words
    /// during segmentation.
    ///
    /// ```
    /// use std::collections::BTreeSet;
    /// use jieba_rs::{TextRank, KeywordExtractConfig};
    ///
    /// let stop_words : BTreeSet<String> =
    ///     BTreeSet::from(["a", "the", "of"].map(|s| s.to_string()));
    ///
    /// TextRank::new(5, KeywordExtractConfig::default());
    /// ```
    pub fn new(span: usize, config: KeywordExtractConfig) -> Self {
        TextRank { span, config }
    }
}

impl Default for TextRank {
    /// Creates TextRank with 5 Unicode Scalar Value spans
    fn default() -> Self {
        TextRank::new(5, KeywordExtractConfigBuilder::default().build())
    }
}

impl KeywordExtract for TextRank {
    /// Uses TextRank algorithm to extract the `top_k` keywords from `sentence`.
    ///
    /// If `allowed_pos` is not empty, then only terms matching those parts if
    /// speech are considered.
    ///
    /// # Examples
    ///
    /// ```
    /// use jieba_rs::{Jieba, KeywordExtract, TextRank};
    ///
    /// let jieba = Jieba::new();
    /// let keyword_extractor = TextRank::default();
    /// let mut top_k = keyword_extractor.extract_keywords(
    ///     &jieba,
    ///     "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
    ///     6,
    ///     vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
    /// );
    /// assert_eq!(
    ///     top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///     vec!["吉林", "欧亚", "置业", "实现", "收入", "子公司"],
    /// );
    ///
    /// top_k = keyword_extractor.extract_keywords(
    ///     &jieba,
    ///     "It is nice weather in New York City. and今天纽约的天气真好啊，and京华大酒店的张尧经理吃了一只北京烤鸭。and后天纽约的天气不好，and昨天纽约的天气也不好，and北京烤鸭真好吃",
    ///     3,
    ///     vec![],
    /// );
    /// assert_eq!(
    ///     top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
    ///     vec!["纽约", "天气", "不好"],
    /// );
    /// ```
    fn extract_keywords(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        let tags = jieba.tag(sentence, self.config.use_hmm());
        let mut allowed_pos_set = BTreeSet::new();

        for s in allowed_pos {
            allowed_pos_set.insert(s);
        }

        let mut word2id: HashMap<&str, usize> =
            HashMap::with_capacity_and_hasher(tags.len() / 2, rustc_hash::FxBuildHasher);
        // Each candidate word with the tag of its first occurrence, by id.
        let mut unique_words: Vec<(&str, &str)> = Vec::with_capacity(tags.len() / 2);
        for t in &tags {
            if !allowed_pos_set.is_empty() && !allowed_pos_set.contains(t.tag) {
                continue;
            }
            if !self.config.is_keyword(t.word) {
                continue;
            }

            let next_id = unique_words.len();
            word2id.entry(t.word).or_insert_with(|| {
                unique_words.push((t.word, t.tag));
                next_id
            });
        }

        let candidate_ids: Vec<Option<usize>> = tags
            .iter()
            .map(|t| {
                if !allowed_pos_set.is_empty() && !allowed_pos_set.contains(t.tag) {
                    return None;
                }

                word2id.get(t.word).copied()
            })
            .collect();

        let mut cooccurence: HashMap<(usize, usize), usize> = HashMap::default();
        for (i, &u) in candidate_ids.iter().enumerate() {
            let Some(u) = u else { continue };

            for &v in candidate_ids.iter().take(i + self.span).skip(i + 1) {
                let Some(v) = v else { continue };
                let entry = cooccurence.entry((u, v)).or_insert(0);
                *entry += 1;
            }
        }

        let edges: Vec<(usize, usize, Weight)> = cooccurence.iter().map(|(&(u, v), &c)| (u, v, c as f64)).collect();
        let ranking_vector = StateDiagram::new(unique_words.len(), &edges).rank();

        let mut heap = BinaryHeap::new();
        for (k, v) in ranking_vector.iter().enumerate() {
            heap.push(HeapNode {
                rank: OrderedFloat(v * 1e10),
                word_id: k,
            });

            if k >= top_k {
                heap.pop();
            }
        }

        let mut res = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            if let Some(w) = heap.pop() {
                let (word, tag) = unique_words[w.word_id];
                res.push(Keyword {
                    keyword: word.to_string(),
                    weight: w.rank.into_inner(),
                    tag: String::from(tag),
                });
            }
        }

        res.reverse();
        res
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapNode {
    rank: OrderedFloat<f64>,
    word_id: usize,
}

impl Ord for HeapNode {
    fn cmp(&self, other: &HeapNode) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| self.word_id.cmp(&other.word_id))
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &HeapNode) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_state_diagram() {
        let diagram = StateDiagram::new(10, &[]);
        assert_eq!(diagram.offsets.len(), 11);
        assert!(diagram.dst.is_empty());
    }

    #[test]
    fn test_extract_keywords_filters_invalid_candidates() {
        let jieba = Jieba::new();
        let config = KeywordExtractConfig::builder()
            .min_keyword_length(2)
            .add_stop_word("股票")
            .build();
        let extractor = TextRank::new(5, config);

        let keywords = extractor.extract_keywords(&jieba, "今天股票跌很厉害，股票又跌", 100, vec![]);

        assert!(!keywords.is_empty());
        assert!(keywords.iter().all(|keyword| keyword.keyword.chars().count() >= 2));
        assert!(keywords.iter().all(|keyword| keyword.keyword != "股票"));
    }
}
