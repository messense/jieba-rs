use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};

use ordered_float::OrderedFloat;

use super::{JiebaKeywordExtract, Keyword, KeywordExtract, KeywordExtractConfig};
use crate::FxHashMap as HashMap;
use crate::Jieba;

type Weight = f64;

#[derive(Clone)]
struct Edge {
    dst: usize,
    weight: Weight,
}

impl Edge {
    fn new(dst: usize, weight: Weight) -> Edge {
        Edge { dst, weight }
    }
}

type Edges = Vec<Edge>;
type Graph = Vec<Edges>;

struct StateDiagram {
    damping_factor: Weight,
    g: Graph,
}

impl StateDiagram {
    fn new(size: usize) -> Self {
        StateDiagram {
            damping_factor: 0.85,
            g: vec![Vec::new(); size],
        }
    }

    fn add_undirected_edge(&mut self, src: usize, dst: usize, weight: Weight) {
        self.g[src].push(Edge::new(dst, weight));
        self.g[dst].push(Edge::new(src, weight));
    }

    fn rank(&mut self) -> Vec<Weight> {
        let n = self.g.len();
        let default_weight = 1.0 / (n as f64);

        let mut ranking_vector = vec![default_weight; n];

        let mut outflow_weights = vec![0.0; n];
        for (i, v) in self.g.iter().enumerate() {
            outflow_weights[i] = v.iter().map(|e| e.weight).sum();
        }

        for _ in 0..20 {
            for (i, v) in self.g.iter().enumerate() {
                let s: f64 = v
                    .iter()
                    .map(|e| e.weight / outflow_weights[e.dst] * ranking_vector[e.dst])
                    .sum();

                ranking_vector[i] = (1.0 - self.damping_factor) + self.damping_factor * s;
            }
        }

        ranking_vector
    }
}

/// Text rank keywords extraction
///
/// Requires `textrank` feature to be enabled
#[derive(Debug)]
pub struct UnboundTextRank {
    span: usize,
    config: KeywordExtractConfig,
}

impl UnboundTextRank {
    /// Creates an UnboundTextRank.
    ///
    /// # Examples
    ///
    /// New instance with custom stop words. Also uses hmm for unknown words
    /// during segmentation.
    /// ```
    ///    use std::collections::BTreeSet;
    ///    use jieba_rs::{UnboundTextRank, KeywordExtractConfig};
    ///
    ///    let stop_words : BTreeSet<String> =
    ///        BTreeSet::from(["a", "the", "of"].map(|s| s.to_string()));
    ///    UnboundTextRank::new(
    ///        5,
    ///        KeywordExtractConfig::default());
    /// ```
    pub fn new(span: usize, config: KeywordExtractConfig) -> Self {
        UnboundTextRank { span, config }
    }
}

impl Default for UnboundTextRank {
    /// Creates UnboundTextRank with 5 Unicode Scalar Value spans
    fn default() -> Self {
        UnboundTextRank::new(5, KeywordExtractConfig::default())
    }
}

impl JiebaKeywordExtract for UnboundTextRank {
    fn extract_tags(&self, jieba: &Jieba, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        let tags = jieba.tag(sentence, self.config.get_use_hmm());
        let mut allowed_pos_set = BTreeSet::new();

        for s in allowed_pos {
            allowed_pos_set.insert(s);
        }

        let mut word2id: HashMap<String, usize> = HashMap::default();
        let mut unique_words = Vec::new();
        for t in &tags {
            if !allowed_pos_set.is_empty() && !allowed_pos_set.contains(t.tag) {
                continue;
            }

            if word2id.get(t.word).is_none() {
                unique_words.push(String::from(t.word));
                word2id.insert(String::from(t.word), unique_words.len() - 1);
            }
        }

        let mut cooccurence: HashMap<(usize, usize), usize> = HashMap::default();
        for (i, t) in tags.iter().enumerate() {
            if !allowed_pos_set.is_empty() && !allowed_pos_set.contains(t.tag) {
                continue;
            }

            if !self.config.filter(t.word) {
                continue;
            }

            for j in (i + 1)..(i + self.span) {
                if j >= tags.len() {
                    break;
                }

                if !allowed_pos_set.is_empty() && !allowed_pos_set.contains(tags[j].tag) {
                    continue;
                }

                if !self.config.filter(tags[j].word) {
                    continue;
                }

                let u = word2id.get(t.word).unwrap().to_owned();
                let v = word2id.get(tags[j].word).unwrap().to_owned();
                let entry = cooccurence.entry((u, v)).or_insert(0);
                *entry += 1;
            }
        }

        let mut diagram = StateDiagram::new(unique_words.len());
        for (k, &v) in cooccurence.iter() {
            diagram.add_undirected_edge(k.0, k.1, v as f64);
        }

        let ranking_vector = diagram.rank();

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

        let mut res = Vec::new();
        for _ in 0..top_k {
            if let Some(w) = heap.pop() {
                res.push(Keyword {
                    keyword: unique_words[w.word_id].clone(),
                    weight: w.rank.into_inner(),
                });
            }
        }

        res.reverse();
        res
    }
}

/// Text rank keywords extraction with a Jieba instance bound to the type.
///
/// Requires `textrank` feature to be enabled
#[derive(Debug)]
pub struct TextRank<'a> {
    jieba: &'a Jieba,
    unbound_text_rank: UnboundTextRank,
}

impl<'a> TextRank<'a> {
    pub fn new_with_jieba(jieba: &'a Jieba) -> Self {
        TextRank {
            jieba,
            unbound_text_rank: Default::default(),
        }
    }

    /// Add a new stop word
    pub fn add_stop_word(&mut self, word: String) -> bool {
        self.unbound_text_rank.config.add_stop_word(word)
    }

    /// Remove an existing stop word
    pub fn remove_stop_word(&mut self, word: &str) -> bool {
        self.unbound_text_rank.config.remove_stop_word(word)
    }

    /// Replace all stop words with new stop words set
    pub fn set_stop_words(&mut self, stop_words: BTreeSet<String>) {
        self.unbound_text_rank.config.set_stop_words(stop_words)
    }
}

impl<'a> KeywordExtract for TextRank<'a> {
    fn extract_tags(&self, sentence: &str, top_k: usize, allowed_pos: Vec<String>) -> Vec<Keyword> {
        self.unbound_text_rank
            .extract_tags(self.jieba, sentence, top_k, allowed_pos)
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
    fn test_init_textrank() {
        let jieba = Jieba::new();
        let _ = TextRank::new_with_jieba(&jieba);
    }

    #[test]
    fn test_init_state_diagram() {
        let diagram = StateDiagram::new(10);
        assert_eq!(diagram.g.len(), 10);
    }

    #[test]
    fn test_extract_tags() {
        let jieba = Jieba::new();
        let keyword_extractor = TextRank::new_with_jieba(&jieba);
        let mut top_k = keyword_extractor.extract_tags(
            "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。",
            6,
            vec![String::from("ns"), String::from("n"), String::from("vn"), String::from("v")],
        );
        assert_eq!(
            top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
            vec!["吉林", "欧亚", "置业", "实现", "收入", "子公司"]
        );

        top_k = keyword_extractor.extract_tags(
            "It is nice weather in New York City. and今天纽约的天气真好啊，and京华大酒店的张尧经理吃了一只北京烤鸭。and后天纽约的天气不好，and昨天纽约的天气也不好，and北京烤鸭真好吃",
            3,
            vec![],
        );
        assert_eq!(
            top_k.iter().map(|x| &x.keyword).collect::<Vec<&String>>(),
            vec!["纽约", "天气", "不好"]
        );
    }
}
