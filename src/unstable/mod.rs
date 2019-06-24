use crate::hmm;
use crate::{SplitMatches, SplitState, DEFAULT_DICT, RE_HAN_DEFAULT, RE_SKIP_DEAFULT};
use darts::{DoubleArrayTrie, DoubleArrayTrieBuilder};
use regex::Regex;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::io::{self, BufRead, BufReader};

type DAG = Vec<SmallVec<[(usize, Option<usize>); 5]>>;

struct IndexBuilder {}

impl IndexBuilder {
    pub fn new() -> Self {
        IndexBuilder {}
    }

    // Require the dictionary to be sorted in lexicographical order
    pub fn build<R: BufRead>(&mut self, dict: &mut R) -> io::Result<Index> {
        let mut buf = String::new();
        let mut records: Vec<(String, usize, String)> = Vec::new();
        let mut prev_word = String::new();

        while dict.read_line(&mut buf)? > 0 {
            {
                let parts: Vec<&str> = buf.trim().split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }

                let word = parts[0];
                let freq = parts.get(1).map(|x| x.parse::<usize>().unwrap()).unwrap_or(0);
                let tag = parts.get(2).cloned().unwrap_or("");

                assert!(
                    (&*prev_word < word),
                    "the dictionary has to be sorted in lexicographical order."
                );
                prev_word = word.to_string();

                records.push((String::from(word), freq, String::from(tag)));
            }
            buf.clear();
        }

        let strs: Vec<&str> = records.iter().map(|n| n.0.as_ref()).collect();
        let total: usize = records.iter().map(|n| n.1).sum();
        let da = DoubleArrayTrieBuilder::new().build(&strs);

        let index = Index {
            da: da,
            records: records,
            total: total,
        };

        Ok(index)
    }
}

#[derive(Debug)]
struct Index {
    da: DoubleArrayTrie,
    records: Vec<(String, usize, String)>,
    total: usize,
}

#[derive(Debug)]
pub struct JiebaUnstable {
    index: Index,
}

impl JiebaUnstable {
    /// Create a new instance with empty dict
    pub fn new() -> Self {
        let mut f = BufReader::new(DEFAULT_DICT.as_bytes());
        let index = IndexBuilder::new().build(&mut f).unwrap();

        JiebaUnstable { index: index }
    }

    fn calc(&self, sentence: &str, dag: &DAG) -> Vec<(f64, usize)> {
        let str_len = sentence.len();
        let mut route = Vec::with_capacity(str_len + 1);
        for _ in 0..=str_len {
            route.push((0.0, 0));
        }

        let logtotal = (self.index.total as f64).ln();
        for i in (0..str_len).rev() {
            let pair = dag[i]
                .iter()
                .map(|x| {
                    let end_index = x.0;
                    let freq = if let Some(word_id) = x.1 {
                        self.index.records[word_id].1
                    } else {
                        1
                    };

                    ((freq as f64).ln() - logtotal + route[i + end_index].0, i + end_index)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
            route[i] = pair.unwrap_or((0.0, 0));
        }
        route
    }

    fn dag(&self, sentence: &str) -> DAG {
        let str_len = sentence.len();
        let mut dag = Vec::with_capacity(str_len);

        for _ in 0..str_len {
            dag.push(SmallVec::new());
        }

        let mut iter = sentence.char_indices().peekable();
        while let Some((byte_start, _)) = iter.next() {
            let mut tmplist = SmallVec::new();
            let haystack = &sentence[byte_start..];

            for (end_index, word_id) in self.index.da.common_prefix_iter(haystack) {
                tmplist.push((end_index, Some(word_id)));
            }

            if tmplist.is_empty() {
                let diff = if let Some((next, _)) = iter.peek() {
                    next - byte_start
                } else {
                    sentence.len() - byte_start
                };

                tmplist.push((diff, None));
            }

            dag[byte_start] = tmplist;
        }
        dag
    }

    fn cut_dag_no_hmm<'a>(&self, sentence: &'a str, buf_indices: &mut Vec<usize>, words: &mut Vec<&'a str>) {
        let dag = self.dag(sentence);
        let route = self.calc(sentence, &dag);
        let mut x = 0;
        while x < sentence.len() {
            let y = if route[x].1 == 0 { sentence.len() } else { route[x].1 };

            let l_str = if y < sentence.len() {
                &sentence[x..y]
            } else {
                &sentence[x..]
            };

            if l_str.chars().count() == 1 && l_str.chars().all(|ch| ch.is_ascii_alphanumeric()) {
                buf_indices.push(x);
            } else {
                if !buf_indices.is_empty() {
                    let byte_start = buf_indices[0];
                    let byte_end = x;

                    let word = if byte_end < sentence.len() {
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };

                    words.push(word);
                    buf_indices.clear();
                }

                let word = if y < sentence.len() {
                    &sentence[x..y]
                } else {
                    &sentence[x..]
                };

                words.push(word);
            }

            x = y;
        }

        if !buf_indices.is_empty() {
            let byte_start = buf_indices[0];
            let word = &sentence[byte_start..];
            words.push(word);
            buf_indices.clear();
        }
    }

    fn cut_dag_hmm<'a>(&self, sentence: &'a str, buf_indices: &mut Vec<usize>, words: &mut Vec<&'a str>) {
        let dag = self.dag(sentence);
        let route = self.calc(sentence, &dag);
        let mut x = 0;

        while x < sentence.len() {
            let y = if route[x].1 == 0 { sentence.len() } else { route[x].1 };

            if sentence[x..y].chars().count() == 1 {
                buf_indices.push(x);
            } else {
                if !buf_indices.is_empty() {
                    let byte_start = buf_indices[0];
                    let byte_end = x;
                    let word = if byte_end < sentence.len() {
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };

                    if buf_indices.len() == 1 {
                        words.push(word);
                    } else if self.index.da.exact_match_search(word).is_none() {
                        words.extend(hmm::cut(word));
                    } else {
                        let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                        while let Some(byte_start) = word_indices.next() {
                            if let Some(byte_end) = word_indices.peek() {
                                words.push(&word[byte_start..*byte_end]);
                            } else {
                                words.push(&word[byte_start..]);
                            }
                        }
                    }
                    buf_indices.clear();
                }
                let word = if y < sentence.len() {
                    &sentence[x..y]
                } else {
                    &sentence[x..]
                };
                words.push(word);
            }
            x = y;
        }

        if !buf_indices.is_empty() {
            let byte_start = buf_indices[0];
            let word = &sentence[byte_start..];

            if buf_indices.len() == 1 {
                words.push(word);
            } else if self.index.da.exact_match_search(word).is_none() {
                words.extend(hmm::cut(word));
            } else {
                let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                while let Some(byte_start) = word_indices.next() {
                    if let Some(byte_end) = word_indices.peek() {
                        words.push(&word[byte_start..*byte_end]);
                    } else {
                        words.push(&word[byte_start..]);
                    }
                }
            }

            buf_indices.clear();
        }
    }

    fn cut_internal<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        let mut words = Vec::with_capacity(sentence.len() / 2);
        let re_han: &Regex = &*RE_HAN_DEFAULT;
        let re_skip: &Regex = &*RE_SKIP_DEAFULT;
        let splitter = SplitMatches::new(&re_han, sentence);
        let mut buf_indices = Vec::with_capacity(sentence.len() / 2);

        for state in splitter {
            match state {
                SplitState::Matched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());

                    if hmm {
                        self.cut_dag_hmm(block, &mut buf_indices, &mut words);
                    } else {
                        self.cut_dag_no_hmm(block, &mut buf_indices, &mut words);
                    }
                }
                SplitState::Unmatched(_) => {
                    let block = state.into_str();
                    assert!(!block.is_empty());

                    let skip_splitter = SplitMatches::new(&re_skip, block);
                    for skip_state in skip_splitter {
                        let word = skip_state.into_str();
                        if word.is_empty() {
                            continue;
                        }
                        if re_skip.is_match(word) {
                            words.push(word);
                        } else {
                            let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                            while let Some(byte_start) = word_indices.next() {
                                if let Some(byte_end) = word_indices.peek() {
                                    words.push(&word[byte_start..*byte_end]);
                                } else {
                                    words.push(&word[byte_start..]);
                                }
                            }
                        }
                    }
                }
            }
        }
        words
    }

    pub fn cut<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        self.cut_internal(sentence, hmm)
    }
}

#[cfg(test)]
mod tests {
    use super::JiebaUnstable;

    #[test]
    fn test_cut_no_hmm() {
        let jieba = JiebaUnstable::new();
        let words1 = jieba.cut("网球拍卖会", false);
        assert_eq!(words1, vec!["网球", "拍卖会"]);

        let words2 = jieba.cut(
            "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            false,
        );
        assert_eq!(
            words2,
            vec![
                "我",
                "是",
                "拖拉机",
                "学院",
                "手扶拖拉机",
                "专业",
                "的",
                "。",
                "不用",
                "多久",
                "，",
                "我",
                "就",
                "会",
                "升职",
                "加薪",
                "，",
                "当",
                "上",
                "CEO",
                "，",
                "走上",
                "人生",
                "巅峰",
                "。"
            ]
        );
    }

    #[test]
    fn test_cut_no_hmm_with_alphabet() {
        let jieba = JiebaUnstable::new();
        let words = jieba.cut("abc网球拍卖会def", false);
        assert_eq!(words, vec!["abc", "网球", "拍卖会", "def"]);
    }

    #[test]
    fn test_cut_with_hmm() {
        let jieba = JiebaUnstable::new();
        let words = jieba.cut("我们中出了一个叛徒", false);
        assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
        let words = jieba.cut("我们中出了一个叛徒", true);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒"]);
        let words = jieba.cut("我们中出了一个叛徒👪", true);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒", "👪"]);
    }

    #[test]
    fn test_cut_weicheng() {
        static WEICHENG_TXT: &str = include_str!("../../examples/weicheng/src/weicheng.txt");
        let jieba = JiebaUnstable::new();
        for line in WEICHENG_TXT.split('\n') {
            let _ = jieba.cut(line, true);
        }
    }
}
