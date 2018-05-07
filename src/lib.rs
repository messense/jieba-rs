extern crate radix_trie;
extern crate smallvec;

use std::io::{self, BufRead, BufReader};
use std::collections::BTreeMap;
use std::cmp::Ordering;

use radix_trie::Trie;
use smallvec::SmallVec;

static DEFAULT_DICT: &str = include_str!("data/dict.txt");

type DAG = BTreeMap<usize, SmallVec<[usize; 5]>>;

#[derive(Debug)]
pub struct Jieba {
    freq: Trie<String, usize>,
    total: usize
}

impl Default for Jieba {
    fn default() -> Self {
        Jieba::new()
    }
}

impl Jieba {
    pub fn new() -> Self {
        let mut instance = Jieba {
            freq: Trie::new(),
            total: 0
        };
        let mut default_dict = BufReader::new(DEFAULT_DICT.as_bytes());
        instance.load_dict(&mut default_dict).unwrap();
        instance
    }

    pub fn load_dict<R: BufRead>(&mut self, dict: &mut R) -> io::Result<()> {
        let mut buf = String::new();
        let mut total = 0;
        while dict.read_line(&mut buf)? > 0 {
            {
                let parts: Vec<&str> = buf.trim().split(' ').collect();
                let word = parts[0];
                let freq: usize = parts[1].parse().unwrap();
                total += freq;
                self.freq.insert(word.to_string(), freq);
                let char_indices: Vec<usize> = word.char_indices().map(|x| x.0).collect();
                for i in 1..char_indices.len() {
                    let index = char_indices[i];
                    let wfrag = &word[0..index];
                    // XXX: this will do double hashing, should be avoided
                    if self.freq.get(wfrag).is_none() {
                        self.freq.insert(wfrag.to_string(), 0);
                    }
                }
            }
            buf.clear();
        }
        self.total = total;
        Ok(())
    }

    fn calc(&self, sentence: &str, char_indices: &[(usize, char)], dag: &DAG) -> Vec<(f64, usize)> {
        let word_count = char_indices.len();
        let mut route = Vec::with_capacity(word_count + 1);
        for _ in 0..word_count + 1 {
            route.push((0.0, 0));
        }
        let logtotal = (self.total as f64).ln();
        for i in (0..word_count).rev() {
            let pair = dag[&i].iter().map(|x| {
                let byte_start = char_indices[i].0;
                let end_index = x + 1;
                let byte_end = if end_index < char_indices.len() {
                    char_indices[end_index].0
                } else {
                    sentence.len()
                };
                let wfrag = &sentence[byte_start..byte_end];
                let freq = self.freq.get(wfrag).cloned().unwrap_or(1);
                ((freq as f64).ln() - logtotal + route[x + 1].0, *x)
            }).max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
            route[i] = pair.unwrap();
        }
        route
    }

    // FIXME: Use a proper DAG impl?
    fn dag(&self, sentence: &str, char_indices: &[(usize, char)]) -> DAG {
        let mut dag = BTreeMap::new();
        let word_count = char_indices.len();
        let mut char_buf = [0; 4];
        for (k, &(byte_start, chr)) in char_indices.iter().enumerate() {
            let mut tmplist = SmallVec::new();
            let mut i = k;
            let mut wfrag: &str = chr.encode_utf8(&mut char_buf);
            while i < word_count {
                if let Some(freq) = self.freq.get(wfrag) {
                    if *freq > 0 {
                        tmplist.push(i);
                    }
                    i += 1;
                    wfrag = if i + 1 < word_count {
                        let byte_end = char_indices[i + 1].0;
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };
                } else {
                    break;
                }
            }
            if tmplist.is_empty() {
                tmplist.push(k);
            }
            dag.insert(k, tmplist);
        }
        dag
    }

    fn cut_all_internal<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        let char_indices: Vec<(usize, char)> = sentence.char_indices().collect();
        let dag = self.dag(sentence, &char_indices);
        let mut words = Vec::new();
        let mut old_j = -1;
        for (k, list) in dag.into_iter() {
            if list.len() == 1 && k as isize > old_j {
                let byte_start = char_indices[k].0;
                let end_index = list[0] + 1;
                let byte_end = if end_index < char_indices.len() {
                    char_indices[end_index].0
                } else {
                    sentence.len()
                };
                words.push(&sentence[byte_start..byte_end]);
                old_j = list[0] as isize;
            } else {
                for j in list.into_iter() {
                    if j > k {
                        let byte_start = char_indices[k].0;
                        let end_index = j + 1;
                        let byte_end = if end_index < char_indices.len() {
                            char_indices[end_index].0
                        } else {
                            sentence.len()
                        };
                        words.push(&sentence[byte_start..byte_end]);
                        old_j = j as isize;
                    }
                }
            }
        }
        words
    }

    fn cut_dag_no_hmm(&self, sentence: &str) -> Vec<String> {
        let char_indices: Vec<(usize, char)> = sentence.char_indices().collect();
        let dag = self.dag(sentence, &char_indices);
        let route = self.calc(sentence, &char_indices, &dag);
        let mut words = Vec::new();
        let mut x = 0;
        let mut buf = String::new();
        while x < char_indices.len() {
            let y = route[x].1 + 1;
            let l_indices = &char_indices[x..y];
            if l_indices.len() == 1 && l_indices.iter().all(|ch| ch.1.is_ascii_alphanumeric()) {
                buf.push(l_indices[0].1);
                x = y;
            } else {
                if !buf.is_empty() {
                    words.push(buf.clone());
                    buf.clear();
                }
                words.push(l_indices.iter().map(|ch| ch.1).collect());
                x = y;
            }
        }
        if !buf.is_empty() {
            words.push(buf.clone());
            buf.clear();
        }
        words
    }
}

#[cfg(test)]
mod tests {
    use smallvec::SmallVec;
    use super::Jieba;

    #[test]
    fn test_init_with_default_dict() {
        let _ = Jieba::new();
    }

    #[test]
    fn test_dag() {
        let jieba = Jieba::new();
        let sentence = "网球拍卖会";
        let char_indices: Vec<(usize, char)> = sentence.char_indices().collect();
        let dag = jieba.dag(sentence, &char_indices);
        assert_eq!(dag[&0], SmallVec::from_buf([0, 1, 2]));
        assert_eq!(dag[&1], SmallVec::from_buf([1, 2]));
        assert_eq!(dag[&2], SmallVec::from_buf([2, 3, 4]));
        assert_eq!(dag[&3], SmallVec::from_buf([3]));
        assert_eq!(dag[&4], SmallVec::from_buf([4]));
    }

    #[test]
    fn test_cut_all_internal() {
        let jieba = Jieba::new();
        let words = jieba.cut_all_internal("网球拍卖会");
        assert_eq!(words, vec!["网球", "网球拍", "球拍", "拍卖", "拍卖会"]);
    }

    #[test]
    fn test_cut_dag_no_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut_dag_no_hmm("网球拍卖会");
        assert_eq!(words, vec!["网球", "拍卖会"]);
    }
}
