extern crate radix_trie;

use std::io::{self, BufRead, BufReader};
use std::collections::BTreeMap;

use radix_trie::{Trie, TrieCommon};

static DEFAULT_DICT: &str = include_str!("dict/dict.txt");

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

    // FIXME: Use a proper DAG impl?
    fn dag(&self, sentence: &str) -> BTreeMap<usize, Vec<usize>> {
        let mut dag = BTreeMap::new();
        let char_indices: Vec<(usize, char)> = sentence.char_indices().collect();
        let word_count = char_indices.len();
        let mut char_buf = [0; 4];
        for (k, &(start_index, chr)) in char_indices.iter().enumerate() {
            let mut tmplist = Vec::new();
            let mut i = k;
            let mut wfrag: &str = chr.encode_utf8(&mut char_buf);
            while i < word_count {
                if let Some(freq) = self.freq.get(wfrag) {
                    if *freq > 0 {
                        tmplist.push(i);
                    }
                    i += 1;
                    wfrag = if i + 1 < word_count {
                        let end_index = char_indices[i + 1].0;
                        &sentence[start_index..end_index]
                    } else {
                        &sentence[start_index..]
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
}

#[cfg(test)]
mod tests {
    use super::Jieba;

    #[test]
    fn test_init_with_default_dict() {
        let _ = Jieba::new();
    }

    #[test]
    fn test_dag() {
        let jieba = Jieba::new();
        let dag = jieba.dag("网球拍卖会");
        assert_eq!(dag[&0], vec![0, 1, 2]);
        assert_eq!(dag[&1], vec![1, 2]);
        assert_eq!(dag[&2], vec![2, 3, 4]);
        assert_eq!(dag[&3], vec![3]);
        assert_eq!(dag[&4], vec![4]);
    }
}
