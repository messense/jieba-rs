extern crate radix_trie;

use std::io::{self, BufRead, BufReader};

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
}

#[cfg(test)]
mod tests {
    use super::Jieba;

    #[test]
    fn init_with_default_dict() {
        let _ = Jieba::new();
    }
}
