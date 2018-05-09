extern crate smallvec;
extern crate regex;
#[macro_use]
extern crate lazy_static;
extern crate phf;
extern crate fxhash;

use std::io::{self, BufRead, BufReader};
use std::collections::BTreeMap;
use std::cmp::Ordering;

use regex::{Regex, Captures, CaptureMatches};
use smallvec::SmallVec;
use fxhash::FxHashMap;

mod hmm;

static DEFAULT_DICT: &str = include_str!("data/dict.txt");

type DAG = BTreeMap<usize, SmallVec<[usize; 5]>>;

lazy_static! {
    static ref RE_HAN_DEFAULT: Regex = Regex::new(r"([\u{4E00}-\u{9FD5}a-zA-Z0-9+#&\._%]+)").unwrap();
    static ref RE_SKIP_DEAFULT: Regex = Regex::new(r"(\r\n|\s)").unwrap();
    static ref RE_HAN_CUT_ALL: Regex = Regex::new("([\u{4E00}-\u{9FD5}]+)").unwrap();
    static ref RE_SKIP_CUT_ALL: Regex = Regex::new(r"[^a-zA-Z0-9+#\n]").unwrap();
}

struct SplitCaptures<'r, 't> {
    finder: CaptureMatches<'r, 't>,
    text: &'t str,
    last: usize,
    caps: Option<Captures<'t>>,
}

impl<'r, 't> SplitCaptures<'r, 't> {
    #[inline]
    fn new(re: &'r Regex, text: &'t str) -> SplitCaptures<'r, 't> {
        SplitCaptures {
            finder: re.captures_iter(text),
            text: text,
            last: 0,
            caps: None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum SplitState<'t> {
    Unmatched(&'t str),
    Captured(Captures<'t>),
}

impl<'t> SplitState<'t> {
    #[inline]
    fn as_str(self) -> &'t str {
        match self {
            SplitState::Unmatched(t) => t,
            SplitState::Captured(caps) => caps.get(0).unwrap().as_str(),
        }
    }
}

impl<'r, 't> Iterator for SplitCaptures<'r, 't> {
    type Item = SplitState<'t>;

    fn next(&mut self) -> Option<SplitState<'t>> {
        if let Some(caps) = self.caps.take() {
            return Some(SplitState::Captured(caps));
        }
        match self.finder.next() {
            None => {
                if self.last >= self.text.len() {
                    None
                } else {
                    let s = &self.text[self.last..];
                    self.last = self.text.len();
                    Some(SplitState::Unmatched(s))
                }
            }
            Some(caps) => {
                let m = caps.get(0).unwrap();
                let unmatched = &self.text[self.last..m.start()];
                self.last = m.end();
                self.caps = Some(caps);
                Some(SplitState::Unmatched(unmatched))
            }
        }
    }
}

#[derive(Debug)]
pub struct Jieba {
    freq: FxHashMap<String, usize>,
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
            freq: FxHashMap::default(),
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
                    self.freq.entry(wfrag.to_string()).or_insert(0);
                }
            }
            buf.clear();
        }
        self.total = total;
        Ok(())
    }

    fn calc(&self, sentence: &str, char_indices: &[usize], dag: &DAG) -> Vec<(f64, usize)> {
        let word_count = char_indices.len();
        let mut route = Vec::with_capacity(word_count + 1);
        for _ in 0..word_count + 1 {
            route.push((0.0, 0));
        }
        let logtotal = (self.total as f64).ln();
        for i in (0..word_count).rev() {
            let pair = dag[&i].iter().map(|x| {
                let byte_start = char_indices[i];
                let end_index = x + 1;
                let byte_end = if end_index < char_indices.len() {
                    char_indices[end_index]
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
    fn dag(&self, sentence: &str, char_indices: &[usize]) -> DAG {
        let mut dag = BTreeMap::new();
        let word_count = char_indices.len();
        for (k, &byte_start) in char_indices.iter().enumerate() {
            let mut tmplist = SmallVec::new();
            let mut i = k;
            let mut wfrag = if k + 1 < char_indices.len() {
                &sentence[byte_start..char_indices[k + 1]]
            } else {
                &sentence[byte_start..]
            };
            while i < word_count {
                if let Some(freq) = self.freq.get(wfrag) {
                    if *freq > 0 {
                        tmplist.push(i);
                    }
                    i += 1;
                    wfrag = if i + 1 < word_count {
                        let byte_end = char_indices[i + 1];
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
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = self.dag(sentence, &char_indices);
        let mut words = Vec::with_capacity(char_indices.len() / 2);
        let mut old_j = -1;
        for (k, list) in dag.into_iter() {
            if list.len() == 1 && k as isize > old_j {
                let byte_start = char_indices[k];
                let end_index = list[0] + 1;
                let byte_end = if end_index < char_indices.len() {
                    char_indices[end_index]
                } else {
                    sentence.len()
                };
                words.push(&sentence[byte_start..byte_end]);
                old_j = list[0] as isize;
            } else {
                for j in list.into_iter() {
                    if j > k {
                        let byte_start = char_indices[k];
                        let end_index = j + 1;
                        let byte_end = if end_index < char_indices.len() {
                            char_indices[end_index]
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

    fn cut_dag_no_hmm<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = self.dag(sentence, &char_indices);
        let route = self.calc(sentence, &char_indices, &dag);
        let mut words = Vec::with_capacity(char_indices.len() / 2);
        let mut x = 0;
        let mut buf_indices = Vec::new();
        while x < char_indices.len() {
            let y = route[x].1 + 1;
            let l_indices = &char_indices[x..y];
            let l_str = if y < char_indices.len() {
                &sentence[char_indices[x]..char_indices[y]]
            } else {
                &sentence[char_indices[x]..]
            };
            if l_indices.len() == 1 && l_str.chars().all(|ch| ch.is_ascii_alphanumeric()) {
                buf_indices.push(x);
            } else {
                if !buf_indices.is_empty() {
                    let byte_start = char_indices[buf_indices[0]];
                    let end_index = buf_indices[buf_indices.len() - 1] + 1;
                    let word = if end_index < char_indices.len() {
                        let byte_end = char_indices[end_index];
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };
                    words.push(word);
                    buf_indices.clear();
                }
                let word = if y < char_indices.len() {
                    &sentence[char_indices[x]..char_indices[y]]
                } else {
                    &sentence[char_indices[x]..]
                };
                words.push(word);
            }
            x = y;
        }
        if !buf_indices.is_empty() {
            let byte_start = char_indices[buf_indices[0]];
            let end_index = buf_indices[buf_indices.len() - 1] + 1;
            let word = if end_index < char_indices.len() {
                let byte_end = char_indices[end_index];
                &sentence[byte_start..byte_end]
            } else {
                &sentence[byte_start..]
            };
            words.push(word);
            buf_indices.clear();
        }
        words
    }

    fn cut_dag_hmm<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = self.dag(sentence, &char_indices);
        let route = self.calc(sentence, &char_indices, &dag);
        let mut words = Vec::with_capacity(char_indices.len() / 2);
        let mut x = 0;
        let mut buf_indices = Vec::new();
        while x < char_indices.len() {
            let y = route[x].1 + 1;
            let l_indices = &char_indices[x..y];
            if l_indices.len() == 1 {
                buf_indices.push(x);
            } else {
                if !buf_indices.is_empty() {
                    let byte_start = char_indices[buf_indices[0]];
                    let end_index = buf_indices[buf_indices.len() - 1] + 1;
                    let word = if end_index < char_indices.len() {
                        let byte_end = char_indices[end_index];
                        &sentence[byte_start..byte_end]
                    } else {
                        &sentence[byte_start..]
                    };
                    if buf_indices.len() == 1 {
                        words.push(word);
                    } else {
                        if !self.freq.get(word).map(|x| *x > 0).unwrap_or(false) {
                            words.extend(hmm::cut(word));
                        } else {
                            let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                            loop {
                                if let Some(byte_start) = word_indices.next() {
                                    if let Some(byte_end) = word_indices.peek() {
                                        words.push(&word[byte_start..*byte_end]);
                                    } else {
                                        words.push(&word[byte_start..]);
                                    }
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                    buf_indices.clear();
                }
                let word = if y < char_indices.len() {
                    &sentence[char_indices[x]..char_indices[y]]
                } else {
                    &sentence[char_indices[x]..]
                };
                words.push(word);
            }
            x = y;
        }
        if !buf_indices.is_empty() {
            let byte_start = char_indices[buf_indices[0]];
            let end_index = buf_indices[buf_indices.len() - 1] + 1;
            let word = if end_index < char_indices.len() {
                let byte_end = char_indices[end_index];
                &sentence[byte_start..byte_end]
            } else {
                &sentence[byte_start..]
            };
            if buf_indices.len() == 1 {
                words.push(word);
            } else {
                if !self.freq.get(word).map(|x| *x > 0).unwrap_or(false) {
                    words.extend(hmm::cut(word));
                } else {
                    let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                    loop {
                        if let Some(byte_start) = word_indices.next() {
                            if let Some(byte_end) = word_indices.peek() {
                                words.push(&word[byte_start..*byte_end]);
                            } else {
                                words.push(&word[byte_start..]);
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
            buf_indices.clear();
        }
        words
    }

    pub fn cut_internal<'a>(&self, sentence: &'a str, cut_all: bool, hmm: bool) -> Vec<&'a str> {
        let mut words = Vec::new();
        let re_han: &Regex = if cut_all { &*RE_HAN_CUT_ALL } else { &*RE_HAN_DEFAULT };
        let re_skip: &Regex = if cut_all { &*RE_SKIP_CUT_ALL } else { &*RE_SKIP_DEAFULT };
        let splitter = SplitCaptures::new(&re_han, sentence);
        for state in splitter {
            let block = state.as_str();
            if block.is_empty() {
                continue;
            }
            if re_han.is_match(block) {
                if cut_all {
                    words.extend(self.cut_all_internal(block));
                } else {
                    if hmm {
                        words.extend(self.cut_dag_hmm(block));
                    } else {
                        words.extend(self.cut_dag_no_hmm(block));
                    }
                }
            } else {
                let skip_splitter = SplitCaptures::new(&re_skip, block);
                for skip_state in skip_splitter {
                    let word = skip_state.as_str();
                    if word.is_empty() {
                        continue;
                    }
                    if re_skip.is_match(word) {
                        words.push(word);
                    } else if !cut_all {
                        let mut word_indices = word.char_indices().map(|x| x.0).peekable();
                        loop {
                            if let Some(byte_start) = word_indices.next() {
                                if let Some(byte_end) = word_indices.peek() {
                                    words.push(&word[byte_start..*byte_end]);
                                } else {
                                    words.push(&word[byte_start..]);
                                }
                            } else {
                                break;
                            }
                        }
                    } else {
                        words.push(word);
                    }
                }
            }
        }
        words
    }

    pub fn cut<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        self.cut_internal(sentence, false, hmm)
    }

    pub fn cut_all<'a>(&self, sentence: &'a str) -> Vec<&'a str> {
        self.cut_internal(sentence, true, false)
    }

    pub fn cut_for_search<'a>(&self, sentence: &'a str, hmm: bool) -> Vec<&'a str> {
        let words = self.cut(sentence, hmm);
        let mut new_words = Vec::with_capacity(words.len());
        for word in words {
            let char_indices: Vec<usize> = word.char_indices().map(|x| x.0).collect();
            let char_count = char_indices.len();
            if char_count > 2 {
                for i in 0..char_count - 1 {
                    let byte_start = char_indices[i];
                    let gram2 = if i + 2 < char_count {
                        &word[byte_start..char_indices[i + 2]]
                    } else {
                        &word[byte_start..]
                    };
                    if self.freq.get(gram2).map(|x| *x > 0).unwrap_or(false) {
                        new_words.push(gram2);
                    }
                }
            }
            if char_count > 3 {
                for i in 0..char_count - 2 {
                    let byte_start = char_indices[i];
                    let gram3 = if i + 3 < char_count {
                        &word[byte_start..char_indices[i + 3]]
                    } else {
                        &word[byte_start..]
                    };
                    if self.freq.get(gram3).map(|x| *x > 0).unwrap_or(false) {
                        new_words.push(gram3);
                    }
                }
            }
            new_words.push(word);
        }
        new_words
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
        let char_indices: Vec<usize> = sentence.char_indices().map(|x| x.0).collect();
        let dag = jieba.dag(sentence, &char_indices);
        assert_eq!(dag[&0], SmallVec::from_buf([0, 1, 2]));
        assert_eq!(dag[&1], SmallVec::from_buf([1, 2]));
        assert_eq!(dag[&2], SmallVec::from_buf([2, 3, 4]));
        assert_eq!(dag[&3], SmallVec::from_buf([3]));
        assert_eq!(dag[&4], SmallVec::from_buf([4]));
    }

    #[test]
    fn test_cut_all() {
        let jieba = Jieba::new();
        let words = jieba.cut_all("网球拍卖会");
        assert_eq!(words, vec!["网球", "网球拍", "球拍", "拍卖", "拍卖会"]);
    }

    #[test]
    fn test_cut_dag_no_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut_dag_no_hmm("网球拍卖会");
        assert_eq!(words, vec!["网球", "拍卖会"]);
    }

    #[test]
    fn test_cut_no_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut("abc网球拍卖会def", false);
        assert_eq!(words, vec!["abc", "网球", "拍卖会", "def"]);
    }

    #[test]
    fn test_cut_with_hmm() {
        let jieba = Jieba::new();
        let words = jieba.cut("我们中出了一个叛徒", false);
        assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
        let words = jieba.cut("我们中出了一个叛徒", true);
        assert_eq!(words, vec!["我们", "中出", "了", "一个", "叛徒"]);
    }

    #[test]
    fn test_cut_for_search() {
        let jieba = Jieba::new();
        let words = jieba.cut_for_search("南京市长江大桥", true);
        assert_eq!(words, vec!["南京", "京市", "南京市", "长江", "大桥", "长江大桥"]);
    }
}
