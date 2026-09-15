//! A character-level trie for the dictionary, built for the segmenter's
//! access pattern: from every character of the input, enumerate the
//! dictionary words that start there.
//!
//! Nodes are prefixes of dictionary words. The root's transitions for BMP
//! characters live in a directly indexed table, so the first step of every
//! walk is a cache-resident array load. Every other transition is one entry
//! in an open-addressing hash table keyed by `(parent node, character)`, and
//! the entry carries everything a walk needs to know about the child: its
//! id, the id of the word that ends there, and a small Bloom-style filter of
//! the characters its own children start with. One probe per character, no
//! separate lookup to test for a word boundary, and no probe at all for the
//! common case of a character that is not a child, against the several
//! dependent loads per *byte* of a byte-level double-array trie.

/// Word id stored on a node that does not end a dictionary word.
const NO_WORD: i32 = -1;

/// Widest character the key packing allows: 21 bits, all of Unicode.
const CHAR_BITS: u32 = 21;

/// A slot key is `(parent << CHAR_BITS | char) + 1` in the low bits; the
/// high 16 bits hold the first child filter.
const KEY_BITS: u32 = 48;
const KEY_MASK: u64 = (1 << KEY_BITS) - 1;

/// A child pointer is the node id in the low bits and the second child
/// filter in the high 8 bits. A node has children iff its filter bits are
/// not all zero.
const ID_BITS: u32 = 24;
const ID_MASK: u32 = (1 << ID_BITS) - 1;
const MAX_NODES: u32 = 1 << ID_BITS;

/// Characters with a direct root transition.
const ROOT_TABLE_LEN: usize = 0x1_0000;

const MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

/// Load factor above which the table doubles, in eighths.
const MAX_LOAD_EIGHTHS: usize = 6;

#[derive(Clone, Copy, Default)]
struct Slot {
    /// Packed `(parent, char)` plus one, and the first child filter; 0
    /// marks an empty slot.
    key: u64,
    /// Child node id and the second child filter.
    child: u32,
    word_id: i32,
}

/// A transition out of the root: child pointer (0 for none) and word id.
#[derive(Clone, Copy)]
struct RootEntry {
    child: u32,
    word_id: i32,
}

/// Reusable buffers for [`CharTrie::for_each_prefix_at_every_char`].
#[derive(Clone, Default)]
pub(crate) struct WalkScratch {
    /// Byte offset and value of each character of the text being walked.
    chars: Vec<(u32, char)>,
}

/// Where a node's child pointer is stored.
#[derive(Clone, Copy)]
enum ParentRef {
    /// The node is the root itself, which has no pointer.
    Root,
    RootEntry(usize),
    Slot(usize),
}

#[derive(Clone)]
pub(crate) struct CharTrie {
    root: Vec<RootEntry>,
    slots: Vec<Slot>,
    /// Occupied slots.
    len: usize,
    /// `64 - log2(slots.len())`, so a hash shifted right by it indexes the table.
    shift: u32,
    /// Next node id; ids start at 1 because 0 is the root.
    next_node: u32,
}

#[inline(always)]
fn key(parent: u32, ch: char) -> u64 {
    (((parent as u64) << CHAR_BITS) | ch as u64) + 1
}

/// The bits a character sets in a node's two child filters: one of 16 in
/// the key's high bits, one of 8 in the child pointer's high bits. A
/// lookup probes only when both are set; with the fan-out this dictionary
/// has (median 1, p90 3) that rejects most characters that are not
/// children without touching the hash table.
#[inline(always)]
fn filter_bits(ch: char) -> (u64, u32) {
    let h = (ch as u32).wrapping_mul(0x9E37_79B1);
    (1u64 << (KEY_BITS + (h >> 28)), 1u32 << (ID_BITS + ((h >> 25) & 7)))
}

impl Default for CharTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl CharTrie {
    pub(crate) fn new() -> Self {
        Self::with_slots(1 << 10)
    }

    fn with_slots(slots: usize) -> Self {
        debug_assert!(slots.is_power_of_two());
        CharTrie {
            root: vec![
                RootEntry {
                    child: 0,
                    word_id: NO_WORD
                };
                ROOT_TABLE_LEN
            ],
            slots: vec![Slot::default(); slots],
            len: 0,
            shift: 64 - slots.trailing_zeros(),
            next_node: 1,
        }
    }

    /// Make room for about `words` more words without rehashing.
    pub(crate) fn reserve(&mut self, words: usize) {
        // The default dictionary has 1.43 nodes per word; leave headroom.
        let nodes = self.len + words + words / 2;
        let needed = (nodes * 8).div_ceil(MAX_LOAD_EIGHTHS).next_power_of_two();
        if needed > self.slots.len() {
            self.rehash(needed);
        }
    }

    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        (key.wrapping_mul(MULTIPLIER) >> self.shift) as usize
    }

    /// Index of the slot holding the (filter-free) `key`, or of the empty
    /// slot where it would go.
    #[inline(always)]
    fn probe(&self, key: u64) -> usize {
        let mask = self.slots.len() - 1;
        let mut i = self.index(key);
        loop {
            let stored = self.slots[i].key;
            if stored & KEY_MASK == key || stored == 0 {
                return i;
            }
            i = (i + 1) & mask;
        }
    }

    /// The root's transition on `ch`: `(child pointer, word id)`.
    #[inline(always)]
    fn root_step(&self, ch: char) -> Option<(u32, i32)> {
        if (ch as usize) < ROOT_TABLE_LEN {
            let entry = self.root[ch as usize];
            return (entry.child != 0).then_some((entry.child, entry.word_id));
        }
        let slot = &self.slots[self.probe(key(0, ch))];
        (slot.key != 0).then_some((slot.child, slot.word_id))
    }

    /// The id of `word`, if it is in the dictionary.
    pub(crate) fn get(&self, word: &str) -> Option<i32> {
        let mut chars = word.chars();
        let (mut child, mut word_id) = self.root_step(chars.next()?)?;
        // The root has no key filter; only the pointer filter applies.
        let mut key_filter = u64::MAX;
        for ch in chars {
            let (a, b) = filter_bits(ch);
            if key_filter & a == 0 || child & b == 0 {
                return None;
            }
            let slot = &self.slots[self.probe(key(child & ID_MASK, ch))];
            if slot.key == 0 {
                return None;
            }
            key_filter = slot.key;
            child = slot.child;
            word_id = slot.word_id;
        }
        (word_id != NO_WORD).then_some(word_id)
    }

    /// Call `emit(char_index, byte_end, word_id)` for every dictionary word
    /// starting at every character of `text`, position by position and
    /// shortest word first within a position.
    pub(crate) fn for_each_prefix_at_every_char(
        &self,
        text: &str,
        scratch: &mut WalkScratch,
        mut emit: impl FnMut(usize, usize, i32),
    ) {
        // Walking a position is a chain of dependent loads that the processor
        // cannot overlap with the next position's chain, so the slot the
        // second character of a position a few steps ahead will need is
        // touched now. By the time the walk arrives it is in cache.
        const LOOKAHEAD: usize = 3;

        let chars = &mut scratch.chars;
        chars.clear();
        chars.extend(text.char_indices().map(|(offset, ch)| (offset as u32, ch)));
        let n = chars.len();
        let end_of_text = text.len();

        for pos in 0..n {
            if let Some(&[(_, c1), (_, c2)]) = chars.get(pos + LOOKAHEAD..pos + LOOKAHEAD + 2)
                && (c1 as usize) < ROOT_TABLE_LEN
            {
                let child = self.root[c1 as usize].child;
                if child & filter_bits(c2).1 != 0 {
                    let i = self.index(key(child & ID_MASK, c2));
                    std::hint::black_box(self.slots[i].key);
                }
            }

            let (_, first) = chars[pos];
            let Some((mut child, mut word_id)) = self.root_step(first) else {
                continue;
            };
            let mut key_filter = u64::MAX;
            let mut next = pos + 1;
            loop {
                let end = chars.get(next).map_or(end_of_text, |&(offset, _)| offset as usize);
                if word_id != NO_WORD {
                    emit(pos, end, word_id);
                }
                if next >= n {
                    break;
                }
                let (_, ch) = chars[next];
                let (a, b) = filter_bits(ch);
                if key_filter & a == 0 || child & b == 0 {
                    break;
                }
                let slot = &self.slots[self.probe(key(child & ID_MASK, ch))];
                if slot.key == 0 {
                    break;
                }
                key_filter = slot.key;
                child = slot.child;
                word_id = slot.word_id;
                next += 1;
            }
        }
    }

    /// Set the id of `word`, creating its nodes as needed. Returns the id it
    /// had before, if any. An empty word is ignored and returns `None`.
    pub(crate) fn insert(&mut self, word: &str, word_id: i32) -> Option<i32> {
        debug_assert!(word_id >= 0, "word ids are non-negative");
        let mut chars = word.chars().peekable();
        let mut parent = 0u32;
        // Where the pointer to `parent` is stored, so its child filters can
        // be updated when `parent` gains a child.
        let mut parent_ref = ParentRef::Root;
        while let Some(ch) = chars.next() {
            let last = chars.peek().is_none();
            if parent == 0 && (ch as usize) < ROOT_TABLE_LEN {
                let i = ch as usize;
                if self.root[i].child == 0 {
                    self.root[i].child = self.alloc_node();
                }
                if last {
                    return Some(std::mem::replace(&mut self.root[i].word_id, word_id)).filter(|&old| old != NO_WORD);
                }
                parent = self.root[i].child & ID_MASK;
                parent_ref = ParentRef::RootEntry(i);
                continue;
            }

            let (a, b) = filter_bits(ch);
            match parent_ref {
                ParentRef::Root => {}
                ParentRef::RootEntry(i) => self.root[i].child |= b,
                ParentRef::Slot(i) => {
                    self.slots[i].key |= a;
                    self.slots[i].child |= b;
                }
            }
            let key = key(parent, ch);
            let mut i = self.probe(key);
            if self.slots[i].key == 0 {
                if self.grow_if_needed() {
                    i = self.probe(key);
                }
                self.slots[i] = Slot {
                    key,
                    child: self.alloc_node(),
                    word_id: NO_WORD,
                };
                self.len += 1;
            }
            if last {
                return Some(std::mem::replace(&mut self.slots[i].word_id, word_id)).filter(|&old| old != NO_WORD);
            }
            parent = self.slots[i].child & ID_MASK;
            parent_ref = ParentRef::Slot(i);
        }
        None
    }

    #[inline]
    fn alloc_node(&mut self) -> u32 {
        let id = self.next_node;
        assert!(id < MAX_NODES, "dictionary has too many distinct prefixes");
        self.next_node += 1;
        id
    }

    /// Double the table if adding one entry would exceed the load limit.
    /// Returns whether it did, which invalidates slot indices.
    fn grow_if_needed(&mut self) -> bool {
        if (self.len + 1) * 8 > self.slots.len() * MAX_LOAD_EIGHTHS {
            self.rehash(self.slots.len() * 2);
            true
        } else {
            false
        }
    }

    fn rehash(&mut self, slots: usize) {
        let old = std::mem::replace(&mut self.slots, vec![Slot::default(); slots]);
        self.shift = 64 - slots.trailing_zeros();
        for slot in old.into_iter().filter(|s| s.key != 0) {
            let i = self.probe(slot.key & KEY_MASK);
            self.slots[i] = slot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dictionary words that are prefixes of `text`, shortest first.
    fn prefixes<'a>(trie: &CharTrie, text: &'a str) -> Vec<(i32, &'a str)> {
        let mut out = Vec::new();
        let mut scratch = WalkScratch::default();
        trie.for_each_prefix_at_every_char(text, &mut scratch, |pos, end, id| {
            if pos == 0 {
                out.push((id, &text[..end]));
            }
        });
        out
    }

    #[test]
    fn test_prefixes_at_every_char() {
        let mut trie = CharTrie::new();
        for (i, w) in ["a", "ab", "abc", "b", "bc", "c", "中", "中国", "国"]
            .iter()
            .enumerate()
        {
            trie.insert(w, i as i32);
        }
        let text = "abc中国x";
        let mut out = Vec::new();
        let mut scratch = WalkScratch::default();
        trie.for_each_prefix_at_every_char(text, &mut scratch, |pos, end, id| out.push((pos, end, id)));
        // Position by position, shortest word first.
        assert_eq!(
            out,
            vec![
                (0, 1, 0),
                (0, 2, 1),
                (0, 3, 2),
                (1, 2, 3),
                (1, 3, 4),
                (2, 3, 5),
                (3, 6, 6),
                (3, 9, 7),
                (4, 9, 8)
            ]
        );
    }

    #[test]
    fn test_insert_get_prefixes() {
        let mut trie = CharTrie::with_slots(4);
        let words = [
            "网",
            "网球",
            "网球拍",
            "球拍",
            "拍卖",
            "拍卖会",
            "a",
            "ab",
            "𦡦",
            "𦡦𦡦",
            "中",
        ];
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.insert(w, i as i32), None);
        }
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.get(w), Some(i as i32), "{w}");
        }
        assert_eq!(trie.get(""), None);
        assert_eq!(trie.get("网球拍卖"), None);
        assert_eq!(trie.get("球"), None); // prefix node without a word
        assert_eq!(trie.get("x"), None);
        assert_eq!(trie.get("abc"), None);
        assert_eq!(trie.get("𦡦x"), None);

        assert_eq!(
            prefixes(&trie, "网球拍卖会"),
            vec![(0, "网"), (1, "网球"), (2, "网球拍")]
        );
        assert_eq!(prefixes(&trie, "拍卖会def"), vec![(4, "拍卖"), (5, "拍卖会")]);
        assert_eq!(prefixes(&trie, "球"), vec![]);
        assert_eq!(prefixes(&trie, "abc"), vec![(6, "a"), (7, "ab")]);
        assert_eq!(prefixes(&trie, "𦡦𦡦𦡦"), vec![(8, "𦡦"), (9, "𦡦𦡦")]);
        assert_eq!(prefixes(&trie, ""), vec![]);
        assert_eq!(prefixes(&trie, "zzz"), vec![]);
    }

    #[test]
    fn test_reinsert_replaces_id() {
        let mut trie = CharTrie::new();
        assert_eq!(trie.insert("中国", 1), None);
        assert_eq!(trie.insert("中国", 2), Some(1));
        assert_eq!(trie.get("中国"), Some(2));
        assert_eq!(trie.insert("中", 3), None);
        assert_eq!(trie.insert("中", 4), Some(3));
        assert_eq!(prefixes(&trie, "中国人"), vec![(4, "中"), (2, "中国")]);
    }

    /// Many children of one node fill its filters; every child must still be
    /// found and non-children rejected.
    #[test]
    fn test_wide_fanout() {
        let mut trie = CharTrie::with_slots(2);
        let children: Vec<String> = (0x4E00..0x4E00 + 300)
            .map(|c| format!("大{}", char::from_u32(c).unwrap()))
            .collect();
        for (i, w) in children.iter().enumerate() {
            trie.insert(w, i as i32);
        }
        for (i, w) in children.iter().enumerate() {
            assert_eq!(trie.get(w), Some(i as i32));
        }
        assert_eq!(trie.get("大x"), None);
        assert_eq!(trie.get("大"), None);
        assert_eq!(prefixes(&trie, "大a"), vec![]);
        assert_eq!(prefixes(&trie, "大一"), vec![(0, "大一")]);
    }

    #[test]
    fn test_grows_and_survives_rehash() {
        let mut trie = CharTrie::with_slots(2);
        let words: Vec<String> = (0..5000).map(|i| format!("w{i}x{}", i % 7)).collect();
        for (i, w) in words.iter().enumerate() {
            trie.insert(w, i as i32);
        }
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.get(w), Some(i as i32));
        }
        assert_eq!(prefixes(&trie, "w12x5zzz"), vec![(12, "w12x5")]);
        assert!(trie.slots.len() * MAX_LOAD_EIGHTHS >= trie.len * 8);
    }

    #[test]
    fn test_reserve_avoids_growth() {
        let mut trie = CharTrie::new();
        trie.reserve(10_000);
        let slots = trie.slots.len();
        for i in 0..10_000 {
            trie.insert(&format!("{i}"), i);
        }
        assert_eq!(trie.slots.len(), slots);
        assert_eq!(trie.get("9999"), Some(9999));
    }
}
