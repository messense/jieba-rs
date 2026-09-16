/// Word candidates of one block, as edges from a character to the index of
/// the character after a dictionary word starting there, carrying the id of
/// that word.
///
/// Edges arrive through `push_edge` grouped by character in increasing
/// order and `finish` closes the lists of the remaining characters. Lists
/// are then read back by character index. This is the compressed sparse
/// row layout: one flat edge array and, per character, the offset where
/// its list starts, with the next character's offset ending it.
#[derive(Default)]
pub(crate) struct StaticSparseDAG {
    /// Edge lists, one per character, back to back.
    array: Vec<u64>,
    /// Maps character index → index into `array`; one more entry than
    /// characters once finished, so every list has an end.
    start_pos: Vec<usize>,
}

/// Maximum end index that can be encoded in the upper 32 bits of a u64.
const MAX_ENCODED_END: usize = u32::MAX as usize;

/// Encodes (end, word_id) into a single u64: the end index in the upper 32
/// bits, the word id in the lower 32.
/// word_id uses i32::MIN as "no match" sentinel.
#[inline(always)]
fn encode_edge(end: usize, word_id: i32) -> u64 {
    debug_assert!(
        end <= MAX_ENCODED_END,
        "end {end} exceeds encodable range {MAX_ENCODED_END}",
    );
    ((end as u64) << 32) | (word_id as u32 as u64)
}

#[inline(always)]
fn decode_edge(val: u64) -> (usize, i32) {
    let end = (val >> 32) as usize;
    let word_id = val as u32 as i32;
    (end, word_id)
}

/// word_id sentinel meaning "no dictionary match"
pub(crate) const NO_MATCH: i32 = i32::MIN;

impl StaticSparseDAG {
    pub(crate) fn release_if_huge(&mut self) {
        // Edge lists are the largest scratch by far when the dictionary has
        // many long words, and regrowing them costs more than keeping them,
        // so they get several times the usual budget.
        const MAX_RETAINED_BYTES: usize = 8 * crate::SCRATCH_BUDGET;
        crate::release_if_huge(&mut self.array, MAX_RETAINED_BYTES);
        crate::release_if_huge(&mut self.start_pos, MAX_RETAINED_BYTES);
    }

    /// The id of the dictionary word spanning characters `start..end`, if
    /// there is one. Edges are ordered by end, so the scan stops at the
    /// first one reaching `end`.
    #[inline]
    pub(crate) fn word_at(&self, start: usize, end: usize) -> Option<i32> {
        self.iter_edges(start)
            .find(|&(e, _)| e >= end)
            .filter(|&(e, _)| e == end)
            .map(|(_, word_id)| word_id)
    }

    /// Record a word spanning characters `char_idx..end`.
    /// Edges must arrive grouped by character, in increasing character
    /// order, and in the order they are to be iterated in.
    #[inline]
    pub(crate) fn push_edge(&mut self, char_idx: usize, end: usize, word_id: i32) {
        self.open_through(char_idx);
        self.array.push(encode_edge(end, word_id));
    }

    /// Open the lists of every character up to and including `char_idx`
    /// that has none yet; a list opened now that gets no edge stays empty.
    #[inline]
    fn open_through(&mut self, char_idx: usize) {
        debug_assert!(
            self.start_pos.len() <= char_idx + 1,
            "edges must arrive in character order"
        );
        while self.start_pos.len() <= char_idx {
            self.start_pos.push(self.array.len());
        }
    }

    /// Terminate the lists once all edges of the `chars` characters are in.
    pub(crate) fn finish(&mut self, chars: usize) {
        self.open_through(chars);
    }

    /// Number of characters whose edge lists have been built.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.start_pos.len().saturating_sub(1)
    }

    /// Edges of the `char_idx`-th character.
    #[inline]
    pub(crate) fn iter_edges(&self, char_idx: usize) -> impl Iterator<Item = (usize, i32)> + '_ {
        self.array[self.start_pos[char_idx]..self.start_pos[char_idx + 1]]
            .iter()
            .map(|&val| decode_edge(val))
    }

    pub(crate) fn clear(&mut self) {
        self.array.clear();
        self.start_pos.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_sparse_dag() {
        let mut dag = StaticSparseDAG::default();
        let mut ans: Vec<Vec<usize>> = vec![Vec::new(); 5];
        for (i, item) in ans.iter_mut().enumerate().take(4) {
            for j in (i + 1)..=4 {
                item.push(j);
                dag.push_edge(i, j, j as i32);
            }
        }
        dag.finish(5);
        assert_eq!(dag.len(), 5);

        for (i, item) in ans.iter().enumerate() {
            let edges: Vec<usize> = dag.iter_edges(i).map(|(to, _)| to).collect();
            assert_eq!(item, &edges, "character {i}");
        }
    }

    #[test]
    fn test_clear_and_rebuild() {
        let mut dag = StaticSparseDAG::default();

        dag.push_edge(0, 1, 1);
        dag.push_edge(2, 4, 2);
        dag.finish(4);
        assert_eq!(dag.len(), 4);
        assert_eq!(dag.iter_edges(1).count(), 0);
        assert_eq!(dag.iter_edges(2).collect::<Vec<_>>(), vec![(4, 2)]);
        assert_eq!(dag.iter_edges(3).count(), 0);

        dag.clear();

        assert!(dag.array.is_empty());
        assert_eq!(dag.len(), 0);

        dag.push_edge(0, 2, 3);
        dag.finish(1);
        let edges: Vec<(usize, i32)> = dag.iter_edges(0).collect();
        assert_eq!(edges, vec![(2, 3)]);
    }
}
