pub(crate) struct StaticSparseDAG {
    array: Vec<u64>,
    /// Maps byte offset → index into `array`. Uses `usize::MAX` as sentinel for "no entry".
    start_pos: Vec<usize>,
    size_hint_for_iterator: usize,
    curr_insertion_len: usize,
}

const NO_ENTRY: usize = usize::MAX;

/// Encodes (byte_end + 1, word_id) into a single u64.
/// byte_end is stored as byte_end + 1 so that 0 can serve as the sentinel.
/// word_id uses i32::MIN as "no match" sentinel.
#[inline(always)]
fn encode_edge(byte_end: usize, word_id: i32) -> u64 {
    ((byte_end as u64 + 1) << 32) | (word_id as u32 as u64)
}

#[inline(always)]
fn decode_edge(val: u64) -> (usize, i32) {
    let byte_end = (val >> 32) as usize - 1;
    let word_id = val as u32 as i32;
    (byte_end, word_id)
}

pub struct EdgeIter<'a> {
    dag: &'a StaticSparseDAG,
    cursor: usize,
}

impl Iterator for EdgeIter<'_> {
    type Item = (usize, i32);

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.dag.array[self.cursor];
        if val == 0 {
            self.cursor += 1;
            None
        } else {
            self.cursor += 1;
            Some(decode_edge(val))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.dag.size_hint_for_iterator))
    }
}

/// word_id sentinel meaning "no dictionary match"
pub(crate) const NO_MATCH: i32 = i32::MIN;

impl StaticSparseDAG {
    pub(crate) fn with_size_hint(hint: usize) -> Self {
        // Cap the allocation to prevent memory issues with very large inputs
        // Use a more conservative multiplier to reduce memory overhead
        const MAX_CAPACITY: usize = 4_000_000;
        const MULTIPLIER: usize = 4; // Reduced from 5 to 4
        const MIN_CAPACITY: usize = 32; // Minimum useful capacity

        let capacity = (hint * MULTIPLIER).clamp(MIN_CAPACITY, MAX_CAPACITY);

        StaticSparseDAG {
            array: Vec::with_capacity(capacity),
            start_pos: vec![NO_ENTRY; hint + 1],
            size_hint_for_iterator: 0,
            curr_insertion_len: 0,
        }
    }

    #[inline]
    pub(crate) fn start(&mut self, from: usize) {
        let idx = self.array.len();
        self.curr_insertion_len = 0;
        if from >= self.start_pos.len() {
            self.start_pos.resize(from + 1, NO_ENTRY);
        }
        self.start_pos[from] = idx;
    }

    #[inline]
    pub(crate) fn insert(&mut self, to: usize, word_id: i32) {
        self.curr_insertion_len += 1;
        self.array.push(encode_edge(to, word_id));
    }

    #[inline]
    pub(crate) fn commit(&mut self) {
        self.size_hint_for_iterator = std::cmp::max(self.curr_insertion_len, self.size_hint_for_iterator);
        self.array.push(0);
    }

    #[inline]
    pub(crate) fn iter_edges(&self, from: usize) -> EdgeIter<'_> {
        let cursor = self.start_pos[from];
        debug_assert!(cursor != NO_ENTRY);

        EdgeIter { dag: self, cursor }
    }

    pub(crate) fn clear(&mut self) {
        self.array.clear();
        self.start_pos.fill(NO_ENTRY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_sparse_dag() {
        let mut dag = StaticSparseDAG::with_size_hint(5);
        let mut ans: Vec<Vec<usize>> = vec![Vec::new(); 5];
        for (i, item) in ans.iter_mut().enumerate().take(4) {
            dag.start(i);
            for j in (i + 1)..=4 {
                item.push(j);
                dag.insert(j, j as i32);
            }

            dag.commit()
        }

        assert_eq!(dag.size_hint_for_iterator, 4);

        for (i, item) in ans.iter().enumerate().take(4) {
            let edges: Vec<usize> = dag.iter_edges(i).map(|(to, _)| to).collect();
            assert_eq!(item, &edges);
        }
    }
}
