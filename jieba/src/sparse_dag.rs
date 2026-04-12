pub(crate) struct StaticSparseDAG {
    array: Vec<u64>,
    /// Maps byte offset → index into `array`. Uses `usize::MAX` as sentinel for "no entry".
    start_pos: Vec<usize>,
    size_hint_for_iterator: usize,
    curr_insertion_len: usize,
}

const NO_ENTRY: usize = usize::MAX;

/// Maximum byte_end value that can be encoded in the upper 32 bits of a u64.
const MAX_ENCODED_BYTE_END: usize = u32::MAX as usize - 1;

/// Encodes (byte_end + 1, word_id) into a single u64.
/// byte_end is stored as byte_end + 1 in the upper 32 bits so that 0 can
/// serve as the sentinel, which limits `byte_end` to `u32::MAX - 1`.
/// word_id uses i32::MIN as "no match" sentinel.
#[inline(always)]
fn encode_edge(byte_end: usize, word_id: i32) -> u64 {
    debug_assert!(
        byte_end <= MAX_ENCODED_BYTE_END,
        "byte_end {byte_end} exceeds encodable range {MAX_ENCODED_BYTE_END}",
    );
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
    done: bool,
}

impl Iterator for EdgeIter<'_> {
    type Item = (usize, i32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let val = self.dag.array[self.cursor];
        if val == 0 {
            self.done = true;
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

impl std::iter::FusedIterator for EdgeIter<'_> {}

/// word_id sentinel meaning "no dictionary match"
pub(crate) const NO_MATCH: i32 = i32::MIN;

impl StaticSparseDAG {
    pub(crate) fn with_size_hint(hint: usize) -> Self {
        const MAX_CAPACITY: usize = 4_000_000;
        const MULTIPLIER: usize = 4;
        const MIN_CAPACITY: usize = 32;

        let capacity = (hint * MULTIPLIER).clamp(MIN_CAPACITY, MAX_CAPACITY);
        let start_pos_len = hint.min(MAX_CAPACITY) + 1;

        StaticSparseDAG {
            array: Vec::with_capacity(capacity),
            start_pos: vec![NO_ENTRY; start_pos_len],
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
        assert!(
            from < self.start_pos.len(),
            "iter_edges: byte offset {from} out of bounds (len {})",
            self.start_pos.len()
        );
        let cursor = self.start_pos[from];
        assert!(
            cursor != NO_ENTRY,
            "iter_edges: byte offset {from} was never recorded via start()"
        );

        EdgeIter {
            dag: self,
            cursor,
            done: false,
        }
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
