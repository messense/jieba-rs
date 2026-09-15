/// Word candidates of one block, as edges `byte_start -> byte_end` carrying
/// the dictionary id of the word they span.
///
/// Edges are appended block by block in increasing `byte_start` order:
/// `start(from)` opens the edge list of a position, `insert` appends to it,
/// and `commit` terminates it with a 0 sentinel.
#[derive(Default)]
pub(crate) struct StaticSparseDAG {
    array: Vec<u64>,
    /// Maps byte offset → index into `array`. Uses `usize::MAX` as sentinel for "no entry".
    start_pos: Vec<usize>,
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
    edges: &'a [u64],
    cursor: usize,
}

impl Iterator for EdgeIter<'_> {
    type Item = (usize, i32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Every list is 0-terminated, so the cursor stops on the sentinel.
        let val = self.edges[self.cursor];
        if val == 0 {
            None
        } else {
            self.cursor += 1;
            Some(decode_edge(val))
        }
    }
}

impl std::iter::FusedIterator for EdgeIter<'_> {}

/// word_id sentinel meaning "no dictionary match"
pub(crate) const NO_MATCH: i32 = i32::MIN;

impl StaticSparseDAG {
    /// Drop the edge storage if a very large block grew it past what is
    /// worth keeping around between calls.
    pub(crate) fn release_if_huge(&mut self) {
        const MAX_RETAINED_EDGES: usize = 4_000_000;
        if self.array.capacity() > MAX_RETAINED_EDGES {
            self.array = Vec::new();
            self.start_pos = Vec::new();
        }
    }

    #[inline]
    pub(crate) fn start(&mut self, from: usize) {
        debug_assert!(from >= self.start_pos.len(), "start offsets must increase");
        let idx = self.array.len();
        // Offsets are opened in increasing order, so the table only ever
        // grows at the end; the gap is the continuation bytes of the
        // previous character.
        self.start_pos.resize(from + 1, NO_ENTRY);
        self.start_pos[from] = idx;
    }

    #[inline]
    pub(crate) fn insert(&mut self, to: usize, word_id: i32) {
        self.array.push(encode_edge(to, word_id));
    }

    #[inline]
    pub(crate) fn commit(&mut self) {
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
            edges: &self.array,
            cursor,
        }
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
            dag.start(i);
            for j in (i + 1)..=4 {
                item.push(j);
                dag.insert(j, j as i32);
            }

            dag.commit()
        }

        for (i, item) in ans.iter().enumerate().take(4) {
            let edges: Vec<usize> = dag.iter_edges(i).map(|(to, _)| to).collect();
            assert_eq!(item, &edges);
        }
    }

    #[test]
    fn test_clear_resets_touched_offsets() {
        let mut dag = StaticSparseDAG::default();

        dag.start(0);
        dag.insert(1, 1);
        dag.commit();
        dag.start(3);
        dag.insert(4, 2);
        dag.commit();

        assert_ne!(dag.start_pos[0], NO_ENTRY);
        assert_ne!(dag.start_pos[3], NO_ENTRY);

        dag.clear();

        assert!(dag.array.is_empty());
        assert!(dag.start_pos.is_empty());

        dag.start(0);
        dag.insert(2, 3);
        dag.commit();
        let edges: Vec<(usize, i32)> = dag.iter_edges(0).collect();
        assert_eq!(edges, vec![(2, 3)]);
    }
}
