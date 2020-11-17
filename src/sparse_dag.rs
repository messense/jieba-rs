use crate::FxHashMap as HashMap;

pub(crate) struct StaticSparseDAG {
    array: Vec<usize>,
    start_pos: HashMap<usize, usize>,
    size_hint_for_iterator: usize,
    curr_insertion_len: usize,
}

pub struct EdgeIter<'a> {
    dag: &'a StaticSparseDAG,
    cursor: usize,
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = usize;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.dag.size_hint_for_iterator))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.dag.array[self.cursor] == 0 {
            self.cursor += 1;
            None
        } else {
            let v = self.dag.array[self.cursor] - 1;
            self.cursor += 1;
            Some(v)
        }
    }
}

impl StaticSparseDAG {
    pub(crate) fn with_size_hint(hint: usize) -> Self {
        StaticSparseDAG {
            array: Vec::with_capacity(hint * 5),
            start_pos: HashMap::default(),
            size_hint_for_iterator: 0,
            curr_insertion_len: 0,
        }
    }

    #[inline]
    pub(crate) fn start(&mut self, from: usize) {
        let idx = self.array.len();
        self.curr_insertion_len = 0;
        self.start_pos.insert(from, idx);
    }

    #[inline]
    pub(crate) fn insert(&mut self, to: usize) {
        self.curr_insertion_len += 1;
        self.array.push(to + 1);
    }

    #[inline]
    pub(crate) fn commit(&mut self) {
        self.size_hint_for_iterator = std::cmp::max(self.curr_insertion_len, self.size_hint_for_iterator);
        self.array.push(0);
    }

    #[inline]
    pub(crate) fn iter_edges(&self, from: usize) -> EdgeIter {
        let cursor = self.start_pos.get(&from).unwrap().to_owned();

        EdgeIter { dag: self, cursor }
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
        let mut dag = StaticSparseDAG::with_size_hint(5);
        let mut ans: Vec<Vec<usize>> = vec![Vec::new(); 5];
        for i in 0..=3 {
            dag.start(i);
            for j in (i + 1)..=4 {
                ans[i].push(j);
                dag.insert(j);
            }

            dag.commit()
        }

        assert_eq!(dag.size_hint_for_iterator, 4);

        for i in 0..=3 {
            let edges: Vec<usize> = dag.iter_edges(i).collect();
            assert_eq!(ans[i], edges);
        }
    }
}
