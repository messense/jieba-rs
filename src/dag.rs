use std::collections::BTreeMap;
use smallvec::SmallVec;

type Value = SmallVec<[usize; 5]>;

pub enum DAG {
    Small(Vec<Value>),
    Large(BTreeMap<usize, Value>),
}

impl DAG {
    pub fn with_capacity(cap: usize) -> Self {
        if cap > 1024 {
            DAG::Large(BTreeMap::new())
        } else {
            DAG::Small(Vec::with_capacity(cap))
        }
    }

    pub fn insert(&mut self, index: usize, val: Value) {
        match *self {
            DAG::Small(ref mut d) => d.insert(index, val),
            DAG::Large(ref mut d) => {
                d.insert(index, val);
                ()
            },
        }
    }
}

impl std::ops::Index<usize> for DAG {
    type Output = Value;

    fn index(&self, idx: usize) -> &Self::Output {
        match *self {
            DAG::Small(ref d) => &d[idx],
            DAG::Large(ref d) => &d[&idx],
        }
    }
}

pub enum IntoIter {
    Small(::std::vec::IntoIter<Value>),
    Large(::std::collections::btree_map::IntoIter<usize, Value>),
}

impl Iterator for IntoIter {
    // FIXME: this really should be `type Item = (usize, Value);`
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match *self {
            IntoIter::Small(ref mut d) => d.next(),
            IntoIter::Large(ref mut d) => d.next().map(|x| x.1),
        }
    }
}

impl IntoIterator for DAG {
    type Item = Value;
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            DAG::Small(d) => IntoIter::Small(d.into_iter()),
            DAG::Large(d) => IntoIter::Large(d.into_iter()),
        }
    }
}
