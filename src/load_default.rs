use std::str::SplitWhitespace;

pub struct LoadDefault {
    iter:SplitWhitespace<'static>,
}

impl LoadDefault {
    pub fn new(s:&'static str)  -> Self {
        Self{
            iter: s.split_whitespace(),
        }
    }
}

impl Iterator for LoadDefault {
    type Item = (&'static str, usize, &'static str);

    fn next(&mut self) -> Option<Self::Item> {
        let word = self.iter.next()?;
        let freq:usize = self.iter.next().unwrap().parse().unwrap();
        let tag = self.iter.next().unwrap();

        Some((word,freq,tag))
    }
}
