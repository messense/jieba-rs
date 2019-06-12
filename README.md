# jieba-rs

[![Build Status](https://travis-ci.com/messense/jieba-rs.svg?branch=master)](https://travis-ci.com/messense/jieba-rs)
[![codecov](https://codecov.io/gh/messense/jieba-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/messense/jieba-rs)
[![Crates.io](https://img.shields.io/crates/v/jieba-rs.svg)](https://crates.io/crates/jieba-rs)
[![docs.rs](https://docs.rs/jieba-rs/badge.svg)](https://docs.rs/jieba-rs/)

The Jieba Chinese Word Segmentation Implemented in Rust

## Installation

Add it to your ``Cargo.toml``:

```toml
[dependencies]
jieba-rs = "0.4"
```

then you are good to go. If you are using Rust 2015 you have to ``extern crate jieba_rs`` to your crate root as well. 

## Example

```rust
use jieba_rs::Jieba;

fn main() {
    let jieba = Jieba::new();
    let words = jieba.cut("我们中出了一个叛徒", false);
    assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
}
```

## Enabling Additional Features

* `tfidf` feature enables TF-IDF keywords extractor
* `textrank` feature enables TextRank keywords extractor

```toml
[dependencies]
jieba-rs = { version = "0.4", features = ["tfidf", "textrank"] }
```

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.
