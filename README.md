# jieba-rs

[![GitHub Actions](https://github.com/messense/jieba-rs/workflows/CI/badge.svg)](https://github.com/messense/jieba-rs/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/messense/jieba-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/messense/jieba-rs)
[![Crates.io](https://img.shields.io/crates/v/jieba-rs.svg)](https://crates.io/crates/jieba-rs)
[![docs.rs](https://docs.rs/jieba-rs/badge.svg)](https://docs.rs/jieba-rs/)

> 🚀 Help me to become a full-time open-source developer by [sponsoring me on GitHub](https://github.com/sponsors/messense)

The Jieba Chinese Word Segmentation Implemented in Rust

## Installation

Add it to your ``Cargo.toml``:

```toml
[dependencies]
jieba-rs = "0.7"
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

* `default-dict` feature enables embedded dictionary, this features is enabled by default
* `tfidf` feature enables TF-IDF keywords extractor
* `textrank` feature enables TextRank keywords extractor

```toml
[dependencies]
jieba-rs = { version = "0.7", features = ["tfidf", "textrank"] }
```

## Run benchmark

```bash
cargo bench --all-features
```

## Benchmark: Compare with cppjieba 

* [Optimizing jieba-rs to be 33% faster than cppjieba](https://blog.paulme.ng/posts/2019-06-30-optimizing-jieba-rs-to-be-33percents-faster-than-cppjieba.html)
* [优化 jieba-rs 中文分词性能评测](https://blog.paulme.ng/posts/2019-07-01-%E4%BC%98%E5%8C%96-jieba-rs-%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D-%E6%80%A7%E8%83%BD%E8%AF%84%E6%B5%8B%EF%BC%88%E5%BF%AB%E4%BA%8E-cppjieba-33percent%29.html)
* [最佳化 jieba-rs 中文斷詞性能測試](https://blog.paulme.ng/posts/2019-07-01-%E6%9C%80%E4%BD%B3%E5%8C%96jieba-rs%E4%B8%AD%E6%96%87%E6%96%B7%E8%A9%9E%E6%80%A7%E8%83%BD%E6%B8%AC%E8%A9%A6%28%E5%BF%AB%E4%BA%8Ecppjieba-33%25%29.html)

## `jieba-rs` bindings

* [`@node-rs/jieba` NodeJS binding](https://github.com/napi-rs/node-rs/tree/main/packages/jieba)
* [`jieba-php` PHP binding](https://github.com/binaryoung/jieba-php)
* [`rjieba-py` Python binding](https://github.com/messense/rjieba-py)
* [`cang-jie` Chinese tokenizer for tantivy](https://github.com/DCjanus/cang-jie)
* [`tantivy-jieba` An adapter that bridges between tantivy and jieba-rs](https://github.com/jiegec/tantivy-jieba)
* [`jieba-wasm` the WebAssembly binding](https://github.com/fengkx/jieba-wasm)

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.
