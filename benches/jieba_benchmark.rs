#[macro_use]
extern crate criterion;

use criterion::{black_box, Benchmark, Criterion, ParameterizedBenchmark, Throughput};
use jieba_rs::{Jieba, KeywordExtract, TextRank, TokenizeMode, TFIDF};
use lazy_static::lazy_static;

#[cfg(unix)]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

lazy_static! {
    static ref JIEBA: Jieba = Jieba::new();
    static ref TFIDF_EXTRACTOR: TFIDF<'static> = TFIDF::new_with_jieba(&JIEBA);
    static ref TEXTRANK_EXTRACTOR: TextRank<'static> = TextRank::new_with_jieba(&JIEBA);
}
static SENTENCE: &str =
    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。";

fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "jieba cut",
        ParameterizedBenchmark::new(
            "no hmm",
            |b, i| b.iter(|| JIEBA.cut(black_box(i), false)),
            vec![SENTENCE],
        )
        .with_function("with hmm", |b, i| b.iter(|| JIEBA.cut(black_box(i), true)))
        .with_function("cut_all", |b, i| b.iter(|| JIEBA.cut_all(black_box(i))))
        .with_function("cut_for_search", |b, i| {
            b.iter(|| JIEBA.cut_for_search(black_box(i), true))
        })
        .throughput(|i| Throughput::Bytes(i.len() as u32)),
    );

    c.bench(
        "jieba tokenize",
        ParameterizedBenchmark::new(
            "default mode",
            |b, i| b.iter(|| JIEBA.tokenize(black_box(i), TokenizeMode::Default, true)),
            vec![SENTENCE],
        )
        .with_function("search mode", |b, i| {
            b.iter(|| JIEBA.tokenize(black_box(i), TokenizeMode::Search, true))
        })
        .throughput(|i| Throughput::Bytes(i.len() as u32)),
    );

    c.bench(
        "jieba",
        Benchmark::new("tag", |b| b.iter(|| JIEBA.tag(black_box(SENTENCE), true)))
            .throughput(Throughput::Bytes(SENTENCE.len() as u32)),
    );

    c.bench(
        "jieba extract keywords",
        ParameterizedBenchmark::new(
            "tfidf",
            |b, i| b.iter(|| TFIDF_EXTRACTOR.extract_tags(black_box(i), 3, Vec::new())),
            vec![SENTENCE],
        )
        .with_function("textrank", |b, i| {
            b.iter(|| TEXTRANK_EXTRACTOR.extract_tags(black_box(i), 3, Vec::new()))
        })
        .throughput(|i| Throughput::Bytes(i.len() as u32)),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
