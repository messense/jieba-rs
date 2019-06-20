#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;
#[cfg(any(feature = "tfidf", feature = "textrank"))]
use jieba_rs::KeywordExtract;
#[cfg(feature = "textrank")]
use jieba_rs::TextRank;
#[cfg(feature = "tfidf")]
use jieba_rs::TFIDF;
use jieba_rs::{Jieba, TokenizeMode};
use lazy_static::lazy_static;

lazy_static! {
    static ref JIEBA: Jieba = Jieba::new();
}
#[cfg(feature = "tfidf")]
lazy_static! {
    static ref TFIDF_EXTRACTOR: TFIDF<'static> = TFIDF::new_with_jieba(&JIEBA);
}
#[cfg(feature = "textrank")]
lazy_static! {
    static ref TEXTRANK_EXTRACTOR: TextRank<'static> = TextRank::new_with_jieba(&JIEBA);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("jieba cut no hmm", |b| {
        b.iter(|| {
            JIEBA.cut(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                false,
            )
        })
    });

    c.bench_function("jieba cut with hmm", |b| {
        b.iter(|| {
            JIEBA.cut(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                true,
            )
        })
    });

    c.bench_function("jieba cut_all", |b| {
        b.iter(|| {
            JIEBA.cut_all(black_box(
                "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            ))
        })
    });

    c.bench_function("jieba cut_for_search", |b| {
        b.iter(|| {
            JIEBA.cut_for_search(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                true,
            )
        })
    });

    c.bench_function("jieba tokenize default mode", |b| {
        b.iter(|| {
            JIEBA.tokenize(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                TokenizeMode::Default,
                true,
            )
        })
    });

    c.bench_function("jieba tokenize search mode", |b| {
        b.iter(|| {
            JIEBA.tokenize(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                TokenizeMode::Search,
                true,
            )
        })
    });

    c.bench_function("jieba tag", |b| {
        b.iter(|| {
            JIEBA.tag(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                true,
            )
        })
    });

    #[cfg(feature = "tfidf")]
    c.bench_function("jieba tfidf", |b| {
        b.iter(|| {
            TFIDF_EXTRACTOR.extract_tags(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                3,
                Vec::new(),
            )
        })
    });

    #[cfg(feature = "textrank")]
    c.bench_function("jieba textrank", |b| {
        b.iter(|| {
            TEXTRANK_EXTRACTOR.extract_tags(
                black_box(
                    "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
                ),
                3,
                Vec::new(),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
