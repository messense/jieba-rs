use codspeed_criterion_compat::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use jieba_rs::{Jieba, KeywordExtract, TextRank, TfIdf, TokenizeMode};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::LazyLock;

#[cfg(unix)]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);
static TFIDF_EXTRACTOR: LazyLock<TfIdf> = LazyLock::new(TfIdf::default);
static TEXTRANK_EXTRACTOR: LazyLock<TextRank> = LazyLock::new(TextRank::default);
static SENTENCE: &str = "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。";

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("jieba");
    let dict_len = include_bytes!("../src/data/dict.txt").len() as u64;
    group.throughput(Throughput::Bytes(dict_len));
    group.bench_function("new", |b| {
        b.iter(|| {
            black_box(Jieba::new());
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cut");
    group.throughput(Throughput::Bytes(SENTENCE.len() as u64));
    group.bench_function("no_hmm", |b| b.iter(|| JIEBA.cut(black_box(SENTENCE), false)));
    group.bench_function("with_hmm", |b| b.iter(|| JIEBA.cut(black_box(SENTENCE), true)));
    group.bench_function("cut_all", |b| b.iter(|| JIEBA.cut_all(black_box(SENTENCE))));
    group.bench_function("cut_for_search", |b| {
        b.iter(|| JIEBA.cut_for_search(black_box(SENTENCE), true))
    });
    group.finish();

    let mut group = c.benchmark_group("tokenize");
    group.throughput(Throughput::Bytes(SENTENCE.len() as u64));
    group.bench_function("default_mode", |b| {
        b.iter(|| JIEBA.tokenize(black_box(SENTENCE), TokenizeMode::Default, true))
    });
    group.bench_function("search_mode", |b| {
        b.iter(|| JIEBA.tokenize(black_box(SENTENCE), TokenizeMode::Search, true))
    });
    group.finish();

    let mut group = c.benchmark_group("jieba");
    group.throughput(Throughput::Bytes(SENTENCE.len() as u64));
    group.bench_function("tag", |b| b.iter(|| JIEBA.tag(black_box(SENTENCE), true)));
    group.finish();

    let mut group = c.benchmark_group("keywords");
    group.throughput(Throughput::Bytes(SENTENCE.len() as u64));
    group.bench_function("tfidf", |b| {
        b.iter(|| TFIDF_EXTRACTOR.extract_keywords(&JIEBA, black_box(SENTENCE), 3, Vec::new()))
    });
    group.bench_function("textrank", |b| {
        b.iter(|| TEXTRANK_EXTRACTOR.extract_keywords(&JIEBA, black_box(SENTENCE), 3, Vec::new()))
    });
    group.finish();

    let mut group = c.benchmark_group("multithreaded");
    let repeat = 1000usize;
    group.throughput(Throughput::Bytes(SENTENCE.len() as u64 * repeat as u64));
    group.bench_function("single_thread", |b| {
        b.iter(|| {
            for _ in 0..repeat {
                let _words = JIEBA.cut(black_box(SENTENCE), true);
            }
        })
    });
    group.bench_function("multi_thread", |b| {
        b.iter(|| {
            (0..repeat).into_par_iter().for_each(|_| {
                let _words = JIEBA.cut(black_box(SENTENCE), true);
            });
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
