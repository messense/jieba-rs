#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;
use jieba_rs::unstable::JiebaUnstable;
use jieba_rs::{Jieba, KeywordExtract, TextRank, TokenizeMode, TFIDF};
use lazy_static::lazy_static;
use rand::Rng;
use smallvec::SmallVec;
use std::collections::btree_map::BTreeMap;

lazy_static! {
    static ref JIEBA: Jieba = Jieba::new();
}

lazy_static! {
    static ref JIEBA_UNSTABLE: JiebaUnstable = JiebaUnstable::new();
}

#[cfg(feature = "tfidf")]
lazy_static! {
    static ref TFIDF_EXTRACTOR: TFIDF<'static> = TFIDF::new_with_jieba(&JIEBA);
}
lazy_static! {
    static ref TEXTRANK_EXTRACTOR: TextRank<'static> = TextRank::new_with_jieba(&JIEBA);
}

fn bench_dag_with_btree(sentence: &str) {
    let mut dag: BTreeMap<usize, SmallVec<[usize; 5]>> = BTreeMap::new();
    let word_count = sentence.len();
    let mut rng = rand::thread_rng();

    for i in 0..(word_count - 1) {
        let mut tmplist = SmallVec::new();

        let number_of_node = rng.gen_range(0, 6);
        for _ in 0..number_of_node {
            let x = rng.gen_range(i + 1, word_count + 1);
            tmplist.push(x);
        }

        dag.insert(i, tmplist);
    }

    dag.insert(word_count - 1, SmallVec::new());

    let mut route = Vec::with_capacity(word_count + 1);
    for _ in 0..=word_count {
        route.push(0);
    }

    for i in (0..word_count).rev() {
        let x = dag[&i].iter().map(|x| x + 1).max().unwrap_or(0);
        route.push(x);
    }
}

fn bench_dag_with_vec(sentence: &str) {
    let word_count = sentence.len();
    let mut dag: Vec<SmallVec<[usize; 5]>> = Vec::with_capacity(word_count);
    let mut rng = rand::thread_rng();

    for i in 0..(word_count - 1) {
        let mut tmplist = SmallVec::new();

        let number_of_node = rng.gen_range(0, 6);
        for _ in 0..number_of_node {
            let x = rng.gen_range(i + 1, word_count + 1);
            tmplist.push(x);
        }

        dag.push(tmplist);
    }

    dag.insert(word_count - 1, SmallVec::new());

    let mut route = Vec::with_capacity(word_count + 1);
    for _ in 0..=word_count {
        route.push(0);
    }

    for i in (0..word_count).rev() {
        let x = dag[i].iter().map(|x| x + 1).max().unwrap_or(0);
        route.push(x);
    }
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

    c.bench_function("jieba unstable cut no hmm", |b| {
        b.iter(|| {
            JIEBA_UNSTABLE.cut(
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
        });
    });

    c.bench_function("dag with btree", |b| {
        b.iter(|| {
            bench_dag_with_btree(black_box(
                "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            ))
        });
    });

    c.bench_function("dag with vec", |b| {
        b.iter(|| {
            bench_dag_with_vec(black_box(
                "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            ))
        });
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
