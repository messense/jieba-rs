#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;
use jieba_rs::Jieba;
use lazy_static::lazy_static;
use rand::Rng;
use smallvec::SmallVec;
use std::collections::btree_map::BTreeMap;

lazy_static! {
    static ref jieba: Jieba = Jieba::new();
}

fn bench_cut_no_hmm(sentence: &str) {
    jieba.cut(sentence, false);
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
    c.bench_function("jieba cut", |b| {
        b.iter(|| {
            bench_cut_no_hmm(black_box(
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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
