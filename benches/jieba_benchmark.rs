#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;
use jieba_rs::Jieba;
use lazy_static::lazy_static;

lazy_static! {
    static ref jieba: Jieba = Jieba::new();
}

fn bench_cut_no_hmm(sentence: &str) {
    jieba.cut(sentence, false);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("jieba cut", |b| {
        b.iter(|| {
            bench_cut_no_hmm(black_box(
                "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。",
            ))
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
