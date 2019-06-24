use jieba_rs::unstable::JiebaUnstable;
use jieba_rs::Jieba;
use std::time::{SystemTime, UNIX_EPOCH};

static WEICHENG_TXT: &str = include_str!("weicheng.txt");

fn bench_jieba() {
    let jieba = Jieba::new();
    let lines: Vec<&str> = WEICHENG_TXT.split('\n').collect();

    let start = SystemTime::now();
    let start_since_the_epoch = start.duration_since(UNIX_EPOCH).expect("Time went backwards");
    for _ in 0..50 {
        for line in &lines {
            let _ = jieba.cut(line, true);
        }
    }
    let end = SystemTime::now();
    let end_since_the_epoch = end.duration_since(UNIX_EPOCH).expect("Time went backwards");
    println!(
        "Jieba Elapsed: {:?} ms",
        end_since_the_epoch.as_millis() - start_since_the_epoch.as_millis()
    );
}

fn bench_jieba_unstable() {
    let jieba = JiebaUnstable::new();
    let lines: Vec<&str> = WEICHENG_TXT.split('\n').collect();

    let start = SystemTime::now();
    let start_since_the_epoch = start.duration_since(UNIX_EPOCH).expect("Time went backwards");
    for _ in 0..50 {
        for line in &lines {
            let _ = jieba.cut(line, true);
        }
    }
    let end = SystemTime::now();
    let end_since_the_epoch = end.duration_since(UNIX_EPOCH).expect("Time went backwards");
    println!(
        "Jieba Unstable Elapsed: {:?} ms",
        end_since_the_epoch.as_millis() - start_since_the_epoch.as_millis()
    );
}

fn main() {
    bench_jieba();
    bench_jieba_unstable();
}
