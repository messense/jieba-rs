extern crate jieba_rs;

use std::time;
use jieba_rs::Jieba;

static WEICHENG_TXT: &str = include_str!("weicheng.txt");

fn main() {
    let jieba = Jieba::new();
    let now = time::Instant::now();
    for _ in 0..50 {
        for line in WEICHENG_TXT.split('\n') {
            let _ = jieba.cut(line, true);
        }
    }
    println!("{}", now.elapsed().as_secs());
}
