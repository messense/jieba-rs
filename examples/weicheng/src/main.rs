use jieba_rs::Jieba;
use std::time;

#[cfg(unix)]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

static WEICHENG_TXT: &str = include_str!("weicheng.txt");

fn main() {
    let jieba = Jieba::new();
    let lines: Vec<&str> = WEICHENG_TXT.split('\n').collect();
    let now = time::Instant::now();
    for _ in 0..50 {
        for line in &lines {
            let _ = jieba.cut(line, true);
        }
    }
    println!("{}ms", now.elapsed().as_millis());
}
