#![cfg(target_arch = "wasm32")]

use jieba_rs::Jieba;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_jieba_cut() {
    let jieba = Jieba::new();
    let words = jieba.cut("我们中出了一个叛徒", false);
    assert_eq!(words, vec!["我们", "中", "出", "了", "一个", "叛徒"]);
}
