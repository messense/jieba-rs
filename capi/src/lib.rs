use jieba_rs::Jieba;
use std::boxed::Box;

pub struct CJieba;

#[no_mangle]
pub unsafe extern "C" fn jieba_new() -> *mut CJieba {
    let jieba = Jieba::new();
    Box::into_raw(Box::new(jieba)) as *mut CJieba
}

#[no_mangle]
pub unsafe extern "C" fn jieba_empty() -> *mut CJieba {
    let jieba = Jieba::empty();
    Box::into_raw(Box::new(jieba)) as *mut CJieba
}

#[no_mangle]
pub unsafe extern "C" fn jieba_free(j: *mut CJieba) {
    if !j.is_null() {
        let jieba = j as *mut Jieba;
        Box::from_raw(jieba);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_jieba_new_and_free() {
        unsafe {
            let jieba = jieba_new();
            jieba_free(jieba);
        }
    }

    #[test]
    fn test_jieba_empty_and_free() {
        unsafe {
            let jieba = jieba_empty();
            jieba_free(jieba);
        }
    }
}
