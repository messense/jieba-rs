use c_fixed_string::CFixedStr;
use jieba_rs::Jieba;
use std::boxed::Box;
use std::os::raw::c_char;
use std::{mem, ptr};

pub struct CJieba;

#[repr(C)]
pub struct CJiebaWords {
    pub words: *mut FfiStr,
    pub len: usize,
}

#[repr(C)]
pub struct CJiebaToken {
    pub word: FfiStr,
    pub start: usize,
    pub end: usize,
}

#[repr(C)]
pub struct CJiebaTokens {
    pub tokens: *mut CJiebaToken,
    pub len: usize,
}

/// Tokenize mode
#[repr(C)]
pub enum TokenizeMode {
    /// Default mode
    Default = 0,
    /// Search mode
    Search = 1,
}

impl From<TokenizeMode> for jieba_rs::TokenizeMode {
    fn from(mode: TokenizeMode) -> Self {
        match mode {
            TokenizeMode::Default => jieba_rs::TokenizeMode::Default,
            TokenizeMode::Search => jieba_rs::TokenizeMode::Search,
        }
    }
}

/// Represents a string.
#[repr(C)]
pub struct FfiStr {
    pub data: *mut c_char,
    pub len: usize,
    pub owned: bool,
}

impl Default for FfiStr {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
            owned: false,
        }
    }
}

impl FfiStr {
    pub fn from_string(mut s: String) -> Self {
        s.shrink_to_fit();
        let rv = Self {
            data: s.as_ptr() as *mut c_char,
            len: s.len(),
            owned: true,
        };
        mem::forget(s);
        rv
    }

    pub unsafe fn free(&mut self) {
        if self.owned && !self.data.is_null() {
            String::from_raw_parts(self.data as *mut _, self.len, self.len);
            self.data = ptr::null_mut();
            self.len = 0;
            self.owned = false;
        }
    }
}

impl Drop for FfiStr {
    fn drop(&mut self) {
        unsafe {
            self.free();
        }
    }
}

/// Frees a ffi str.
///
/// If the string is marked as not owned then this function does not
/// do anything.
#[no_mangle]
pub unsafe extern "C" fn jieba_str_free(s: *mut FfiStr) {
    if !s.is_null() {
        (*s).free()
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn jieba_cut(j: *mut CJieba, sentence: *const c_char, len: usize, hmm: bool) -> *mut CJiebaWords {
    let jieba = j as *mut Jieba;
    let c_str = CFixedStr::from_ptr(sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let words = (*jieba).cut(&s, hmm);
    let mut c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.to_string())).collect();
    let words_len = c_words.len();
    let ptr = c_words.as_mut_ptr();
    mem::forget(c_words);
    Box::into_raw(Box::new(CJiebaWords {
        words: ptr,
        len: words_len,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn jieba_cut_all(j: *mut CJieba, sentence: *const c_char, len: usize) -> *mut CJiebaWords {
    let jieba = j as *mut Jieba;
    let c_str = CFixedStr::from_ptr(sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let words = (*jieba).cut_all(&s);
    let mut c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.to_string())).collect();
    let words_len = c_words.len();
    let ptr = c_words.as_mut_ptr();
    mem::forget(c_words);
    Box::into_raw(Box::new(CJiebaWords {
        words: ptr,
        len: words_len,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn jieba_cut_for_search(
    j: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    hmm: bool,
) -> *mut CJiebaWords {
    let jieba = j as *mut Jieba;
    let c_str = CFixedStr::from_ptr(sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let words = (*jieba).cut_for_search(&s, hmm);
    let mut c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.to_string())).collect();
    let words_len = c_words.len();
    let ptr = c_words.as_mut_ptr();
    mem::forget(c_words);
    Box::into_raw(Box::new(CJiebaWords {
        words: ptr,
        len: words_len,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn jieba_words_free(c_words: *mut CJiebaWords) {
    if !c_words.is_null() {
        Vec::from_raw_parts((*c_words).words, (*c_words).len, (*c_words).len);
        Box::from_raw(c_words);
    }
}

#[no_mangle]
pub unsafe extern "C" fn jieba_tokenize(
    j: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    mode: TokenizeMode,
    hmm: bool,
) -> *mut CJiebaTokens {
    let jieba = j as *mut Jieba;
    let c_str = CFixedStr::from_ptr(sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let tokens = (*jieba).tokenize(&s, mode.into(), hmm);
    let mut c_tokens: Vec<CJiebaToken> = tokens
        .into_iter()
        .map(|x| CJiebaToken {
            word: FfiStr::from_string(x.word.to_string()),
            start: x.start,
            end: x.end,
        })
        .collect();
    let tokens_len = c_tokens.len();
    let ptr = c_tokens.as_mut_ptr();
    mem::forget(c_tokens);
    Box::into_raw(Box::new(CJiebaTokens {
        tokens: ptr,
        len: tokens_len,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn jieba_tokens_free(c_tokens: *mut CJiebaTokens) {
    if !c_tokens.is_null() {
        Vec::from_raw_parts((*c_tokens).tokens, (*c_tokens).len, (*c_tokens).len);
        Box::from_raw(c_tokens);
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
