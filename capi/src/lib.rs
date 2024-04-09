use c_fixed_string::CFixedStr;
use jieba_rs::{Jieba, KeywordExtract, TextRank, TfIdf};
use std::boxed::Box;
use std::os::raw::c_char;
use std::{mem, ptr};

#[repr(C)]
pub struct CJieba {
    jieba: Jieba,
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct CJiebaTFIDF {
    cjieba: *mut CJieba,
    tfidf: TfIdf,
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

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

#[repr(C)]
pub struct CJiebaTag {
    pub word: FfiStr,
    pub tag: FfiStr,
}

#[repr(C)]
pub struct CJiebaTags {
    pub tags: *mut CJiebaTag,
    pub len: usize,
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

    /// # Safety
    /// Frees the underlying data. After this call, the internal pointer is invalid.
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
///
/// # Safety
/// Used to release strings returned as results of function calls.
#[no_mangle]
pub unsafe extern "C" fn jieba_str_free(s: *mut FfiStr) {
    if !s.is_null() {
        (*s).free()
    }
}

unsafe fn params_unwrap(cjieba_ref: &*mut CJieba, s: *const c_char, len: usize) -> (&Jieba, &CFixedStr) {
    let jieba = &(*(*cjieba_ref)).jieba;
    let c_str = CFixedStr::from_ptr(s, len);
    (jieba, c_str)
}

unsafe fn params_unwrap_mut(cjieba_ref: &*mut CJieba, s: *const c_char, len: usize) -> (&mut Jieba, &CFixedStr) {
    let jieba = &mut (*(*cjieba_ref)).jieba;
    let c_str = CFixedStr::from_ptr(s, len);
    (jieba, c_str)
}

/// # Safety
/// Returned value must be freed by `jieba_free()`.
#[no_mangle]
pub extern "C" fn jieba_new() -> *mut CJieba {
    let cjieba = CJieba {
        jieba: Jieba::new(),
        _marker: Default::default(),
    };
    Box::into_raw(Box::new(cjieba))
}

/// Returns a Jieba instance with an empty dictionary.
///
/// # Safety
/// Returned value must be freed by `jieba_free()`.
#[no_mangle]
pub extern "C" fn jieba_empty() -> *mut CJieba {
    let cjieba = CJieba {
        jieba: Jieba::empty(),
        _marker: Default::default(),
    };
    Box::into_raw(Box::new(cjieba))
}

/// # Safety
/// cjieba is result from `jieba_new()` call.
#[no_mangle]
pub unsafe extern "C" fn jieba_free(cjieba: *mut CJieba) {
    if !cjieba.is_null() {
        drop(Box::from_raw(cjieba));
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
#[no_mangle]
pub unsafe extern "C" fn jieba_cut(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    hmm: bool,
) -> *mut CJiebaWords {
    let (jieba, c_str) = params_unwrap(&cjieba, sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let words = jieba.cut(&s, hmm);
    let mut c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.to_string())).collect();
    let words_len = c_words.len();
    let ptr = c_words.as_mut_ptr();
    mem::forget(c_words);
    Box::into_raw(Box::new(CJiebaWords {
        words: ptr,
        len: words_len,
    }))
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
#[no_mangle]
pub unsafe extern "C" fn jieba_cut_all(cjieba: *mut CJieba, sentence: *const c_char, len: usize) -> *mut CJiebaWords {
    let (jieba, c_str) = params_unwrap(&cjieba, sentence, len);
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

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
#[no_mangle]
pub unsafe extern "C" fn jieba_cut_for_search(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    hmm: bool,
) -> *mut CJiebaWords {
    let (jieba, c_str) = params_unwrap(&cjieba, sentence, len);
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

/// # Safety
/// cjieba must be valid object from `jieba_new()` and must outlive the returned CJiebaTFIDF instance.
///
/// Returned value must be freed by `jieba_tfidf_free()`.
#[no_mangle]
pub extern "C" fn jieba_tfidf_new(cjieba: *mut CJieba) -> *mut CJiebaTFIDF {
    let cjieba_tfidf = CJiebaTFIDF {
        cjieba,
        tfidf: Default::default(),
        _marker: Default::default(),
    };
    Box::into_raw(Box::new(cjieba_tfidf))
}

/// # Safety
/// cjieba_tfidf is result from `jieba_tfidf_new()` call.
#[no_mangle]
pub unsafe extern "C" fn jieba_tfidf_free(cjieba_tfidf: *mut CJiebaTFIDF) {
    if !cjieba_tfidf.is_null() {
        drop(Box::from_raw(cjieba_tfidf));
    }
}

/// # Safety
/// cjieba_tfidf must be valid object from `jieba_tfidf_new()`. `sentence` must be `len` or larger.
///
/// Returned value must be freed by `jieba_words_free()`.
#[no_mangle]
pub unsafe extern "C" fn jieba_tfidf_extract(
    cjieba_tfidf: *mut CJiebaTFIDF,
    sentence: *const c_char,
    len: usize,
    top_k: usize,
    allowed_pos: *const *mut c_char,
    allowed_pos_len: usize,
) -> *mut CJiebaWords {
    let cjieba_tfidf_ref = &(*cjieba_tfidf);
    let tfidf = &cjieba_tfidf_ref.tfidf;
    let (jieba, c_str) = params_unwrap(&cjieba_tfidf_ref.cjieba, sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());

    let allowed_pos: Vec<String> = if allowed_pos_len == 0 || allowed_pos.is_null() {
        Vec::new()
    } else {
        let mut v = Vec::with_capacity(allowed_pos_len);

        let slice: &[*mut c_char] = std::slice::from_raw_parts(allowed_pos, allowed_pos_len);
        for ptr in slice.iter() {
            let cstring_allowed_pos = std::ffi::CString::from_raw(*ptr);
            let string_allowed_pos = cstring_allowed_pos.into_string().expect("into_string().err() failed");
            v.push(string_allowed_pos);
        }

        v
    };

    let words = tfidf.extract_keywords(jieba, &s, top_k, allowed_pos);
    let mut c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.keyword)).collect();
    let words_len = c_words.len();
    let ptr = c_words.as_mut_ptr();
    mem::forget(c_words);
    Box::into_raw(Box::new(CJiebaWords {
        words: ptr,
        len: words_len,
    }))
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
///
/// Returned value must be freed by `jieba_words_free()`.
#[no_mangle]
pub unsafe extern "C" fn jieba_textrank_extract(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    top_k: usize,
    allowed_pos: *const *mut c_char,
    allowed_pos_len: usize,
) -> *mut CJiebaWords {
    let (jieba, c_str) = params_unwrap(&cjieba, sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());

    let allowed_pos: Vec<String> = if allowed_pos_len == 0 || allowed_pos.is_null() {
        Vec::new()
    } else {
        let mut v = Vec::with_capacity(allowed_pos_len);

        let slice: &[*mut c_char] = std::slice::from_raw_parts(allowed_pos, allowed_pos_len);
        for ptr in slice.iter() {
            let cstring_allowed_pos = std::ffi::CString::from_raw(*ptr);
            let string_allowed_pos = cstring_allowed_pos.into_string().expect("into_string().err() failed");
            v.push(string_allowed_pos);
        }

        v
    };

    let textrank = TextRank::default();
    let words = textrank.extract_keywords(jieba, &s, top_k, allowed_pos);
    let mut c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.keyword)).collect();
    let words_len = c_words.len();
    let ptr = c_words.as_mut_ptr();
    mem::forget(c_words);
    Box::into_raw(Box::new(CJiebaWords {
        words: ptr,
        len: words_len,
    }))
}

/// # Safety
/// c_tags is result from `jieba_textrank_extract()` or `jieba_tfidf_extract()` call.
#[no_mangle]
pub unsafe extern "C" fn jieba_words_free(c_words: *mut CJiebaWords) {
    if !c_words.is_null() {
        Vec::from_raw_parts((*c_words).words, (*c_words).len, (*c_words).len);
        drop(Box::from_raw(c_words));
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
///
/// Returned value must be freed by `jieba_tokens_free()`.
#[no_mangle]
pub unsafe extern "C" fn jieba_tokenize(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    mode: TokenizeMode,
    hmm: bool,
) -> *mut CJiebaTokens {
    let (jieba, c_str) = params_unwrap(&cjieba, sentence, len);
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

/// # Safety
/// c_tokens is result from `jieba_tokenize()` call.
#[no_mangle]
pub unsafe extern "C" fn jieba_tokens_free(c_tokens: *mut CJiebaTokens) {
    if !c_tokens.is_null() {
        Vec::from_raw_parts((*c_tokens).tokens, (*c_tokens).len, (*c_tokens).len);
        drop(Box::from_raw(c_tokens));
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
///
/// Returned value must be freed by `jieba_tags_free()`.
#[no_mangle]
pub unsafe extern "C" fn jieba_tag(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    hmm: bool,
) -> *mut CJiebaTags {
    let (jieba, c_str) = params_unwrap(&cjieba, sentence, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let tags = (*jieba).tag(&s, hmm);
    let mut c_tags: Vec<CJiebaTag> = tags
        .into_iter()
        .map(|x| CJiebaTag {
            word: FfiStr::from_string(x.word.to_string()),
            tag: FfiStr::from_string(x.tag.to_string()),
        })
        .collect();
    let tags_len = c_tags.len();
    let ptr = c_tags.as_mut_ptr();
    mem::forget(c_tags);
    Box::into_raw(Box::new(CJiebaTags {
        tags: ptr,
        len: tags_len,
    }))
}

/// # Safety
/// c_tags is result from `jieba_tag()` call.
#[no_mangle]
pub unsafe extern "C" fn jieba_tags_free(c_tags: *mut CJiebaTags) {
    if !c_tags.is_null() {
        Vec::from_raw_parts((*c_tags).tags, (*c_tags).len, (*c_tags).len);
        drop(Box::from_raw(c_tags));
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `word` must be `len` or larger.
#[no_mangle]
pub unsafe extern "C" fn jieba_add_word(cjieba: *mut CJieba, word: *const c_char, len: usize) -> usize {
    let (jieba, c_str) = params_unwrap_mut(&cjieba, word, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    jieba.add_word(&s, None, None)
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `segment` must be `len` or larger.
#[no_mangle]
pub unsafe extern "C" fn jieba_suggest_freq(cjieba: *mut CJieba, segment: *const c_char, len: usize) -> usize {
    let (jieba, c_str) = params_unwrap(&cjieba, segment, len);
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());

    (*jieba).suggest_freq(&s)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_jieba_new_and_free() {
        let jieba = jieba_new();
        unsafe { jieba_free(jieba) };
    }

    #[test]
    fn test_jieba_empty_and_free() {
        let jieba = jieba_empty();
        unsafe { jieba_free(jieba) };
    }

    #[test]
    fn test_jieba_add_word() {
        let jieba = jieba_empty();
        let word = "今天";
        let c_word = CString::new(word).unwrap();
        unsafe {
            jieba_add_word(jieba, c_word.as_ptr(), word.len());
            jieba_free(jieba)
        };
    }
}
