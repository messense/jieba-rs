use c_fixed_string::CFixedStr;
use jieba_rs::{Jieba, KeywordExtract, TextRank, TfIdf};
use std::boxed::Box;
use std::os::raw::c_char;
use std::ptr;

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
    pub fn from_string(s: String) -> Self {
        // A boxed str is exactly `len` bytes, so `free` can rebuild it from
        // the pointer and length alone. `shrink_to_fit` gives no such
        // guarantee: the allocator may leave spare capacity, and freeing
        // with a smaller size than was allocated is undefined behaviour.
        let s: Box<str> = s.into_boxed_str();
        let len = s.len();
        Self {
            data: Box::into_raw(s) as *mut c_char,
            len,
            owned: true,
        }
    }

    /// # Safety
    /// Frees the underlying data. After this call, the internal pointer is invalid.
    pub unsafe fn free(&mut self) {
        if self.owned && !self.data.is_null() {
            unsafe {
                let slice = ptr::slice_from_raw_parts_mut(self.data as *mut u8, self.len);
                drop(Box::from_raw(slice as *mut str));
            }
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_str_free(s: *mut FfiStr) {
    if !s.is_null() {
        unsafe { (*s).free() }
    }
}

/// Hand a result array to C as a pointer and length.
///
/// The Vec is turned into a boxed slice first, so the allocation is exactly
/// `len` elements and `free_slice` can rebuild it from the two values the C
/// struct keeps. A Vec collected from an iterator often has spare capacity
/// (`collect` reuses the source allocation when it can), and freeing it as
/// if its capacity were `len` is undefined behaviour.
fn leak_slice<T>(v: Vec<T>) -> (*mut T, usize) {
    let slice: Box<[T]> = v.into_boxed_slice();
    let len = slice.len();
    (Box::into_raw(slice) as *mut T, len)
}

/// Free a result array produced by `leak_slice`, dropping its elements.
///
/// # Safety
/// `ptr` and `len` must be exactly what `leak_slice` returned, and the array
/// must not have been freed before.
unsafe fn free_slice<T>(ptr: *mut T, len: usize) {
    unsafe { drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len))) }
}

/// The allowed-POS list a caller passes as `len` C strings, copied out.
///
/// The strings stay the caller's: they are read, never freed.
///
/// # Safety
/// `allowed_pos` must point to `len` valid NUL-terminated strings, or `len`
/// must be 0.
unsafe fn allowed_pos_vec(allowed_pos: *const *const c_char, len: usize) -> Vec<String> {
    if len == 0 || allowed_pos.is_null() {
        return Vec::new();
    }
    let slice: &[*const c_char] = unsafe { std::slice::from_raw_parts(allowed_pos, len) };
    slice
        .iter()
        .map(|&p| unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned())
        .collect()
}

unsafe fn params_unwrap(cjieba_ref: &*mut CJieba, s: *const c_char, len: usize) -> (&Jieba, &CFixedStr) {
    let jieba = unsafe { &(*(*cjieba_ref)).jieba };
    let c_str = unsafe { CFixedStr::from_ptr(s, len) };
    (jieba, c_str)
}

unsafe fn params_unwrap_mut<'a>(
    cjieba_ref: *mut CJieba,
    s: *const c_char,
    len: usize,
) -> (&'a mut Jieba, &'a CFixedStr) {
    let jieba = unsafe { &mut (*cjieba_ref).jieba };
    let c_str = unsafe { CFixedStr::from_ptr(s, len) };
    (jieba, c_str)
}

/// # Safety
/// Returned value must be freed by `jieba_free()`.
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
pub extern "C" fn jieba_empty() -> *mut CJieba {
    let cjieba = CJieba {
        jieba: Jieba::empty(),
        _marker: Default::default(),
    };
    Box::into_raw(Box::new(cjieba))
}

/// # Safety
/// cjieba is result from `jieba_new()` call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_free(cjieba: *mut CJieba) {
    if !cjieba.is_null() {
        unsafe {
            drop(Box::from_raw(cjieba));
        }
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_cut(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    hmm: bool,
) -> *mut CJiebaWords {
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba, sentence, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let words = jieba.cut(&s, hmm);
    let c_words: Vec<FfiStr> = words
        .into_iter()
        .map(|x| FfiStr::from_string(x.word.to_string()))
        .collect();
    let (words, len) = leak_slice(c_words);
    Box::into_raw(Box::new(CJiebaWords { words, len }))
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_cut_all(cjieba: *mut CJieba, sentence: *const c_char, len: usize) -> *mut CJiebaWords {
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba, sentence, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let words = (*jieba).cut_all(&s);
    let c_words: Vec<FfiStr> = words
        .into_iter()
        .map(|x| FfiStr::from_string(x.word.to_string()))
        .collect();
    let (words, len) = leak_slice(c_words);
    Box::into_raw(Box::new(CJiebaWords { words, len }))
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_cut_for_search(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    hmm: bool,
) -> *mut CJiebaWords {
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba, sentence, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let words = (*jieba).cut_for_search(&s, hmm);
    let c_words: Vec<FfiStr> = words
        .into_iter()
        .map(|x| FfiStr::from_string(x.word.to_string()))
        .collect();
    let (words, len) = leak_slice(c_words);
    Box::into_raw(Box::new(CJiebaWords { words, len }))
}

/// # Safety
/// cjieba must be valid object from `jieba_new()` and must outlive the returned CJiebaTFIDF instance.
///
/// Returned value must be freed by `jieba_tfidf_free()`.
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_tfidf_free(cjieba_tfidf: *mut CJiebaTFIDF) {
    if !cjieba_tfidf.is_null() {
        unsafe {
            drop(Box::from_raw(cjieba_tfidf));
        }
    }
}

/// # Safety
/// cjieba_tfidf must be valid object from `jieba_tfidf_new()`. `sentence` must be `len` or larger.
///
/// Returned value must be freed by `jieba_words_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_tfidf_extract(
    cjieba_tfidf: *mut CJiebaTFIDF,
    sentence: *const c_char,
    len: usize,
    top_k: usize,
    allowed_pos: *const *const c_char,
    allowed_pos_len: usize,
) -> *mut CJiebaWords {
    let cjieba_tfidf_ref = unsafe { &(*cjieba_tfidf) };
    let tfidf = &cjieba_tfidf_ref.tfidf;
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba_tfidf_ref.cjieba, sentence, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());

    let allowed_pos = unsafe { allowed_pos_vec(allowed_pos, allowed_pos_len) };

    let words = tfidf.extract_keywords(jieba, &s, top_k, allowed_pos);
    let c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.keyword)).collect();
    let (words, len) = leak_slice(c_words);
    Box::into_raw(Box::new(CJiebaWords { words, len }))
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
/// `allowed_pos` must point to `allowed_pos_len` NUL-terminated strings, which stay owned by
/// the caller, or `allowed_pos_len` must be 0.
///
/// Returned value must be freed by `jieba_words_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_textrank_extract(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    top_k: usize,
    allowed_pos: *const *const c_char,
    allowed_pos_len: usize,
) -> *mut CJiebaWords {
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba, sentence, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());

    let allowed_pos = unsafe { allowed_pos_vec(allowed_pos, allowed_pos_len) };

    let textrank = TextRank::default();
    let words = textrank.extract_keywords(jieba, &s, top_k, allowed_pos);
    let c_words: Vec<FfiStr> = words.into_iter().map(|x| FfiStr::from_string(x.keyword)).collect();
    let (words, len) = leak_slice(c_words);
    Box::into_raw(Box::new(CJiebaWords { words, len }))
}

/// # Safety
/// c_words is result from `jieba_cut()`, `jieba_cut_all()`, `jieba_cut_for_search()`,
/// `jieba_textrank_extract()` or `jieba_tfidf_extract()` call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_words_free(c_words: *mut CJiebaWords) {
    if !c_words.is_null() {
        unsafe {
            let c_words = Box::from_raw(c_words);
            free_slice(c_words.words, c_words.len);
        }
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
///
/// Returned value must be freed by `jieba_tokens_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_tokenize(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    mode: TokenizeMode,
    hmm: bool,
) -> *mut CJiebaTokens {
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba, sentence, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let tokens = (*jieba).tokenize(&s, mode.into(), hmm);
    let c_tokens: Vec<CJiebaToken> = tokens
        .into_iter()
        .map(|x| CJiebaToken {
            word: FfiStr::from_string(x.word.to_string()),
            start: x.start,
            end: x.end,
        })
        .collect();
    let (tokens, len) = leak_slice(c_tokens);
    Box::into_raw(Box::new(CJiebaTokens { tokens, len }))
}

/// # Safety
/// c_tokens is result from `jieba_tokenize()` call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_tokens_free(c_tokens: *mut CJiebaTokens) {
    if !c_tokens.is_null() {
        unsafe {
            let c_tokens = Box::from_raw(c_tokens);
            free_slice(c_tokens.tokens, c_tokens.len);
        }
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `sentence` must be `len` or larger.
///
/// Returned value must be freed by `jieba_tags_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_tag(
    cjieba: *mut CJieba,
    sentence: *const c_char,
    len: usize,
    hmm: bool,
) -> *mut CJiebaTags {
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba, sentence, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    let tags = (*jieba).tag(&s, hmm);
    let c_tags: Vec<CJiebaTag> = tags
        .into_iter()
        .map(|x| CJiebaTag {
            word: FfiStr::from_string(x.word.to_string()),
            tag: FfiStr::from_string(x.tag.to_string()),
        })
        .collect();
    let (tags, len) = leak_slice(c_tags);
    Box::into_raw(Box::new(CJiebaTags { tags, len }))
}

/// # Safety
/// c_tags is result from `jieba_tag()` call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_tags_free(c_tags: *mut CJiebaTags) {
    if !c_tags.is_null() {
        unsafe {
            let c_tags = Box::from_raw(c_tags);
            free_slice(c_tags.tags, c_tags.len);
        }
    }
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `word` must be `len` or larger.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_add_word(cjieba: *mut CJieba, word: *const c_char, len: usize) -> usize {
    let (jieba, c_str) = unsafe { params_unwrap_mut(cjieba, word, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());
    jieba.add_word(&s, None, None)
}

/// # Safety
/// cjieba must be valid object from `jieba_new()`. `segment` must be `len` or larger.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jieba_suggest_freq(cjieba: *mut CJieba, segment: *const c_char, len: usize) -> usize {
    let (jieba, c_str) = unsafe { params_unwrap(&cjieba, segment, len) };
    // FIXME: remove allocation
    let s = String::from_utf8_lossy(c_str.as_bytes_full());

    (*jieba).suggest_freq(&s)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::ffi::CString;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// An allocator that remembers the size of every live allocation and
    /// aborts when one is freed with a different size, which the system
    /// allocator would silently accept. Every test in this crate runs on
    /// it, so a result freed through the wrong layout fails the test run.
    struct CheckedAlloc;

    /// Open-addressed table from allocation address to size. TfIdf::default
    /// alone holds ~270k allocations, so it is sized well above that, and
    /// a freed slot becomes a tombstone that a later insert may reuse.
    const SLOTS: usize = 1 << 20;
    const EMPTY: usize = 0;
    const TOMBSTONE: usize = usize::MAX;
    /// Probes before giving up on an address, so a pathological run can
    /// never hang the tests; an untracked address is simply not checked.
    const MAX_PROBES: usize = 128;
    static PTRS: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(EMPTY) }; SLOTS];
    static SIZES: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];

    fn slot(ptr: usize) -> usize {
        (ptr.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) & (SLOTS - 1)
    }

    fn record(ptr: *mut u8, size: usize) {
        let key = ptr as usize;
        let mut i = slot(key);
        for _ in 0..MAX_PROBES {
            let stored = PTRS[i].load(Ordering::Acquire);
            if (stored == EMPTY || stored == TOMBSTONE)
                && PTRS[i]
                    .compare_exchange(stored, key, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                SIZES[i].store(size, Ordering::Release);
                return;
            }
            i = (i + 1) & (SLOTS - 1);
        }
    }

    fn check_and_forget(ptr: *mut u8, size: usize) {
        let key = ptr as usize;
        let mut i = slot(key);
        for _ in 0..MAX_PROBES {
            let stored = PTRS[i].load(Ordering::Acquire);
            if stored == key {
                let recorded = SIZES[i].load(Ordering::Acquire);
                if recorded != size {
                    eprintln!("freeing {size} bytes at {ptr:p} that were allocated as {recorded} bytes");
                    std::process::abort();
                }
                PTRS[i].store(TOMBSTONE, Ordering::Release);
                return;
            }
            if stored == EMPTY {
                return; // never tracked
            }
            i = (i + 1) & (SLOTS - 1);
        }
    }

    unsafe impl GlobalAlloc for CheckedAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(layout) };
            if !p.is_null() {
                record(p, layout.size());
            }
            p
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            check_and_forget(ptr, layout.size());
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            check_and_forget(ptr, layout.size());
            let p = unsafe { System.realloc(ptr, layout, new_size) };
            if !p.is_null() {
                record(p, new_size);
            }
            p
        }
    }

    #[global_allocator]
    static ALLOC: CheckedAlloc = CheckedAlloc;

    const SENTENCE: &str = "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。";

    fn c_sentence() -> CString {
        CString::new(SENTENCE).unwrap()
    }

    unsafe fn words_to_vec(c_words: *mut CJiebaWords) -> Vec<String> {
        let c_words = unsafe { &*c_words };
        let words = unsafe { std::slice::from_raw_parts(c_words.words, c_words.len) };
        words
            .iter()
            .map(|w| {
                let bytes = unsafe { std::slice::from_raw_parts(w.data as *const u8, w.len) };
                String::from_utf8(bytes.to_vec()).unwrap()
            })
            .collect()
    }

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

    #[test]
    fn test_ffi_str_round_trip() {
        for s in ["", "a", "拖拉机", &"x".repeat(1000)] {
            let mut f = FfiStr::from_string(s.to_string());
            assert_eq!(f.len, s.len());
            unsafe { f.free() };
            assert!(f.data.is_null());
            assert!(!f.owned);
        }
        unsafe { jieba_str_free(ptr::null_mut()) };
    }

    #[test]
    fn test_cut_results_free_with_their_own_layout() {
        let jieba = jieba_new();
        let s = c_sentence();
        unsafe {
            for hmm in [false, true] {
                let words = jieba_cut(jieba, s.as_ptr(), SENTENCE.len(), hmm);
                let got = words_to_vec(words);
                assert_eq!(got[..3], ["我", "是", "拖拉机"]);
                jieba_words_free(words);

                let words = jieba_cut_for_search(jieba, s.as_ptr(), SENTENCE.len(), hmm);
                assert!(!words_to_vec(words).is_empty());
                jieba_words_free(words);
            }
            let words = jieba_cut_all(jieba, s.as_ptr(), SENTENCE.len());
            assert!(words_to_vec(words).contains(&"拖拉机".to_string()));
            jieba_words_free(words);

            let empty = jieba_cut(jieba, s.as_ptr(), 0, true);
            assert_eq!((*empty).len, 0);
            jieba_words_free(empty);
            jieba_words_free(ptr::null_mut());
            jieba_free(jieba);
        }
    }

    #[test]
    fn test_tokens_and_tags_free_with_their_own_layout() {
        let jieba = jieba_new();
        let s = c_sentence();
        unsafe {
            for mode in [TokenizeMode::Default, TokenizeMode::Search] {
                let tokens = jieba_tokenize(jieba, s.as_ptr(), SENTENCE.len(), mode, true);
                let first = &*(*tokens).tokens;
                assert_eq!((first.start, first.end), (0, 1));
                jieba_tokens_free(tokens);
            }
            let tags = jieba_tag(jieba, s.as_ptr(), SENTENCE.len(), true);
            let first = &*(*tags).tags;
            let tag = std::slice::from_raw_parts(first.tag.data as *const u8, first.tag.len);
            assert_eq!(tag, b"r");
            jieba_tags_free(tags);
            jieba_tokens_free(ptr::null_mut());
            jieba_tags_free(ptr::null_mut());
            jieba_free(jieba);
        }
    }

    #[test]
    fn test_keyword_results_leave_allowed_pos_to_the_caller() {
        let jieba = jieba_new();
        let tfidf = jieba_tfidf_new(jieba);
        let s = c_sentence();
        let allowed: Vec<CString> = ["n", "v"].iter().map(|p| CString::new(*p).unwrap()).collect();
        let allowed_ptrs: Vec<*const c_char> = allowed.iter().map(|p| p.as_ptr()).collect();
        unsafe {
            for (ptrs, len) in [(allowed_ptrs.as_ptr(), allowed_ptrs.len()), (ptr::null(), 0)] {
                let words = jieba_tfidf_extract(tfidf, s.as_ptr(), SENTENCE.len(), 3, ptrs, len);
                assert_eq!(words_to_vec(words).len(), 3);
                jieba_words_free(words);

                let words = jieba_textrank_extract(jieba, s.as_ptr(), SENTENCE.len(), 3, ptrs, len);
                assert_eq!(words_to_vec(words).len(), 3);
                jieba_words_free(words);
            }
            jieba_tfidf_free(tfidf);
            jieba_free(jieba);
        }
        // The caller's strings are still intact and still the caller's to free.
        assert_eq!(allowed[0].to_str().unwrap(), "n");
        drop(allowed);
    }
}
