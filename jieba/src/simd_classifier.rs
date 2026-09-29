//! Bounded SIMD acceleration of the default Jieba character class, ported
//! from the jieba-rs copy vendored in Brooooooklyn/pinyin.
//!
//! aarch64 uses baseline NEON. x86_64 dispatches at runtime through
//! AVX-512BW/VBMI, AVX2 and SSSE3 tiers, keeping an SSE2 ASCII path that is
//! baseline on every x86_64 CPU. Other targets return 0 and the caller keeps
//! the scalar loop.

/// Length in bytes of a prefix of `input` made up only of characters that
/// [`is_han_default`](crate::is_han_default) accepts: either ASCII bytes of
/// `[a-zA-Z0-9+#&._%-]`, or three-byte characters of U+4E00..=U+9FFF, never a
/// mix of the two. The prefix is not maximal, and 0 only means that the fast
/// path has nothing to say about the first character.
///
/// `input` must be valid UTF-8; the prefix then ends at a character boundary.
pub(crate) fn prefix(input: &[u8]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: every path checks for a complete 16-, 24- or 48-byte block
        // before loading it. NEON is baseline on the supported ARM64 targets.
        unsafe {
            use std::arch::aarch64::*;
            if input.len() >= 16 && input[0].is_ascii() {
                let v = vld1q_u8(input.as_ptr());
                let between = |lo, hi| vandq_u8(vcgeq_u8(v, vdupq_n_u8(lo)), vcleq_u8(v, vdupq_n_u8(hi)));
                let mut mask = vorrq_u8(between(b'a', b'z'), vorrq_u8(between(b'A', b'Z'), between(b'0', b'9')));
                for c in *b"+#&._%-" {
                    mask = vorrq_u8(mask, vceqq_u8(v, vdupq_n_u8(c)));
                }
                let bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(mask), 4)), 0);
                return bits.trailing_ones() as usize / 4;
            }
            // U+4E00..=U+9FFF in UTF-8 is lead E4..=E9, where E4 needs a second
            // byte >= 0xB8; E9 BF BF is exactly U+9FFF. `input` is valid UTF-8
            // and starts at a character boundary, so as long as every triplet
            // so far had such a lead, the next one starts at a boundary too:
            // the leading run of accepted triplets needs no continuation-byte
            // check, and whatever a misaligned triplet after it looks like
            // does not matter.
            if input.len() >= 48 {
                let bytes = vld3q_u8(input.as_ptr());
                let mask = vorrq_u8(
                    vcleq_u8(vsubq_u8(bytes.0, vdupq_n_u8(0xe5)), vdupq_n_u8(4)),
                    vandq_u8(vceqq_u8(bytes.0, vdupq_n_u8(0xe4)), vcgeq_u8(bytes.1, vdupq_n_u8(0xb8))),
                );
                let bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(mask), 4)), 0);
                return bits.trailing_ones() as usize / 4 * 3;
            }
            if input.len() >= 24 {
                let bytes = vld3_u8(input.as_ptr());
                let mask = vorr_u8(
                    vcle_u8(vsub_u8(bytes.0, vdup_n_u8(0xe5)), vdup_n_u8(4)),
                    vand_u8(vceq_u8(bytes.0, vdup_n_u8(0xe4)), vcge_u8(bytes.1, vdup_n_u8(0xb8))),
                );
                let bits = vget_lane_u64(vreinterpret_u64_u8(mask), 0);
                return bits.trailing_ones() as usize / 8 * 3;
            }
        }
        0
    }

    #[cfg(target_arch = "x86_64")]
    {
        x86::prefix(input)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = input; // Portable builds compile out the SIMD branches.
        0
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    /// Highest SIMD tier supported by the running CPU.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub(super) enum Tier {
        /// No SSSE3: only the baseline SSE2 ASCII block is available.
        Scalar,
        /// SSSE3: 48-byte CJK blocks through `pshufb` deinterleaving.
        Ssse3,
        /// AVX2: 32-byte ASCII and 96-byte CJK blocks.
        Avx2,
        /// AVX-512F/BW/VBMI: 64-byte ASCII and 192-byte CJK blocks.
        Avx512,
    }

    pub(super) fn tier() -> Tier {
        // Probe once per process; is_x86_feature_detected is an atomic load after
        // the first call, and the OnceLock keeps this to a single predictable read.
        static TIER: OnceLock<Tier> = OnceLock::new();
        *TIER.get_or_init(|| {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vbmi")
            {
                Tier::Avx512
            } else if std::is_x86_feature_detected!("avx2") {
                Tier::Avx2
            } else if std::is_x86_feature_detected!("ssse3") {
                Tier::Ssse3
            } else {
                Tier::Scalar
            }
        })
    }

    pub fn prefix(input: &[u8]) -> usize {
        let tier = tier();
        if input.len() >= 16 && input[0].is_ascii() {
            if tier >= Tier::Avx512 && input.len() >= 64 {
                // SAFETY: AVX-512F/BW were detected at runtime; the block load is
                // bounded by the length check above.
                return unsafe { ascii64(input) };
            }
            if tier >= Tier::Avx2 && input.len() >= 32 {
                // SAFETY: AVX2 was detected at runtime; the block load is bounded.
                return unsafe { ascii32(input) };
            }
            return ascii16(input);
        }
        // A `None` gate failure (a non-three-byte character inside the block)
        // retries at narrower widths so a single emoji does not lose the run.
        // SAFETY: AVX-512F/BW/VBMI were detected at runtime; loads are bounded.
        if tier >= Tier::Avx512
            && input.len() >= 192
            && let Some(n) = unsafe { cjk192(input) }
        {
            return n;
        }
        // SAFETY: AVX2 was detected at runtime; loads are bounded.
        if tier >= Tier::Avx2
            && input.len() >= 96
            && let Some(n) = unsafe { cjk96(input) }
        {
            return n;
        }
        // SAFETY: SSSE3 was detected at runtime; loads are bounded.
        if tier >= Tier::Ssse3
            && input.len() >= 48
            && let Some(n) = unsafe { cjk48(input) }
        {
            return n;
        }
        0
    }

    /// Accepted ASCII run, up to 16 bytes. SSE2 is baseline on x86_64.
    pub(super) fn ascii16(input: &[u8]) -> usize {
        debug_assert!(input.len() >= 16);
        // SAFETY: the caller checked the 16-byte block length.
        unsafe {
            let v = _mm_loadu_si128(input.as_ptr().cast());
            let bits = _mm_movemask_epi8(ascii_mask128(v)) as u16;
            (bits.trailing_ones() as usize).min(16)
        }
    }

    /// Accepted ASCII run, up to 32 bytes.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ascii32(input: &[u8]) -> usize {
        debug_assert!(input.len() >= 32);
        // SAFETY: the caller checked the 32-byte block length.
        unsafe {
            let v = _mm256_loadu_si256(input.as_ptr().cast());
            let bits = _mm256_movemask_epi8(ascii_mask256(v)) as u32;
            (bits.trailing_ones() as usize).min(32)
        }
    }

    /// Accepted ASCII run, up to 64 bytes.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    pub(super) unsafe fn ascii64(input: &[u8]) -> usize {
        debug_assert!(input.len() >= 64);
        // SAFETY: the caller checked the 64-byte block length.
        unsafe {
            let v = _mm512_loadu_si512(input.as_ptr().cast());
            (ascii_mask512(v).trailing_ones() as usize).min(64)
        }
    }

    /// `[a-zA-Z0-9+#&._%-]` membership as a byte mask. Unsigned range checks use
    /// the `max(v, lo) == v == min(v, hi)` idiom; bytes >= 0x80 never match.
    #[inline]
    fn ascii_mask128(v: __m128i) -> __m128i {
        // SAFETY: register-only SSE2, baseline on x86_64.
        unsafe {
            let splat = |x: u8| _mm_set1_epi8(x as i8);
            let ge = |x: u8| _mm_cmpeq_epi8(_mm_max_epu8(v, splat(x)), v);
            let le = |x: u8| _mm_cmpeq_epi8(_mm_min_epu8(v, splat(x)), v);
            let range = |lo: u8, hi: u8| _mm_and_si128(ge(lo), le(hi));
            let mut mask = _mm_or_si128(range(b'a', b'z'), _mm_or_si128(range(b'A', b'Z'), range(b'0', b'9')));
            for c in *b"+#&._%-" {
                mask = _mm_or_si128(mask, _mm_cmpeq_epi8(v, splat(c)));
            }
            mask
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ascii_mask256(v: __m256i) -> __m256i {
        // Register-only AVX2; the caller detected the feature.
        let splat = |x: u8| _mm256_set1_epi8(x as i8);
        let ge = |x: u8| _mm256_cmpeq_epi8(_mm256_max_epu8(v, splat(x)), v);
        let le = |x: u8| _mm256_cmpeq_epi8(_mm256_min_epu8(v, splat(x)), v);
        let range = |lo: u8, hi: u8| _mm256_and_si256(ge(lo), le(hi));
        let mut mask = _mm256_or_si256(range(b'a', b'z'), _mm256_or_si256(range(b'A', b'Z'), range(b'0', b'9')));
        for c in *b"+#&._%-" {
            mask = _mm256_or_si256(mask, _mm256_cmpeq_epi8(v, splat(c)));
        }
        mask
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn ascii_mask512(v: __m512i) -> __mmask64 {
        // Register-only AVX-512; the caller detected the features.
        let splat = |x: u8| _mm512_set1_epi8(x as i8);
        let ge = |x: u8| _mm512_cmpge_epu8_mask(v, splat(x));
        let le = |x: u8| _mm512_cmple_epu8_mask(v, splat(x));
        let range = |lo: u8, hi: u8| ge(lo) & le(hi);
        let mut mask = range(b'a', b'z') | range(b'A', b'Z') | range(b'0', b'9');
        for c in *b"+#&._%-" {
            mask |= _mm512_cmpeq_epu8_mask(v, splat(c));
        }
        mask
    }

    /// `pshufb` index selecting bytes of one deinterleaved stream from a
    /// 16-byte block with the given phase in the 48-byte repeating unit.
    /// Unselected lanes are zeroed (index high bit set).
    const fn deinterleave_idx(stream: usize, phase: usize) -> [u8; 16] {
        let mut idx = [0x80u8; 16];
        let mut t = 0usize;
        while t < 16 {
            let g = 3 * t + stream;
            if g / 16 == phase {
                idx[t] = (g % 16) as u8;
            }
            t += 1;
        }
        idx
    }

    /// IDX[stream][phase] for the three 16-byte blocks of a 48-byte unit.
    const IDX: [[[u8; 16]; 3]; 3] = [
        [deinterleave_idx(0, 0), deinterleave_idx(0, 1), deinterleave_idx(0, 2)],
        [deinterleave_idx(1, 0), deinterleave_idx(1, 1), deinterleave_idx(1, 2)],
        [deinterleave_idx(2, 0), deinterleave_idx(2, 1), deinterleave_idx(2, 2)],
    ];

    /// `vpermt2b`/`vpermb` indices deinterleaving 192 bytes held in three
    /// 64-byte registers: the first permute covers byte positions below 128,
    /// the masked second permute covers positions in the third register.
    const fn perm_idx(stream: usize) -> ([u8; 64], [u8; 64], u64) {
        let mut first = [0u8; 64];
        let mut second = [0u8; 64];
        let mut mask = 0u64;
        let mut t = 0usize;
        while t < 64 {
            let g = 3 * t + stream;
            if g < 128 {
                first[t] = g as u8;
            } else {
                second[t] = (g - 128) as u8;
                mask |= 1 << t;
            }
            t += 1;
        }
        (first, second, mask)
    }

    const PERM: [([u8; 64], [u8; 64], u64); 3] = [perm_idx(0), perm_idx(1), perm_idx(2)];

    /// Common-CJK prefix of a 48-byte block: 16 triplets through SSSE3.
    /// `None` reports a non-three-byte character inside the block (gate),
    /// letting the caller retry narrower; `Some(0)` is a real class change.
    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn cjk48(input: &[u8]) -> Option<usize> {
        debug_assert!(input.len() >= 48);
        // SAFETY: the caller checked the 48-byte block length.
        unsafe {
            let m0 = _mm_loadu_si128(input.as_ptr().cast());
            let m1 = _mm_loadu_si128(input.as_ptr().add(16).cast());
            let m2 = _mm_loadu_si128(input.as_ptr().add(32).cast());
            let gather = |stream: usize| {
                let a = _mm_shuffle_epi8(m0, _mm_loadu_si128(IDX[stream][0].as_ptr().cast()));
                let b = _mm_shuffle_epi8(m1, _mm_loadu_si128(IDX[stream][1].as_ptr().cast()));
                let c = _mm_shuffle_epi8(m2, _mm_loadu_si128(IDX[stream][2].as_ptr().cast()));
                _mm_or_si128(_mm_or_si128(a, b), c)
            };
            let b0 = gather(0);
            let b1 = gather(1);
            let b2 = gather(2);
            let splat = |x: u8| _mm_set1_epi8(x as i8);
            let eq = |v: __m128i, x: u8| _mm_cmpeq_epi8(v, splat(x));
            let cont = |v: __m128i| eq(_mm_and_si128(v, splat(0xc0)), 0x80);
            let valid = _mm_and_si128(
                eq(_mm_and_si128(b0, splat(0xf0)), 0xe0),
                _mm_and_si128(cont(b1), cont(b2)),
            );
            if _mm_movemask_epi8(valid) as u16 != 0xffff {
                return None;
            }
            Some(cjk_count128(b0, b1))
        }
    }

    /// Common-CJK prefix of a 96-byte block: 32 triplets through AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn cjk96(input: &[u8]) -> Option<usize> {
        debug_assert!(input.len() >= 96);
        // SAFETY: the caller checked the 96-byte block length.
        unsafe {
            let y0 = _mm256_loadu_si256(input.as_ptr().cast());
            let y1 = _mm256_loadu_si256(input.as_ptr().add(32).cast());
            let y2 = _mm256_loadu_si256(input.as_ptr().add(64).cast());
            // `vpshufb` shuffles within each 128-bit lane, so pair up the
            // 16-byte blocks that share a phase: the low lanes hold the first
            // 48-byte unit and the high lanes the second.
            let phase = [
                _mm256_permute2x128_si256::<0x30>(y0, y1),
                _mm256_permute2x128_si256::<0x21>(y0, y2),
                _mm256_permute2x128_si256::<0x30>(y1, y2),
            ];
            let gather = |stream: usize| {
                let idx = |p: usize| _mm256_broadcastsi128_si256(_mm_loadu_si128(IDX[stream][p].as_ptr().cast()));
                let a = _mm256_shuffle_epi8(phase[0], idx(0));
                let b = _mm256_shuffle_epi8(phase[1], idx(1));
                let c = _mm256_shuffle_epi8(phase[2], idx(2));
                _mm256_or_si256(_mm256_or_si256(a, b), c)
            };
            let b0 = gather(0);
            let b1 = gather(1);
            let b2 = gather(2);
            let splat = |x: u8| _mm256_set1_epi8(x as i8);
            let eq = |v: __m256i, x: u8| _mm256_cmpeq_epi8(v, splat(x));
            let cont = |v: __m256i| eq(_mm256_and_si256(v, splat(0xc0)), 0x80);
            let valid = _mm256_and_si256(
                eq(_mm256_and_si256(b0, splat(0xf0)), 0xe0),
                _mm256_and_si256(cont(b1), cont(b2)),
            );
            if _mm256_movemask_epi8(valid) as u32 != u32::MAX {
                return None;
            }
            Some(cjk_count256(b0, b1))
        }
    }

    /// Common-CJK prefix of a 192-byte block: 64 triplets through AVX-512.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    pub(super) unsafe fn cjk192(input: &[u8]) -> Option<usize> {
        debug_assert!(input.len() >= 192);
        // SAFETY: the caller checked the 192-byte block length.
        unsafe {
            let z0 = _mm512_loadu_si512(input.as_ptr().cast());
            let z1 = _mm512_loadu_si512(input.as_ptr().add(64).cast());
            let z2 = _mm512_loadu_si512(input.as_ptr().add(128).cast());
            let gather = |stream: usize| {
                let (first, second, mask) = &PERM[stream];
                let t = _mm512_permutex2var_epi8(z0, _mm512_loadu_si512(first.as_ptr().cast()), z1);
                _mm512_mask_permutexvar_epi8(t, *mask, _mm512_loadu_si512(second.as_ptr().cast()), z2)
            };
            let b0 = gather(0);
            let b1 = gather(1);
            let b2 = gather(2);
            let splat = |x: u8| _mm512_set1_epi8(x as i8);
            let cont = |v: __m512i| _mm512_cmpeq_epi8_mask(_mm512_and_si512(v, splat(0xc0)), splat(0x80));
            let valid = _mm512_cmpeq_epi8_mask(_mm512_and_si512(b0, splat(0xf0)), splat(0xe0)) & cont(b1) & cont(b2);
            if valid != u64::MAX {
                return None;
            }
            // U+4E00..=U+9FFF in UTF-8 is lead E4..=E9, where E4 needs b1 >= 0xB8;
            // E9 BF BF is exactly U+9FFF, so E9 imposes no extra b1 bound.
            let inrange = (_mm512_cmpge_epu8_mask(b0, splat(0xe5)) & _mm512_cmple_epu8_mask(b0, splat(0xe9)))
                | (_mm512_cmpeq_epu8_mask(b0, splat(0xe4)) & _mm512_cmpge_epu8_mask(b1, splat(0xb8)));
            Some(inrange.trailing_ones() as usize * 3)
        }
    }

    /// Leading triplets inside U+4E00..=U+9FFF, counted in bytes, 128-bit.
    #[inline]
    fn cjk_count128(b0: __m128i, b1: __m128i) -> usize {
        // SAFETY: register-only SSE2, baseline on x86_64.
        unsafe {
            let splat = |x: u8| _mm_set1_epi8(x as i8);
            let ge = |v: __m128i, x: u8| _mm_cmpeq_epi8(_mm_max_epu8(v, splat(x)), v);
            let le = |v: __m128i, x: u8| _mm_cmpeq_epi8(_mm_min_epu8(v, splat(x)), v);
            let inrange = _mm_or_si128(
                _mm_and_si128(ge(b0, 0xe5), le(b0, 0xe9)),
                _mm_and_si128(_mm_cmpeq_epi8(b0, splat(0xe4)), ge(b1, 0xb8)),
            );
            let bits = _mm_movemask_epi8(inrange) as u16;
            bits.trailing_ones() as usize * 3
        }
    }

    /// Leading triplets inside U+4E00..=U+9FFF, counted in bytes, 256-bit.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cjk_count256(b0: __m256i, b1: __m256i) -> usize {
        // Register-only AVX2; the caller detected the feature.
        let splat = |x: u8| _mm256_set1_epi8(x as i8);
        let ge = |v: __m256i, x: u8| _mm256_cmpeq_epi8(_mm256_max_epu8(v, splat(x)), v);
        let le = |v: __m256i, x: u8| _mm256_cmpeq_epi8(_mm256_min_epu8(v, splat(x)), v);
        let inrange = _mm256_or_si256(
            _mm256_and_si256(ge(b0, 0xe5), le(b0, 0xe9)),
            _mm256_and_si256(_mm256_cmpeq_epi8(b0, splat(0xe4)), ge(b1, 0xb8)),
        );
        let bits = _mm256_movemask_epi8(inrange) as u32;
        bits.trailing_ones() as usize * 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    fn ascii_accepted(b: u8) -> bool {
        matches!(
          b,
          b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'+' | b'#' | b'&' | b'.' | b'_' | b'%' | b'-'
        )
    }

    #[cfg(target_arch = "x86_64")]
    /// Scalar model of the ASCII block: leading accepted bytes, capped.
    fn ascii_ref(input: &[u8], cap: usize) -> usize {
        if input.len() < cap || !input[0].is_ascii() {
            return 0;
        }
        input.iter().take(cap).take_while(|&&b| ascii_accepted(b)).count()
    }

    #[cfg(target_arch = "x86_64")]
    /// Scalar model of the CJK block: all `cap` triplets must be three-byte
    /// characters; the result counts the leading common-CJK ones in bytes.
    fn cjk_ref(s: &str, cap: usize) -> Option<usize> {
        if s.len() < cap * 3 {
            return None;
        }
        let mut it = s.chars();
        let mut count = 0;
        for i in 0..cap {
            let c = it.next()? as u32;
            if !(0x800..=0xffff).contains(&c) {
                return None;
            }
            if count == i && (0x4e00..0xa000).contains(&c) {
                count += 1;
            }
        }
        Some(count * 3)
    }

    /// Full matched-run oracle for the splitter contract.
    fn run_ref(s: &str) -> usize {
        let mut n = 0;
        for c in s.chars() {
            let ok = c.is_ascii_alphanumeric()
                || matches!(c, '+' | '#' | '&' | '.' | '_' | '%' | '-')
                || matches!(c as u32,
          0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff
          | 0x20000..=0x2a6df | 0x2a700..=0x2b73f | 0x2b740..=0x2b81f
          | 0x2b820..=0x2ceaf | 0x2ceb0..=0x2ebef | 0x2f800..=0x2fa1f);
            if !ok {
                break;
            }
            n += c.len_utf8();
        }
        n
    }

    /// Every dispatch result must be a char-boundary prefix of the true run.
    fn check_contract(s: &str) {
        let n = prefix(s.as_bytes());
        assert!(n <= run_ref(s), "{s:?}: prefix {n} exceeds run {}", run_ref(s));
        assert!(s.is_char_boundary(n), "{s:?}: prefix {n} splits a character");
    }

    #[test]
    fn contract_over_crafted_inputs() {
        let cjk = "中文测试字符串"; // 21 bytes
        let ext_a = "㐀㐁"; // U+3400, three-byte but outside the fast range
        for text in [
            "",
            "a",
            "abc",
            "abc def",
            "ABC123+#&._%-x",
            "hello world this is a fairly long ascii run indeed",
            "中文",
            cjk,
            "中a文",
            "中文🙂测试",
            "🙂中文",
            ext_a,
            "中文，标点。后续",
            "０１２全角数字",
            &"x".repeat(100),
            &cjk.repeat(20),
            &"中文🙂".repeat(30),
            &format!("{}🙂{}", cjk.repeat(10), cjk.repeat(10)),
            &"阿".repeat(17),
            "çak拾得", // two-byte character first
            "\u{10ffff}",
        ] {
            check_contract(text);
        }
    }

    #[test]
    fn contract_over_random_inputs() {
        let pieces = [
            "中",
            "文",
            "测",
            "a",
            "b9",
            "+",
            " ",
            "，",
            "。",
            "🙂",
            "é",
            "\u{3400}",
            "\u{f900}",
            "\0",
            "z9_",
            "&x",
            "\u{10ffff}",
        ];
        let mut seed = 0x243f6a88u32;
        for _ in 0..2000 {
            let mut s = String::new();
            for _ in 0..40 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                s.push_str(pieces[(seed >> 16) as usize % pieces.len()]);
            }
            check_contract(&s);
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod x86_tiers {
        use super::super::x86::{Tier, ascii16, ascii32, ascii64, cjk48, cjk96, cjk192, tier};
        use super::{ascii_ref, cjk_ref, prefix};

        #[test]
        fn ascii_blocks_match_scalar() {
            for b in 0u16..=255 {
                for pos in [0, 1, 15, 16, 31, 33, 63] {
                    let mut input = vec![b'a'; 80];
                    input[pos] = b as u8;
                    let expected16 = ascii_ref(&input, 16);
                    assert_eq!(ascii16(&input), expected16, "sse2 byte {b:#x} at {pos}");
                    if std::is_x86_feature_detected!("avx2") {
                        // SAFETY: detected above.
                        assert_eq!(
                            unsafe { ascii32(&input) },
                            ascii_ref(&input, 32),
                            "avx2 byte {b:#x} at {pos}"
                        );
                    }
                    if std::is_x86_feature_detected!("avx512f")
                        && std::is_x86_feature_detected!("avx512bw")
                        && std::is_x86_feature_detected!("avx512vbmi")
                    {
                        // SAFETY: detected above.
                        assert_eq!(
                            unsafe { ascii64(&input) },
                            ascii_ref(&input, 64),
                            "avx512 byte {b:#x} at {pos}"
                        );
                    }
                }
            }
        }

        #[test]
        fn cjk_blocks_sweep_all_three_byte_scalars() {
            for code in (0x800u32..=0xffff).step_by(1) {
                let Some(c) = char::from_u32(code) else { continue };
                let s: String = std::iter::repeat_n(c, 70).collect();
                if std::is_x86_feature_detected!("ssse3") {
                    // SAFETY: detected above; the input holds 210 bytes.
                    assert_eq!(unsafe { cjk48(s.as_bytes()) }, cjk_ref(&s, 16), "ssse3 U+{code:04X}");
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: detected above.
                    assert_eq!(unsafe { cjk96(s.as_bytes()) }, cjk_ref(&s, 32), "avx2 U+{code:04X}");
                }
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vbmi")
                {
                    // SAFETY: detected above.
                    assert_eq!(unsafe { cjk192(s.as_bytes()) }, cjk_ref(&s, 64), "avx512 U+{code:04X}");
                }
            }
        }

        #[test]
        fn cjk_blocks_track_prefix_length() {
            let run = "中文测试字符串长度验证abcdefghij"; // mixed tail stops the count
            for k in 0..10 {
                let s = format!("{}{}", "中".repeat(k), &run[run.len().min(k * 3)..]);
                let pad = "一".repeat(80); // keep every block length check satisfied
                let s = format!("{s}{pad}");
                let expect = |cap: usize| cjk_ref(&s, cap);
                if std::is_x86_feature_detected!("ssse3") {
                    // SAFETY: detected above; input exceeds 48 bytes.
                    assert_eq!(unsafe { cjk48(s.as_bytes()) }, expect(16), "ssse3 k={k}");
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: detected above; input exceeds 96 bytes.
                    assert_eq!(unsafe { cjk96(s.as_bytes()) }, expect(32), "avx2 k={k}");
                }
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vbmi")
                {
                    // SAFETY: detected above; input exceeds 192 bytes.
                    assert_eq!(unsafe { cjk192(s.as_bytes()) }, expect(64), "avx512 k={k}");
                }
            }
        }

        #[test]
        fn cjk_gate_reports_none_for_wider_or_shorter_encodings() {
            // "，" passes the three-byte gate but ends the common-CJK count.
            for intruder in ["a", "é", "🙂", "\u{7ff}", "，"] {
                // Every triplet position of the widest block, and one past it.
                for pos in 0usize..=64 {
                    let mut s = "中".repeat(80);
                    s.insert_str(pos * 3, intruder);
                    let expected = |cap: usize| cjk_ref(&s, cap);
                    if std::is_x86_feature_detected!("ssse3") {
                        // SAFETY: detected above; input exceeds 48 bytes.
                        assert_eq!(
                            unsafe { cjk48(s.as_bytes()) },
                            expected(16),
                            "ssse3 {intruder:?} at {pos}"
                        );
                    }
                    if std::is_x86_feature_detected!("avx2") {
                        // SAFETY: detected above; input exceeds 96 bytes.
                        assert_eq!(
                            unsafe { cjk96(s.as_bytes()) },
                            expected(32),
                            "avx2 {intruder:?} at {pos}"
                        );
                    }
                    if std::is_x86_feature_detected!("avx512f")
                        && std::is_x86_feature_detected!("avx512bw")
                        && std::is_x86_feature_detected!("avx512vbmi")
                    {
                        // SAFETY: detected above; input exceeds 192 bytes.
                        assert_eq!(
                            unsafe { cjk192(s.as_bytes()) },
                            expected(64),
                            "avx512 {intruder:?} at {pos}"
                        );
                    }
                }
            }
        }

        #[test]
        fn dispatch_matches_tier_model() {
            let model = |s: &str| {
                let input = s.as_bytes();
                let tier = tier();
                if input.len() >= 16 && input[0].is_ascii() {
                    if tier >= Tier::Avx512 && input.len() >= 64 {
                        return ascii_ref(input, 64);
                    }
                    if tier >= Tier::Avx2 && input.len() >= 32 {
                        return ascii_ref(input, 32);
                    }
                    return ascii_ref(input, 16);
                }
                if tier >= Tier::Avx512
                    && input.len() >= 192
                    && let Some(n) = cjk_ref(s, 64)
                {
                    return n;
                }
                if tier >= Tier::Avx2
                    && input.len() >= 96
                    && let Some(n) = cjk_ref(s, 32)
                {
                    return n;
                }
                if tier >= Tier::Ssse3
                    && input.len() >= 48
                    && let Some(n) = cjk_ref(s, 16)
                {
                    return n;
                }
                0
            };
            for text in [
                "hello world this is a fairly long ascii run indeed, over sixty-four bytes for sure!",
                "中文测试字符串",
                &"中".repeat(100),
                &"中文🙂".repeat(40),
                &format!("{}🙂{}", "文".repeat(30), "字".repeat(30)),
                "㐀㐁㐂㐃㐄㐅㐆㐇㐈㐉㐊㐌㐍㐎㐏㐐㐑",
            ] {
                assert_eq!(prefix(text.as_bytes()), model(text), "{text:?}");
            }
        }
    }
}
