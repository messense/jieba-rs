use proc_macro::TokenStream;
use std::collections::BTreeMap;

#[proc_macro]
pub fn generate_hmm_data(_input: TokenStream) -> TokenStream {
    let hmm_data = include_str!("hmm.model");
    let mut output = String::new();
    let mut lines = hmm_data.lines().skip_while(|x| x.starts_with('#'));

    // Initial probabilities
    let init_probs = lines
        .next()
        .expect("Failed to read initial probabilities from hmm.model");

    output.push_str("#[allow(clippy::style)]\n");
    output.push_str("pub static INITIAL_PROBS: [f64; 4] = [");
    output.push_str(&init_probs.replace(' ', ", "));
    output.push_str("];\n\n");

    // Transition probabilities
    output.push_str("#[allow(clippy::style)]\n");
    output.push_str("pub static TRANS_PROBS: [[f64; 4]; 4] = [");
    for line in lines
        .by_ref()
        .skip_while(|x| x.starts_with('#'))
        .take_while(|x| !x.starts_with('#'))
    {
        output.push('[');
        output.push_str(&line.replace(' ', ", "));
        output.push_str("],\n");
    }
    output.push_str("];\n\n");

    // Emission probabilities
    let mut emit_probs: BTreeMap<char, [String; 4]> = BTreeMap::new();
    for (i, line) in lines.filter(|x| !x.starts_with('#')).enumerate() {
        for word_prob in line.split(',') {
            let mut parts = word_prob.split(':');
            let word = parts.next().unwrap();
            let prob = parts.next().unwrap();
            // All emit keys are single characters
            let ch = word.chars().next().unwrap();
            let probs = emit_probs
                .entry(ch)
                .or_insert_with(|| std::array::from_fn(|_| "MIN_FLOAT".to_string()));
            probs[i] = prob.to_string();
        }
    }

    // The segmenter only runs the HMM over `[\u{4E00}-\u{9FD5}]` blocks, so
    // the emission table is a direct index over that range: `EMIT_INDEX` maps
    // `ch - EMIT_MIN_CHAR` to a row of `EMIT_PROBS`, or `EMIT_NONE`. Keys
    // outside the range can never be looked up and are dropped.
    const HMM_HAN_MIN: u32 = 0x4E00;
    const HMM_HAN_MAX: u32 = 0x9FD5;
    let in_range: Vec<(u32, &[String; 4])> = emit_probs
        .iter()
        .map(|(ch, probs)| (*ch as u32, probs))
        .filter(|(ch, _)| (HMM_HAN_MIN..=HMM_HAN_MAX).contains(ch))
        .collect();
    let min_char = in_range.first().map_or(HMM_HAN_MIN, |(ch, _)| *ch);
    let max_char = in_range.last().map_or(HMM_HAN_MIN, |(ch, _)| *ch);
    let index_len = (max_char - min_char + 1) as usize;
    assert!(in_range.len() < u16::MAX as usize, "too many HMM emission entries for a u16 index");

    let mut index = vec![u16::MAX; index_len];
    for (row, (ch, _)) in in_range.iter().enumerate() {
        index[(ch - min_char) as usize] = row as u16;
    }

    output.push_str(&format!("pub const EMIT_MIN_CHAR: u32 = {min_char:#x};\n"));
    output.push_str("pub const EMIT_NONE: u16 = u16::MAX;\n\n");

    output.push_str("#[allow(clippy::style)]\n");
    output.push_str(&format!("pub static EMIT_INDEX: [u16; {index_len}] = ["));
    for (i, row) in index.iter().enumerate() {
        if i % 32 == 0 {
            output.push('\n');
        }
        output.push_str(&row.to_string());
        output.push(',');
    }
    output.push_str("\n];\n\n");

    output.push_str("#[allow(clippy::style)]\n");
    output.push_str(&format!("pub static EMIT_PROBS: [[f64; 4]; {}] = [\n", in_range.len()));
    for (_, probs) in &in_range {
        output.push_str(&format!("[{}, {}, {}, {}],\n", probs[0], probs[1], probs[2], probs[3]));
    }
    output.push_str("];\n\n");

    output.parse().unwrap()
}
