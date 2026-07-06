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

    output.push_str("#[allow(clippy::style)]\n");
    output.push_str("pub static EMIT_PROBS: phf::Map<char, [f64; 4]> = ");
    let mut map = phf_codegen::Map::new();
    for (ch, probs) in emit_probs {
        map.entry(ch, format!("[{}, {}, {}, {}]", probs[0], probs[1], probs[2], probs[3]));
    }
    output.push_str(&map.build().to_string());
    output.push_str(";\n\n");

    output.parse().unwrap()
}
