use proc_macro::TokenStream;

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
    for (i, line) in lines.filter(|x| !x.starts_with('#')).enumerate() {
        output.push_str("#[allow(clippy::style)]\n");
        output.push_str(&format!("pub static EMIT_PROB_{}: phf::Map<&'static str, f64> = ", i));

        let mut map = phf_codegen::Map::new();
        for word_prob in line.split(',') {
            let mut parts = word_prob.split(':');
            let word = parts.next().unwrap();
            let prob = parts.next().unwrap();
            map.entry(word, prob);
        }
        output.push_str(&map.build().to_string());
        output.push_str(";\n\n");
    }

    output.push_str("#[allow(clippy::style)]\n");
    output.push_str("pub static EMIT_PROBS: [&'static phf::Map<&'static str, f64>; 4] = [&EMIT_PROB_0, &EMIT_PROB_1, &EMIT_PROB_2, &EMIT_PROB_3];\n\n");

    output.parse().unwrap()
}
