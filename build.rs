extern crate phf_codegen;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

fn main() {
    let path = Path::new(&env::var("OUT_DIR").unwrap()).join("hmm_prob.rs");
    let hmm_file = File::open("src/data/hmm.model").expect("cannot open hmm.model");
    let mut file = BufWriter::new(File::create(path).unwrap());
    let reader = BufReader::new(hmm_file);
    let mut lines = reader.lines().map(|x| x.unwrap()).skip_while(|x| x.starts_with('#'));
    let prob_start = lines.next().unwrap();
    writeln!(&mut file, "#[allow(clippy::style)]").unwrap();
    write!(&mut file, "static INITIAL_PROBS: StateSet = [").unwrap();
    for prob in prob_start.split(' ') {
        write!(&mut file, "{}, ", prob).unwrap();
    }
    write!(&mut file, "];\n\n").unwrap();
    writeln!(&mut file, "#[allow(clippy::style)]").unwrap();
    write!(&mut file, "static TRANS_PROBS: [StateSet; crate::hmm::NUM_STATES] = [").unwrap();
    for line in lines
        .by_ref()
        .skip_while(|x| x.starts_with('#'))
        .take_while(|x| !x.starts_with('#'))
    {
        write!(&mut file, "[").unwrap();
        for prob in line.split(' ') {
            write!(&mut file, "{}, ", prob).unwrap();
        }
        writeln!(&mut file, "],").unwrap();
    }
    write!(&mut file, "];\n\n").unwrap();
    let mut i = 0;
    for line in lines {
        if line.starts_with('#') {
            continue;
        }
        writeln!(&mut file, "#[allow(clippy::style)]").unwrap();
        write!(&mut file, "static EMIT_PROB_{}: phf::Map<&'static str, f64> = ", i).unwrap();
        let mut map = phf_codegen::Map::new();
        for word_prob in line.split(',') {
            let mut parts = word_prob.split(':');
            let word = parts.next().unwrap();
            let prob = parts.next().unwrap();
            map.entry(word, prob);
        }
        writeln!(&mut file, "{};", map.build()).unwrap();
        i += 1;
    }
    writeln!(&mut file, "#[allow(clippy::style)]").unwrap();
    writeln!(&mut file, "static EMIT_PROBS: [&'static phf::Map<&'static str, f64>; crate::hmm::NUM_STATES] = [&EMIT_PROB_0, &EMIT_PROB_1, &EMIT_PROB_2, &EMIT_PROB_3];").unwrap();
}
