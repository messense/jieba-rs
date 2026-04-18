#!/usr/bin/env python3
"""
Train and evaluate an HMM model for jieba-rs from a word-segmented corpus.

Usage:
    # Download corpora, train, and evaluate:
    python scripts/train_hmm.py --download --eval --output jieba-macros/src/hmm.model

    # Train from a custom corpus:
    python scripts/train_hmm.py --corpus my_corpus.utf8 --output hmm.model

    # Evaluate an existing model against SIGHAN gold standard:
    python scripts/train_hmm.py --eval --model hmm.model

Corpus format:
    Each line is a sentence with words separated by one or more spaces.
    Example:
        中国 人民 站 起来 了
        我 是 中国 人
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict


MIN_FLOAT = -3.14e100

# State order must match jieba-rs: B=0, E=1, M=2, S=3
STATE_ORDER = ['B', 'E', 'M', 'S']
STATE_IDX = {s: i for i, s in enumerate(STATE_ORDER)}

ALLOWED_PREV = {
    'B': ['E', 'S'],
    'E': ['B', 'M'],
    'M': ['M', 'B'],
    'S': ['S', 'E'],
}

RE_HAN = re.compile(r'([\u4E00-\u9FD5]+)')
RE_SKIP = re.compile(r'([a-zA-Z0-9]+(?:.\d+)?%?)')


def word_to_bmes(word):
    """Convert a word to BMES tags for each character."""
    chars = list(word)
    n = len(chars)
    if n == 0:
        return []
    if n == 1:
        return [(chars[0], 'S')]
    tags = []
    for i, ch in enumerate(chars):
        if i == 0:
            tags.append((ch, 'B'))
        elif i == n - 1:
            tags.append((ch, 'E'))
        else:
            tags.append((ch, 'M'))
    return tags


def train_hmm(corpus_files):
    """
    Train HMM parameters from word-segmented corpus files.

    Returns (start_prob, trans_prob, emit_prob).
    """
    start_count = defaultdict(int)
    trans_count = defaultdict(lambda: defaultdict(int))
    emit_count = defaultdict(lambda: defaultdict(int))

    total_sentences = 0

    for corpus_file in corpus_files:
        print(f"Processing {corpus_file}...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                words = line.split()
                if not words:
                    continue

                tagged = []
                for word in words:
                    word = word.strip()
                    if not word:
                        continue
                    tagged.extend(word_to_bmes(word))

                if not tagged:
                    continue

                total_sentences += 1
                start_count[tagged[0][1]] += 1

                prev_tag = None
                for char, tag in tagged:
                    emit_count[tag][char] += 1
                    if prev_tag is not None:
                        trans_count[prev_tag][tag] += 1
                    prev_tag = tag

    print(f"\nTotal sentences: {total_sentences}")

    # Compute log probabilities
    start_prob = {}
    total_start = sum(start_count.values())
    for state in STATE_ORDER:
        count = start_count.get(state, 0)
        start_prob[state] = math.log(count / total_start) if count > 0 else MIN_FLOAT

    trans_prob = {}
    for from_state in STATE_ORDER:
        trans_prob[from_state] = {}
        total_from = sum(trans_count[from_state].values())
        for to_state in STATE_ORDER:
            count = trans_count[from_state].get(to_state, 0)
            if count > 0 and total_from > 0:
                trans_prob[from_state][to_state] = math.log(count / total_from)
            else:
                trans_prob[from_state][to_state] = MIN_FLOAT

    emit_prob = {}
    for state in STATE_ORDER:
        emit_prob[state] = {}
        total_emit = sum(emit_count[state].values())
        for char, count in emit_count[state].items():
            emit_prob[state][char] = math.log(count / total_emit)

    for state in STATE_ORDER:
        print(f"  Emit {state}: {len(emit_prob[state])} unique characters")

    return start_prob, trans_prob, emit_prob


def write_hmm_model(start_prob, trans_prob, emit_prob, output_path):
    """Write HMM model in cppjieba/jieba-rs format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('#\n')
        f.write('#0:B\n#1:E\n#2:M\n#3:S\n#\n')
        for state in STATE_ORDER:
            f.write(f'#{state}:{start_prob[state]}\n')
        f.write('#prob_start\n')

        vals = [str(start_prob[s]) for s in STATE_ORDER]
        f.write(' '.join(vals) + '\n')

        f.write('#prob_trans 4x4 matrix\n')
        for from_state in STATE_ORDER:
            vals = [str(trans_prob[from_state][s]) for s in STATE_ORDER]
            f.write(' '.join(vals) + '\n')

        valid_trans = {'B': ['E', 'M'], 'E': ['B', 'S'], 'M': ['E', 'M'], 'S': ['B', 'S']}
        for state in STATE_ORDER:
            f.write(f'#{state}\n')
            parts = [f'{t}:{trans_prob[state][t]}' for t in valid_trans[state]]
            f.write(f'#{",".join(parts)}\n')

        f.write('#prob_emit 4 lines\n')
        for state in STATE_ORDER:
            f.write(f'#{state}\n')
            pairs = [f'{ch}:{prob:.6f}' for ch, prob in sorted(emit_prob[state].items())]
            f.write(','.join(pairs) + '\n')

    print(f"\nModel written to {output_path}")


def load_hmm_model(model_path):
    """Load HMM model from cppjieba/jieba-rs format file."""
    with open(model_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    # Line 0: start probs
    start_vals = lines[0].split()
    start_prob = {STATE_ORDER[i]: float(v) for i, v in enumerate(start_vals)}

    # Lines 1-4: transition matrix
    trans_prob = {}
    for i, from_state in enumerate(STATE_ORDER):
        vals = lines[1 + i].split()
        trans_prob[from_state] = {STATE_ORDER[j]: float(v) for j, v in enumerate(vals)}

    # Lines 5-8: emission probs
    emit_prob = {}
    for i, state in enumerate(STATE_ORDER):
        emit_prob[state] = {}
        for pair in lines[5 + i].split(','):
            char, prob = pair.rsplit(':', 1)
            emit_prob[state][char] = float(prob)

    return start_prob, trans_prob, emit_prob


# ---------------------------------------------------------------------------
# Viterbi decoder (mirrors jieba-rs hmm.rs)
# ---------------------------------------------------------------------------

def viterbi(chars, start_prob, trans_prob, emit_prob):
    """Run Viterbi on a sequence of characters, return list of states."""
    if not chars:
        return []

    n = len(chars)
    V = [{} for _ in range(n)]
    path = {}

    # Init
    for s in STATE_ORDER:
        V[0][s] = start_prob[s] + emit_prob[s].get(chars[0], MIN_FLOAT)
        path[s] = [s]

    # Recurse
    for t in range(1, n):
        newpath = {}
        for s in STATE_ORDER:
            em = emit_prob[s].get(chars[t], MIN_FLOAT)
            best_prob, best_prev = max(
                (V[t - 1][prev] + trans_prob[prev][s] + em, prev)
                for prev in ALLOWED_PREV[s]
            )
            V[t][s] = best_prob
            newpath[s] = path[best_prev] + [s]
        path = newpath

    # Termination: only E or S can end a sentence
    prob_e = V[n - 1].get('E', MIN_FLOAT)
    prob_s = V[n - 1].get('S', MIN_FLOAT)
    final = 'E' if prob_e >= prob_s else 'S'
    return path[final]


def hmm_cut(sentence, start_prob, trans_prob, emit_prob):
    """Segment a sentence using the HMM, same logic as jieba-rs hmm::cut."""
    words = []
    for m in RE_HAN.split(sentence):
        if not m:
            continue
        if RE_HAN.match(m):
            if len(m) == 1:
                words.append(m)
                continue
            chars = list(m)
            states = viterbi(chars, start_prob, trans_prob, emit_prob)
            begin = 0
            for i, (ch, st) in enumerate(zip(chars, states)):
                if st == 'B':
                    begin = i
                elif st == 'E':
                    words.append(''.join(chars[begin:i + 1]))
                elif st == 'S':
                    words.append(ch)
                # M: do nothing
            # Handle trailing B or M (incomplete word at end)
            if states and states[-1] in ('B', 'M'):
                words.append(''.join(chars[begin:]))
        else:
            for sk in RE_SKIP.split(m):
                if sk:
                    words.append(sk)
    return words


# ---------------------------------------------------------------------------
# Evaluation (Precision / Recall / F1)
# ---------------------------------------------------------------------------

def evaluate_model(start_prob, trans_prob, emit_prob, test_file, gold_file):
    """Evaluate the model against a gold-standard segmentation."""
    with open(test_file, 'r', encoding='utf-8') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_lines = [line.strip() for line in f if line.strip()]

    if len(test_lines) != len(gold_lines):
        print(f"WARNING: test ({len(test_lines)} lines) and gold ({len(gold_lines)} lines) differ in length")

    total_correct = 0
    total_pred = 0
    total_gold = 0

    n = min(len(test_lines), len(gold_lines))
    for i in range(n):
        pred_words = hmm_cut(test_lines[i], start_prob, trans_prob, emit_prob)
        gold_words = gold_lines[i].split()

        # Convert to sets of (start_offset, end_offset) spans for comparison
        pred_spans = _words_to_spans(pred_words)
        gold_spans = _words_to_spans(gold_words)

        total_correct += len(pred_spans & gold_spans)
        total_pred += len(pred_spans)
        total_gold += len(gold_spans)

    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def _words_to_spans(words):
    """Convert a list of words to a set of (start, end) character offset spans."""
    spans = set()
    offset = 0
    for w in words:
        w = w.strip()
        if not w:
            continue
        spans.add((offset, offset + len(w)))
        offset += len(w)
    return spans


def convert_conllu_to_corpus(conllu_path, output_path):
    """Convert a CoNLL-U file to space-separated word corpus (one sentence per line)."""
    sentences = []
    current = []
    skip_until = -1
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                continue
            if not line:
                if current:
                    sentences.append(' '.join(current))
                    current = []
                skip_until = -1
                continue
            fields = line.split('\t')
            token_id = fields[0]
            if '-' in token_id:
                # Multi-word token: use this surface form, skip sub-tokens
                start, end = token_id.split('-')
                skip_until = int(end)
                current.append(fields[1])
            elif '.' in token_id:
                # Empty node, skip
                continue
            else:
                idx = int(token_id)
                if idx <= skip_until:
                    continue
                current.append(fields[1])
    if current:
        sentences.append(' '.join(current))

    with open(output_path, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
    print(f"Converted {conllu_path} -> {output_path} ({len(sentences)} sentences)")
    return output_path


def download_icwb2():
    """Download SIGHAN Bakeoff 2005 data and return the repo directory."""
    cache_dir = os.path.join(tempfile.gettempdir(), 'jieba_rs_icwb2')
    repo_dir = os.path.join(cache_dir, 'icwb2-data')

    if not os.path.exists(repo_dir):
        print("Downloading icwb2-data from GitHub...")
        os.makedirs(cache_dir, exist_ok=True)
        subprocess.run(
            ['git', 'clone', '--depth=1', 'https://github.com/yuikns/icwb2-data.git', repo_dir],
            check=True,
        )
    else:
        print(f"Using cached icwb2-data at {repo_dir}")

    return repo_dir


def download_ud_gsdsimp():
    """Download UD Chinese-GSDSimp and convert training data to corpus format."""
    cache_dir = os.path.join(tempfile.gettempdir(), 'jieba_rs_icwb2')
    repo_dir = os.path.join(cache_dir, 'UD_Chinese-GSDSimp')

    if not os.path.exists(repo_dir):
        print("Downloading UD_Chinese-GSDSimp from GitHub...")
        os.makedirs(cache_dir, exist_ok=True)
        subprocess.run(
            ['git', 'clone', '--depth=1',
             'https://github.com/UniversalDependencies/UD_Chinese-GSDSimp.git', repo_dir],
            check=True,
        )
    else:
        print(f"Using cached UD_Chinese-GSDSimp at {repo_dir}")

    conllu_path = os.path.join(repo_dir, 'zh_gsdsimp-ud-train.conllu')
    corpus_path = os.path.join(cache_dir, 'ud_gsdsimp_train.utf8')
    if not os.path.exists(corpus_path):
        convert_conllu_to_corpus(conllu_path, corpus_path)
    else:
        print(f"Using cached converted corpus at {corpus_path}")
    return corpus_path


def get_training_files(repo_dir):
    """Return default MSR+PKU training file paths."""
    training_dir = os.path.join(repo_dir, 'training')
    files = [
        os.path.join(training_dir, 'msr_training.utf8'),
        os.path.join(training_dir, 'pku_training.utf8'),
    ]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Expected corpus file not found: {f}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate HMM model for jieba-rs'
    )
    parser.add_argument(
        '--corpus', nargs='+',
        help='Path(s) to corpus files (space-separated words, one sentence per line)'
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Download and use SIGHAN Bakeoff 2005 MSR+PKU corpora'
    )
    parser.add_argument(
        '--output', default='hmm.model',
        help='Output path for the trained HMM model (default: hmm.model)'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='Evaluate the model against SIGHAN Bakeoff 2005 test sets'
    )
    parser.add_argument(
        '--model',
        help='Load an existing model for evaluation (skip training)'
    )
    args = parser.parse_args()

    if not args.corpus and not args.download and not args.model:
        parser.error('Specify --corpus, --download, or --model')

    repo_dir = None
    if args.download or (args.eval and not args.model and not args.corpus):
        repo_dir = download_icwb2()

    # Train or load
    if args.model:
        print(f"Loading model from {args.model}...")
        start_prob, trans_prob, emit_prob = load_hmm_model(args.model)
        print("Model loaded.")
    else:
        corpus_files = []
        if args.download:
            corpus_files.extend(get_training_files(repo_dir))
        if args.corpus:
            corpus_files.extend(args.corpus)

        if not corpus_files:
            parser.error('No corpus files specified')

        print(f"Training from {len(corpus_files)} corpus file(s)...")
        start_prob, trans_prob, emit_prob = train_hmm(corpus_files)
        write_hmm_model(start_prob, trans_prob, emit_prob, args.output)

    # Evaluate
    if args.eval:
        if not repo_dir:
            repo_dir = download_icwb2()

        datasets = [
            ('MSR', 'msr_test.utf8', 'msr_test_gold.utf8'),
            ('PKU', 'pku_test.utf8', 'pku_test_gold.utf8'),
        ]

        print("\n" + "=" * 60)
        print("Evaluation Results (HMM-only, no dictionary)")
        print("=" * 60)
        print(f"{'Dataset':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 42)

        for name, test_name, gold_name in datasets:
            test_file = os.path.join(repo_dir, 'testing', test_name)
            gold_file = os.path.join(repo_dir, 'gold', gold_name)

            if not os.path.exists(test_file) or not os.path.exists(gold_file):
                print(f"{name:<10} SKIPPED (files not found)")
                continue

            p, r, f1 = evaluate_model(start_prob, trans_prob, emit_prob, test_file, gold_file)
            print(f"{name:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

        print("-" * 42)
        print("Note: HMM-only scores are expected to be lower than")
        print("full jieba (which combines dictionary + HMM for OOV).")


if __name__ == '__main__':
    main()
