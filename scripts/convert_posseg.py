#!/usr/bin/env python3
"""Convert Python jieba posseg HMM pickle data to text format for Rust embedding."""
import os, pickle, sys

import jieba
base = os.path.dirname(jieba.__file__)
posseg_dir = os.path.join(base, 'posseg')

with open(os.path.join(posseg_dir, 'prob_start.p'), 'rb') as f:
    prob_start = pickle.load(f)
with open(os.path.join(posseg_dir, 'prob_trans.p'), 'rb') as f:
    prob_trans = pickle.load(f)
with open(os.path.join(posseg_dir, 'prob_emit.p'), 'rb') as f:
    prob_emit = pickle.load(f)
with open(os.path.join(posseg_dir, 'char_state_tab.p'), 'rb') as f:
    char_state_tab = pickle.load(f)

# Collect all POS tags
all_tags = set()
for (pos, tag) in prob_start:
    all_tags.add(tag)
all_tags = sorted(all_tags)

pos_map = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
tag_map = {t: i for i, t in enumerate(all_tags)}

print(f"# {len(all_tags)} POS tags, 4 position tags (B=0 M=1 E=2 S=3)", file=sys.stderr)

out = sys.stdout

out.write("@TAGS\n")
out.write(",".join(all_tags) + "\n")

out.write("@START\n")
for (pos, tag), prob in sorted(prob_start.items(), key=lambda x: (pos_map[x[0][0]], tag_map[x[0][1]])):
    if prob < -1e90:
        continue
    out.write(f"{pos_map[pos]},{tag_map[tag]},{prob}\n")

out.write("@TRANS\n")
for (from_pos, from_tag), targets in sorted(prob_trans.items(), key=lambda x: (pos_map[x[0][0]], tag_map[x[0][1]])):
    if not targets:
        continue
    from_idx = f"{pos_map[from_pos]},{tag_map[from_tag]}"
    parts = []
    for (to_pos, to_tag), prob in sorted(targets.items(), key=lambda x: (pos_map[x[0][0]], tag_map[x[0][1]])):
        parts.append(f"{pos_map[to_pos]},{tag_map[to_tag]},{prob}")
    out.write(f"{from_idx}|{'|'.join(parts)}\n")

out.write("@EMIT\n")
for (pos, tag), chars in sorted(prob_emit.items(), key=lambda x: (pos_map[x[0][0]], tag_map[x[0][1]])):
    if not chars:
        continue
    idx = f"{pos_map[pos]},{tag_map[tag]}"
    parts = []
    for ch, prob in sorted(chars.items()):
        parts.append(f"{ch},{prob}")
    out.write(f"{idx}|{'|'.join(parts)}\n")

out.write("@CHAR_STATE\n")
for ch, states in sorted(char_state_tab.items()):
    parts = []
    for (pos, tag) in sorted(states, key=lambda x: (pos_map[x[0]], tag_map[x[1]])):
        parts.append(f"{pos_map[pos]},{tag_map[tag]}")
    out.write(f"{ch}|{'|'.join(parts)}\n")
