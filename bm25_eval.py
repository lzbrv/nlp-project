"""
BM25 baseline (our own vectorized implementation, Okapi BM25).

Ranks ALL premises in processed/premises.jsonl for every test proof state
(the "all premises" setting; accessible-premise pools are future work) and
reports R@1, R@10, MRR on the random and novel_premises test splits.

Requires: pip install numpy scipy scikit-learn
Run from the repo root after build_data.py:  python bm25_eval.py
Writes results/bm25_all_premises.json
"""

import json
import os
import re
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from metrics import example_metrics, average_metrics

DATA = "processed"
SPLITS = ["random", "novel_premises"]
K1, B = 1.5, 0.75          # standard BM25 parameters (same defaults as rank_bm25)
BATCH = 64                 # queries scored at once


def tokenize(text):
    # Words/identifiers incl. Unicode letters; dotted names split into parts
    # (Nat.add_comm -> Nat, add_comm). Case is kept.
    return re.findall(r"\w+", text)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Index premises
# ---------------------------------------------------------------------------
t0 = time.time()
premises = read_jsonl(f"{DATA}/premises.jsonl")
id_to_idx = {p["id"]: i for i, p in enumerate(premises)}

vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
tf = vectorizer.fit_transform(p["statement"] for p in premises).tocsr().astype(np.float32)
n_docs = tf.shape[0]

doc_len = np.asarray(tf.sum(axis=1)).ravel()
avgdl = doc_len.mean()
df = np.bincount(tf.indices, minlength=tf.shape[1])
idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

# Per (doc, term) BM25 weight: idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
row_of_entry = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
norm = K1 * (1 - B + B * doc_len[row_of_entry] / avgdl)
weights = tf.copy()
weights.data = idf[tf.indices] * tf.data * (K1 + 1) / (tf.data + norm)
print(f"Indexed {n_docs} premises, vocab {tf.shape[1]} ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
results = {}
for split in SPLITS:
    t0 = time.time()
    examples = read_jsonl(f"{DATA}/{split}/test.jsonl")
    per_example = []
    for start in range(0, len(examples), BATCH):
        batch = examples[start:start + BATCH]
        # query term counts; terms not in the premise vocabulary are ignored
        q = vectorizer.transform(ex["state"] for ex in batch).astype(np.float32)
        scores = (weights @ q.T).toarray()            # (n_docs, batch)
        for j, ex in enumerate(batch):
            pos = [id_to_idx[pid] for pid in ex["positive_premise_ids"]]
            per_example.append(example_metrics(scores[:, j], pos))
    results[split] = average_metrics(per_example)
    results[split]["n_examples"] = len(examples)
    m = results[split]
    print(f"{split:15s} R@1 {100*m['R@1']:.1f}  R@10 {100*m['R@10']:.1f}  "
          f"MRR {m['MRR']:.3f}  (n={len(examples)}, {time.time() - t0:.0f}s)")

with open("results/bm25_all_premises.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved results/bm25_all_premises.json")
