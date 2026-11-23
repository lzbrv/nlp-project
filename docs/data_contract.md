# Data contract: `build_data.py` → `retrieval/`

The file formats are defined by **[`DATA.md`](../DATA.md)** (Leo's
`build_data.py`). `retrieval/` reads them as-is from `processed/` — nothing to
convert. This page records exactly what the retriever relies on, and proposes
the format for the one piece that is not built yet: accessible-premise pools.

## What `retrieval/` reads

| file | fields used |
|---|---|
| `processed/premises.jsonl` | `id` (must be unique), `full_name`, `statement` (encoder input; prefixed with `full_name :` if the name is not in it), `file_path` (in-file negatives) |
| `processed/{random,novel_premises}/{train,val,test}.jsonl` | `state`, `positive_premise_ids`, `file_path`, `theorem` |

- **Premise ids are the Lean full names** (`id` from `premises.jsonl`;
  for the 46 duplicated names that is `name@file:line:col`). All outputs
  (`per_example.jsonl`: `positive_ids`, `top_ids`, `top_wrong_id`; disagreement
  dumps) use these ids. Only inside the code is a premise also addressed by its
  line number, so embeddings and candidate pools can be integer arrays.
- Each example gets the id `"<part>:<line number>"` (e.g. `test:17`), which
  `per_example.jsonl` and the paired analyses use to align models. This relies
  on `build_data.py` writing lines in a deterministic order (it does). If an
  example line ever carries an `id` field, that is used instead.
- An unknown premise id in a split file is a hard error (it means the files
  came from different builds).

## Proposed: accessible-premise pools (not built yet)

Until these exist, every premise is a candidate for both negative sampling
and ranking (the proposal's fallback and DATA.md's instruction). To switch
to scope-correct pools, the retriever accepts **either** of:

1. **Per example** — add `"accessible_ids": [premise id, ...]` to a split line.
   Simple, but lists are large (thousands of ids per tactic).
2. **Shared by key** (recommended) — add `"accessible_key": "<key>"` to each
   split line and write `processed/accessible.jsonl` with lines
   `{"key": "<key>", "ids": [premise id, ...]}`.

A natural key is `"{file_path}:{theorem_start[0]}:{theorem_start[1]}"`: the
pool is every premise in the transitive imports of `file_path` (from the raw
`corpus.jsonl` `imports`) plus the premises in `file_path` whose `end` comes
before `theorem_start`. All tactics of one theorem share a key, so the file
stays small. Ids are the same string ids as in `premises.jsonl`.

Nothing in `retrieval/` needs to change when this lands: if
`accessible.jsonl` exists it is loaded automatically.

## Plugging in BM25 (or any other scorer)

`retrieval.evaluate.evaluate_with_scorer(examples, scorer, corpus, accessible)`
takes any callable `scorer(batch_of_examples) -> array[B, len(corpus)]` of
scores over all premises in `premises.jsonl` line order (higher = better;
non-accessible entries are masked for you). It returns the same per-example
records as the dense model; save them with `retrieval.evaluate.write_results`
and `summarize`, and `python -m retrieval.analysis compare` then works across
BM25 and DPR runs. Sketch:

```python
from rank_bm25 import BM25Okapi
from retrieval.data import Corpus, AccessibleIndex, load_examples, split_path
from retrieval.evaluate import evaluate_with_scorer, summarize, write_results

corpus = Corpus.load("processed")
bm25 = BM25Okapi([tokenize(p.text()) for p in corpus.premises])
exs, _ = load_examples(split_path("processed", "random", "test"), corpus)
per_ex = evaluate_with_scorer(
    exs, lambda b: np.stack([bm25.get_scores(tokenize(e.state)) for e in b]),
    corpus, AccessibleIndex.load("processed", corpus))
write_results("runs/bm25/eval_random_test", per_ex, summarize(per_ex))
```
