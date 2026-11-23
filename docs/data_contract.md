# Data contract: preprocessing → retriever

This is the interface between the **preprocessing / accessible-premise pipeline
(Leo)** and the **DPR training / evaluation code (Joel)**. Anything that writes
these files can be trained on and evaluated by `retrieval/`, and the BM25
baseline can be scored through the same evaluator (see bottom), so all systems
are compared on identical candidate pools and metrics.

Everything is UTF-8 JSON Lines (one JSON object per line).

## Directory layout

```
data/leandojo_benchmark_4_v7/          # = config `data.root`
  corpus.jsonl                         # all premises, shared by both splits
  accessible.jsonl                     # OPTIONAL, see "Accessible premises"
  random/
    train.jsonl  val.jsonl  test.jsonl
  novel_premises/
    train.jsonl  val.jsonl  test.jsonl
  preprocessing_stats.json             # OPTIONAL but encouraged: counts/exclusions
```

## `corpus.jsonl` — one premise per line

| field       | type | notes |
|-------------|------|-------|
| `id`        | int  | **contiguous, 0..N-1**, line order = id order |
| `full_name` | str  | e.g. `Nat.add_comm` |
| `path`      | str  | Lean source file the premise is defined in, e.g. `Mathlib/Algebra/Group/Basic.lean`. Used to build **in-file negatives**, so it must be the same string format as `file` in examples. |
| `code`      | str  | the text the encoder sees (statement / serialized premise). If it does not contain `full_name`, the encoder input is `"{full_name} : {code}"`. |

## `{split}/{train,val,test}.jsonl` — one proof step per line

One line per annotated tactic **that uses ≥ 1 premise** (tactics with no
premises have no retrieval target; please drop them and report the count).

| field            | type        | notes |
|------------------|-------------|-------|
| `id`             | str         | unique within the file, stable across runs (e.g. `"{theorem}#{tactic_idx}"`) |
| `theorem`        | str         | full name of the theorem being proved |
| `file`           | str         | Lean file containing the theorem |
| `state`          | str         | pretty-printed proof state before the tactic (the query) |
| `positive_ids`   | list[int]   | corpus ids of premises used by the tactic (de-duplicated) |
| `accessible_ids` | list[int]   | OPTIONAL — explicit candidate pool for this step |
| `accessible_key` | str         | OPTIONAL — key into `accessible.jsonl` (use instead of `accessible_ids` to avoid repeating huge lists) |

## Accessible premises

The candidate pool for an example is resolved in this order:

1. `accessible_ids` on the example, if present;
2. else `accessible.jsonl[accessible_key]`, if the example has a key;
3. else **every premise in the corpus** — this is the "all accessible premises"
   fallback named in the proposal, so the pipeline works end to end before
   scope-specific sets exist.

`accessible.jsonl` lines look like `{"key": "<any string>", "ids": [int, ...]}`.
A natural key is "file + position" (premises imported by the file plus those
defined earlier in it), but the retriever does not care how keys are chosen.

The same pool is used for (a) ranking at evaluation time and (b) drawing
random and in-file negatives during training.

## Plugging in BM25 (or any other scorer)

`retrieval.evaluate.evaluate_with_scorer(examples, scorer, corpus, accessible)`
takes any callable `scorer(batch_of_examples) -> array[B, N]` of scores over the
whole corpus (higher = better; non-accessible entries are masked for you) and
writes the same `per_example.jsonl` / `metrics.json` as the dense model, so
`python -m retrieval.analysis compare` works across BM25 and DPR runs.
