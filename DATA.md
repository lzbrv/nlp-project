# Data

## Getting the data

1. Download `leandojo_benchmark_4_v7.tar.gz` from Zenodo: https://zenodo.org/records/10811294
2. Put it in the repo root and unpack it: `tar -xzf leandojo_benchmark_4_v7.tar.gz`
3. Run `python build_data.py` (about 30 seconds). It writes everything into `processed/`.

The raw data and `processed/` are in `.gitignore` (too large for git). Regenerate them locally with the steps above.

## Files in `processed/`

### `premises.jsonl` — one line per premise (158,266 total)

| field | meaning |
|---|---|
| `id` | **Join key used everywhere.** The Lean full name, e.g. `Nat.add_comm` |
| `full_name` | Lean full name (same as `id` except for the 46 duplicates below) |
| `statement` | Lean source code of the declaration (what the encoder embeds) |
| `file_path` | File the premise is defined in (use this for in-file negatives) |
| `start`, `end` | `[line, col]` span of the declaration in its file |

### `{random,novel_premises}/{train,val,test}.jsonl` — one line per tactic

| field | meaning |
|---|---|
| `state` | Proof state before the tactic (the query) |
| `positive_premise_ids` | `id`s of premises the tactic used (deduplicated, at least one) |
| `file_path` | File of the theorem being proved |
| `theorem` | Full name of the theorem being proved |
| `theorem_start` | `[line, col]` where the theorem starts (for accessibility) |

### `sample_1000.jsonl`

First 1,000 lines of `random/train.jsonl`, for building and debugging the training loop.

### `stats.json`

All preprocessing counts (for the Data section of the writeup).

## Preprocessing decisions

- **Premise matching follows ReProver** (`common.py`: `get_all_pos_premises` / `Corpus.locate_premise`): each used premise is matched to the corpus declaration in its `def_path` whose span contains its `def_pos`. So a use of an auto-generated declaration maps to its containing declaration (e.g. `Prod.fst` → `Prod`).
- **Unlocatable premises are dropped** (~3% of used premises in every split), as in ReProver. Examples: `zero_lt_one`, `one_ne_zero`. A tactic left with no located premise is dropped.
- **Tactics that use no premise are dropped** (~35%), as in the LeanDojo paper.
- **46 full names occur more than once** in the corpus (all Lean-internal metaprogramming, e.g. `Lean.Meta.preprocess`). Their `id` is `name@file_path:line:col` so every `id` is unique.

## Example counts

| split | train | val | test |
|---|---|---|---|
| random | 135,415 | 2,458 | 2,393 |
| novel_premises | 131,957 | 4,174 | 4,135 |

## Not built yet

- Accessible-premise candidate pools (needed for evaluation). Until then, sample training negatives from all premises.
- Shared metrics function (R@1, R@10, MRR) and the BM25 baseline.
