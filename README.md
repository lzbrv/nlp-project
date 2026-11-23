# Do In-File Negatives Improve Generalization in Lean 4 Premise Retrieval?

COMS 4705 project — Leo Zubarev, Joel Lazcano.

We reimplement the retrieval component of ReProver (a ByT5-small DPR-style dual
encoder) and run a controlled ablation of its negative sampling — **in-file**
negatives vs. **uniformly random** accessible negatives — on LeanDojo Benchmark 4
(v7), evaluating on both the `random` and `novel_premises` splits.

## Who owns what (from the proposal)

| Area | Lead | Where |
|---|---|---|
| LeanDojo preprocessing, accessible-premise sets, BM25 | Leo | `build_data.py` → `processed/`, see [`DATA.md`](DATA.md) |
| DPR training loop, experiment management, error analysis | Joel | `retrieval/`, `configs/`, `scripts/` |

`retrieval/` reads `build_data.py`'s `processed/` files directly. What it relies
on, the proposed format for accessible-premise pools, and how to run BM25
through the shared evaluator are in
**[`docs/data_contract.md`](docs/data_contract.md)**.

## Layout

```
retrieval/
  data.py        load the contract files; resolve accessible-premise pools
  negatives.py   in-file vs random negative sampling  (the ablation variable)
  model.py       shared ByT5 encoder, mean pooling, cosine similarity
  losses.py      ReProver MSE loss (default) and multi-positive InfoNCE
  train.py       training loop: in-batch + 3 explicit negatives, warmup, bf16,
                 val-based checkpoint selection, resumable checkpoints
  evaluate.py    rank all accessible premises: R@1, R@10, MRR + bootstrap CIs,
                 seen/unseen-premise breakdown, same-file error rate
  analysis.py    paired comparison, diff-in-diff across splits,
                 model disagreements, results table across seeds
configs/base.yaml   ReProver-style defaults      configs/smoke.yaml  tiny run
scripts/run_grid.sh full {split} x {negatives} x {seed} grid + evaluation
tests/              offline tests on synthetic data with a tiny random encoder
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # ~5 s on CPU, no downloads
```

Then build the data as described in [`DATA.md`](DATA.md) (download the
LeanDojo tarball, `python build_data.py` → `processed/`).

## Running

```bash
# 1. sanity check on real data (a few minutes on one GPU)
python -m retrieval.train --config configs/smoke.yaml
python -m retrieval.evaluate --run-dir runs_smoke/random__in_file__seed0 --part val --limit 500

# 2. one real run
python -m retrieval.train --config configs/base.yaml --set data.split=novel_premises negatives.strategy=random seed=0
python -m retrieval.evaluate --run-dir runs/novel_premises__random__seed0 --part test

# 3. the whole ablation (resumes interrupted runs, skips finished ones)
SEEDS="0 1 2" scripts/run_grid.sh
```

Each run directory stores `config.yaml`, `env.json` (git commit, versions,
GPU), `log.jsonl` (loss, lr, in-file fill rate, sec/step, validation metrics),
`best/` and `last/` checkpoints, and `eval_<split>_<part>/` with
`metrics.json` and `per_example.jsonl`.

## Analysis

```bash
# in-file (a) minus random (b) on the same test steps, with paired bootstrap CIs,
# plus a dump of steps where exactly one model gets a positive into its top 10
python -m retrieval.analysis compare \
  runs/novel_premises__in_file__seed0/eval_novel_premises_test \
  runs/novel_premises__random__seed0/eval_novel_premises_test \
  --data-root processed --split novel_premises

# main hypothesis: is the in-file gap larger on novel_premises than on random?
python -m retrieval.analysis did \
  --random-a runs/random__in_file__seed0/eval_random_test \
  --random-b runs/random__random__seed0/eval_random_test \
  --novel-a  runs/novel_premises__in_file__seed0/eval_novel_premises_test \
  --novel-b  runs/novel_premises__random__seed0/eval_novel_premises_test

python -m retrieval.analysis table runs      # markdown table, mean ± sd over seeds
```

`metrics.json` also reports `top1_wrong_same_file_rate` (how often the
highest-ranked wrong premise comes from the same file as a target — the failure
mode in-file negatives target) and `unseen_recall@10` / `seen_recall@10`
(recall restricted to premises that never appear as positives in training).

## Implementation notes

- One training item per (proof step, used premise) pair, as in ReProver; a
  batch column counts as positive for a row if it is *any* premise that step uses.
- Negatives are re-drawn each epoch from an RNG seeded by (seed, epoch, index):
  for a given seed both strategies see identical pairs in identical order, so
  the negative source is the only thing that differs.
- If a file has no eligible in-file negative, the slot is filled uniformly and
  this is logged as `in_file_fill_rate` (report it in the paper).
- Without scope-specific accessible sets, every premise is a candidate (the
  proposal's fallback); this makes the task harder than ReProver's setting.
- `configs/base.yaml` values should be checked against ReProver's released
  retrieval config, and `max_steps` calibrated from `sec_per_step`.
