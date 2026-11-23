"""Rank every accessible premise for each test step; report R@1, R@10, MRR.

Usage:
    python -m retrieval.evaluate --run-dir runs/random__in_file__seed0 --part test
    python -m retrieval.evaluate --checkpoint <dir> --data-root ... --split ... --out <dir>

Writes `<out>/per_example.jsonl` (one line per step, for paired comparisons and
error analysis) and `<out>/metrics.json` (means with bootstrap 95% CIs).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
import torch

from .data import AccessibleIndex, Corpus, Example, load_examples, split_path, write_jsonl
from .metrics import bootstrap_mean_ci, recall_at_k, reciprocal_rank

Scorer = Callable[[Sequence[Example]], "torch.Tensor | np.ndarray"]


class DenseScorer:
    """Cosine scores of states against precomputed premise embeddings."""

    def __init__(self, model, premise_emb: torch.Tensor):
        self.model = model
        self.premise_emb = premise_emb

    @torch.no_grad()
    def __call__(self, batch: Sequence[Example]) -> torch.Tensor:
        q = self.model.encode_many([ex.state for ex in batch], batch_size=len(batch),
                                   dtype=self.premise_emb.dtype)
        return (q @ self.premise_emb.T).float()


def rank_one(scores: torch.Tensor, ex: Example, pool: Optional[np.ndarray],
             corpus: Corpus, k_save: int, train_positive_ids: Optional[Set[int]]) -> dict:
    s = scores
    if pool is not None:
        masked = torch.full_like(s, float("-inf"))
        idx = torch.as_tensor(pool, device=s.device)
        masked[idx] = s[idx]
        s = masked
    pos = torch.as_tensor(ex.positive_ids, device=s.device)
    pos_scores = s[pos]
    accessible = torch.isfinite(pos_scores)
    # 1 + number of candidates scored strictly higher (ties resolved optimistically).
    ranks = (s.unsqueeze(0) > pos_scores.unsqueeze(1)).sum(1) + 1
    pos_ranks = [float(r) if ok else math.inf for r, ok in zip(ranks.tolist(), accessible.tolist())]

    n_cand = len(s) if pool is None else len(pool)
    top = torch.topk(s, min(k_save, n_cand)).indices.tolist()
    pos_set = set(ex.positive_ids)
    top_wrong = next((p for p in top if p not in pos_set), None)
    pos_paths = {corpus.paths[p] for p in ex.positive_ids}

    rec = {
        "id": ex.id,
        "theorem": ex.theorem,
        "n_pos": len(ex.positive_ids),
        "n_candidates": n_cand,
        # premise ids are build_data.py ids, i.e. Lean full names
        "positive_ids": [corpus.name(p) for p in ex.positive_ids],
        "pos_ranks": [None if math.isinf(r) else int(r) for r in pos_ranks],
        "n_pos_inaccessible": int((~accessible).sum()),
        "recall@1": recall_at_k(pos_ranks, 1),
        "recall@10": recall_at_k(pos_ranks, 10),
        "mrr": reciprocal_rank(pos_ranks),
        "top_ids": [corpus.name(p) for p in top],
        "top_wrong_id": None if top_wrong is None else corpus.name(top_wrong),
        "top_wrong_same_file": top_wrong is not None and corpus.paths[top_wrong] in pos_paths,
        "top1_correct": bool(top) and top[0] in pos_set,
    }
    if train_positive_ids is not None:
        unseen = [r for p, r in zip(ex.positive_ids, pos_ranks) if p not in train_positive_ids]
        seen = [r for p, r in zip(ex.positive_ids, pos_ranks) if p in train_positive_ids]
        rec["n_unseen_pos"] = len(unseen)
        rec["unseen_recall@10"] = recall_at_k(unseen, 10) if unseen else None
        rec["seen_recall@10"] = recall_at_k(seen, 10) if seen else None
    return rec


def evaluate_with_scorer(examples: Sequence[Example], scorer: Scorer, corpus: Corpus,
                         accessible: AccessibleIndex, batch_size: int = 64, k_save: int = 100,
                         train_positive_ids: Optional[Set[int]] = None,
                         device: Optional[torch.device] = None,
                         progress: bool = False) -> List[dict]:
    """Generic evaluator: works for the dense model, BM25, or any other scorer
    returning a [B, len(corpus)] score matrix (higher = better)."""
    out = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        scores = scorer(batch)
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        if device is not None:
            scores = scores.to(device)
        scores = scores.float()
        for i, ex in enumerate(batch):
            out.append(rank_one(scores[i], ex, accessible.get(ex), corpus, k_save, train_positive_ids))
        if progress and (start // batch_size) % 20 == 0:
            print(f"  ranked {start + len(batch)}/{len(examples)}", flush=True)
    return out


def summarize(per_example: List[dict], n_boot: int = 1000, seed: int = 0) -> Dict:
    res: Dict = {"n_examples": len(per_example)}
    for m in ("recall@1", "recall@10", "mrr"):
        res[m] = bootstrap_mean_ci([r[m] for r in per_example], n_boot=n_boot, seed=seed)
    wrong = [r for r in per_example if not r["top1_correct"] and r["top_wrong_id"] is not None]
    res["top1_wrong_same_file_rate"] = (
        sum(r["top_wrong_same_file"] for r in wrong) / len(wrong) if wrong else None)
    res["n_top1_wrong"] = len(wrong)
    res["n_pos_inaccessible"] = sum(r["n_pos_inaccessible"] for r in per_example)
    for m in ("unseen_recall@10", "seen_recall@10"):
        vals = [r[m] for r in per_example if r.get(m) is not None]
        if vals:
            res[m] = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
            res[m]["n_examples"] = len(vals)
    return res


def write_results(out_dir, per_example: List[dict], metrics: Dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_example.jsonl", per_example)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def train_positive_set(data_root, split: str, corpus: Corpus) -> Set[int]:
    exs, _ = load_examples(split_path(data_root, split, "train"), corpus)
    return {p for ex in exs for p in ex.positive_ids}


def evaluate_dense(model, corpus: Corpus, examples: Sequence[Example], accessible: AccessibleIndex,
                   batch_size: int = 64, premise_batch_size: int = 128, k_save: int = 100,
                   train_positive_ids: Optional[Set[int]] = None,
                   premise_emb: Optional[torch.Tensor] = None, progress: bool = False):
    """Returns (per_example, premise_emb) so callers can cache the embeddings."""
    if premise_emb is None:
        if progress:
            print(f"Encoding {len(corpus)} premises...", flush=True)
        premise_emb = model.encode_many([p.text() for p in corpus.premises],
                                        batch_size=premise_batch_size, progress=progress)
    per_ex = evaluate_with_scorer(examples, DenseScorer(model, premise_emb), corpus, accessible,
                                  batch_size=batch_size, k_save=k_save,
                                  train_positive_ids=train_positive_ids, progress=progress)
    return per_ex, premise_emb


def main(argv=None):
    from .config import load_config
    from .model import DualEncoder

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="training run directory (uses its config and best checkpoint)")
    ap.add_argument("--checkpoint", help="model directory (overrides the run's best checkpoint)")
    ap.add_argument("--data-root")
    ap.add_argument("--split", help="random | novel_premises (defaults to the run's split)")
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="output directory (default: <run-dir>/eval_<split>_<part>)")
    ap.add_argument("--no-seen-unseen", action="store_true",
                    help="skip the seen/unseen-premise breakdown (needs the train file)")
    args = ap.parse_args(argv)

    cfg = load_config(str(Path(args.run_dir) / "config.yaml")) if args.run_dir else load_config()
    data_root = args.data_root or cfg.data.root
    split = args.split or cfg.data.split
    ckpt = args.checkpoint or (Path(args.run_dir) / "best" if args.run_dir else None)
    if ckpt is None:
        ap.error("give --run-dir or --checkpoint")
    out = args.out or (Path(args.run_dir) / f"eval_{split}_{args.part}" if args.run_dir else None)
    if out is None:
        ap.error("give --out when not using --run-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoder.from_pretrained(str(ckpt), cfg.model.max_len).to(device)
    corpus = Corpus.load(data_root)
    accessible = AccessibleIndex.load(data_root, corpus)
    examples, stats = load_examples(split_path(data_root, split, args.part), corpus, args.limit)
    print(f"{split}/{args.part}: {stats}", flush=True)
    train_pos = None if args.no_seen_unseen else train_positive_set(data_root, split, corpus)

    per_ex, _ = evaluate_dense(model, corpus, examples, accessible, cfg.eval.batch_size,
                               cfg.eval.premise_batch_size, cfg.eval.k_save, train_pos, progress=True)
    metrics = summarize(per_ex, cfg.eval.n_boot, cfg.seed)
    metrics.update({"checkpoint": str(ckpt), "split": split, "part": args.part, "data_stats": stats})
    write_results(out, per_ex, metrics)
    print(json.dumps({k: metrics[k] for k in ("recall@1", "recall@10", "mrr")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
