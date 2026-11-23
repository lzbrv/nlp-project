"""Compare evaluation outputs: paired gaps, diff-in-diff, disagreements, tables.

    # in-file (A) vs random (B) on one split, paired over test steps
    python -m retrieval.analysis compare runs/random__in_file__seed0/eval_random_test \
        runs/random__random__seed0/eval_random_test --data-root data/... --split random

    # is the in-file advantage larger on novel_premises than on random?
    python -m retrieval.analysis did \
        --random-a .../eval_random_test --random-b .../eval_random_test \
        --novel-a  .../eval_novel_premises_test --novel-b .../eval_novel_premises_test

    # results table over every run (mean ± sd across seeds)
    python -m retrieval.analysis table runs/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import Corpus, load_examples, read_jsonl, split_path, write_jsonl
from .metrics import diff_in_diff, paired_bootstrap_diff

METRICS = ("recall@1", "recall@10", "mrr")


def load_per_example(eval_dir) -> Dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(Path(eval_dir) / "per_example.jsonl")}


def align(a: Dict[str, dict], b: Dict[str, dict]) -> List[str]:
    if a.keys() != b.keys():
        raise ValueError(f"evaluations cover different steps ({len(a)} vs {len(b)}); "
                         "were they run on the same split/limit?")
    return sorted(a)


def compare(a_dir, b_dir, n_boot=1000, seed=0) -> dict:
    a, b = load_per_example(a_dir), load_per_example(b_dir)
    ids = align(a, b)
    out = {"n_examples": len(ids)}
    for m in METRICS:
        out[f"{m} (a-b)"] = paired_bootstrap_diff([a[i][m] for i in ids], [b[i][m] for i in ids],
                                                  n_boot=n_boot, seed=seed)
    for name, d in (("a", a), ("b", b)):
        wrong = [d[i] for i in ids if not d[i]["top1_correct"] and d[i]["top_wrong_id"] is not None]
        out[f"top1_wrong_same_file_rate_{name}"] = (
            sum(r["top_wrong_same_file"] for r in wrong) / len(wrong) if wrong else None)
    return out


def disagreements(a_dir, b_dir, data_root, split, part="test", k=10) -> List[dict]:
    """Steps where exactly one model gets any positive into its top k."""
    a, b = load_per_example(a_dir), load_per_example(b_dir)
    ids = align(a, b)
    corpus = Corpus.load(Path(data_root) / "corpus.jsonl")
    exs = {e.id: e for e in load_examples(split_path(data_root, split, part), len(corpus))[0]}

    def hit(r):
        return any(x is not None and x <= k for x in r["pos_ranks"])

    rows = []
    for i in ids:
        ha, hb = hit(a[i]), hit(b[i])
        if ha == hb:
            continue
        ex = exs.get(i)
        rows.append({
            "id": i, "winner": "a" if ha else "b",
            "state": ex.state if ex else None,
            "positives": [corpus.name(p) for p in a[i]["positive_ids"]],
            "pos_ranks_a": a[i]["pos_ranks"], "pos_ranks_b": b[i]["pos_ranks"],
            "top5_a": [corpus.name(p) for p in a[i]["top_ids"][:5]],
            "top5_b": [corpus.name(p) for p in b[i]["top_ids"][:5]],
            "top_wrong_same_file_a": a[i]["top_wrong_same_file"],
            "top_wrong_same_file_b": b[i]["top_wrong_same_file"],
        })
    return rows


def did(ra, rb, na, nb, n_boot=1000, seed=0) -> dict:
    RA, RB, NA, NB = (load_per_example(x) for x in (ra, rb, na, nb))
    r_ids, n_ids = align(RA, RB), align(NA, NB)
    return {m: diff_in_diff([RA[i][m] for i in r_ids], [RB[i][m] for i in r_ids],
                            [NA[i][m] for i in n_ids], [NB[i][m] for i in n_ids],
                            n_boot=n_boot, seed=seed) for m in METRICS}


def table(runs_root) -> str:
    """Markdown table: rows = (eval split, strategy), mean ± sd over seeds."""
    import yaml
    groups = defaultdict(list)
    for mpath in sorted(Path(runs_root).glob("*/eval_*/metrics.json")):
        run = mpath.parent.parent
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        m = json.loads(mpath.read_text())
        key = (cfg["data"]["split"], m.get("split"), m.get("part"), cfg["negatives"]["strategy"])
        groups[key].append(m)
    lines = ["| train split | eval split | part | negatives | seeds | R@1 | R@10 | MRR | unseen R@10 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, ms in sorted(groups.items()):
        def cell(name):
            vals = [m[name]["mean"] for m in ms if name in m]
            if not vals:
                return "–"
            return f"{np.mean(vals):.3f}" + (f" ± {np.std(vals, ddof=1):.3f}" if len(vals) > 1 else "")
        lines.append(f"| {' | '.join(map(str, key))} | {len(ms)} | {cell('recall@1')} | "
                     f"{cell('recall@10')} | {cell('mrr')} | {cell('unseen_recall@10')} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="paired comparison of two eval dirs (a - b)")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--data-root", help="also dump disagreements (needs data)")
    c.add_argument("--split")
    c.add_argument("--part", default="test")
    c.add_argument("--out", help="where to write disagreements.jsonl (default: <a>/vs_<b name>)")
    c.add_argument("--n-boot", type=int, default=1000)
    d = sub.add_parser("did", help="difference-in-differences across splits")
    for n in ("random-a", "random-b", "novel-a", "novel-b"):
        d.add_argument(f"--{n}", required=True)
    d.add_argument("--n-boot", type=int, default=1000)
    t = sub.add_parser("table", help="markdown table over all runs")
    t.add_argument("runs_root")
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        print(json.dumps(compare(args.a, args.b, args.n_boot), indent=2))
        if args.data_root:
            if not args.split:
                ap.error("--split is required with --data-root")
            rows = disagreements(args.a, args.b, args.data_root, args.split, args.part)
            out = Path(args.out or Path(args.a) / f"vs_{Path(args.b).parent.name}")
            out.mkdir(parents=True, exist_ok=True)
            write_jsonl(out / "disagreements.jsonl", rows)
            wins = sum(r["winner"] == "a" for r in rows)
            print(f"{len(rows)} disagreements (a wins {wins}, b wins {len(rows) - wins}) -> {out}")
    elif args.cmd == "did":
        print(json.dumps(did(args.random_a, args.random_b, args.novel_a, args.novel_b,
                             args.n_boot), indent=2))
    else:
        print(table(args.runs_root))


if __name__ == "__main__":
    main()
