"""Train the dual-encoder retriever with in-file or random negatives.

Usage:
    python -m retrieval.train --config configs/base.yaml \
        --set data.split=novel_premises negatives.strategy=random seed=1
    python -m retrieval.train --config ... --resume      # continue from <run>/last

Run directory (`<output_dir>/<split>__<strategy>__seed<seed>` by default):
    config.yaml      exact config used
    env.json         git commit, library versions, device
    log.jsonl        training / validation log
    best/            checkpoint with the highest validation Recall@10
    last/            latest checkpoint + trainer_state.pt (for --resume)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .config import Config, load_config
from .data import AccessibleIndex, Corpus, Example, load_examples, split_path
from .evaluate import evaluate_dense
from .losses import label_matrix, retrieval_loss
from .negatives import NegativeSampler


class PairDataset(Dataset):
    """One item per (proof step, used premise) pair, as in ReProver.

    Negatives are re-drawn every epoch from a generator seeded by
    (seed, epoch, index), so runs are reproducible and two strategies with the
    same seed see the same pairs in the same order.
    """

    def __init__(self, examples: List[Example], sampler: NegativeSampler,
                 accessible: AccessibleIndex, seed: int):
        self.examples = examples
        self.sampler = sampler
        self.accessible = accessible
        self.seed = seed
        self.epoch = 0
        self.pairs = [(i, p) for i, ex in enumerate(examples) for p in ex.positive_ids]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ex_idx, pos = self.pairs[i]
        ex = self.examples[ex_idx]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        negs, n_in_file = self.sampler.sample(pos, set(ex.positive_ids), self.accessible.get(ex), rng)
        return {"state": ex.state, "pos": pos, "negs": negs,
                "positive_set": set(ex.positive_ids), "n_in_file": n_in_file}


class Collator:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def __call__(self, items):
        columns = [it["pos"] for it in items] + [n for it in items for n in it["negs"]]
        return {
            "states": [it["state"] for it in items],
            "premises": [self.corpus.text(p) for p in columns],
            "labels": label_matrix([it["positive_set"] for it in items], columns),
            "n_in_file": sum(it["n_in_file"] for it in items),
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def env_info() -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    import transformers
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def log(self, row: dict):
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)


def validate(model, corpus, val_examples, accessible, cfg: Config) -> dict:
    per_ex, _ = evaluate_dense(model, corpus, val_examples, accessible,
                               cfg.eval.batch_size, cfg.eval.premise_batch_size, k_save=10)
    return {k: float(np.mean([r[k] for r in per_ex])) for k in ("recall@1", "recall@10", "mrr")}


def train(cfg: Config, resume: bool = False, max_steps_override: Optional[int] = None) -> Path:
    tc = cfg.train
    total_steps = max_steps_override or tc.max_steps
    run_dir = cfg.run_dir()
    if run_dir.exists() and (run_dir / "last").exists() and not resume:
        raise SystemExit(f"{run_dir} already has a checkpoint; pass --resume or change run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    if not resume:
        (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2))
    set_seed(cfg.seed)
    logger = Logger(run_dir / "log.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .model import DualEncoder
    corpus = Corpus.load(cfg.data.root)
    accessible = AccessibleIndex.load(cfg.data.root, corpus)
    train_ex, train_stats = load_examples(split_path(cfg.data.root, cfg.data.split, "train"),
                                          corpus, cfg.data.max_train_examples)
    val_ex, val_stats = load_examples(split_path(cfg.data.root, cfg.data.split, "val"),
                                      corpus, cfg.data.val_limit)
    logger.log({"event": "data", "corpus": len(corpus), "train": train_stats, "val": val_stats})

    start_from = run_dir / "last" if resume and (run_dir / "last").exists() else cfg.model.name
    model = DualEncoder.from_pretrained(str(start_from), cfg.model.max_len).to(device)
    model.train()

    sampler = NegativeSampler(corpus, cfg.negatives.strategy, cfg.negatives.num_negatives,
                              cfg.negatives.num_in_file)
    dataset = PairDataset(train_ex, sampler, accessible, cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup_steps, total_steps))

    step, best = 0, -1.0
    state_path = run_dir / "last" / "trainer_state.pt"
    if resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, best, dataset.epoch = st["step"], st["best"], st["epoch"] + 1
        logger.log({"event": "resume", "step": step, "best_val_recall@10": best})

    use_bf16 = tc.bf16 and device.type == "cuda"

    def save(name: str, epoch: int):
        model.save(run_dir / name)
        if name == "last":
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "step": step, "best": best, "epoch": epoch}, state_path)

    def maybe_validate(epoch: int):
        nonlocal best
        if not val_ex:
            return
        model.eval()
        v = validate(model, corpus, val_ex, accessible, cfg)
        model.train()
        improved = v["recall@10"] > best
        if improved:
            best = v["recall@10"]
            save("best", epoch)
        logger.log({"event": "val", "step": step, **{f"val_{k}": x for k, x in v.items()},
                    "best": improved})

    accum_loss, accum_n, in_file, n_pairs, t0 = 0.0, 0, 0, 0, time.time()
    micro = 0
    while step < total_steps:
        g = torch.Generator().manual_seed(cfg.seed * 100003 + dataset.epoch)
        loader = DataLoader(dataset, batch_size=tc.batch_size, shuffle=True, generator=g,
                            collate_fn=Collator(corpus), num_workers=tc.num_workers,
                            drop_last=True, persistent_workers=False)
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                q = model.encode(batch["states"])
                p = model.encode(batch["premises"])
            sim = q @ p.T
            loss = retrieval_loss(sim, batch["labels"].to(device), tc.loss, tc.temperature)
            (loss / tc.grad_accum).backward()
            accum_loss += loss.item()
            accum_n += 1
            in_file += batch["n_in_file"]
            n_pairs += len(batch["states"])
            micro += 1
            if micro % tc.grad_accum:
                continue
            if tc.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % tc.log_every == 0 or step == total_steps:
                denom = n_pairs * sampler.num_in_file
                logger.log({"event": "train", "step": step, "epoch": dataset.epoch,
                            "loss": accum_loss / accum_n, "lr": sched.get_last_lr()[0],
                            "in_file_fill_rate": (in_file / denom) if denom else None,
                            "sec_per_step": (time.time() - t0) / tc.log_every})
                accum_loss, accum_n, in_file, n_pairs, t0 = 0.0, 0, 0, 0, time.time()
            if step % tc.eval_every == 0 or step == total_steps:
                maybe_validate(dataset.epoch)
            if step % tc.ckpt_every == 0 or step == total_steps:
                save("last", dataset.epoch)
            if step >= total_steps:
                break
        dataset.epoch += 1

    if not (run_dir / "best").exists():  # no validation data: best = last
        save("best", dataset.epoch)
    logger.log({"event": "done", "step": step, "best_val_recall@10": best})
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. negatives.strategy=random seed=1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run_dir = train(cfg, resume=args.resume)
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
