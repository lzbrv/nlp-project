"""Ranking metrics and bootstrap confidence intervals."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def recall_at_k(pos_ranks: Sequence[float], k: int) -> float:
    """Fraction of a step's positives ranked in the top k (ranks are 1-based)."""
    if not pos_ranks:
        return 0.0
    return sum(r <= k for r in pos_ranks) / len(pos_ranks)


def reciprocal_rank(pos_ranks: Sequence[float]) -> float:
    """1 / rank of the first correct premise (0 if none is retrievable)."""
    best = min(pos_ranks) if pos_ranks else math.inf
    return 0.0 if math.isinf(best) else 1.0 / best


def _boot_means(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                chunk: int = 100) -> np.ndarray:
    n = len(values)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(m, n))
        out[s:s + m] = values[idx].mean(axis=1)
    return out


def _ci(mean: float, boots: np.ndarray, alpha: float) -> Dict[str, float]:
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(mean), "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_mean_ci(values, n_boot: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return _ci(v.mean(), _boot_means(v, n_boot, np.random.default_rng(seed)), alpha)


def paired_bootstrap_diff(a, b, n_boot: int = 1000, alpha: float = 0.05,
                          seed: int = 0) -> Dict[str, float]:
    """CI for mean(a - b) over the same test examples, plus the bootstrap
    fraction of resamples where the difference is <= 0 (a one-sided p-value
    for "a > b")."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    boots = _boot_means(d, n_boot, np.random.default_rng(seed))
    res = _ci(d.mean(), boots, alpha)
    res["p_le_0"] = float((boots <= 0).mean())
    return res


def diff_in_diff(a1, b1, a2, b2, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """CI for (mean(a2 - b2) - mean(a1 - b1)).

    Pairs are resampled jointly within each split, independently across splits.
    With a = in-file, b = random, split 1 = `random`, split 2 = `novel_premises`,
    a positive value means the in-file advantage is larger on novel premises.
    """
    d1 = np.asarray(a1, dtype=np.float64) - np.asarray(b1, dtype=np.float64)
    d2 = np.asarray(a2, dtype=np.float64) - np.asarray(b2, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = _boot_means(d2, n_boot, rng) - _boot_means(d1, n_boot, rng)
    res = _ci(d2.mean() - d1.mean(), boots, alpha)
    res["p_le_0"] = float((boots <= 0).mean())
    return res
