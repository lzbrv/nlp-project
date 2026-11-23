"""
Shared retrieval metrics: Recall@k and MRR (as in LeanDojo/ReProver).

Both BM25 and the DPR models use this, so their numbers are comparable.

Input for one proof state:
  scores        1-D numpy array, one score per candidate premise (higher = better)
  positive_idx  indices into `scores` of the premises the tactic actually used

To restrict ranking to a candidate pool (e.g. accessible premises later),
set the scores of excluded premises to -inf before calling.

Ties are counted pessimistically: a premise with the same score as a positive
is treated as ranked above it.
"""

import numpy as np


def positive_ranks(scores, positive_idx):
    """1-based rank of each positive premise."""
    pos_scores = scores[positive_idx]
    # rank of positive p = number of premises scoring >= p's score (includes p itself)
    return np.array([(scores >= s).sum() for s in pos_scores])


def example_metrics(scores, positive_idx, ks=(1, 10)):
    """Metrics for one proof state."""
    ranks = positive_ranks(scores, np.asarray(positive_idx))
    out = {f"R@{k}": float((ranks <= k).mean()) for k in ks}   # fraction of positives in top k
    out["MRR"] = float(1.0 / ranks.min())                       # reciprocal rank of first positive
    return out


def average_metrics(per_example):
    """Average a list of per-example metric dicts."""
    keys = per_example[0].keys()
    return {k: float(np.mean([m[k] for m in per_example])) for k in keys}
