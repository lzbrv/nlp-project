"""Losses over a batch similarity matrix.

Rows are proof states; columns are every premise in the batch (each row's
positive plus all explicit negatives), so other rows' premises act as in-batch
negatives. A column is a positive for row i if it is *any* premise used by
step i (not only the one sampled positive), matching ReProver's labels.
"""

from __future__ import annotations

from typing import List, Sequence, Set

import torch
import torch.nn.functional as F


def label_matrix(positive_sets: Sequence[Set[int]], column_ids: List[int],
                 device=None) -> torch.Tensor:
    labels = torch.zeros(len(positive_sets), len(column_ids), device=device)
    for i, pos in enumerate(positive_sets):
        for j, pid in enumerate(column_ids):
            if pid in pos:
                labels[i, j] = 1.0
    return labels


def retrieval_loss(sim: torch.Tensor, labels: torch.Tensor, kind: str = "mse",
                   temperature: float = 0.05) -> torch.Tensor:
    """`sim` is cosine similarity in [-1, 1].

    * ``mse``: ReProver's objective, regress similarity onto the 0/1 labels.
    * ``infonce``: multi-positive contrastive cross-entropy with temperature.
    """
    if kind == "mse":
        return F.mse_loss(sim, labels)
    if kind == "infonce":
        logits = sim / temperature
        log_denom = torch.logsumexp(logits, dim=1)
        log_num = torch.logsumexp(logits.masked_fill(labels == 0, float("-inf")), dim=1)
        return (log_denom - log_num).mean()
    raise ValueError(f"unknown loss {kind!r}")
