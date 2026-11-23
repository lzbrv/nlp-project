"""Explicit negative sampling — the independent variable of the ablation.

Both strategies draw `num_negatives` premises per (state, positive) pair from the
example's accessible pool, never returning any of the step's positives:

* ``in_file`` (ReProver default): ``num_in_file`` of them come from the same
  source file as the positive premise; the rest are uniform over the pool.
  If the file has too few eligible premises, the shortfall is filled uniformly
  (so the number of negatives is identical across strategies) and reported via
  the returned in-file count.
* ``random`` (ablation): all are uniform over the pool.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from .data import Corpus

STRATEGIES = ("in_file", "random")


def sample_excluding(pool: Optional[np.ndarray], n_total: int, k: int,
                     exclude: Set[int], rng: np.random.Generator) -> List[int]:
    """Up to k distinct ids from `pool` (None = range(n_total)) not in `exclude`."""
    if k <= 0:
        return []
    size = n_total if pool is None else len(pool)
    if size == 0:
        return []
    picked: List[int] = []
    seen = set(exclude)
    # Rejection sampling: pools are usually huge relative to k and exclude.
    for _ in range(20 * k):
        j = int(rng.integers(size))
        pid = j if pool is None else int(pool[j])
        if pid not in seen:
            picked.append(pid)
            seen.add(pid)
            if len(picked) == k:
                return picked
    # Small or nearly exhausted pool: enumerate what is left.
    universe = np.arange(n_total) if pool is None else pool
    rest = np.setdiff1d(universe, np.fromiter(seen, dtype=np.int64, count=len(seen)))
    need = min(k - len(picked), len(rest))
    if need > 0:
        picked.extend(int(x) for x in rng.choice(rest, size=need, replace=False))
    return picked


class NegativeSampler:
    def __init__(self, corpus: Corpus, strategy: str, num_negatives: int = 3,
                 num_in_file: int = 1):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown negative strategy {strategy!r}; choose from {STRATEGIES}")
        if not 0 <= num_in_file <= num_negatives:
            raise ValueError("need 0 <= num_in_file <= num_negatives")
        self.corpus = corpus
        self.strategy = strategy
        self.num_negatives = num_negatives
        self.num_in_file = num_in_file if strategy == "in_file" else 0

    def sample(self, positive_id: int, all_positive_ids: Set[int],
               pool: Optional[np.ndarray], rng: np.random.Generator) -> Tuple[List[int], int]:
        """Returns (negative ids, how many of them are true in-file negatives)."""
        exclude = set(all_positive_ids)
        negs: List[int] = []
        if self.num_in_file:
            same_file = self.corpus.path_to_ids[self.corpus.paths[positive_id]]
            if pool is not None:
                same_file = np.intersect1d(same_file, pool, assume_unique=True)
            negs = sample_excluding(same_file, len(self.corpus), self.num_in_file, exclude, rng)
            exclude.update(negs)
        n_in_file = len(negs)
        negs += sample_excluding(pool, len(self.corpus), self.num_negatives - len(negs), exclude, rng)
        return negs, n_in_file
