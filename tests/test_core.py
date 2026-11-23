import math

import numpy as np
import pytest
import torch

from retrieval.config import load_config
from retrieval.data import AccessibleIndex, Corpus, load_examples, split_path
from retrieval.evaluate import evaluate_with_scorer, summarize
from retrieval.losses import label_matrix, retrieval_loss
from retrieval.metrics import (bootstrap_mean_ci, diff_in_diff, paired_bootstrap_diff,
                               recall_at_k, reciprocal_rank)
from retrieval.negatives import NegativeSampler, sample_excluding


def load(data_root, part="train"):
    corpus = Corpus.load(data_root)
    exs, stats = load_examples(split_path(data_root, "random", part), corpus)
    return corpus, exs, stats, AccessibleIndex.load(data_root, corpus)


def test_loading_drops_empty_and_resolves_accessible(data_root):
    corpus, exs, stats, acc = load(data_root)
    assert stats["dropped_no_positive"] == 1 and stats["examples"] == 40
    assert len(acc.get(exs[0])) == 30          # keyed pool
    assert acc.get(exs[1]) is None             # falls back to all premises
    assert exs[0].id == "train:0" and exs[0].file == "Mathlib/F0.lean"
    assert [corpus.name(i) for i in exs[0].positive_ids] == sorted(
        corpus.name(i) for i in exs[0].positive_ids)


def test_corpus_rejects_duplicate_and_unknown_ids(tmp_path):
    line = '{"id": "a", "full_name": "a", "statement": "x", "file_path": "f"}\n'
    (tmp_path / "premises.jsonl").write_text(line * 2)
    with pytest.raises(ValueError):
        Corpus.load(tmp_path)
    (tmp_path / "premises.jsonl").write_text(line)
    (tmp_path / "t.jsonl").write_text('{"state": "s", "positive_premise_ids": ["zzz"]}\n')
    with pytest.raises(KeyError):
        load_examples(tmp_path / "t.jsonl", Corpus.load(tmp_path))


@pytest.mark.parametrize("pool", [None, np.arange(30)])
def test_in_file_sampler(data_root, pool):
    corpus, *_ = load(data_root)
    s = NegativeSampler(corpus, "in_file", 3, 1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        pos = int(rng.integers(30))
        positives = {pos, (pos + 7) % 30}
        negs, n_in = s.sample(pos, positives, pool, rng)
        assert len(negs) == 3 and len(set(negs)) == 3
        assert not set(negs) & positives
        assert n_in == 1 and corpus.paths[negs[0]] == corpus.paths[pos]
        if pool is not None:
            assert set(negs) <= set(pool.tolist())


def test_random_sampler_never_counts_in_file(data_root):
    corpus, *_ = load(data_root)
    s = NegativeSampler(corpus, "random", 3, 1)
    negs, n_in = s.sample(0, {0}, None, np.random.default_rng(0))
    assert n_in == 0 and len(negs) == 3 and 0 not in negs


def test_in_file_falls_back_when_file_exhausted(data_root):
    corpus, *_ = load(data_root)
    s = NegativeSampler(corpus, "in_file", 3, 1)
    same_file = set(corpus.path_to_ids[corpus.paths[0]].tolist())
    negs, n_in = s.sample(0, same_file, None, np.random.default_rng(0))
    assert n_in == 0 and len(negs) == 3 and not set(negs) & same_file


def test_sample_excluding_small_pool():
    out = sample_excluding(np.array([1, 2, 3]), 10, 5, {2}, np.random.default_rng(0))
    assert sorted(out) == [1, 3]


def test_losses():
    labels = label_matrix([{1}, {2, 3}], [1, 2, 3, 4])
    assert labels.tolist() == [[1, 0, 0, 0], [0, 1, 1, 0]]
    good = labels * 2 - 1
    for kind in ("mse", "infonce"):
        assert retrieval_loss(good, labels, kind) < retrieval_loss(-good, labels, kind)


def test_metrics():
    assert recall_at_k([1, 5, 20], 10) == pytest.approx(2 / 3)
    assert reciprocal_rank([3, 7]) == pytest.approx(1 / 3)
    assert reciprocal_rank([math.inf]) == 0.0
    ci = bootstrap_mean_ci([0.0, 1.0] * 50, n_boot=200)
    assert ci["ci_low"] < 0.5 < ci["ci_high"]
    d = paired_bootstrap_diff([1.0] * 50, [0.0] * 50, n_boot=100)
    assert d["mean"] == 1.0 and d["p_le_0"] == 0.0
    did = diff_in_diff([0.5] * 30, [0.5] * 30, [0.9] * 30, [0.5] * 30, n_boot=100)
    assert did["mean"] == pytest.approx(0.4)


def test_evaluator_with_oracle_and_masking(data_root):
    corpus, exs, _, acc = load(data_root, "test")

    def oracle(batch):
        s = np.zeros((len(batch), len(corpus)))
        for i, ex in enumerate(batch):
            s[i, ex.positive_ids] = 1.0
            s[i, 59] = 2.0   # outside the keyed pool: must be masked for keyed examples
        return s

    per_ex = evaluate_with_scorer(exs, oracle, corpus, acc, batch_size=7)
    keyed = [r for r, ex in zip(per_ex, exs) if ex.accessible_key]
    full = [r for r, ex in zip(per_ex, exs) if not ex.accessible_key]
    assert all(r["recall@10"] == 1.0 and r["mrr"] == 1.0 for r in keyed)
    assert all(r["mrr"] == 0.5 and r["top_ids"][0] == "lemma_59" for r in full)
    assert all(r["positive_ids"][0].startswith("lemma_") for r in per_ex)
    m = summarize(per_ex, n_boot=50)
    assert m["n_examples"] == 40 and m["recall@10"]["mean"] == 1.0


def test_config_overrides(tmp_path):
    cfg = load_config("configs/base.yaml", ["negatives.strategy=random", "seed=3", "train.lr=0.5"])
    assert cfg.negatives.strategy == "random" and cfg.seed == 3 and cfg.train.lr == 0.5
    assert cfg.resolved_run_name() == "random__random__seed3"
    with pytest.raises(KeyError):
        load_config(None, ["train.nope=1"])
