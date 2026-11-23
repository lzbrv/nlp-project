"""Synthetic data in the contract format + a tiny random ByT5-like encoder,
so the whole pipeline can be tested offline on CPU in seconds."""

import random

import pytest

from retrieval.data import write_jsonl

FILES = [f"Mathlib/F{i}.lean" for i in range(5)]


def make_dataset(root, n_premises=60, n_steps=40, seed=0, with_accessible=True):
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    corpus = [{"id": i, "full_name": f"lemma_{i}", "path": FILES[i % len(FILES)],
               "code": f"theorem lemma_{i} : x{i} + 0 = x{i}"} for i in range(n_premises)]
    write_jsonl(root / "corpus.jsonl", corpus)
    if with_accessible:
        write_jsonl(root / "accessible.jsonl",
                    [{"key": "first_half", "ids": list(range(n_premises // 2))}])
    for split in ("random", "novel_premises"):
        (root / split).mkdir(parents=True, exist_ok=True)
        for part in ("train", "val", "test"):
            rows = []
            for j in range(n_steps):
                pos = rng.sample(range(n_premises // 2), rng.randint(1, 2))
                row = {"id": f"{split}-{part}-{j}", "theorem": f"thm_{j}", "file": FILES[0],
                       "state": "⊢ " + " ∧ ".join(f"x{p} + 0 = x{p}" for p in pos),
                       "positive_ids": pos}
                if with_accessible and j % 2 == 0:
                    row["accessible_key"] = "first_half"
                rows.append(row)
            rows.append({"id": f"{split}-{part}-empty", "theorem": "t", "file": FILES[0],
                         "state": "⊢ True", "positive_ids": []})
            write_jsonl(root / split / f"{part}.jsonl", rows)
    return root


@pytest.fixture
def data_root(tmp_path):
    return make_dataset(tmp_path / "data")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from transformers import ByT5Tokenizer, T5Config, T5EncoderModel

    d = tmp_path_factory.mktemp("tiny_byt5")
    cfg = T5Config(vocab_size=384, d_model=32, d_kv=8, d_ff=64, num_layers=2,
                   num_heads=4, feed_forward_proj="gated-gelu", decoder_start_token_id=0)
    T5EncoderModel(cfg).save_pretrained(d)
    ByT5Tokenizer().save_pretrained(d)
    return d
