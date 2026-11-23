"""Loading the preprocessed data described in docs/data_contract.md."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Premise:
    id: int
    full_name: str
    path: str
    code: str

    def text(self) -> str:
        """Encoder input for this premise."""
        if self.full_name and self.full_name not in self.code:
            return f"{self.full_name} : {self.code}"
        return self.code


class Corpus:
    """All premises, indexed by contiguous integer id."""

    def __init__(self, premises: List[Premise]):
        for i, p in enumerate(premises):
            if p.id != i:
                raise ValueError(
                    f"corpus ids must be contiguous 0..N-1 in line order; "
                    f"line {i} has id {p.id}"
                )
        self.premises = premises
        self.paths: List[str] = [p.path for p in premises]
        by_path: Dict[str, List[int]] = defaultdict(list)
        for p in premises:
            by_path[p.path].append(p.id)
        self.path_to_ids: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in by_path.items()
        }

    @classmethod
    def load(cls, path) -> "Corpus":
        return cls([
            Premise(int(r["id"]), r.get("full_name", ""), r["path"], r["code"])
            for r in read_jsonl(path)
        ])

    def __len__(self) -> int:
        return len(self.premises)

    def text(self, pid: int) -> str:
        return self.premises[pid].text()

    def name(self, pid: int) -> str:
        return self.premises[pid].full_name


@dataclass
class Example:
    id: str
    state: str
    positive_ids: List[int]
    file: str = ""
    theorem: str = ""
    accessible_ids: Optional[np.ndarray] = None
    accessible_key: Optional[str] = None


def load_examples(path, corpus_size: Optional[int] = None,
                  limit: Optional[int] = None) -> Tuple[List[Example], dict]:
    """Load one split file. Returns (examples, stats).

    Examples without positives are dropped (they have no retrieval target);
    out-of-range premise ids are an error, since they mean the corpus and the
    split files came from different preprocessing runs.
    """
    examples: List[Example] = []
    stats = {"lines": 0, "dropped_no_positive": 0}
    for r in read_jsonl(path):
        stats["lines"] += 1
        pos = sorted(set(int(x) for x in r.get("positive_ids", [])))
        if not pos:
            stats["dropped_no_positive"] += 1
            continue
        if corpus_size is not None and (pos[0] < 0 or pos[-1] >= corpus_size):
            raise ValueError(f"{path}: example {r.get('id')} has premise id outside corpus")
        acc = r.get("accessible_ids")
        examples.append(Example(
            id=str(r["id"]),
            state=r["state"],
            positive_ids=pos,
            file=r.get("file", ""),
            theorem=r.get("theorem", ""),
            accessible_ids=None if acc is None else np.asarray(sorted(set(acc)), dtype=np.int64),
            accessible_key=r.get("accessible_key"),
        ))
        if limit is not None and len(examples) >= limit:
            break
    stats["examples"] = len(examples)
    return examples, stats


class AccessibleIndex:
    """Resolves the candidate pool of an example.

    `get` returns a sorted id array, or None meaning "every premise".
    """

    def __init__(self, keyed: Optional[Dict[str, np.ndarray]] = None):
        self.keyed = keyed or {}

    @classmethod
    def load(cls, data_root) -> "AccessibleIndex":
        path = Path(data_root) / "accessible.jsonl"
        if not path.exists():
            return cls()
        return cls({
            r["key"]: np.asarray(sorted(set(r["ids"])), dtype=np.int64)
            for r in read_jsonl(path)
        })

    def get(self, ex: Example) -> Optional[np.ndarray]:
        if ex.accessible_ids is not None:
            return ex.accessible_ids
        if ex.accessible_key is not None:
            try:
                return self.keyed[ex.accessible_key]
            except KeyError:
                raise KeyError(
                    f"example {ex.id} has accessible_key {ex.accessible_key!r} "
                    f"which is not in accessible.jsonl"
                ) from None
        return None


def split_path(data_root, split: str, part: str) -> Path:
    return Path(data_root) / split / f"{part}.jsonl"
