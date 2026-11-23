"""Load the files written by build_data.py (see DATA.md).

    processed/premises.jsonl                     id, full_name, statement, file_path, start, end
    processed/{random,novel_premises}/{train,val,test}.jsonl
                                                 state, positive_premise_ids, file_path,
                                                 theorem, theorem_start
    processed/accessible.jsonl                   OPTIONAL candidate pools (docs/data_contract.md)

Premise ids in the files are strings (Lean full names). Internally each premise
is addressed by its 0-based line number in premises.jsonl, so embeddings and
candidate pools can be plain integer arrays.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

PREMISES_FILE = "premises.jsonl"
ACCESSIBLE_FILE = "accessible.jsonl"


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
    idx: int          # line number in premises.jsonl
    id: str           # build_data.py id (full name, suffixed if duplicated)
    full_name: str
    path: str         # file_path: used for in-file negatives
    code: str         # statement

    def text(self) -> str:
        """Encoder input for this premise."""
        if self.full_name and self.full_name not in self.code:
            return f"{self.full_name} : {self.code}"
        return self.code


class Corpus:
    """All premises; integer index = line in premises.jsonl."""

    def __init__(self, premises: List[Premise]):
        self.premises = premises
        self.id_to_idx: Dict[str, int] = {}
        for p in premises:
            if p.id in self.id_to_idx:
                raise ValueError(f"duplicate premise id {p.id!r}")
            self.id_to_idx[p.id] = p.idx
        self.paths: List[str] = [p.path for p in premises]
        by_path: Dict[str, List[int]] = defaultdict(list)
        for p in premises:
            by_path[p.path].append(p.idx)
        self.path_to_ids: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in by_path.items()
        }

    @classmethod
    def load(cls, path) -> "Corpus":
        path = Path(path)
        if path.is_dir():
            path = path / PREMISES_FILE
        return cls([
            Premise(i, r["id"], r.get("full_name", r["id"]), r["file_path"], r["statement"])
            for i, r in enumerate(read_jsonl(path))
        ])

    def __len__(self) -> int:
        return len(self.premises)

    def text(self, idx: int) -> str:
        return self.premises[idx].text()

    def name(self, idx: int) -> str:
        return self.premises[idx].id

    def to_indices(self, ids, where: str = "") -> List[int]:
        try:
            return [self.id_to_idx[i] for i in ids]
        except KeyError as e:
            raise KeyError(f"{where}: premise id {e.args[0]!r} not in premises.jsonl "
                           "(were the files built by the same build_data.py run?)") from None


@dataclass
class Example:
    id: str
    state: str
    positive_ids: List[int]          # corpus indices
    file: str = ""
    theorem: str = ""
    theorem_start: Optional[List[int]] = None
    accessible_ids: Optional[np.ndarray] = None
    accessible_key: Optional[str] = None


def load_examples(path, corpus: Corpus, limit: Optional[int] = None) -> Tuple[List[Example], dict]:
    """Load one split file. Returns (examples, stats).

    Example ids are "<part>:<line number>" unless the line has an "id" field;
    build_data.py is deterministic, so these are stable across rebuilds.
    """
    path = Path(path)
    examples: List[Example] = []
    stats = {"lines": 0, "dropped_no_positive": 0}
    for line_no, r in enumerate(read_jsonl(path)):
        stats["lines"] += 1
        where = f"{path}:{line_no + 1}"
        pos = sorted(set(corpus.to_indices(r.get("positive_premise_ids", []), where)))
        if not pos:
            stats["dropped_no_positive"] += 1
            continue
        acc = r.get("accessible_ids")
        examples.append(Example(
            id=str(r.get("id", f"{path.stem}:{line_no}")),
            state=r["state"],
            positive_ids=pos,
            file=r.get("file_path", ""),
            theorem=r.get("theorem", ""),
            theorem_start=r.get("theorem_start"),
            accessible_ids=None if acc is None else np.asarray(
                sorted(set(corpus.to_indices(acc, where))), dtype=np.int64),
            accessible_key=r.get("accessible_key"),
        ))
        if limit is not None and len(examples) >= limit:
            break
    stats["examples"] = len(examples)
    return examples, stats


class AccessibleIndex:
    """Resolves the candidate pool of an example.

    `get` returns a sorted index array, or None meaning "every premise"
    (DATA.md: until accessible pools exist, use all premises).
    """

    def __init__(self, keyed: Optional[Dict[str, np.ndarray]] = None):
        self.keyed = keyed or {}

    @classmethod
    def load(cls, data_root, corpus: Corpus) -> "AccessibleIndex":
        path = Path(data_root) / ACCESSIBLE_FILE
        if not path.exists():
            return cls()
        return cls({
            r["key"]: np.asarray(sorted(set(corpus.to_indices(r["ids"], str(path)))), dtype=np.int64)
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
                    f"which is not in {ACCESSIBLE_FILE}"
                ) from None
        return None


def split_path(data_root, split: str, part: str) -> Path:
    return Path(data_root) / split / f"{part}.jsonl"
