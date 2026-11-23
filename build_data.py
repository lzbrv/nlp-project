"""
Build the shared data files for the project from LeanDojo Benchmark 4 (v7).

Premises used by a tactic are matched to the corpus the same way ReProver does
(common.py: get_all_pos_premises / Corpus.locate_premise): by file path plus
position, i.e. the corpus premise in `def_path` whose [start, end] span contains
`def_pos`. Used premises that cannot be located are skipped (ReProver logs a
warning and skips them too). Tactics left with no located premise are dropped.

Premise IDs are Lean full names. The few full names that occur more than once
in the corpus get "@<file path>:<line>:<col>" appended so every ID is unique.

Outputs (in processed/):
  premises.jsonl                    id, full_name, statement, file_path, start, end
  random/{train,val,test}.jsonl     one line per tactic with >= 1 located premise
  novel_premises/{train,val,test}.jsonl
  sample_1000.jsonl                 first 1000 lines of random/train.jsonl (for Joel)
  stats.json                        counts for the milestone's Data section

Each example line has:
  state                 proof state before the tactic (the query)
  positive_premise_ids  IDs of the premises the tactic used, deduplicated
  file_path             file of the theorem being proved
  theorem               full name of the theorem
  theorem_start         [line, col] where the theorem starts (needed later for accessibility)
"""

import bisect
import json
import os
from collections import Counter, defaultdict

ROOT = "leandojo_benchmark_4"
OUT = "processed"
SPLITS = ["random", "novel_premises"]
PARTS = ["train", "val", "test"]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


os.makedirs(OUT, exist_ok=True)
stats = {}

# ---------------------------------------------------------------------------
# 1. Premises
# ---------------------------------------------------------------------------
print("Reading corpus.jsonl ...")
raw = []  # (full_name, code, path, start, end)
with open(f"{ROOT}/corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        file_entry = json.loads(line)
        for p in file_entry["premises"]:
            raw.append((p["full_name"], p["code"], file_entry["path"],
                        tuple(p["start"]), tuple(p["end"])))

name_counts = Counter(r[0] for r in raw)
duplicates = {n for n, c in name_counts.items() if c > 1}

premises = []
by_file = defaultdict(list)  # path -> list of (start, end, id), sorted by start
for full_name, code, path, start, end in raw:
    # Some duplicated names occur twice in the SAME file, so the suffix needs
    # the start position as well as the path to be unique.
    pid = f"{full_name}@{path}:{start[0]}:{start[1]}" if full_name in duplicates else full_name
    premises.append({
        "id": pid,
        "full_name": full_name,
        "statement": code,
        "file_path": path,
        "start": list(start),
        "end": list(end),
    })
    by_file[path].append((start, end, pid))

for path in by_file:
    by_file[path].sort()
starts_by_file = {path: [s for s, _, _ in entries] for path, entries in by_file.items()}

id_counts = Counter(p["id"] for p in premises)
still_dup = [pid for pid, c in id_counts.items() if c > 1]
if still_dup:
    print("  IDs still not unique, e.g.:", still_dup[:5])
    raise SystemExit("Stopping: premise IDs are not unique")
print(f"  {len(premises)} premises in {len(by_file)} files")
print(f"  {len(duplicates)} full names occurred more than once -> given '@path:line:col' suffixes")
write_jsonl(f"{OUT}/premises.jsonl", premises)
stats["premises"] = {
    "total": len(premises),
    "files": len(by_file),
    "duplicate_full_names_suffixed": len(duplicates),
}


def locate(def_path, def_pos):
    """Return the ID of the corpus premise in def_path whose span contains def_pos, else None."""
    if def_path not in by_file:
        return None
    pos = tuple(def_pos)
    i = bisect.bisect_right(starts_by_file[def_path], pos) - 1
    if i >= 0:
        start, end, pid = by_file[def_path][i]
        if start <= pos <= end:
            return pid
    return None


# ---------------------------------------------------------------------------
# 2. Examples for each split and part
# ---------------------------------------------------------------------------
unlocated_examples = Counter()

for split in SPLITS:
    os.makedirs(f"{OUT}/{split}", exist_ok=True)
    stats[split] = {}
    for part in PARTS:
        print(f"Reading {split}/{part}.json ...")
        with open(f"{ROOT}/{split}/{part}.json", encoding="utf-8") as f:
            theorems = json.load(f)

        rows = []
        n_tactics = 0
        n_no_premise = 0
        n_none_located = 0
        n_used = 0
        n_unlocated = 0

        for thm in theorems:
            for tac in thm["traced_tactics"]:
                n_tactics += 1
                _, provenances = tac["annotated_tactic"]
                if not provenances:
                    n_no_premise += 1
                    continue

                located = []
                for prov in provenances:
                    n_used += 1
                    pid = locate(prov["def_path"], prov["def_pos"])
                    if pid is None:
                        n_unlocated += 1
                        unlocated_examples[(prov["full_name"], prov["def_path"])] += 1
                    else:
                        located.append(pid)

                located = list(dict.fromkeys(located))  # dedupe, keep order
                if not located:
                    n_none_located += 1
                    continue

                rows.append({
                    "state": tac["state_before"],
                    "positive_premise_ids": located,
                    "file_path": thm["file_path"],
                    "theorem": thm["full_name"],
                    "theorem_start": thm["start"],
                })

        write_jsonl(f"{OUT}/{split}/{part}.jsonl", rows)
        stats[split][part] = {
            "theorems": len(theorems),
            "tactics": n_tactics,
            "dropped_no_premise": n_no_premise,
            "dropped_no_located_premise": n_none_located,
            "examples_kept": len(rows),
            "used_premises": n_used,
            "used_premises_not_located": n_unlocated,
        }
        print(f"  {len(theorems)} theorems, {n_tactics} tactics | dropped: "
              f"{n_no_premise} no premise, {n_none_located} none located | "
              f"{len(rows)} kept | {n_unlocated}/{n_used} used premises not located")

        if split == "random" and part == "train":
            write_jsonl(f"{OUT}/sample_1000.jsonl", rows[:1000])
            print(f"  wrote {OUT}/sample_1000.jsonl")

        del theorems  # free memory before the next file

stats["most_common_unlocated"] = [
    {"full_name": n, "def_path": p, "count": c}
    for (n, p), c in unlocated_examples.most_common(20)
]
print("\nMost common premises that could not be located:")
for (n, p), c in unlocated_examples.most_common(10):
    print(f"  {c:6d}  {n}   ({p})")

with open(f"{OUT}/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"\nDone. Counts saved to {OUT}/stats.json")
