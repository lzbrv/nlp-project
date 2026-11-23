import json

ROOT = "leandojo_benchmark_4"   # change if the folder name is different


def show(obj, indent=0, max_len=120):
    """Print keys and shortened values of a nested dict/list."""
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: ({type(v).__name__}, len {len(v)})")
                show(v[0] if isinstance(v, list) and v else v, indent + 1, max_len)
            else:
                print(f"{pad}{k}: {str(v)[:max_len]!r}")
    elif isinstance(obj, list) and obj:
        show(obj[0], indent, max_len)


# One line of the premise corpus (= one Lean file)
with open(f"{ROOT}/corpus.jsonl", encoding="utf-8") as f:
    first_file = json.loads(f.readline())
print("=== corpus.jsonl: one file ===")
show(first_file)

# One theorem from the test set (smaller than train, loads faster)
with open(f"{ROOT}/random/test.json", encoding="utf-8") as f:
    theorems = json.load(f)
print(f"\n=== random/test.json: {len(theorems)} theorems; first one ===")
show(theorems[0])

# --- Find a tactic that actually uses premises and print annotated_tactic fully ---
print("\n=== first tactic with premises ===")
done = False
for thm in theorems:
    for tac in thm["traced_tactics"]:
        tactic_text, provenances = tac["annotated_tactic"]
        if provenances:  # non-empty list = this tactic uses premises
            print("theorem:", thm["full_name"])
            print("tactic:", tac["tactic"])
            print("annotated tactic text:", tactic_text[:200])
            print("premises used:")
            for p in provenances:
                print("   ", p)
            done = True
            break
    if done:
        break

# --- Show a corpus file that has imports, to see the import format ---
print("\n=== a corpus file with imports ===")
with open(f"{ROOT}/corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        file_entry = json.loads(line)
        if file_entry["imports"]:
            print("path:", file_entry["path"])
            print("imports:", file_entry["imports"][:5])
            break