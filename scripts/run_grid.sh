#!/usr/bin/env bash
# Full ablation: {random, novel_premises} x {in_file, random} x seeds,
# then test-set evaluation of every run's best checkpoint.
#   SEEDS="0 1 2" CONFIG=configs/base.yaml scripts/run_grid.sh [extra --set overrides]
set -euo pipefail
CONFIG=${CONFIG:-configs/base.yaml}
SEEDS=${SEEDS:-0}
for split in random novel_premises; do
  for neg in in_file random; do
    for seed in $SEEDS; do
      run="${split}__${neg}__seed${seed}"
      echo "=== $run ==="
      resume=""
      [ -d "runs/$run/last" ] && resume="--resume"
      if [ ! -f "runs/$run/log.jsonl" ] || ! grep -q '"event": "done"' "runs/$run/log.jsonl"; then
        python -m retrieval.train --config "$CONFIG" $resume \
          --set data.split=$split negatives.strategy=$neg seed=$seed "$@"
      fi
      [ -f "runs/$run/eval_${split}_test/metrics.json" ] || \
        python -m retrieval.evaluate --run-dir "runs/$run" --part test
    done
  done
done
python -m retrieval.analysis table runs
