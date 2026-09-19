#!/usr/bin/env bash
# k sweep for R2.1 and R4.3. Five seeds, three classifiers, k = 3..8.
# XGBoost is skipped at k = 8 (65,536 dense columns after scaling is slow).
set -euo pipefail

OUT="results_abl/ksweep"
mkdir -p "$OUT"

for K in 3 4 5 6 7 8; do
  for M in svm 5nn xgboost; do
    if [[ "$M" == "xgboost" && "$K" -ge 8 ]]; then
      continue
    fi
    for REG in small large; do
      if [[ "$REG" == "small" ]]; then DIR="data"; else DIR="data_large"; fi
      F="$OUT/${REG}_${M}_k${K}.json"
      if [[ -s "$F" ]]; then
        continue
      fi
      python run_ablation.py --datadir "$DIR" --out "$F" \
          --features c bc abc --method "$M" --k "$K" --seeds 5 --n-jobs 8
    done
  done
done
