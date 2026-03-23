#!/bin/bash
set -e

mkdir -p artifacts/landscape

for file in artifacts/binary_pca/*.npz; do
    base=$(basename "$file" .npz)
    echo "Analyzing $base"

    python experiments/analyze_qubo_landscape.py \
      --data "$file" \
      --seed 42 \
      --l2 1e-3 \
      --n-samples 512 \
      --out "artifacts/landscape/${base}_seed_42.json"
done
