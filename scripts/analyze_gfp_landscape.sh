#!/bin/bash
set -e

mkdir -p artifacts/landscape

python experiments/analyze_qubo_landscape.py \
  --data artifacts/binary_pca/gfp_1000_pca_binary_16.npz \
  --seed 42 \
  --l2 1e-3 \
  --n-samples 512 \
  --out artifacts/landscape/gfp_1000_pca_binary_16_seed_42.json
