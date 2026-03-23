#!/bin/bash
set -e

mkdir -p artifacts/figures/merged

python experiments/plot_optimization_landscape_relationships.py \
  --input-csv artifacts/results/merged_landscape_optimization.csv \
  --output-dir artifacts/figures/merged
