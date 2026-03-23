#!/bin/bash
set -e

mkdir -p artifacts/figures/final

python experiments/plot_final_figures.py \
  --input-csv artifacts/results/merged_landscape_optimization.csv \
  --output-dir artifacts/figures/final \
  --dataset gfp \
  --optimizer genetic_algorithm
