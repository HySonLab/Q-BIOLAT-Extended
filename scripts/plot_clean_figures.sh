#!/bin/bash
set -e

mkdir -p artifacts/figures/clean

python experiments/plot_clean_figures.py \
  --input-csv artifacts/results/merged_landscape_optimization.csv \
  --output-dir artifacts/figures/clean
