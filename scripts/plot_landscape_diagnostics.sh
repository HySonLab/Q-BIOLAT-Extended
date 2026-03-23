#!/bin/bash
set -e

mkdir -p artifacts/figures/landscape

python experiments/plot_landscape_diagnostics.py \
  --input-csv artifacts/results/landscape_grouped_summary.csv \
  --output-dir artifacts/figures/landscape
