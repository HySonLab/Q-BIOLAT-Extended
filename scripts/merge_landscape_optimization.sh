#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/merge_landscape_optimization.py \
  --landscape-csv artifacts/results/landscape_grouped_summary.csv \
  --optimization-csv artifacts/results/optimization_grouped_summary.csv \
  --output-csv artifacts/results/merged_landscape_optimization.csv
