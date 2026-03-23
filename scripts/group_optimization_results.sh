#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/group_optimization_results.py \
  --input-csv artifacts/results/optimization_summary.csv \
  --output-csv artifacts/results/optimization_grouped_summary.csv
