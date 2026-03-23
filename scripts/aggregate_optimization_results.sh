#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/aggregate_optimization_results.py \
  --input-glob "artifacts/multiseed/**/*.json" \
  --output-csv artifacts/results/optimization_summary.csv
