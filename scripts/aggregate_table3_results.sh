#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/aggregate_table3_results.py \
  --input-glob "artifacts/latent_models/*.json" \
  --output-csv artifacts/results/table3_latent_summary.csv
