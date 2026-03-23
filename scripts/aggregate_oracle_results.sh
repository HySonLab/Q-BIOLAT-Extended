#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/aggregate_oracle_results.py \
  --input-glob "artifacts/oracle/models/*.json" \
  --output-csv artifacts/results/oracle_summary.csv
