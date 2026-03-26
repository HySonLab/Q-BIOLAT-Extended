#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/aggregate_decoded_results.py \
  --input-glob "artifacts/decoded_scored/*.csv" \
  --output-csv artifacts/results/decoded_summary.csv
