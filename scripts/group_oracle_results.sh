#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/group_oracle_results.py \
  --input-csv artifacts/results/oracle_summary.csv \
  --output-csv artifacts/results/oracle_grouped_summary.csv
