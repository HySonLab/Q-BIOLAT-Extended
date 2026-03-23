#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/group_landscape_results.py \
  --input-csv artifacts/results/landscape_summary.csv \
  --output-csv artifacts/results/landscape_grouped_summary.csv
