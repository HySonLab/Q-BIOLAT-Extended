#!/bin/bash
set -e

mkdir -p artifacts/results

python experiments/aggregate_decoder_results.py \
  --input-glob "artifacts/decoder_models/*.json" \
  --output-csv artifacts/results/decoder_summary.csv
