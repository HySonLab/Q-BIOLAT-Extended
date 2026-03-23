#!/bin/bash
set -e

mkdir -p artifacts/figures/oracle

python experiments/plot_oracle_results.py \
  --input artifacts/results/oracle_grouped_summary.csv \
  --outdir artifacts/figures/oracle
