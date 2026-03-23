#!/bin/bash
set -e

python experiments/export_oracle_table_latex.py \
  --input-csv artifacts/results/oracle_grouped_summary.csv \
  --output-tex artifacts/results/oracle_table_rows.tex
