#!/bin/bash
set -e

mkdir -p artifacts/oracle/eval

python experiments/evaluate_oracle.py \
  --model-path artifacts/oracle/models/gfp_1000_ridge.pkl \
  --input-npz artifacts/dense/gfp_1000_dense.npz \
  --output-prefix artifacts/oracle/eval/gfp_1000_ridge_eval
