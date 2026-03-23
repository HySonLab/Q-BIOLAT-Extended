#!/bin/bash
set -e

mkdir -p artifacts/oracle/models
mkdir -p artifacts/oracle/metrics
mkdir -p artifacts/oracle/preds

python experiments/train_oracle.py \
  --input-npz artifacts/dense/gfp_1000_dense.npz \
  --model-name ridge \
  --seed 42 \
  --output-prefix artifacts/oracle/models/gfp_1000_ridge
