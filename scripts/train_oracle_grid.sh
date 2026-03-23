#!/bin/bash
set -e

mkdir -p artifacts/oracle/models
mkdir -p artifacts/oracle/metrics
mkdir -p artifacts/oracle/preds

for dataset in gfp aav; do
  for n in 1000 2000 5000 10000; do
    for model in ridge xgboost gp; do
      echo "========================================"
      echo "Training oracle: dataset=${dataset}, n=${n}, model=${model}"
      echo "========================================"

      python experiments/train_oracle.py \
        --input-npz artifacts/dense/${dataset}_${n}_dense.npz \
        --model-name ${model} \
        --seed 42 \
        --output-prefix artifacts/oracle/models/${dataset}_${n}_${model}
    done
  done
done
