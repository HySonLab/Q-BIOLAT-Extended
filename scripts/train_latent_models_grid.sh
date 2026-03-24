#!/bin/bash
set -e

mkdir -p artifacts/latent_models
mkdir -p logs

DEVICE=cpu
SEED=42

for DATASET in gfp aav; do
  for N in 1000 2000 5000 10000; do
    for MODEL in ae vae; do
      for DIM in 8 16 32 64; do
        echo "========================================"
        echo "Training latent model: dataset=${DATASET}, n=${N}, model=${MODEL}, dim=${DIM}"
        echo "========================================"

        python experiments/train_latent_models.py \
          --input-npz artifacts/dense/${DATASET}_${N}_dense.npz \
          --model-name ${MODEL} \
          --latent-dim ${DIM} \
          --hidden-dim 256 \
          --batch-size 32 \
          --epochs 200 \
          --lr 1e-3 \
          --weight-decay 1e-5 \
          --beta-kl 1e-3 \
          --seed ${SEED} \
          --device ${DEVICE} \
          --output-prefix artifacts/latent_models/${DATASET}_${N}_${MODEL}_${DIM}
      done
    done
  done
done
