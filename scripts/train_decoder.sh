#!/bin/bash
set -e

mkdir -p artifacts/decoder_models

DATASET=gfp
N=1000
DEVICE=cpu
SEED=42

for LATENT_MODEL in ae vae; do
  for DIM in 16 32; do
    echo "========================================"
    echo "Training decoder: dataset=${DATASET}, n=${N}, latent_model=${LATENT_MODEL}, dim=${DIM}"
    echo "========================================"

    python experiments/train_decoder.py \
      --dense-npz artifacts/dense/${DATASET}_${N}_dense.npz \
      --latent-model-ckpt artifacts/latent_models/${DATASET}_${N}_${LATENT_MODEL}_${DIM}.pt \
      --latent-model-name ${LATENT_MODEL} \
      --latent-dim ${DIM} \
      --latent-hidden-dim 256 \
      --decoder-hidden-dim 256 \
      --batch-size 32 \
      --epochs 100 \
      --lr 1e-3 \
      --weight-decay 1e-5 \
      --mutation-loss-weight 2.0 \
      --aa-loss-weight 1.0 \
      --mask-pos-weight-scale 1.0 \
      --seed ${SEED} \
      --device ${DEVICE} \
      --output-prefix artifacts/decoder_models/${DATASET}_${N}_${LATENT_MODEL}_${DIM}_decoder
  done
done
