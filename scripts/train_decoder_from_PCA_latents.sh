#!/bin/bash
set -e

mkdir -p artifacts/decoder_models

DATASET=gfp
N=1000

for DIM in 16 32 64; do
  echo "========================================"
  echo "PCA decoder: ${DATASET}, n=${N}, dim=${DIM}"
  echo "========================================"

  python experiments/train_decoder_from_PCA_latents.py \
    --latent-npz artifacts/binary_pca/${DATASET}_${N}_pca_binary_${DIM}.npz \
    --latent-dim ${DIM} \
    --batch-size 32 \
    --epochs 100 \
    --output-prefix artifacts/decoder_models/${DATASET}_${N}_pca_${DIM}_decoder
done
