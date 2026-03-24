#!/bin/bash
set -e

mkdir -p artifacts/decoder_models

for DATASET in gfp aav; do
  for N in 1000 2000 5000 10000; do
    for LATENT_TYPE in pca random; do
      for DIM in 8 16 32 64; do

        if [ "$LATENT_TYPE" = "pca" ]; then
          LATENT_FILE="artifacts/binary_pca/${DATASET}_${N}_pca_binary_${DIM}.npz"
          OUT_PREFIX="artifacts/decoder_models/${DATASET}_${N}_pca_${DIM}_decoder"
        else
          LATENT_FILE="artifacts/binary/${DATASET}_${N}_esm_binary_${DIM}.npz"
          OUT_PREFIX="artifacts/decoder_models/${DATASET}_${N}_random_${DIM}_decoder"
        fi

        echo "========================================"
        echo "Decoder from projection latents:"
        echo "dataset=${DATASET}, n=${N}, latent=${LATENT_TYPE}, dim=${DIM}"
        echo "========================================"

        python experiments/train_decoder_from_projection_latents.py \
          --latent-npz ${LATENT_FILE} \
          --latent-dim ${DIM} \
          --batch-size 32 \
          --epochs 100 \
          --lr 1e-3 \
          --hidden-dim 256 \
          --mask-loss-weight 2.0 \
          --aa-loss-weight 1.0 \
          --seed 42 \
          --output-prefix ${OUT_PREFIX}

      done
    done
  done
done
