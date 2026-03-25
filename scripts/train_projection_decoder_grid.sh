#!/bin/bash
set -e

mkdir -p artifacts/decoder_models
mkdir -p logs

DEVICE=cpu
BATCH_SIZE=32
EPOCHS=100
SEED=42

for DATASET in gfp aav; do
  for N in 1000 2000 5000 10000; do
    for LATENT in pca random; do
      for DIM in 8 16 32 64; do

        if [ "$LATENT" = "pca" ]; then
          LATENT_FILE="artifacts/binary_pca/${DATASET}_${N}_pca_binary_${DIM}.npz"
          OUT_PREFIX="artifacts/decoder_models/${DATASET}_${N}_pca_${DIM}_decoder"
        else
          LATENT_FILE="artifacts/binary/${DATASET}_${N}_esm_binary_${DIM}.npz"
          OUT_PREFIX="artifacts/decoder_models/${DATASET}_${N}_random_${DIM}_decoder"
        fi

        if [ ! -f "$LATENT_FILE" ]; then
          echo "[SKIP] Missing latent file: $LATENT_FILE"
          continue
        fi

        echo "=================================================="
        echo "Training decoder:"
        echo "dataset=${DATASET}, n=${N}, latent=${LATENT}, dim=${DIM}"
        echo "=================================================="

        python experiments/train_decoder_from_projection_latents.py \
          --latent-npz ${LATENT_FILE} \
          --latent-dim ${DIM} \
          --batch-size ${BATCH_SIZE} \
          --epochs ${EPOCHS} \
          --device ${DEVICE} \
          --seed ${SEED} \
          --output-prefix ${OUT_PREFIX}

      done
    done
  done
done
