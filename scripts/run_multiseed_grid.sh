#!/bin/bash
set -e

mkdir -p artifacts/multiseed

for DATASET in gfp aav; do
  for NUM_SAMPLES in 1000 2000 5000 10000; do
    for DIM in 8 16 32 64; do
      for LATENT in random pca; do

        if [ "$LATENT" = "random" ]; then
          DATA_PATH="artifacts/binary/${DATASET}_${NUM_SAMPLES}_esm_binary_${DIM}.npz"
          OUTPUT="artifacts/multiseed/${DATASET}_${NUM_SAMPLES}_${DIM}_multiseed.json"
        else
          DATA_PATH="artifacts/binary_pca/${DATASET}_${NUM_SAMPLES}_pca_binary_${DIM}.npz"
          OUTPUT="artifacts/multiseed/${DATASET}_${NUM_SAMPLES}_${DIM}_pca_multiseed.json"
        fi

        echo "========================================"
        echo "Running multiseed:"
        echo "dataset=${DATASET}, samples=${NUM_SAMPLES}, dim=${DIM}, latent=${LATENT}"
        echo "========================================"

        python experiments/benchmark_multiseed.py \
          --data ${DATA_PATH} \
          --seeds 0 1 2 3 4 \
          --out ${OUTPUT}

      done
    done
  done
done
