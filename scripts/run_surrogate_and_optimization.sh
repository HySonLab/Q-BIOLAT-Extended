#!/bin/bash
set -e

mkdir -p artifacts/results

for num_samples in 1000 2000 5000 10000; do
    for dim in 8 16 32 64; do

        dataset="artifacts/binary/gfp_${num_samples}_esm_binary_${dim}.npz"

        echo "======================================"
        echo "Dataset: ${dataset}"
        echo "======================================"

        echo "Train QUBO surrogate"

        python experiments/train_surrogate.py \
            --data ${dataset} \
            --model qubo \
            --out artifacts/results/train_gfp_${num_samples}_${dim}.json

        echo "Run optimization"

        python experiments/optimize_latent.py \
            --data ${dataset} \
            --out artifacts/results/optimize_gfp_${num_samples}_${dim}.json

    done
done
