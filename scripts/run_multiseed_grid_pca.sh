#!/bin/bash
set -e

mkdir -p artifacts/multiseed

for num_samples in 1000 2000 5000 10000; do
    for dim in 8 16 32 64; do

        dataset="artifacts/binary_pca/gfp_${num_samples}_pca_binary_${dim}.npz"
        output="artifacts/multiseed/gfp_${num_samples}_${dim}_pca_multiseed.json"

        echo "========================================"
        echo "Running PCA multiseed: samples=${num_samples}, dim=${dim}"
        echo "========================================"

        python experiments/benchmark_multiseed.py \
            --data ${dataset} \
            --seeds 0 1 2 3 4 \
            --out ${output}

    done
done
