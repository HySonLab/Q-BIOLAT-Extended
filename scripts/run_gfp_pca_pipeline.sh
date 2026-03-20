#!/bin/bash
set -e

mkdir -p artifacts/binary_pca

for num_samples in 1000 2000 5000 10000; do

    echo "======================================"
    echo "Processing ${num_samples} samples"
    echo "======================================"

    dense_file="artifacts/dense/gfp_${num_samples}_dense.npz"

    for dim in 8 16 32 64; do

        echo "PCA binarization → ${dim} dimensions"

        python examples/pca_binarize_embeddings.py \
            --input-npz ${dense_file} \
            --output-npz artifacts/binary_pca/gfp_${num_samples}_pca_binary_${dim}.npz \
            --dim ${dim} \
            --binarize median

    done

done
