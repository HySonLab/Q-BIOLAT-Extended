#!/bin/bash
set -e

seed=42

mkdir -p data/proteingym
mkdir -p artifacts/dense
mkdir -p artifacts/binary

for num_samples in 1000 2000 5000 10000; do
    echo "========================================"
    echo "Sample ${num_samples} proteins from AAV"
    echo "========================================"

    python examples/make_subset_csv.py \
        --input-csv data/proteingym/aav.csv \
        --output-csv data/proteingym/aav_${num_samples}.csv \
        --n "${num_samples}" \
        --seed "${seed}"

    echo "Compute dense ESM embeddings once for ${num_samples} samples"
    python examples/build_real_peptide_dataset_esm_dense.py \
        --input-csv data/proteingym/aav_${num_samples}.csv \
        --output-npz artifacts/dense/aav_${num_samples}_dense.npz \
        --device cpu

    for dim in 8 16 32 64; do
        echo "Build ${dim}-dim binary latent codes for ${num_samples} samples"

        python examples/project_binarize_embeddings.py \
            --input-npz artifacts/dense/aav_${num_samples}_dense.npz \
            --output-npz artifacts/binary/aav_${num_samples}_esm_binary_${dim}.npz \
            --project-dim "${dim}" \
            --binarize median \
            --seed "${seed}"
    done
done
