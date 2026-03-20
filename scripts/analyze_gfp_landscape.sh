#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${1:-data/gfp_latent_pca_16.npz}
OUT_DIR=${2:-artifacts/landscape}
SEED=${3:-42}

mkdir -p "${OUT_DIR}"

python experiments/analyze_qubo_landscape.py \
  --data "${DATA_PATH}" \
  --seed "${SEED}" \
  --l2 1e-3 \
  --n-samples 512 \
  --out "${OUT_DIR}/landscape_seed_${SEED}.json"
