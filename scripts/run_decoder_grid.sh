#!/bin/bash
set -e

mkdir -p artifacts/decoded_scored

for DATASET in gfp aav; do
  for N in 1000 2000 5000 10000; do
    for LATENT in pca random; do
      for DIM in 8 16 32 64; do

        if [ "$LATENT" = "pca" ]; then
          MULTISEED_JSON="artifacts/multiseed/${DATASET}_${N}_${DIM}_pca_multiseed.json"
          DECODER_CKPT="artifacts/decoder_models/${DATASET}_${N}_pca_${DIM}_decoder.pt"
        else
          MULTISEED_JSON="artifacts/multiseed/${DATASET}_${N}_${DIM}_multiseed.json"
          DECODER_CKPT="artifacts/decoder_models/${DATASET}_${N}_random_${DIM}_decoder.pt"
        fi

        if [ "$DATASET" = "gfp" ]; then
          if [ "$N" = "1000" ] || [ "$N" = "2000" ] || [ "$N" = "5000" ] || [ "$N" = "10000" ]; then
            if [ "$N" = "1000" ] || [ "$N" = "2000" ] || [ "$N" = "5000" ] || [ "$N" = "10000" ]; then
              if [ "$N" = "1000" ] || [ "$N" = "2000" ] || [ "$N" = "5000" ]; then
                ORACLE_MODEL="artifacts/oracle/models/${DATASET}_${N}_gp.pkl"
              else
                ORACLE_MODEL="artifacts/oracle/models/${DATASET}_${N}_ridge.pkl"
              fi
            fi
          fi
        else
          if [ "$N" = "1000" ]; then
            ORACLE_MODEL="artifacts/oracle/models/${DATASET}_${N}_ridge.pkl"
          else
            ORACLE_MODEL="artifacts/oracle/models/${DATASET}_${N}_gp.pkl"
          fi
        fi

        if [ ! -f "$MULTISEED_JSON" ]; then
          echo "[SKIP] missing multiseed file: $MULTISEED_JSON"
          continue
        fi

        if [ ! -f "$DECODER_CKPT" ]; then
          echo "[SKIP] missing decoder checkpoint: $DECODER_CKPT"
          continue
        fi

        if [ ! -f "$ORACLE_MODEL" ]; then
          echo "[SKIP] missing oracle model: $ORACLE_MODEL"
          continue
        fi

        for METHOD in simulated_annealing genetic_algorithm random_search greedy_hill_climb latent_bo; do
          echo "=================================================="
          echo "Decoding+scoring: dataset=${DATASET}, n=${N}, latent=${LATENT}, dim=${DIM}, method=${METHOD}"
          echo "=================================================="

          python experiments/decode_optimized_latents.py \
            --optimized-codes ${MULTISEED_JSON} \
            --decoder-ckpt ${DECODER_CKPT} \
            --latent-dim ${DIM} \
            --method ${METHOD} \
            --deduplicate \
            --oracle-model ${ORACLE_MODEL} \
            --output-csv artifacts/decoded_scored/${DATASET}_${N}_${LATENT}_${DIM}_${METHOD}_decoded.csv
        done

      done
    done
  done
done
