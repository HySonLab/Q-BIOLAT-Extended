import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def parse_filename(path):
    name = os.path.basename(path).replace(".csv", "")
    m = re.match(r"([a-z]+)_(\d+)_(pca|random)_(\d+)_(.+)_decoded", name)
    if m is None:
        return None

    dataset = m.group(1)
    train_size = int(m.group(2))
    latent_type = m.group(3)
    latent_dim = int(m.group(4))
    method = m.group(5)

    return dataset, train_size, latent_type, latent_dim, method


def compute_metrics(df, top_k=10):
    scores = df["oracle_score"].values
    best = float(np.max(scores))
    topk = float(np.mean(np.sort(scores)[-top_k:]))
    unique_sequences = len(set(df["sequence"]))
    total = len(df)
    uniqueness = unique_sequences / total if total > 0 else 0.0

    return {
        "best_score": best,
        "topk_mean": topk,
        "uniqueness": uniqueness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="artifacts/decoded_scored/*.csv")
    parser.add_argument("--output-csv", default="artifacts/results/decoded_summary.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise ValueError(f"No decoded CSV files found for glob: {args.input_glob}")

    rows = []

    for path in files:
        meta = parse_filename(path)
        if meta is None:
            continue

        dataset, train_size, latent_type, latent_dim, method = meta
        df = pd.read_csv(path)

        if "oracle_score" not in df.columns:
            print(f"[WARN] Missing oracle_score in {path}, skipping")
            continue

        if len(df) == 0:
            print(f"[WARN] Empty CSV in {path}, skipping")
            continue

        metrics = compute_metrics(df)

        rows.append({
            "dataset": dataset,
            "train_size": train_size,
            "latent_type": latent_type,
            "latent_dim": latent_dim,
            "method": method,
            "best_score": metrics["best_score"],
            "top10_mean": metrics["topk_mean"],
            "uniqueness": metrics["uniqueness"],
            "source_file": path,
        })

    if len(rows) == 0:
        raise ValueError(
            "No scored decoded CSVs were found. "
            "Make sure you ran decode_optimized_latents.py with --oracle-model."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "train_size", "latent_type", "latent_dim", "method"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved summary to: {args.output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
