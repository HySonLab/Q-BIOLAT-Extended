import argparse
import json
import os
import re
from glob import glob

import pandas as pd


FILE_PATTERN = re.compile(
    r"(gfp|aav)_(1000|2000|5000|10000)_(8|16|32|64)_(multiseed)\.json$"
)


def infer_metadata_from_path(path: str):
    base = os.path.basename(path).lower()

    dataset = "gfp" if "gfp" in base else "aav" if "aav" in base else None

    representation = "pca" if "pca" in base else "random"

    train_size = None
    m = re.search(r'_(1000|2000|5000|10000)_', base)
    if m:
        train_size = int(m.group(1))

    latent_dim = None
    # works for:
    # gfp_1000_16_multiseed.json
    # gfp_1000_16_pca_multiseed.json
    m = re.search(r'_(8|16|32|64)(?:_pca)?_multiseed\.json$', base)
    if m:
        latent_dim = int(m.group(1))

    return dataset, representation, train_size, latent_dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        type=str,
        default="artifacts/multiseed/**/*.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/results/optimization_summary.csv",
    )
    args = parser.parse_args()

    files = sorted(glob(args.input_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No optimization JSON files found: {args.input_glob}")

    rows = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        dataset, representation, train_size, latent_dim = infer_metadata_from_path(path)

        summary = obj.get("summary", {})
        for optimizer, metrics in summary.items():
            row = {
                "dataset": dataset,
                "representation": representation,
                "train_size": train_size,
                "latent_dim": latent_dim,
                "optimizer": optimizer,
                "improvement": metrics.get("improvement", {}).get("mean"),
                "nn_true_fitness": metrics.get("nearest_neighbor_true_fitness", {}).get("mean"),
                "nn_percentile": metrics.get("nearest_neighbor_percentile", {}).get("mean"),
                "score": metrics.get("score", {}).get("mean"),
                "min_hamming_to_train": metrics.get("min_hamming_to_train", {}).get("mean"),
                "source_file": path,
            }
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["dataset", "representation", "train_size", "latent_dim", "optimizer"]
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved optimization summary to: {args.output_csv}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
