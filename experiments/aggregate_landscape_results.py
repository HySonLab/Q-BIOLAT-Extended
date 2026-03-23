import argparse
import json
import os
import re
from glob import glob

import pandas as pd


def infer_metadata_from_path(path: str):
    """
    Infer dataset / representation / train_size / latent_dim from filename.
    Supports patterns like:
      gfp_10000_esm_binary_16.npz
      gfp_10000_pca_binary_16.npz
      aav_5000_esm_binary_32.npz
      aav_5000_pca_binary_32.npz
    and landscape JSON filenames that may embed those paths or names.
    """
    base = os.path.basename(path).lower()

    dataset = None
    if "gfp" in base:
        dataset = "gfp"
    elif "aav" in base:
        dataset = "aav"

    representation = None
    if "pca" in base:
        representation = "pca"
    elif "esm_binary" in base or "random" in base or "project" in base:
        representation = "random"

    train_size = None
    m = re.search(r'_(1000|2000|5000|10000)_', base)
    if m:
        train_size = int(m.group(1))

    latent_dim = None
    m = re.search(r'_(8|16|32|64)(?:\.|_|$)', base)
    if m:
        latent_dim = int(m.group(1))

    seed = None
    m = re.search(r'seed[_\-]?(\d+)', base)
    if m:
        seed = int(m.group(1))

    return {
        "dataset": dataset,
        "representation": representation,
        "train_size": train_size,
        "latent_dim": latent_dim,
        "seed_from_name": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        type=str,
        default="artifacts/landscape/**/*.json",
        help="Glob for landscape JSON files",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/results/landscape_summary.csv",
    )
    args = parser.parse_args()

    files = sorted(glob(args.input_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No JSON files matched: {args.input_glob}")

    rows = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {}
        row.update(infer_metadata_from_path(path))

        # pull keys from JSON if present
        row["json_seed"] = data.get("seed")
        row["train_size_json"] = data.get("train_size")
        row["test_size"] = data.get("test_size")
        row["train_spearman"] = data.get("train_spearman")
        row["test_spearman"] = data.get("test_spearman")

        row["spectral_norm"] = data.get("spectral_norm")
        row["frobenius_norm"] = data.get("frobenius_norm")
        row["infinity_norm"] = data.get("infinity_norm")
        row["effective_rank"] = data.get("effective_rank")
        row["mean_row_norm"] = data.get("mean_row_norm")
        row["max_row_norm"] = data.get("max_row_norm")
        row["min_eigenvalue"] = data.get("min_eigenvalue")
        row["max_eigenvalue"] = data.get("max_eigenvalue")
        row["top1_energy_ratio"] = data.get("top1_energy_ratio")
        row["top3_energy_ratio"] = data.get("top3_energy_ratio")
        row["top5_energy_ratio"] = data.get("top5_energy_ratio")

        row["bit_flip_mean"] = data.get("bit_flip_mean")
        row["bit_flip_std"] = data.get("bit_flip_std")
        row["bit_flip_var"] = data.get("bit_flip_var")
        row["mean_abs_bit_flip_gain"] = data.get("mean_abs_bit_flip_gain")
        row["max_abs_bit_flip_gain"] = data.get("max_abs_bit_flip_gain")
        row["per_bit_variance_mean"] = data.get("per_bit_variance_mean")
        row["per_sample_variance_mean"] = data.get("per_sample_variance_mean")

        row["n_bits"] = data.get("n_bits")
        row["l2"] = data.get("l2")
        row["source_file"] = path

        # prefer JSON values if available
        if row["train_size_json"] is not None:
            row["train_size"] = row["train_size_json"]
        if row["json_seed"] is not None:
            row["seed"] = row["json_seed"]
        else:
            row["seed"] = row["seed_from_name"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # optional cleanup
    preferred_cols = [
        "dataset",
        "representation",
        "train_size",
        "latent_dim",
        "seed",
        "test_spearman",
        "train_spearman",
        "spectral_norm",
        "frobenius_norm",
        "effective_rank",
        "bit_flip_var",
        "mean_abs_bit_flip_gain",
        "top1_energy_ratio",
        "top3_energy_ratio",
        "top5_energy_ratio",
        "source_file",
    ]
    cols = preferred_cols + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved summary to: {args.output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()
