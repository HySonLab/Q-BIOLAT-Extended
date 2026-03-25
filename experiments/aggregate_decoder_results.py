import argparse
import json
import os
import re
from glob import glob

import pandas as pd


def parse_metadata(path: str):
    """
    Expected filename pattern:
      gfp_1000_pca_16_decoder.json
      aav_5000_random_32_decoder.json
    """
    base = os.path.basename(path).replace(".json", "")
    m = re.match(r"([a-zA-Z0-9]+)_(\d+)_(pca|random)_(\d+)_decoder", base)
    if m is None:
        return None
    dataset = m.group(1).lower()
    train_size = int(m.group(2))
    latent_type = m.group(3).lower()
    latent_dim = int(m.group(4))
    return dataset, train_size, latent_type, latent_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="artifacts/decoder_models/*.json",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/results/decoder_summary.csv",
    )
    args = parser.parse_args()

    files = sorted(glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No decoder JSON files matched: {args.input_glob}")

    rows = []
    for path in files:
        meta = parse_metadata(path)
        if meta is None:
            continue

        dataset, train_size, latent_type, latent_dim = meta

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data["metrics"]

        rows.append({
            "dataset": dataset,
            "train_size": train_size,
            "latent_type": latent_type,
            "latent_dim": latent_dim,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "aa_acc": metrics.get("aa_acc"),
            "pos_acc": metrics.get("pos_acc"),
            "wt_length": data.get("wt_length"),
            "train_split_size": data.get("train_size"),
            "test_split_size": data.get("test_size"),
            "source_file": path,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "train_size", "latent_type", "latent_dim"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved decoder summary to: {args.output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
