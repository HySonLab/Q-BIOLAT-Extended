import argparse
import json
import os
import re
from glob import glob

import pandas as pd


def parse_metadata(path: str):
    base = os.path.basename(path).replace(".json", "")
    m = re.match(r"([a-zA-Z0-9]+)_(\d+)_(ae|vae)_(\d+)", base)
    if m is None:
        return None
    dataset = m.group(1).lower()
    train_size = int(m.group(2))
    model = m.group(3).lower()
    latent_dim = int(m.group(4))
    return dataset, train_size, model, latent_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="artifacts/latent_models/*.json",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/results/table3_latent_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    files = sorted(glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No JSON files matched: {args.input_glob}")

    for path in files:
        meta = parse_metadata(path)
        if meta is None:
            continue

        dataset, train_size, model, latent_dim = meta

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {
            "dataset": dataset,
            "train_size": train_size,
            "representation": model.upper(),
            "latent_dim": latent_dim,
            "train_recon_mse": data["train"]["recon_mse"],
            "val_recon_mse": data["val"]["recon_mse"],
            "test_recon_mse": data["test"]["recon_mse"],
            "train_bit_entropy": data["train"]["bit_entropy"],
            "val_bit_entropy": data["val"]["bit_entropy"],
            "test_bit_entropy": data["test"]["bit_entropy"],
            "train_active_dims": data["train"]["active_dims"],
            "val_active_dims": data["val"]["active_dims"],
            "test_active_dims": data["test"]["active_dims"],
            "train_frac_ones": data["train"]["frac_ones"],
            "val_frac_ones": data["val"]["frac_ones"],
            "test_frac_ones": data["test"]["frac_ones"],
            "source_file": path,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "train_size", "representation", "latent_dim"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved Table 3 latent summary to: {args.output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
