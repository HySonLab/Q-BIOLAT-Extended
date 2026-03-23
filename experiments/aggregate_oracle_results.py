import argparse
import json
import os
import re
from glob import glob

import pandas as pd


def infer_metadata_from_path(path: str):
    base = os.path.basename(path).lower()

    dataset = None
    if "gfp" in base:
        dataset = "gfp"
    elif "aav" in base:
        dataset = "aav"

    model = None
    if "ridge" in base:
        model = "ridge"
    elif "xgboost" in base:
        model = "xgboost"
    elif "gp" in base:
        model = "gp"

    train_size = None
    m = re.search(r'_(1000|2000|5000|10000)_', base)
    if m:
        train_size = int(m.group(1))

    return dataset, train_size, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        type=str,
        default="artifacts/oracle/models/*.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/results/oracle_summary.csv",
    )
    args = parser.parse_args()

    files = sorted(glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No oracle JSON files found: {args.input_glob}")

    rows = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset, train_size, model = infer_metadata_from_path(path)

        for split in ["train", "val", "test"]:
            if split not in data:
                continue

            metrics = data[split]

            row = {
                "dataset": dataset,
                "train_size": train_size,
                "model": model,
                "split": split,
                "spearman": metrics.get("spearman"),
                "pearson": metrics.get("pearson"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "source_file": path,
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["dataset", "train_size", "model", "split"]
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved oracle summary to: {args.output_csv}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
