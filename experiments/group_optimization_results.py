import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        default="artifacts/results/optimization_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/results/optimization_grouped_summary.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = [
        "dataset",
        "representation",
        "train_size",
        "latent_dim",
        "optimizer",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    agg_dict = {}

    optional_metrics = [
        "improvement",
        "nn_true_fitness",
        "nn_percentile",
        "surrogate_score",
        "start_score",
        "best_score",
        "runtime_sec",
    ]

    for col in optional_metrics:
        if col in df.columns:
            agg_dict[col] = "mean"

    if not agg_dict:
        raise ValueError(
            "No optimization metric columns found. "
            "Expected at least one of: "
            f"{optional_metrics}"
        )

    summary = (
        df.groupby(
            ["dataset", "representation", "train_size", "latent_dim", "optimizer"],
            as_index=False,
        )
        .agg(agg_dict)
        .sort_values(
            ["dataset", "representation", "train_size", "latent_dim", "optimizer"]
        )
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    print(f"Saved grouped optimization summary to: {args.output_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
