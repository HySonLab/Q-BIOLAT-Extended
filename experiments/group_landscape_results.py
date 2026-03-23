import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        default="artifacts/results/landscape_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/results/landscape_grouped_summary.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = [
        "dataset",
        "representation",
        "train_size",
        "latent_dim",
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
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Convert split train size back to original total dataset size if test_size exists
    if "test_size" in df.columns:
        df["train_size"] = pd.to_numeric(df["train_size"], errors="coerce")
        df["test_size"] = pd.to_numeric(df["test_size"], errors="coerce")
        df["train_size"] = (df["train_size"] + df["test_size"]).astype("Int64")

    summary = (
        df.groupby(
            ["dataset", "representation", "train_size", "latent_dim"],
            as_index=False,
        )
        .agg(
            {
                "test_spearman": "mean",
                "train_spearman": "mean",
                "spectral_norm": "mean",
                "frobenius_norm": "mean",
                "effective_rank": "mean",
                "bit_flip_var": "mean",
                "mean_abs_bit_flip_gain": "mean",
                "top1_energy_ratio": "mean",
                "top3_energy_ratio": "mean",
                "top5_energy_ratio": "mean",
            }
        )
        .sort_values(["dataset", "train_size", "latent_dim", "representation"])
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    print(f"Saved grouped summary to: {args.output_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
