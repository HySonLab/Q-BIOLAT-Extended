import argparse
import os
import pandas as pd


MODEL_ORDER = {
    "gp": 0,
    "ridge": 1,
    "xgboost": 2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="artifacts/results/oracle_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/results/oracle_grouped_summary.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Keep only test split for the paper table
    df = df[df["split"] == "test"].copy()

    summary = (
        df.groupby(["dataset", "train_size", "model"], as_index=False)
        .agg({
            "spearman": "mean",
            "pearson": "mean",
            "rmse": "mean",
            "mae": "mean",
        })
    )

    summary["model_order"] = summary["model"].map(MODEL_ORDER).fillna(999)
    summary = summary.sort_values(["dataset", "train_size", "model_order", "model"])
    summary = summary.drop(columns=["model_order"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    print(f"Saved oracle grouped summary to: {args.output_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
