import argparse
import os
import pandas as pd


def normalize_dataset_name(x):
    if pd.isna(x):
        return x
    x = str(x).lower()
    if "gfp" in x:
        return "gfp"
    if "aav" in x:
        return "aav"
    return x


def normalize_representation_name(x):
    if pd.isna(x):
        return x
    x = str(x).lower()
    if "pca" in x:
        return "pca"
    if "random" in x or "esm_binary" in x or "project" in x:
        return "random"
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landscape-csv",
        type=str,
        default="artifacts/results/landscape_grouped_summary.csv",
    )
    parser.add_argument(
        "--optimization-csv",
        type=str,
        default="artifacts/results/optimization_grouped_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/results/merged_landscape_optimization.csv",
    )
    args = parser.parse_args()

    landscape = pd.read_csv(args.landscape_csv)
    optimization = pd.read_csv(args.optimization_csv)

    # Normalize key columns
    landscape["dataset"] = landscape["dataset"].apply(normalize_dataset_name)
    optimization["dataset"] = optimization["dataset"].apply(normalize_dataset_name)

    landscape["representation"] = landscape["representation"].apply(normalize_representation_name)
    optimization["representation"] = optimization["representation"].apply(normalize_representation_name)

    # Force numeric keys to integer
    landscape["train_size"] = pd.to_numeric(landscape["train_size"], errors="coerce").astype("Int64")
    optimization["train_size"] = pd.to_numeric(optimization["train_size"], errors="coerce").astype("Int64")

    landscape["latent_dim"] = pd.to_numeric(landscape["latent_dim"], errors="coerce").astype("Int64")
    optimization["latent_dim"] = pd.to_numeric(optimization["latent_dim"], errors="coerce").astype("Int64")

    merge_keys = ["dataset", "representation", "train_size", "latent_dim"]

    print("LANDSCAPE merge keys sample:")
    print(landscape[merge_keys].drop_duplicates().head(10).to_string(index=False))

    print("\nOPTIMIZATION merge keys sample:")
    print(optimization[merge_keys].drop_duplicates().head(10).to_string(index=False))

    merged = pd.merge(
        optimization,
        landscape,
        on=merge_keys,
        how="inner",
        suffixes=("_opt", "_land"),
    )

    merged = merged.sort_values(
        by=[c for c in ["dataset", "representation", "train_size", "latent_dim", "optimizer"] if c in merged.columns]
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    print(f"\nSaved merged results to: {args.output_csv}")
    print("Merged shape:", merged.shape)
    print(merged.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
