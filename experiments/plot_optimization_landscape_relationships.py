import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def make_scatter(df, x_col, y_col, out_path, title):
    plt.figure(figsize=(7, 5))

    markers = {"pca": "o", "random": "s"}
    colors = {
        ("gfp", "pca"): "tab:blue",
        ("gfp", "random"): "tab:orange",
        ("aav", "pca"): "tab:green",
        ("aav", "random"): "tab:red",
    }

    for dataset in sorted(df["dataset"].dropna().unique()):
        for representation in sorted(df["representation"].dropna().unique()):
            sub = df[
                (df["dataset"] == dataset) &
                (df["representation"] == representation)
            ]
            if len(sub) == 0:
                continue

            plt.scatter(
                sub[x_col],
                sub[y_col],
                label=f"{dataset}-{representation}",
                marker=markers.get(representation, "o"),
                color=colors.get((dataset, representation), None),
                alpha=0.8,
            )

            for _, row in sub.iterrows():
                txt = f"{row['optimizer']}\n{int(row['train_size'])}-{int(row['latent_dim'])}"
                plt.annotate(
                    txt,
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6,
                )

    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        default="artifacts/results/merged_landscape_optimization.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/figures/merged",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    make_scatter(
        df,
        x_col="spectral_norm",
        y_col="nn_true_fitness",
        out_path=os.path.join(args.output_dir, "spectral_norm_vs_nn_true_fitness.png"),
        title="Spectral Norm vs NN True Fitness",
    )

    make_scatter(
        df,
        x_col="effective_rank",
        y_col="nn_percentile",
        out_path=os.path.join(args.output_dir, "effective_rank_vs_nn_percentile.png"),
        title="Effective Rank vs NN Percentile",
    )

    make_scatter(
        df,
        x_col="test_spearman",
        y_col="nn_true_fitness",
        out_path=os.path.join(args.output_dir, "test_spearman_vs_nn_true_fitness.png"),
        title="Test Spearman vs NN True Fitness",
    )

    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
