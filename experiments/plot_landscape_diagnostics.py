import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def scatter_plot(df, x_col, y_col, out_path, title):
    plt.figure(figsize=(7, 5))

    markers = {"pca": "o", "random": "s"}
    datasets = sorted(df["dataset"].dropna().unique())

    for dataset in datasets:
        sub_dataset = df[df["dataset"] == dataset]

        for representation in sorted(sub_dataset["representation"].dropna().unique()):
            sub = sub_dataset[sub_dataset["representation"] == representation]

            label = f"{dataset}-{representation}"
            marker = markers.get(representation, "o")

            plt.scatter(
                sub[x_col],
                sub[y_col],
                label=label,
                marker=marker,
                alpha=0.8,
            )

            for _, row in sub.iterrows():
                txt = f"{int(row['train_size'])}-{int(row['latent_dim'])}"
                plt.annotate(
                    txt,
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
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
        default="artifacts/results/landscape_grouped_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/figures/landscape",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    scatter_plot(
        df,
        x_col="spectral_norm",
        y_col="test_spearman",
        out_path=os.path.join(args.output_dir, "spectral_norm_vs_spearman.png"),
        title="Spectral Norm vs Test Spearman",
    )

    scatter_plot(
        df,
        x_col="effective_rank",
        y_col="test_spearman",
        out_path=os.path.join(args.output_dir, "effective_rank_vs_spearman.png"),
        title="Effective Rank vs Test Spearman",
    )

    scatter_plot(
        df,
        x_col="bit_flip_var",
        y_col="test_spearman",
        out_path=os.path.join(args.output_dir, "bit_flip_var_vs_spearman.png"),
        title="Bit-Flip Variance vs Test Spearman",
    )

    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
