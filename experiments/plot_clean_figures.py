import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_filtered(df, dataset, optimizer, x, y, out_path, title):
    sub = df[
        (df["dataset"] == dataset) &
        (df["optimizer"] == optimizer)
    ].copy()

    plt.figure(figsize=(6, 5))

    for rep, marker in [("pca", "o"), ("random", "s")]:
        d = sub[sub["representation"] == rep]
        if len(d) == 0:
            continue

        plt.scatter(
            d[x],
            d[y],
            label=rep,
            marker=marker,
            s=70,
            alpha=0.85,
        )

        for _, row in d.iterrows():
            txt = f"{int(row['train_size'])}-{int(row['latent_dim'])}"
            plt.annotate(
                txt,
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.legend()
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
        default="artifacts/figures/clean",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_filtered(
        df,
        dataset="gfp",
        optimizer="genetic_algorithm",
        x="test_spearman",
        y="nn_true_fitness",
        out_path=os.path.join(args.output_dir, "spearman_vs_nn_gfp_ga.png"),
        title="Prediction vs Optimization (GFP, GA)",
    )

    plot_filtered(
        df,
        dataset="gfp",
        optimizer="genetic_algorithm",
        x="spectral_norm",
        y="nn_true_fitness",
        out_path=os.path.join(args.output_dir, "spectral_vs_nn_gfp_ga.png"),
        title="Spectral Norm vs Optimization (GFP, GA)",
    )

    plot_filtered(
        df,
        dataset="gfp",
        optimizer="genetic_algorithm",
        x="effective_rank",
        y="nn_percentile",
        out_path=os.path.join(args.output_dir, "rank_vs_percentile_gfp_ga.png"),
        title="Effective Rank vs Optimization (GFP, GA)",
    )

    print(f"Saved clean plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
