import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter_with_trend(
    df,
    x,
    y,
    dataset,
    optimizer,
    out_path,
    title,
    xlabel,
    ylabel,
    drop_outlier_fn=None,
):
    sub = df[
        (df["dataset"] == dataset) &
        (df["optimizer"] == optimizer)
    ].copy()

    if sub.empty:
        print(f"[WARN] No data for dataset={dataset}, optimizer={optimizer}")
        return

    if drop_outlier_fn is not None:
        before = len(sub)
        sub = sub[~drop_outlier_fn(sub)].copy()
        after = len(sub)
        print(f"[INFO] Removed {before - after} outlier(s) for {title}")

    plt.figure(figsize=(6, 5))

    style = {
        "pca": {
            "marker": "o",
            "color": "tab:blue",
            "label": "PCA (Binary)",
        },
        "random": {
            "marker": "s",
            "color": "tab:orange",
            "label": "Random Projection",
        },
    }

    for rep in ["pca", "random"]:
        d = sub[sub["representation"] == rep].copy()
        if d.empty:
            continue

        plt.scatter(
            d[x],
            d[y],
            s=110,
            alpha=0.85,
            label=style[rep]["label"],
            marker=style[rep]["marker"],
            color=style[rep]["color"],
        )

        # linear trend line
        if len(d) >= 2 and d[x].std() > 1e-12:
            coef = np.polyfit(d[x], d[y], 1)
            poly = np.poly1d(coef)
            xs = np.linspace(d[x].min(), d[x].max(), 100)
            plt.plot(
                xs,
                poly(xs),
                linestyle="--",
                linewidth=2.5,
                color=style[rep]["color"],
            )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="artifacts/results/merged_landscape_optimization.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/figures/final",
    )
    parser.add_argument(
        "--dataset",
        default="gfp",
        help="Dataset to plot, e.g. gfp or aav",
    )
    parser.add_argument(
        "--optimizer",
        default="genetic_algorithm",
        help="Optimizer to plot",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional outlier filter for the prediction figure only
    # This removes the extreme low-fitness point that compresses the y-axis.
    def prediction_outlier_mask(subdf):
        return (subdf["nn_true_fitness"] < 2.0)

    dataset_tag = args.dataset.upper()

    plot_scatter_with_trend(
        df=df,
        x="test_spearman",
        y="nn_true_fitness",
        dataset=args.dataset,
        optimizer=args.optimizer,
        out_path=os.path.join(args.output_dir, f"{args.dataset}_fig1_prediction_vs_optimization.png"),
        title="Prediction vs Optimization",
        xlabel="Spearman Correlation (Test)",
        ylabel="Nearest Neighbor True Fitness",
        drop_outlier_fn=prediction_outlier_mask,
    )

    plot_scatter_with_trend(
        df=df,
        x="spectral_norm",
        y="nn_true_fitness",
        dataset=args.dataset,
        optimizer=args.optimizer,
        out_path=os.path.join(args.output_dir, f"{args.dataset}_fig2_spectral_vs_optimization.png"),
        title="Spectral Norm vs Optimization",
        xlabel="Spectral Norm of J",
        ylabel="Nearest Neighbor True Fitness",
        drop_outlier_fn=None,
    )

    plot_scatter_with_trend(
        df=df,
        x="effective_rank",
        y="nn_percentile",
        dataset=args.dataset,
        optimizer=args.optimizer,
        out_path=os.path.join(args.output_dir, f"{args.dataset}_fig3_rank_vs_optimization.png"),
        title="Effective Rank vs Optimization",
        xlabel="Effective Rank of J",
        ylabel="Nearest Neighbor Percentile",
        drop_outlier_fn=None,
    )

    print(f"Done. Output directory: {args.output_dir}")
    print(f"Dataset: {dataset_tag}, Optimizer: {args.optimizer}")


if __name__ == "__main__":
    main()
