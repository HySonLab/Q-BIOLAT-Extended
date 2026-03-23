import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_dataset(ax, df, dataset):
    sub = df[df["dataset"] == dataset].copy()

    styles = {
        "gp": {
            "label": "Gaussian Process",
            "marker": "o",
            "color": "tab:blue",
        },
        "ridge": {
            "label": "Ridge Regression",
            "marker": "s",
            "color": "tab:orange",
        },
        "xgboost": {
            "label": "XGBoost",
            "marker": "^",
            "color": "tab:green",
        },
    }

    for model in ["gp", "ridge", "xgboost"]:
        d = sub[sub["model"] == model].copy()
        if d.empty:
            continue

        d = d.sort_values("train_size")

        ax.plot(
            d["train_size"],
            d["spearman"],
            marker=styles[model]["marker"],
            color=styles[model]["color"],
            label=styles[model]["label"],
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Spearman Correlation (Test)", fontsize=12)
    ax.set_title(dataset.upper(), fontsize=13)
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="artifacts/results/oracle_grouped_summary.csv"
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/figures/oracle"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    plot_dataset(axes[0], df, "gfp")
    plot_dataset(axes[1], df, "aav")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Oracle Performance Across Data Regimes", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(args.outdir, "oracle_performance_gfp_aav.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved oracle plot to: {output_path}")


if __name__ == "__main__":
    main()
