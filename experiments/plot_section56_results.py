import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def pick_best_per_rep(df: pd.DataFrame) -> pd.DataFrame:
    idx = (
        df.sort_values(
            ["dataset", "train_size", "latent_type", "best_score", "top10_mean"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["dataset", "train_size", "latent_type"], as_index=False)
        .head(1)
        .index
    )
    out = df.loc[idx].copy()
    out = out.sort_values(["dataset", "train_size", "latent_type"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-png", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[df["latent_type"].isin(["pca", "random"])].copy()
    best_df = pick_best_per_rep(df)

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    datasets = ["gfp", "aav"]
    titles = ["GFP", "AAV"]

    for ax, dataset, title in zip(axes, datasets, titles):
        sub = best_df[best_df["dataset"] == dataset].copy()

        for latent_type, label in [("pca", "PCA"), ("random", "Random")]:
            cur = sub[sub["latent_type"] == latent_type].sort_values("train_size")
            ax.plot(cur["train_size"], cur["best_score"], marker="o", label=label)

            # annotate dim + optimizer
            for _, row in cur.iterrows():
                txt = f"{int(row['latent_dim'])}, {row['method']}"
                ax.annotate(
                    txt,
                    (row["train_size"], row["best_score"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )

        ax.set_title(title)
        ax.set_xlabel("Train size")
        ax.set_ylabel("Best oracle score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.savefig(args.output_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {args.output_png}")


if __name__ == "__main__":
    main()
