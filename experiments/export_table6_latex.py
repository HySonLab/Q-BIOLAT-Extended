import argparse
import os
import pandas as pd


def pick_best_per_group(df: pd.DataFrame) -> pd.DataFrame:
    # For each dataset/train_size/latent_type, keep the row with highest best_score.
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


def fmt_method(row):
    latent = "PCA" if row["latent_type"] == "pca" else "Random"
    return latent


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-tex", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Keep only pca/random rows
    df = df[df["latent_type"].isin(["pca", "random"])].copy()

    best_df = pick_best_per_group(df)

    # Determine which row is best overall within each dataset/train_size
    best_overall_idx = (
        best_df.sort_values(
            ["dataset", "train_size", "best_score", "top10_mean"],
            ascending=[True, True, False, False],
        )
        .groupby(["dataset", "train_size"], as_index=False)
        .head(1)
        .index
    )
    best_df["is_best_overall"] = False
    best_df.loc[best_overall_idx, "is_best_overall"] = True

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{End-to-end protein design benchmark under a fixed external-oracle evaluation budget. "
                 r"For each dataset and train size, we report the best-performing random-projection and PCA-based "
                 r"Q-BioLat configuration after optimization, decoding, and oracle scoring.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Train size & Representation & Optimizer & Latent dim & Best score $\uparrow$ & Top-10 mean $\uparrow$ \\")
    lines.append(r"\midrule")

    last_dataset = None
    for _, row in best_df.iterrows():
        dataset = row["dataset"].upper()
        train_size = int(row["train_size"])
        rep = fmt_method(row)
        optimizer = latex_escape(str(row["method"]))
        latent_dim = int(row["latent_dim"])
        best_score = f"{row['best_score']:.3f}"
        top10_mean = f"{row['top10_mean']:.3f}"

        if row["is_best_overall"]:
            best_score = r"\textbf{" + best_score + "}"
            top10_mean = r"\textbf{" + top10_mean + "}"

        if last_dataset is not None and last_dataset != dataset:
            lines.append(r"\midrule")
        last_dataset = dataset

        lines.append(
            f"{dataset} & {train_size} & {rep} & {optimizer} & {latent_dim} & {best_score} & {top10_mean} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(args.output_tex), exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved LaTeX table to: {args.output_tex}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
