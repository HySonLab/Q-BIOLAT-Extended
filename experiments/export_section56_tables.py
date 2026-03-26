import argparse
import os
from typing import List, Tuple

import pandas as pd


def latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def format_representation(latent_type: str, latent_dim: int) -> str:
    if latent_type == "pca":
        return f"PCA-{latent_dim}"
    if latent_type == "random":
        return f"Random-{latent_dim}"
    return f"{latent_type}-{latent_dim}"


def build_table(
    df: pd.DataFrame,
    dataset: str,
    train_sizes: List[int],
    label: str,
    caption: str,
) -> str:
    sub = df[
        (df["dataset"] == dataset) &
        (df["train_size"].isin(train_sizes))
    ].copy()

    sub = sub.sort_values(
        ["train_size", "latent_type", "latent_dim", "method"]
    )

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{l l l c c c}")
    lines.append(r"\toprule")
    lines.append(r"Train size & Representation & Optimizer & Best score $\uparrow$ & Top-10 mean $\uparrow$ & Uniqueness $\uparrow$ \\")
    lines.append(r"\midrule")

    current_train = None
    for _, row in sub.iterrows():
        train_size = int(row["train_size"])
        rep = format_representation(row["latent_type"], int(row["latent_dim"]))
        optimizer = latex_escape(row["method"])
        best_score = f"{row['best_score']:.3f}"
        top10_mean = f"{row['top10_mean']:.3f}"
        uniqueness = f"{row['uniqueness']:.3f}"

        if current_train is not None and train_size != current_train:
            lines.append(r"\midrule")
        current_train = train_size

        lines.append(
            f"{train_size} & {rep} & {optimizer} & {best_score} & {top10_mean} & {uniqueness} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def save_table(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="artifacts/results/decoded_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/results/section56_tables",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Optional: prettier optimizer ordering
    method_order = {
        "greedy_hill_climb": 0,
        "genetic_algorithm": 1,
        "simulated_annealing": 2,
        "random_search": 3,
        "latent_bo": 4,
    }
    df["method_order"] = df["method"].map(method_order).fillna(999)
    latent_order = {"pca": 0, "random": 1}
    df["latent_order"] = df["latent_type"].map(latent_order).fillna(999)

    df = df.sort_values(
        ["dataset", "train_size", "latent_order", "latent_dim", "method_order"]
    ).drop(columns=["method_order", "latent_order"])

    specs: List[Tuple[str, List[int], str, str, str]] = [
        (
            "gfp",
            [1000, 2000],
            "tab:section56_gfp_small",
            "End-to-end sequence design results on GFP for low-data regimes (1000 and 2000 training samples).",
            "table_section56_gfp_1000_2000.tex",
        ),
        (
            "gfp",
            [5000, 10000],
            "tab:section56_gfp_large",
            "End-to-end sequence design results on GFP for moderate-data regimes (5000 and 10000 training samples).",
            "table_section56_gfp_5000_10000.tex",
        ),
        (
            "aav",
            [1000, 2000],
            "tab:section56_aav_small",
            "End-to-end sequence design results on AAV for low-data regimes (1000 and 2000 training samples).",
            "table_section56_aav_1000_2000.tex",
        ),
        (
            "aav",
            [5000, 10000],
            "tab:section56_aav_large",
            "End-to-end sequence design results on AAV for moderate-data regimes (5000 and 10000 training samples).",
            "table_section56_aav_5000_10000.tex",
        ),
    ]

    for dataset, train_sizes, label, caption, filename in specs:
        tex = build_table(
            df=df,
            dataset=dataset,
            train_sizes=train_sizes,
            label=label,
            caption=caption,
        )
        out_path = os.path.join(args.output_dir, filename)
        save_table(out_path, tex)
        print(f"Saved: {out_path}")

        print("\n" + "=" * 80)
        print(filename)
        print("=" * 80)
        print(tex)
        print()

    master_path = os.path.join(args.output_dir, "section56_tables_all.tex")
    with open(master_path, "w", encoding="utf-8") as f:
        for _, _, _, _, filename in specs:
            f.write(rf"\input{{{os.path.join(args.output_dir, filename)}}}" + "\n\n")
    print(f"Saved: {master_path}")


if __name__ == "__main__":
    main()
