import argparse
import os
import pandas as pd


def latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def pick_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dataset, train_size, latent_type, latent_dim),
    keep the best row by best_score, then top10_mean.
    """
    idx = (
        df.sort_values(
            ["dataset", "train_size", "latent_type", "latent_dim", "best_score", "top10_mean"],
            ascending=[True, True, True, True, False, False],
        )
        .groupby(["dataset", "train_size", "latent_type", "latent_dim"], as_index=False)
        .head(1)
        .index
    )
    out = df.loc[idx].copy()
    return out


def build_one_table(df: pd.DataFrame, dataset: str, train_size: int, label: str, caption: str) -> str:
    sub = df[
        (df["dataset"] == dataset) &
        (df["train_size"] == train_size)
    ].copy()

    bit_order = [8, 16, 32, 64]
    latent_order = ["pca", "random"]

    sub["latent_order"] = sub["latent_type"].map({"pca": 0, "random": 1})
    sub = sub.sort_values(["latent_dim", "latent_order"]).drop(columns=["latent_order"])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{l l l c c}")
    lines.append(r"\toprule")
    lines.append(r"Bits & Representation & Best optimizer & Best score $\uparrow$ & Top-10 mean $\uparrow$ \\")
    lines.append(r"\midrule")

    first_block = True
    for bits in bit_order:
        rows_for_bits = sub[sub["latent_dim"] == bits]

        if len(rows_for_bits) == 0:
            continue

        if not first_block:
            lines.append(r"\midrule")
        first_block = False

        for latent_type in latent_order:
            cur = rows_for_bits[rows_for_bits["latent_type"] == latent_type]
            if len(cur) == 0:
                continue

            row = cur.iloc[0]
            rep = "PCA" if latent_type == "pca" else "Random"
            optimizer = latex_escape(row["method"])
            best_score = f"{row['best_score']:.3f}"
            top10_mean = f"{row['top10_mean']:.3f}"

            lines.append(
                f"{bits} & {rep} & {optimizer} & {best_score} & {top10_mean} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[df["latent_type"].isin(["pca", "random"])].copy()

    best_df = pick_best_rows(df)

    os.makedirs(args.output_dir, exist_ok=True)

    specs = [
        ("gfp", 1000,  "tab:appendix_gfp_1000",  "Detailed end-to-end design results on GFP with 1000 training samples."),
        ("gfp", 2000,  "tab:appendix_gfp_2000",  "Detailed end-to-end design results on GFP with 2000 training samples."),
        ("gfp", 5000,  "tab:appendix_gfp_5000",  "Detailed end-to-end design results on GFP with 5000 training samples."),
        ("gfp", 10000, "tab:appendix_gfp_10000", "Detailed end-to-end design results on GFP with 10000 training samples."),
        ("aav", 1000,  "tab:appendix_aav_1000",  "Detailed end-to-end design results on AAV with 1000 training samples."),
        ("aav", 2000,  "tab:appendix_aav_2000",  "Detailed end-to-end design results on AAV with 2000 training samples."),
        ("aav", 5000,  "tab:appendix_aav_5000",  "Detailed end-to-end design results on AAV with 5000 training samples."),
        ("aav", 10000, "tab:appendix_aav_10000", "Detailed end-to-end design results on AAV with 10000 training samples."),
    ]

    master_lines = []

    for dataset, train_size, label, caption in specs:
        tex = build_one_table(
            df=best_df,
            dataset=dataset,
            train_size=train_size,
            label=label,
            caption=caption,
        )

        filename = f"{dataset}_{train_size}_appendix_table.tex"
        out_path = os.path.join(args.output_dir, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)

        master_lines.append(rf"\input{{{os.path.join(args.output_dir, filename)}}}")

        print(f"Saved: {out_path}")
        print()
        print(tex)
        print()

    master_path = os.path.join(args.output_dir, "appendix_tables_all.tex")
    with open(master_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(master_lines))

    print(f"Saved: {master_path}")


if __name__ == "__main__":
    main()
