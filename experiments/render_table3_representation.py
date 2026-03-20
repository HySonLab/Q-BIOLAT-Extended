import argparse
import json
import os
import pandas as pd


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def fmt_pm(mean, std, digits=3):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--summary-csv", default="artifacts/aggregated/full_summary.csv")

    parser.add_argument("--random-json",
                        default="artifacts/multiseed/gfp_10000_16_multiseed.json")

    parser.add_argument("--pca-json",
                        default="artifacts/gfp_10000_pca_16_multiseed.json")

    parser.add_argument("--outdir", default="artifacts/tables")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------
    # Load surrogate Spearman
    # -------------------------
    df = pd.read_csv(args.summary_csv)

    row = df[(df["num_samples"] == 10000) & (df["dim"] == 16)]

    if len(row) == 0:
        raise ValueError("Cannot find (10000, 16) in summary CSV")

    spearman_random = float(row.iloc[0]["test_spearman"])

    # -------------------------
    # Load multiseed JSON
    # -------------------------
    rand_data = load_json(args.random_json)
    pca_data = load_json(args.pca_json)

    def extract(summary):
        m = summary["simulated_annealing"]
        return {
            "nn_mean": m["nearest_neighbor_true_fitness"]["mean"],
            "nn_std": m["nearest_neighbor_true_fitness"]["std"],
            "pct_mean": m["nearest_neighbor_percentile"]["mean"],
            "pct_std": m["nearest_neighbor_percentile"]["std"],
        }

    rand = extract(rand_data["summary"])
    pca = extract(pca_data["summary"])

    # -------------------------
    # Determine best values
    # -------------------------
    best_nn = max(rand["nn_mean"], pca["nn_mean"])
    best_pct = max(rand["pct_mean"], pca["pct_mean"])

    # -------------------------
    # Build LaTeX
    # -------------------------
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Representation} & \\textbf{Spearman} & \\textbf{NN True Fitness} & \\textbf{NN Percentile} \\\\")
    lines.append("\\hline")

    # Random projection row
    nn_rand = fmt_pm(rand["nn_mean"], rand["nn_std"])
    pct_rand = fmt_pm(rand["pct_mean"], rand["pct_std"], digits=2)

    lines.append(
        f"Random Projection & {spearman_random:.3f} & {nn_rand} & {pct_rand} \\\\"
    )

    # PCA row (bold if better)
    nn_pca = fmt_pm(pca["nn_mean"], pca["nn_std"])
    pct_pca = fmt_pm(pca["pct_mean"], pca["pct_std"], digits=2)

    if pca["nn_mean"] == best_nn:
        nn_pca = f"\\textbf{{{nn_pca}}}"
    if pca["pct_mean"] == best_pct:
        pct_pca = f"\\textbf{{{pct_pca}}}"

    lines.append(
        f"PCA Projection & {spearman_random:.3f} & {nn_pca} & {pct_pca} \\\\"
    )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Comparison of latent representation methods on the GFP benchmark (10{,}000 samples, 16 latent bits).}"
    )
    lines.append("\\label{tab:representation}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    out_path = os.path.join(args.outdir, "table3_representation.tex")

    with open(out_path, "w") as f:
        f.write(latex)

    print("Saved:", out_path)
    print("\n=== LaTeX Preview ===\n")
    print(latex)


if __name__ == "__main__":
    main()
