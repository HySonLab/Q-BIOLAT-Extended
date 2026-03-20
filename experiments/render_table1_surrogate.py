import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifacts/aggregated/full_summary.csv")
    parser.add_argument("--outdir", type=str, default="artifacts/tables")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    # Pivot: rows = samples, cols = dim
    pivot = df.pivot(index="num_samples", columns="dim", values="test_spearman")

    # Sort rows/columns
    pivot = pivot.sort_index()
    pivot = pivot[sorted(pivot.columns)]

    # Save CSV
    csv_path = os.path.join(args.outdir, "table1_surrogate.csv")
    pivot.to_csv(csv_path)

    # ===== Generate LaTeX =====
    dims = list(pivot.columns)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(dims) + "}")
    lines.append("\\hline")

    header = "Samples"
    for d in dims:
        header += f" & Dim={d}"
    header += " \\\\"
    lines.append(header)

    lines.append("\\hline")

    for idx, row in pivot.iterrows():
        line = f"{int(idx)}"
        for d in dims:
            val = row[d]
            if pd.isna(val):
                line += " & --"
            else:
                line += f" & {val:.3f}"
        line += " \\\\"
        lines.append(line)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Test Spearman correlation of the QUBO surrogate across dataset sizes and latent dimensions.}")
    lines.append("\\label{tab:surrogate_spearman}")
    lines.append("\\end{table}")

    latex_table = "\n".join(lines)

    tex_path = os.path.join(args.outdir, "table1_surrogate.tex")
    with open(tex_path, "w") as f:
        f.write(latex_table)

    print("Saved:")
    print(" ", csv_path)
    print(" ", tex_path)

    print("\n=== LaTeX Preview ===\n")
    print(latex_table)


if __name__ == "__main__":
    main()
