import argparse
import json
import os

METHOD_ORDER = [
    "simulated_annealing",
    "genetic_algorithm",
    "random_search",
    "greedy_hill_climb",
    "latent_bo",
]

METHOD_NAMES = {
    "simulated_annealing": "Simulated Annealing",
    "genetic_algorithm": "Genetic Algorithm",
    "random_search": "Random Search",
    "greedy_hill_climb": "Greedy Hill Climb",
    "latent_bo": "Latent BO",
}


def fmt(x, digits=3):
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def fmt_pm(mean, std, digits=3):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="artifacts/tables")
    parser.add_argument("--caption", default="")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.input, "r") as f:
        data = json.load(f)

    summary = data["summary"]

    # Extract table values
    rows = []
    for m in METHOD_ORDER:
        if m not in summary:
            continue

        rows.append({
            "name": METHOD_NAMES[m],
            "improvement": summary[m]["improvement"],
            "nn_true": summary[m]["nearest_neighbor_true_fitness"],
            "nn_pct": summary[m]["nearest_neighbor_percentile"],
        })

    # Find best values for bolding
    best_improve = max(r["improvement"]["mean"] for r in rows)
    best_nn = max(r["nn_true"]["mean"] for r in rows)
    best_pct = max(r["nn_pct"]["mean"] for r in rows)

    # Build LaTeX
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Method} & \\textbf{Improvement} & \\textbf{NN True Fitness} & \\textbf{NN Percentile} \\\\")
    lines.append("\\hline")

    for r in rows:
        imp = fmt_pm(r["improvement"]["mean"], r["improvement"]["std"])
        nn = fmt_pm(r["nn_true"]["mean"], r["nn_true"]["std"])
        pct = fmt_pm(r["nn_pct"]["mean"], r["nn_pct"]["std"], digits=2)

        if r["improvement"]["mean"] == best_improve:
            imp = f"\\textbf{{{imp}}}"
        if r["nn_true"]["mean"] == best_nn:
            nn = f"\\textbf{{{nn}}}"
        if r["nn_pct"]["mean"] == best_pct:
            pct = f"\\textbf{{{pct}}}"

        lines.append(f"{r['name']} & {imp} & {nn} & {pct} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    if args.caption:
        lines.append(f"\\caption{{{args.caption}}}")
    else:
        lines.append("\\caption{Multi-seed optimization results.}")

    lines.append("\\label{tab:optimization}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    out_path = os.path.join(args.outdir, "table2_optimization.tex")
    with open(out_path, "w") as f:
        f.write(latex)

    print("Saved:", out_path)
    print("\n=== LaTeX Preview ===\n")
    print(latex)


if __name__ == "__main__":
    main()
