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

METHOD_DISPLAY = {
    "simulated_annealing": "Simulated Annealing",
    "genetic_algorithm": "Genetic Algorithm",
    "random_search": "Random Search",
    "greedy_hill_climb": "Greedy Hill Climbing",
    "latent_bo": "Latent BO",
}


def fmt_mean_std(metric_dict, digits=3):
    mean_val = metric_dict["mean"]
    std_val = metric_dict["std"]
    if mean_val is None:
        return "N/A"
    return f"{mean_val:.{digits}f} $\\pm$ {std_val:.{digits}f}"


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj


def make_latex_table(summary_obj, caption, label):
    summary = summary_obj["summary"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("Method & Surrogate Improvement & NN True Fitness & NN Percentile \\\\")
    lines.append("\\hline")

    for method in METHOD_ORDER:
        if method not in summary:
            continue

        row_name = METHOD_DISPLAY.get(method, method)
        improvement = fmt_mean_std(summary[method]["improvement"], digits=3)
        nn_true = fmt_mean_std(summary[method]["nearest_neighbor_true_fitness"], digits=3)
        nn_pct = fmt_mean_std(summary[method]["nearest_neighbor_percentile"], digits=2)

        lines.append(f"{row_name} & {improvement} & {nn_true} & {nn_pct} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def make_markdown_table(summary_obj):
    summary = summary_obj["summary"]

    lines = []
    lines.append("| Method | Surrogate Improvement | NN True Fitness | NN Percentile |")
    lines.append("|---|---:|---:|---:|")

    for method in METHOD_ORDER:
        if method not in summary:
            continue

        row_name = METHOD_DISPLAY.get(method, method)
        improvement = fmt_mean_std(summary[method]["improvement"], digits=3)
        nn_true = fmt_mean_std(summary[method]["nearest_neighbor_true_fitness"], digits=3)
        nn_pct = fmt_mean_std(summary[method]["nearest_neighbor_percentile"], digits=2)

        lines.append(f"| {row_name} | {improvement} | {nn_true} | {nn_pct} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to multiseed benchmark JSON.")
    parser.add_argument("--outdir", type=str, default="artifacts/tables")
    parser.add_argument(
        "--caption",
        type=str,
        default="Multi-seed optimization results on the GFP benchmark using ESM-derived binary latent codes.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:gfp_multiseed_results",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    summary_obj = load_summary(args.input)

    latex_table = make_latex_table(
        summary_obj=summary_obj,
        caption=args.caption,
        label=args.label,
    )
    markdown_table = make_markdown_table(summary_obj)

    latex_path = os.path.join(args.outdir, "multiseed_results.tex")
    markdown_path = os.path.join(args.outdir, "multiseed_results.md")

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table + "\n")

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_table + "\n")

    print("Saved LaTeX table to:", latex_path)
    print("Saved Markdown table to:", markdown_path)
    print("\n=== LaTeX Preview ===\n")
    print(latex_table)
    print("\n=== Markdown Preview ===\n")
    print(markdown_table)


if __name__ == "__main__":
    main()
