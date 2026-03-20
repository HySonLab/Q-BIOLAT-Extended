import argparse
import json
import os
import matplotlib.pyplot as plt


DIMS = [8, 16, 32, 64]
SAMPLES = [1000, 2000, 5000, 10000]
OFFSETS = {
    1000: -0.9,
    2000: -0.3,
    5000:  0.3,
    10000: 0.9,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(summary, method, metric_name):
    return summary[method][metric_name]["mean"], summary[method][metric_name]["std"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="artifacts/multiseed")
    parser.add_argument("--outdir", type=str, default="artifacts/figures")
    parser.add_argument("--prefix", type=str, default="gfp")
    parser.add_argument("--suffix", type=str, default="_multiseed.json")
    parser.add_argument(
        "--optimization-method",
        type=str,
        default="simulated_annealing",
        choices=[
            "simulated_annealing",
            "genetic_algorithm",
            "random_search",
            "greedy_hill_climb",
            "latent_bo",
        ],
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="figure1_latent_dimension_errorbars.pdf",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for n in SAMPLES:
        x_dims = []
        nn_means = []
        nn_stds = []
        pct_means = []
        pct_stds = []

        for d in DIMS:
            path = os.path.join(args.results_dir, f"{args.prefix}_{n}_{d}{args.suffix}")
            if not os.path.exists(path):
                print(f"[WARN] Missing file: {path}")
                continue

            data = load_json(path)
            summary = data["summary"]

            nn_mean, nn_std = get_metric(
                summary, args.optimization_method, "nearest_neighbor_true_fitness"
            )
            pct_mean, pct_std = get_metric(
                summary, args.optimization_method, "nearest_neighbor_percentile"
            )

            x_dims.append(d + OFFSETS[n])
            nn_means.append(nn_mean)
            nn_stds.append(nn_std)
            pct_means.append(pct_mean)
            pct_stds.append(pct_std)

        if not x_dims:
            continue

        axes[0].errorbar(
            x_dims,
            nn_means,
            yerr=nn_stds,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=f"{n}",
        )

        axes[1].errorbar(
            x_dims,
            pct_means,
            yerr=pct_stds,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=f"{n}",
        )

    axes[0].set_xlabel("Latent dimension")
    axes[0].set_ylabel("NN True Fitness")
    axes[0].set_title("Optimization Performance")
    axes[0].set_xticks(DIMS)
    axes[0].set_xticklabels([str(d) for d in DIMS])

    axes[1].set_xlabel("Latent dimension")
    axes[1].set_ylabel("NN Percentile")
    axes[1].set_title("Retrieved Sequence Quality")
    axes[1].set_xticks(DIMS)
    axes[1].set_xticklabels([str(d) for d in DIMS])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Samples",
        loc="lower center",
        ncol=len(labels) if labels else 1,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()

    out_path = os.path.join(args.outdir, args.output_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
