import argparse
import json
import os
import matplotlib.pyplot as plt


DIMS = [8, 16, 32, 64]
SAMPLES = [5000, 10000]


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
        default="figure1_latent_dimension_shaded.pdf",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for n in SAMPLES:
        nn_means, nn_stds = [], []
        pct_means, pct_stds = [], []

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

            nn_means.append(nn_mean)
            nn_stds.append(nn_std)
            pct_means.append(pct_mean)
            pct_stds.append(pct_std)

        if len(nn_means) != len(DIMS):
            continue

        nn_lower = [m - s for m, s in zip(nn_means, nn_stds)]
        nn_upper = [m + s for m, s in zip(nn_means, nn_stds)]

        pct_lower = [m - s for m, s in zip(pct_means, pct_stds)]
        pct_upper = [m + s for m, s in zip(pct_means, pct_stds)]

        axes[0].plot(DIMS, nn_means, marker="o", linewidth=2, label=f"{n} samples")
        axes[0].fill_between(DIMS, nn_lower, nn_upper, alpha=0.2)

        axes[1].plot(DIMS, pct_means, marker="o", linewidth=2, label=f"{n} samples")
        axes[1].fill_between(DIMS, pct_lower, pct_upper, alpha=0.2)

    axes[0].set_xlabel("Latent dimension")
    axes[0].set_ylabel("NN True Fitness")
    axes[0].set_title("Optimization Performance")
    axes[0].set_xticks(DIMS)

    axes[1].set_xlabel("Latent dimension")
    axes[1].set_ylabel("NN Percentile")
    axes[1].set_title("Retrieved Sequence Quality")
    axes[1].set_xticks(DIMS)

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()

    out_path = os.path.join(args.outdir, args.output_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
