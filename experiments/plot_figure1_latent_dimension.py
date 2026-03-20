import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifacts/aggregated/full_summary.csv")
    parser.add_argument("--outdir", type=str, default="artifacts/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.sort_values(["num_samples", "dim"])

    sample_sizes = sorted(df["num_samples"].unique())
    dims = sorted(df["dim"].unique())

    # -------- Figure 1A: latent dimension vs SA NN true fitness --------
    plt.figure(figsize=(6, 4))
    for n in sample_sizes:
        sub = df[df["num_samples"] == n].sort_values("dim")
        plt.plot(
            sub["dim"],
            sub["simulated_annealing_nn_true_fitness"],
            marker="o",
            label=f"{n} samples"
        )

    plt.xlabel("Latent dimension")
    plt.ylabel("SA nearest-neighbor true fitness")
    plt.title("Effect of latent dimension on optimization performance")
    plt.xticks(dims)
    plt.legend()
    plt.tight_layout()

    out1 = os.path.join(args.outdir, "figure1_latent_dimension_sa_nn_fitness.pdf")
    plt.savefig(out1)
    plt.close()

    # -------- Figure 1B: latent dimension vs test Spearman --------
    plt.figure(figsize=(6, 4))
    for n in sample_sizes:
        sub = df[df["num_samples"] == n].sort_values("dim")
        plt.plot(
            sub["dim"],
            sub["test_spearman"],
            marker="o",
            label=f"{n} samples"
        )

    plt.xlabel("Latent dimension")
    plt.ylabel("Test Spearman")
    plt.title("Effect of latent dimension on surrogate performance")
    plt.xticks(dims)
    plt.legend()
    plt.tight_layout()

    out2 = os.path.join(args.outdir, "figure1_latent_dimension_spearman.pdf")
    plt.savefig(out2)
    plt.close()

    print("Saved:")
    print(" ", out1)
    print(" ", out2)


if __name__ == "__main__":
    main()
