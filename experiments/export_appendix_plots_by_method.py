import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "simulated_annealing": "SA",
    "genetic_algorithm": "GA",
    "random_search": "RS",
    "greedy_hill_climb": "GHC",
    "latent_bo": "LBO",
}

LATENT_LABELS = {
    "pca": "PCA",
    "random": "Random",
}


def make_plot(df: pd.DataFrame, dataset: str, train_size: int, output_path: str, include_latent_bo: bool = False):
    sub = df[
        (df["dataset"] == dataset) &
        (df["train_size"] == train_size) &
        (df["latent_type"].isin(["pca", "random"]))
    ].copy()

    if not include_latent_bo:
        sub = sub[sub["method"] != "latent_bo"].copy()

    if len(sub) == 0:
        print(f"[WARN] No rows for dataset={dataset}, train_size={train_size}")
        return

    sub = sub.sort_values(["method", "latent_type", "latent_dim"])

    plt.figure(figsize=(7, 5))

    method_order = [
        "simulated_annealing",
        "genetic_algorithm",
        "random_search",
        "greedy_hill_climb",
    ]
    if include_latent_bo:
        method_order.append("latent_bo")

    latent_order = ["pca", "random"]

    for method in method_order:
        for latent_type in latent_order:
            cur = sub[
                (sub["method"] == method) &
                (sub["latent_type"] == latent_type)
            ].sort_values("latent_dim")

            if len(cur) == 0:
                continue

            x = cur["latent_dim"].values
            y = cur["best_score"].values

            label = f"{METHOD_LABELS[method]} + {LATENT_LABELS[latent_type]}"
            plt.plot(x, y, marker="o", label=label)

    plt.title(f"{dataset.upper()} ({train_size} samples)")
    plt.xlabel("Number of bits")
    plt.ylabel("Best oracle score")
    plt.xticks([8, 16, 32, 64])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-latent-bo", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    specs = [
        ("gfp", 1000),
        ("gfp", 2000),
        ("gfp", 5000),
        ("gfp", 10000),
        ("aav", 1000),
        ("aav", 2000),
        ("aav", 5000),
        ("aav", 10000),
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset, train_size in specs:
        filename = f"{dataset}_{train_size}_methods_plot.png"
        output_path = os.path.join(args.output_dir, filename)
        make_plot(
            df=df,
            dataset=dataset,
            train_size=train_size,
            output_path=output_path,
            include_latent_bo=args.include_latent_bo,
        )


if __name__ == "__main__":
    main()
