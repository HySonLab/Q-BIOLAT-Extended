import argparse
import json
import os
import re
import pandas as pd


TRAIN_PATTERN = re.compile(r"train_gfp_(\d+)_(\d+)\.json$")
OPT_PATTERN = re.compile(r"optimize_gfp_(\d+)_(\d+)\.json$")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_train_file(filename):
    m = TRAIN_PATTERN.match(filename)
    if not m:
        return None
    num_samples = int(m.group(1))
    dim = int(m.group(2))
    return num_samples, dim


def parse_opt_file(filename):
    m = OPT_PATTERN.match(filename)
    if not m:
        return None
    num_samples = int(m.group(1))
    dim = int(m.group(2))
    return num_samples, dim


def flatten_train_metrics(obj):
    return {
        "model": obj.get("model"),
        "train_rmse": obj.get("train_rmse"),
        "test_rmse": obj.get("test_rmse"),
        "train_r2": obj.get("train_r2"),
        "test_r2": obj.get("test_r2"),
        "train_spearman": obj.get("train_spearman"),
        "test_spearman": obj.get("test_spearman"),
    }


def flatten_opt_metrics(obj):
    row = {
        "start_score": obj.get("start_score"),
        "start_true_fitness": obj.get("start_true_fitness"),
        "best_train_true_fitness": obj.get("best_train_true", {}).get("true_fitness"),
        "best_train_predicted_score": obj.get("best_train_predicted", {}).get("predicted_score"),
        "best_train_predicted_true_fitness": obj.get("best_train_predicted", {}).get("true_fitness"),
    }

    for method, metrics in obj.get("methods", {}).items():
        prefix = method
        row[f"{prefix}_score"] = metrics.get("score")
        row[f"{prefix}_improvement"] = metrics.get("improvement")
        row[f"{prefix}_min_hamming"] = metrics.get("min_hamming_to_train")

        nn = metrics.get("nearest_neighbor", {})
        row[f"{prefix}_nn_true_fitness"] = nn.get("true_fitness")
        row[f"{prefix}_nn_predicted_fitness"] = nn.get("predicted_fitness")
        row[f"{prefix}_nn_percentile"] = nn.get("true_fitness_percentile_in_train")
        row[f"{prefix}_nn_hamming"] = nn.get("hamming_distance")

        if "ucb" in metrics:
            row[f"{prefix}_ucb"] = metrics.get("ucb")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="artifacts/results")
    parser.add_argument("--outdir", type=str, default="artifacts/aggregated")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_rows = []
    opt_rows = []

    for filename in sorted(os.listdir(args.results_dir)):
        path = os.path.join(args.results_dir, filename)

        parsed = parse_train_file(filename)
        if parsed is not None:
            num_samples, dim = parsed
            obj = load_json(path)
            row = {
                "num_samples": num_samples,
                "dim": dim,
                "train_file": filename,
            }
            row.update(flatten_train_metrics(obj))
            train_rows.append(row)
            continue

        parsed = parse_opt_file(filename)
        if parsed is not None:
            num_samples, dim = parsed
            obj = load_json(path)
            row = {
                "num_samples": num_samples,
                "dim": dim,
                "opt_file": filename,
            }
            row.update(flatten_opt_metrics(obj))
            opt_rows.append(row)
            continue

    train_df = pd.DataFrame(train_rows).sort_values(["num_samples", "dim"]).reset_index(drop=True)
    opt_df = pd.DataFrame(opt_rows).sort_values(["num_samples", "dim"]).reset_index(drop=True)

    merged_df = pd.merge(
        train_df,
        opt_df,
        on=["num_samples", "dim"],
        how="outer",
    ).sort_values(["num_samples", "dim"]).reset_index(drop=True)

    train_csv = os.path.join(args.outdir, "train_summary.csv")
    opt_csv = os.path.join(args.outdir, "optimization_summary.csv")
    merged_csv = os.path.join(args.outdir, "full_summary.csv")

    train_df.to_csv(train_csv, index=False)
    opt_df.to_csv(opt_csv, index=False)
    merged_df.to_csv(merged_csv, index=False)

    # Compact view for quick inspection
    compact_cols = [
        "num_samples",
        "dim",
        "test_spearman",
        "test_r2",
        "simulated_annealing_improvement",
        "simulated_annealing_nn_true_fitness",
        "simulated_annealing_nn_percentile",
        "genetic_algorithm_improvement",
        "genetic_algorithm_nn_true_fitness",
        "genetic_algorithm_nn_percentile",
        "random_search_nn_true_fitness",
        "greedy_hill_climb_nn_true_fitness",
        "latent_bo_nn_true_fitness",
    ]
    compact_cols = [c for c in compact_cols if c in merged_df.columns]
    compact_df = merged_df[compact_cols].copy()

    compact_csv = os.path.join(args.outdir, "compact_summary.csv")
    compact_df.to_csv(compact_csv, index=False)

    print("Saved:")
    print(" ", train_csv)
    print(" ", opt_csv)
    print(" ", merged_csv)
    print(" ", compact_csv)

    print("\n=== Compact Summary ===")
    print(compact_df.to_string(index=False))

    # Best configs by test Spearman
    if "test_spearman" in merged_df.columns:
        best_spearman = merged_df.sort_values("test_spearman", ascending=False).head(5)
        best_spearman_csv = os.path.join(args.outdir, "best_by_test_spearman.csv")
        best_spearman.to_csv(best_spearman_csv, index=False)
        print("\nSaved:", best_spearman_csv)
        print("\n=== Top by Test Spearman ===")
        print(best_spearman[["num_samples", "dim", "test_spearman", "test_r2"]].to_string(index=False))

    # Best configs by SA nearest-neighbor true fitness
    col = "simulated_annealing_nn_true_fitness"
    if col in merged_df.columns:
        best_sa = merged_df.sort_values(col, ascending=False).head(5)
        best_sa_csv = os.path.join(args.outdir, "best_by_sa_nn_true_fitness.csv")
        best_sa.to_csv(best_sa_csv, index=False)
        print("\nSaved:", best_sa_csv)
        print("\n=== Top by SA NN True Fitness ===")
        print(
            best_sa[
                ["num_samples", "dim", "test_spearman", "simulated_annealing_improvement", "simulated_annealing_nn_true_fitness", "simulated_annealing_nn_percentile"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
