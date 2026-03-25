import argparse
import json
import os
import subprocess
import sys
import tempfile
from statistics import mean, stdev


METHODS = [
    "simulated_annealing",
    "genetic_algorithm",
    "random_search",
    "greedy_hill_climb",
    "latent_bo",
]


def safe_mean(values):
    return mean(values) if len(values) > 0 else None


def safe_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def run_one_seed(data_path, seed, sa_steps, random_samples, ga_generations):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        sys.executable,
        "experiments/optimize_latent.py",
        "--data", data_path,
        "--seed", str(seed),
        "--sa-steps", str(sa_steps),
        "--random-samples", str(random_samples),
        "--ga-generations", str(ga_generations),
        "--out", tmp_path,
    ]

    print(f"\n[RUN] seed={seed}")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    with open(tmp_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    os.remove(tmp_path)
    return result


def maybe_copy_code_fields(method_result):
    """
    Preserve optimized codes if optimize_latent.py returns them.
    Supports several common field names.
    """
    out = {}

    for key in [
        "best_code",
        "best_x",
        "x",
        "code",
        "z_bin",
        "top_codes",
        "candidate_codes",
    ]:
        if key in method_result:
            out[key] = method_result[key]

    return out


def extract_metrics(result):
    extracted = {}
    for method in METHODS:
        m = result["methods"][method]

        extracted[method] = {
            "score": m["score"],
            "improvement": m["improvement"],
            "min_hamming_to_train": m["min_hamming_to_train"],
            "nearest_neighbor_true_fitness": m["nearest_neighbor"]["true_fitness"],
            "nearest_neighbor_predicted_fitness": m["nearest_neighbor"]["predicted_fitness"],
            "nearest_neighbor_percentile": m["nearest_neighbor"]["true_fitness_percentile_in_train"],
        }

        if method == "latent_bo" and "ucb" in m:
            extracted[method]["ucb"] = m["ucb"]

        # Also preserve optimized codes if available
        extracted[method].update(maybe_copy_code_fields(m))

    return extracted


def summarize_across_seeds(all_results):
    summary = {}

    for method in METHODS:
        summary[method] = {}

        metric_names = [
            k for k in all_results[0][method].keys()
            if k not in {"best_code", "best_x", "x", "code", "z_bin", "top_codes", "candidate_codes"}
        ]

        for metric in metric_names:
            values = [run[method][metric] for run in all_results]
            summary[method][metric] = {
                "mean": safe_mean(values),
                "std": safe_std(values),
                "values": values,
            }

    return summary


def rank_methods(summary, key="nearest_neighbor_true_fitness"):
    ranking = []
    for method in METHODS:
        val = summary[method][key]["mean"]
        ranking.append((method, val))
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


def build_decoder_codes(raw_runs):
    """
    Collect optimized binary codes across seeds/methods in a flat structure
    that decode_optimized_latents.py can use later.

    Returns:
      {
        "simulated_annealing": [[...], [...], ...],
        "genetic_algorithm": [[...], ...],
        ...
      }
    """
    out = {method: [] for method in METHODS}

    for run in raw_runs:
        methods = run.get("methods", {})
        for method in METHODS:
            if method not in methods:
                continue
            m = methods[method]

            # single best code
            for key in ["best_code", "best_x", "x", "code", "z_bin"]:
                if key in m:
                    out[method].append(m[key])
                    break

            # optional multiple codes
            for key in ["top_codes", "candidate_codes"]:
                if key in m and isinstance(m[key], list):
                    for code in m[key]:
                        out[method].append(code)

    # remove empty methods
    out = {k: v for k, v in out.items() if len(v) > 0}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sa-steps", type=int, default=20000)
    parser.add_argument("--random-samples", type=int, default=10000)
    parser.add_argument("--ga-generations", type=int, default=150)
    parser.add_argument("--out", type=str, default="artifacts/multiseed_benchmark.json")
    args = parser.parse_args()

    raw_runs = []
    extracted_runs = []

    for seed in args.seeds:
        result = run_one_seed(
            data_path=args.data,
            seed=seed,
            sa_steps=args.sa_steps,
            random_samples=args.random_samples,
            ga_generations=args.ga_generations,
        )
        raw_runs.append(result)
        extracted_runs.append(extract_metrics(result))

    summary = summarize_across_seeds(extracted_runs)
    decoder_codes = build_decoder_codes(raw_runs)

    output = {
        "dataset": args.data,
        "seeds": args.seeds,
        "summary": summary,
        "ranking_by_nearest_neighbor_true_fitness": rank_methods(
            summary, key="nearest_neighbor_true_fitness"
        ),
        "ranking_by_improvement": rank_methods(
            summary, key="improvement"
        ),
        "decoder_codes": decoder_codes,
        "raw_runs": raw_runs,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Multi-seed Summary ===")
    for method in METHODS:
        imp_mean = summary[method]["improvement"]["mean"]
        imp_std = summary[method]["improvement"]["std"]
        fit_mean = summary[method]["nearest_neighbor_true_fitness"]["mean"]
        fit_std = summary[method]["nearest_neighbor_true_fitness"]["std"]
        pct_mean = summary[method]["nearest_neighbor_percentile"]["mean"]
        pct_std = summary[method]["nearest_neighbor_percentile"]["std"]

        print(
            f"{method:20s} | "
            f"improvement = {imp_mean:.4f} ± {imp_std:.4f} | "
            f"nn_true_fitness = {fit_mean:.4f} ± {fit_std:.4f} | "
            f"nn_percentile = {pct_mean:.2f} ± {pct_std:.2f}"
        )

    if len(decoder_codes) == 0:
        print("\n[WARN] No optimized binary codes were found in optimize_latent.py outputs.")
        print("       Retrieval benchmarking still works, but decoder-based evaluation will need code export.")

    print(f"\nSaved benchmark summary to: {args.out}")


if __name__ == "__main__":
    main()
