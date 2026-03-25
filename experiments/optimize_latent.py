import argparse
import json
import os
import numpy as np

from src.data.dataset import load_dataset, train_test_split_numpy
from src.models.qubo_model import QUBOSurrogate
from src.optimization.simulated_annealing import SimulatedAnnealer
from src.optimization.greedy_hill_climb import greedy_hill_climb
from src.optimization.random_search import random_search
from src.optimization.genetic_algorithm import genetic_algorithm
from src.optimization.latent_bo import LatentBayesSearch
from src.utils.metrics import min_hamming_distance
from src.utils.retrieval import retrieve_nearest_items


def find_nearest_index(X, x):
    dists = np.sum(np.abs(X - x[None, :]), axis=1)
    idx = int(np.argmin(dists))
    return idx, int(dists[idx])


def rank_percentile_desc(values, target):
    values = np.asarray(values, dtype=np.float64)
    rank = 1 + int(np.sum(values > target))
    n = len(values)
    return 100.0 * (n - rank + 1) / n


def summarize_candidate(name, x, score, x0_score, X_train, y_train, items_train, model):
    nn_idx, nn_dist = find_nearest_index(X_train, x)
    nn_item = str(items_train[nn_idx])
    nn_true_fitness = float(y_train[nn_idx])
    nn_pred_fitness = float(model.energy(X_train[nn_idx]))
    nn_percentile = rank_percentile_desc(y_train, nn_true_fitness)

    return {
        "score": float(score),
        "improvement": float(score - x0_score),

        # 🔥 NEW: SAVE OPTIMIZED CODE
        "best_code": [int(v) for v in x.tolist()],

        "min_hamming_to_train": int(min_hamming_distance(x, X_train)),
        "nearest_items": retrieve_nearest_items(X_train, items_train, x, top_k=3),
        "nearest_neighbor": {
            "item": nn_item,
            "hamming_distance": nn_dist,
            "true_fitness": nn_true_fitness,
            "predicted_fitness": nn_pred_fitness,
            "true_fitness_percentile_in_train": nn_percentile,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sa-steps", type=int, default=20000)
    parser.add_argument("--random-samples", type=int, default=10000)
    parser.add_argument("--ga-generations", type=int, default=150)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    X, y, items = load_dataset(args.data)
    X_train, X_test, y_train, y_test, items_train, items_test = train_test_split_numpy(
        X, y, items, test_ratio=0.2, seed=args.seed
    )

    model = QUBOSurrogate(n_bits=X.shape[1], l2=1e-3).fit(X_train, y_train)

    start_idx = int(np.argmax(y_train))
    x0 = X_train[start_idx].copy()
    x0_score = float(model.energy(x0))
    x0_true = float(y_train[start_idx])
    x0_item = str(items_train[start_idx])

    # === optimization ===
    sa = SimulatedAnnealer(model, seed=args.seed)
    sa_x, sa_score = sa.run(x0, n_steps=args.sa_steps)

    greedy_x, greedy_score = greedy_hill_climb(model, x0)

    random_x, random_score = random_search(
        model,
        n_bits=X.shape[1],
        n_samples=args.random_samples,
        seed=args.seed,
    )

    ga_x, ga_score = genetic_algorithm(
        model,
        n_bits=X.shape[1],
        n_generations=args.ga_generations,
        seed=args.seed,
    )

    bo = LatentBayesSearch(seed=args.seed)
    max_seed = min(64, len(X_train))
    bo_x, bo_score, bo_ucb = bo.optimize(
        model,
        X_train[:max_seed],
        y_train[:max_seed],
    )

    results = {
        "dataset": args.data,
        "seed": args.seed,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "methods": {
            "simulated_annealing": summarize_candidate(
                "simulated_annealing", sa_x, sa_score, x0_score, X_train, y_train, items_train, model
            ),
            "greedy_hill_climb": summarize_candidate(
                "greedy_hill_climb", greedy_x, greedy_score, x0_score, X_train, y_train, items_train, model
            ),
            "random_search": summarize_candidate(
                "random_search", random_x, random_score, x0_score, X_train, y_train, items_train, model
            ),
            "genetic_algorithm": summarize_candidate(
                "genetic_algorithm", ga_x, ga_score, x0_score, X_train, y_train, items_train, model
            ),
            "latent_bo": summarize_candidate(
                "latent_bo", bo_x, bo_score, x0_score, X_train, y_train, items_train, model
            ),
        },
    }

    results["methods"]["latent_bo"]["ucb"] = float(bo_ucb)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
