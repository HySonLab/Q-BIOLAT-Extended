import argparse
import json
import os

from src.data.dataset import load_dataset, train_test_split_numpy
from src.models.qubo_model import QUBOSurrogate
from src.optimization.simulated_annealing import SimulatedAnnealer
from src.optimization.greedy_hill_climb import greedy_hill_climb
from src.optimization.random_search import random_search
from src.optimization.genetic_algorithm import genetic_algorithm
from src.optimization.latent_bo import LatentBayesSearch
from src.utils.metrics import min_hamming_distance
from src.utils.retrieval import retrieve_nearest_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="examples/synthetic_peptides.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sa-steps", type=int, default=20000)
    parser.add_argument("--random-samples", type=int, default=10000)
    parser.add_argument("--ga-generations", type=int, default=150)
    parser.add_argument("--out", type=str, default="artifacts/optimization_results.json")
    args = parser.parse_args()

    X, y, items = load_dataset(args.data)
    X_train, X_test, y_train, y_test, items_train, items_test = train_test_split_numpy(
        X, y, items, test_ratio=0.2, seed=args.seed
    )

    model = QUBOSurrogate(n_bits=X.shape[1], l2=1e-3).fit(X_train, y_train)

    start_idx = int(y_train.argmax())
    x0 = X_train[start_idx].copy()
    x0_score = float(model.energy(x0))

    sa = SimulatedAnnealer(model, seed=args.seed)
    sa_x, sa_score = sa.run(x0, n_steps=args.sa_steps)

    greedy_x, greedy_score = greedy_hill_climb(model, x0)
    random_x, random_score = random_search(model, n_bits=X.shape[1], n_samples=args.random_samples, seed=args.seed)
    ga_x, ga_score = genetic_algorithm(
        model,
        n_bits=X.shape[1],
        n_generations=args.ga_generations,
        seed=args.seed,
    )
    bo = LatentBayesSearch(seed=args.seed)
    max_seed = min(64, len(X_train))
    bo_x, bo_score, bo_ucb = bo.optimize(model, X_train[:max_seed], y_train[:max_seed])

    results = {
        "start_score": x0_score,
        "start_item": str(items_train[start_idx]),
        "methods": {
            "simulated_annealing": {
                "score": sa_score,
                "improvement": sa_score - x0_score,
                "min_hamming_to_train": min_hamming_distance(sa_x, X_train),
                "nearest_items": retrieve_nearest_items(X_train, items_train, sa_x, top_k=3),
            },
            "greedy_hill_climb": {
                "score": greedy_score,
                "improvement": greedy_score - x0_score,
                "min_hamming_to_train": min_hamming_distance(greedy_x, X_train),
                "nearest_items": retrieve_nearest_items(X_train, items_train, greedy_x, top_k=3),
            },
            "random_search": {
                "score": random_score,
                "improvement": random_score - x0_score,
                "min_hamming_to_train": min_hamming_distance(random_x, X_train),
                "nearest_items": retrieve_nearest_items(X_train, items_train, random_x, top_k=3),
            },
            "genetic_algorithm": {
                "score": ga_score,
                "improvement": ga_score - x0_score,
                "min_hamming_to_train": min_hamming_distance(ga_x, X_train),
                "nearest_items": retrieve_nearest_items(X_train, items_train, ga_x, top_k=3),
            },
            "latent_bo": {
                "score": bo_score,
                "ucb": bo_ucb,
                "improvement": bo_score - x0_score,
                "min_hamming_to_train": min_hamming_distance(bo_x, X_train),
                "nearest_items": retrieve_nearest_items(X_train, items_train, bo_x, top_k=3),
            },
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
