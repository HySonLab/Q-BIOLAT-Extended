import argparse
import json
import os

from src.data.dataset import load_dataset, train_test_split_numpy
from src.models.qubo_model import QUBOSurrogate
from src.models.mlp_surrogate import MLPRegressorNumpy
from src.utils.metrics import rmse, r2_score, spearman_rank_corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="examples/synthetic_peptides.npz")
    parser.add_argument("--model", type=str, choices=["qubo", "mlp"], default="qubo")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="artifacts/train_metrics.json")
    args = parser.parse_args()

    X, y, items = load_dataset(args.data)
    X_train, X_test, y_train, y_test, items_train, items_test = train_test_split_numpy(
        X, y, items, test_ratio=0.2, seed=args.seed
    )

    if args.model == "qubo":
        model = QUBOSurrogate(n_bits=X.shape[1], l2=args.l2).fit(X_train, y_train)
    else:
        model = MLPRegressorNumpy(input_dim=X.shape[1], seed=args.seed).fit(X_train, y_train, verbose=True)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "model": args.model,
        "train_rmse": rmse(y_train, y_train_pred),
        "test_rmse": rmse(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_spearman": float(spearman_rank_corr(y_train, y_train_pred)),
        "test_spearman": float(spearman_rank_corr(y_test, y_test_pred)),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
