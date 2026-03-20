import argparse
import json
import os

from src.data.dataset import load_dataset, train_test_split_numpy
from src.models.qubo_model import QUBOSurrogate
from src.utils.metrics import spearman_rank_corr
from src.analysis.landscape import qubo_landscape_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    X, y, items = load_dataset(args.data)
    X_train, X_test, y_train, y_test, items_train, items_test = train_test_split_numpy(
        X, y, items, test_ratio=args.test_ratio, seed=args.seed
    )

    model = QUBOSurrogate(n_bits=X.shape[1], l2=args.l2).fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    report = qubo_landscape_report(model, n_samples=args.n_samples, seed=args.seed)
    report["dataset"] = args.data
    report["seed"] = int(args.seed)
    report["train_size"] = int(len(X_train))
    report["test_size"] = int(len(X_test))
    report["train_spearman"] = float(spearman_rank_corr(y_train, y_train_pred))
    report["test_spearman"] = float(spearman_rank_corr(y_test, y_test_pred))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
