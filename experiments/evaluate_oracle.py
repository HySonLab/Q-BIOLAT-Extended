import argparse
import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_dense_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)

    if "embeddings" not in data.files:
        raise ValueError(f"{path} does not contain 'embeddings'")
    if "y" not in data.files:
        raise ValueError(f"{path} does not contain 'y'")
    if "items" not in data.files:
        raise ValueError(f"{path} does not contain 'items'")

    X = data["embeddings"].astype(np.float32)
    y = data["y"].astype(np.float64)
    items = data["items"]

    return X, y, items


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    sp = spearmanr(y_true, y_pred).correlation
    pr = pearsonr(y_true, y_pred)[0]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "spearman": float(sp if sp is not None else np.nan),
        "pearson": float(pr if pr is not None else np.nan),
        "rmse": rmse,
        "mae": mae,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    X, y, items = load_dense_npz(args.input_npz)
    y_pred = model.predict(X)

    metrics = {
        "model_path": args.model_path,
        "input_npz": args.input_npz,
        "n_samples": int(len(y)),
        "metrics": compute_metrics(y, y_pred),
    }

    metrics_path = args.output_prefix + ".json"
    preds_path = args.output_prefix + "_predictions.csv"

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "sequence": items,
        "y_true": y,
        "y_pred": y_pred,
    })
    pred_df.to_csv(preds_path, index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation metrics to: {metrics_path}")
    print(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()
