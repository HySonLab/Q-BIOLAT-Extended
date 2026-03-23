import argparse
import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


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


def build_model(model_name: str, seed: int):
    if model_name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, solver="svd")),
        ])

    if model_name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed, but model_name='xgboost' was requested.")
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )

    if model_name == "gp":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=seed,
                n_restarts_optimizer=2,
            )),
        ])

    raise ValueError(f"Unknown model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--model-name", type=str, choices=["ridge", "xgboost", "gp"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--output-prefix", type=str, required=True)
    args = parser.parse_args()

    X, y, items = load_dense_npz(args.input_npz)

    X_trainval, X_test, y_trainval, y_test, items_trainval, items_test = train_test_split(
        X, y, items,
        test_size=args.test_size,
        random_state=args.seed,
    )

    val_ratio_adjusted = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val, items_train, items_val = train_test_split(
        X_trainval, y_trainval, items_trainval,
        test_size=val_ratio_adjusted,
        random_state=args.seed,
    )

    model = build_model(args.model_name, args.seed)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics = {
        "input_npz": args.input_npz,
        "model_name": args.model_name,
        "seed": args.seed,
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train": compute_metrics(y_train, pred_train),
        "val": compute_metrics(y_val, pred_val),
        "test": compute_metrics(y_test, pred_test),
    }

    model_path = args.output_prefix + ".pkl"
    metrics_path = args.output_prefix + ".json"
    preds_path = args.output_prefix + "_test_predictions.csv"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "sequence": items_test,
        "y_true": y_test,
        "y_pred": pred_test,
    })
    pred_df.to_csv(preds_path, index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved test predictions to: {preds_path}")


if __name__ == "__main__":
    main()
