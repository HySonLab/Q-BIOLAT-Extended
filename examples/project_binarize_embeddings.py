import argparse
import os
import numpy as np


def random_project(X: np.ndarray, out_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    in_dim = X.shape[1]
    W = rng.normal(loc=0.0, scale=1.0 / np.sqrt(in_dim), size=(in_dim, out_dim))
    Z = X @ W
    return Z, W


def binarize_embeddings(Z: np.ndarray, method: str = "median"):
    if method == "median":
        thresholds = np.median(Z, axis=0, keepdims=True)
        X_bin = (Z > thresholds).astype(np.int32)
    elif method == "sign":
        thresholds = np.zeros((1, Z.shape[1]), dtype=Z.dtype)
        X_bin = (Z > 0.0).astype(np.int32)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    return X_bin, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)
    parser.add_argument("--project-dim", type=int, required=True)
    parser.add_argument("--binarize", type=str, choices=["median", "sign"], default="median")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)

    data = np.load(args.input_npz, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    y = data["y"].astype(np.float64)
    items = data["items"]

    print(f"Loaded dense embeddings: {embeddings.shape}")

    if args.project_dim > 0 and args.project_dim < embeddings.shape[1]:
        projected, projection = random_project(embeddings, out_dim=args.project_dim, seed=args.seed)
        print("Projected embedding shape:", projected.shape)
    else:
        projected = embeddings
        projection = None
        print("Skipping projection; using full embedding dimension.")

    X_bin, thresholds = binarize_embeddings(projected, method=args.binarize)

    print("Binary latent shape:", X_bin.shape)
    print("Mean bit activation:", float(X_bin.mean()))

    np.savez(
        args.output_npz,
        X=X_bin.astype(np.int32),
        y=y.astype(np.float64),
        items=items,
        embeddings=embeddings.astype(np.float32),
        projected=projected.astype(np.float32),
        thresholds=thresholds.astype(np.float32),
        projection=None if projection is None else projection.astype(np.float32),
    )

    print(f"Saved binarized dataset to: {args.output_npz}")


if __name__ == "__main__":
    main()
