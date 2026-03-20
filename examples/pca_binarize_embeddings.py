import argparse
import os
import numpy as np
from sklearn.decomposition import PCA


def binarize(Z, method="median"):
    if method == "median":
        thresholds = np.median(Z, axis=0, keepdims=True)
        X_bin = (Z > thresholds).astype(np.int32)
    elif method == "sign":
        thresholds = np.zeros((1, Z.shape[1]))
        X_bin = (Z > 0).astype(np.int32)
    else:
        raise ValueError("Unknown binarization method")
    return X_bin, thresholds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--binarize", default="median")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)

    data = np.load(args.input_npz, allow_pickle=True)

    embeddings = data["embeddings"]
    y = data["y"]
    items = data["items"]

    print("Loaded embeddings:", embeddings.shape)

    print("Running PCA →", args.dim, "dimensions")

    pca = PCA(n_components=args.dim)
    Z = pca.fit_transform(embeddings)

    print("Projected shape:", Z.shape)

    X_bin, thresholds = binarize(Z, method=args.binarize)

    print("Binary latent shape:", X_bin.shape)
    print("Mean bit activation:", X_bin.mean())

    np.savez(
        args.output_npz,
        X=X_bin,
        y=y,
        items=items,
        embeddings=embeddings,
        projected=Z,
        thresholds=thresholds,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
    )

    print("Saved:", args.output_npz)


if __name__ == "__main__":
    main()
