import numpy as np


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.int32)
    y = data["y"].astype(np.float64)
    items = data["items"]
    return X, y, items


def train_test_split_numpy(X, y, items, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        items[train_idx], items[test_idx],
    )
