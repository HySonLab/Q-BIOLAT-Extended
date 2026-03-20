import numpy as np
from itertools import combinations


class QUBOSurrogate:
    # Linear + pairwise QUBO surrogate:
    # y_hat(x) = h^T x + sum_{i<j} J_ij x_i x_j

    def __init__(self, n_bits: int, l2: float = 1e-4):
        self.n_bits = int(n_bits)
        self.l2 = float(l2)
        self.h = None
        self.J = None
        self.pair_indices = list(combinations(range(self.n_bits), 2))

    def _featurize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_bits:
            raise ValueError(f"Expected X shape (N, {self.n_bits}), got {X.shape}")
        linear = X
        pairwise = np.empty((X.shape[0], len(self.pair_indices)), dtype=np.float64)
        for k, (i, j) in enumerate(self.pair_indices):
            pairwise[:, k] = X[:, i] * X[:, j]
        return np.concatenate([linear, pairwise], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        Phi = self._featurize(X)
        D = Phi.shape[1]
        A = Phi.T @ Phi + self.l2 * np.eye(D)
        b = Phi.T @ y
        w = np.linalg.solve(A, b)

        self.h = w[: self.n_bits].copy()
        self.J = np.zeros((self.n_bits, self.n_bits), dtype=np.float64)
        offset = self.n_bits
        for k, (i, j) in enumerate(self.pair_indices):
            self.J[i, j] = w[offset + k]
            self.J[j, i] = w[offset + k]
        np.fill_diagonal(self.J, 0.0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.h is None or self.J is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self.h
        for i, j in self.pair_indices:
            scores += self.J[i, j] * X[:, i] * X[:, j]
        return scores

    def energy(self, x: np.ndarray) -> float:
        if self.h is None or self.J is None:
            raise RuntimeError("Model has not been fitted yet.")
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_bits:
            raise ValueError(f"Expected vector of length {self.n_bits}, got {x.shape[0]}")
        return float(x @ self.h + 0.5 * x @ self.J @ x)

    def save(self, path: str) -> None:
        if self.h is None or self.J is None:
            raise RuntimeError("Model has not been fitted yet.")
        np.savez(path, n_bits=self.n_bits, l2=self.l2, h=self.h, J=self.J)

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        model = cls(int(data["n_bits"]), float(data["l2"]))
        model.h = data["h"]
        model.J = data["J"]
        return model
