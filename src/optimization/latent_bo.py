import numpy as np


class LatentBayesSearch:
    # Lightweight Bayesian-style baseline.

    def __init__(self, beta: float = 1.0, length_scale: float = 4.0, seed: int = 0):
        self.beta = float(beta)
        self.length_scale = float(length_scale)
        self.rng = np.random.default_rng(seed)

    def _kernel(self, X, x):
        d = np.sum(np.abs(X - x[None, :]), axis=1)
        return np.exp(-(d ** 2) / (2.0 * self.length_scale ** 2))

    def optimize(self, model, X_seed, y_seed=None, n_candidates: int = 5000):
        X_seed = np.asarray(X_seed, dtype=np.int32)
        n_bits = X_seed.shape[1]

        if y_seed is None:
            y_seed = np.array([model.energy(x) for x in X_seed], dtype=np.float64)
        else:
            y_seed = np.asarray(y_seed, dtype=np.float64)

        cand = self.rng.integers(0, 2, size=(n_candidates, n_bits), dtype=np.int32)
        best_ucb = -np.inf
        best_x = None

        for x in cand:
            w = self._kernel(X_seed, x)
            w_sum = np.sum(w) + 1e-8
            mu = float(np.sum(w * y_seed) / w_sum)
            sigma = float(np.sqrt(max(1e-8, 1.0 / w_sum)))
            ucb = mu + self.beta * sigma
            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x.copy()

        return best_x, float(model.energy(best_x)), float(best_ucb)
