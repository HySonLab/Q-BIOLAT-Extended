import numpy as np


def random_search(model, n_bits: int, n_samples: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    best_x = None
    best_score = -np.inf
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=n_bits, dtype=np.int32)
        score = model.energy(x)
        if score > best_score:
            best_score = score
            best_x = x.copy()
    return best_x, float(best_score)
