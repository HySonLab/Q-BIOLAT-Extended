import numpy as np


class SimulatedAnnealer:
    def __init__(self, model, seed: int = 0):
        self.model = model
        self.rng = np.random.default_rng(seed)

    def _delta_energy_flip(self, x: np.ndarray, k: int) -> float:
        old = x[k]
        new = 1 - old
        delta = new - old
        dE = self.model.h[k] * delta
        interaction_sum = np.dot(self.model.J[k], x) - self.model.J[k, k] * x[k]
        dE += delta * interaction_sum
        return float(dE)

    def run(
        self,
        x0: np.ndarray,
        n_steps: int = 20000,
        T0: float = 1.0,
        Tmin: float = 1e-4,
        alpha: float = 0.999,
        keep_best: bool = True,
    ):
        x = np.asarray(x0, dtype=np.int32).copy()
        best_x = x.copy()
        current_E = self.model.energy(x)
        best_E = current_E

        for t in range(n_steps):
            T = max(Tmin, T0 * (alpha ** t))
            k = self.rng.integers(0, len(x))
            dE = self._delta_energy_flip(x, k)

            if dE >= 0:
                accept = True
            else:
                accept_prob = np.exp(dE / max(T, 1e-12))
                accept = self.rng.random() < accept_prob

            if accept:
                x[k] = 1 - x[k]
                current_E += dE
                if keep_best and current_E > best_E:
                    best_E = current_E
                    best_x = x.copy()

        return best_x, float(best_E)
