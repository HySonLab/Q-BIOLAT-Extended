import numpy as np


def greedy_hill_climb(model, x0, max_passes: int = 100):
    x = np.asarray(x0, dtype=np.int32).copy()
    current = model.energy(x)

    for _ in range(max_passes):
        improved = False
        best_x = x.copy()
        best_score = current

        for k in range(len(x)):
            cand = x.copy()
            cand[k] = 1 - cand[k]
            score = model.energy(cand)
            if score > best_score:
                best_score = score
                best_x = cand
                improved = True

        x = best_x
        current = best_score
        if not improved:
            break

    return x, float(current)
