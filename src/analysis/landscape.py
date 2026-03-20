import numpy as np


def bit_flip_gain(model, x: np.ndarray, k: int) -> float:
    """
    Exact fitness gain after flipping bit k:
        Delta_k(x) = f(x^(k)) - f(x)
    """
    x = np.asarray(x, dtype=np.int32).copy()
    x_flip = x.copy()
    x_flip[k] = 1 - x_flip[k]
    return float(model.energy(x_flip) - model.energy(x))


def sample_bit_flip_statistics(model, n_samples: int = 256, seed: int = 0):
    """
    Sample random binary codes and estimate bit-flip ruggedness statistics.
    """
    rng = np.random.default_rng(seed)
    m = model.n_bits

    deltas = []
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=m, dtype=np.int32)
        row = [bit_flip_gain(model, x, k) for k in range(m)]
        deltas.append(row)

    deltas = np.asarray(deltas, dtype=np.float64)

    return {
        "bit_flip_mean": float(np.mean(deltas)),
        "bit_flip_std": float(np.std(deltas)),
        "bit_flip_var": float(np.var(deltas)),
        "mean_abs_bit_flip_gain": float(np.mean(np.abs(deltas))),
        "max_abs_bit_flip_gain": float(np.max(np.abs(deltas))),
        "per_bit_variance_mean": float(np.mean(np.var(deltas, axis=0))),
        "per_sample_variance_mean": float(np.mean(np.var(deltas, axis=1))),
    }


def spectral_diagnostics(J: np.ndarray):
    """
    Spectral and matrix diagnostics of the QUBO interaction matrix J.
    """
    J = np.asarray(J, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(J)
    abs_eigvals = np.sort(np.abs(eigvals))[::-1]

    spectral_norm = float(abs_eigvals[0]) if len(abs_eigvals) > 0 else 0.0
    fro_norm = float(np.linalg.norm(J, ord="fro"))
    inf_norm = float(np.linalg.norm(J, ord=np.inf))
    row_norms = np.linalg.norm(J, axis=1)

    effective_rank = float((fro_norm ** 2) / (spectral_norm ** 2 + 1e-12))

    energy_total = float(np.sum(abs_eigvals ** 2))
    top1_ratio = float((abs_eigvals[0] ** 2) / (energy_total + 1e-12)) if len(abs_eigvals) >= 1 else 0.0
    top3_ratio = float(np.sum(abs_eigvals[:3] ** 2) / (energy_total + 1e-12)) if len(abs_eigvals) >= 3 else top1_ratio
    top5_ratio = float(np.sum(abs_eigvals[:5] ** 2) / (energy_total + 1e-12)) if len(abs_eigvals) >= 5 else top3_ratio

    return {
        "spectral_norm": spectral_norm,
        "frobenius_norm": fro_norm,
        "infinity_norm": inf_norm,
        "effective_rank": effective_rank,
        "mean_row_norm": float(np.mean(row_norms)),
        "max_row_norm": float(np.max(row_norms)),
        "min_eigenvalue": float(np.min(eigvals)),
        "max_eigenvalue": float(np.max(eigvals)),
        "top1_energy_ratio": top1_ratio,
        "top3_energy_ratio": top3_ratio,
        "top5_energy_ratio": top5_ratio,
        "eigenvalues": eigvals.tolist(),
    }


def qubo_landscape_report(model, n_samples: int = 256, seed: int = 0):
    """
    Full theoretical-diagnostics report for a fitted QUBO surrogate.
    """
    report = {}
    report.update(spectral_diagnostics(model.J))
    report.update(sample_bit_flip_statistics(model, n_samples=n_samples, seed=seed))
    report["n_bits"] = int(model.n_bits)
    report["l2"] = float(model.l2)
    return report
