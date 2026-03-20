import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_peptide(rng, length=12):
    return "".join(rng.choice(list(AMINO_ACIDS), size=length))


def generate_dataset(
    N=300,
    M=16,
    peptide_length=12,
    noise_std=0.1,
    interaction_scale=0.2,
    sparse_interactions=False,
    interaction_keep_prob=0.25,
    seed=0,
):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(N, M), dtype=np.int32)

    true_h = rng.normal(size=M)
    true_J = rng.normal(scale=interaction_scale, size=(M, M))
    true_J = np.triu(true_J, 1)
    true_J = true_J + true_J.T
    np.fill_diagonal(true_J, 0.0)

    if sparse_interactions:
        mask = rng.random(size=(M, M)) < interaction_keep_prob
        mask = np.triu(mask, 1)
        mask = mask + mask.T
        true_J = true_J * mask

    y = X @ true_h + 0.5 * np.sum((X @ true_J) * X, axis=1)
    y += noise_std * rng.normal(size=N)

    peptides = np.array([random_peptide(rng, length=peptide_length) for _ in range(N)], dtype=object)
    return X, y, peptides, true_h, true_J


def save_dataset(path, **kwargs):
    X, y, items, true_h, true_J = generate_dataset(**kwargs)
    np.savez(path, X=X, y=y, items=items, true_h=true_h, true_J=true_J)
