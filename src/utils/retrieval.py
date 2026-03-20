import numpy as np


def hamming_distance_matrix(X, x):
    return np.sum(np.abs(X - x[None, :]), axis=1)


def retrieve_nearest_items(X_latent, items, x_star, top_k=5):
    dists = hamming_distance_matrix(X_latent, x_star)
    idx = np.argsort(dists)[:top_k]
    return [(str(items[i]), int(dists[i])) for i in idx]
