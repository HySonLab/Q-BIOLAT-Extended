import numpy as np


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def spearman_rank_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r1 = np.argsort(np.argsort(y_true))
    r2 = np.argsort(np.argsort(y_pred))
    return float(np.corrcoef(r1, r2)[0, 1])


def hamming_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return int(np.sum(np.abs(x - y)))


def min_hamming_distance(x, X):
    X = np.asarray(X)
    return int(np.min(np.sum(np.abs(X - x[None, :]), axis=1)))
