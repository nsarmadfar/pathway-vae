"""Comparison methods and evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .genesets import GeneSetMask


def geneset_scores(X: np.ndarray, mask: GeneSetMask) -> np.ndarray:
    """Classic gene set scoring: mean z-scored expression of each set's genes.

    The usual "transparent" alternative to a model (similar in spirit to
    scanpy's score_genes or AUCell). Returns cells x n_sets.
    """
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    M = mask.mask[: mask.n_sets]
    return Z @ M.T / M.sum(1)


def pca_embedding(X: np.ndarray, n_components: int, seed: int = 0) -> np.ndarray:
    return PCA(n_components=n_components, random_state=seed).fit_transform(X)


def per_programme_correlation(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pearson r between estimated and true activity, column by column (named nodes)."""
    e = (est - est.mean(0)) / (est.std(0) + 1e-8)
    t = (truth - truth.mean(0)) / (truth.std(0) + 1e-8)
    return (e * t).mean(0)


def matched_correlation(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Best-case score for unlabelled embeddings (PCA, dense VAE).

    Their dimensions carry no names, so we match each true programme to the
    embedding dimension it correlates with most (Hungarian matching on |r|),
    using the ground truth. This is an upper bound that a user with real data
    could never achieve, which makes it a generous baseline.
    """
    e = (est - est.mean(0)) / (est.std(0) + 1e-8)
    t = (truth - truth.mean(0)) / (truth.std(0) + 1e-8)
    C = np.abs(t.T @ e) / len(t)  # truth x est
    rows, cols = linear_sum_assignment(-C)
    out = np.zeros(truth.shape[1])
    out[rows] = C[rows, cols]
    return out


def celltype_accuracy(Z: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """5-fold cross-validated balanced accuracy of a linear classifier on an embedding."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, Z, labels, cv=cv, scoring="balanced_accuracy").mean())
