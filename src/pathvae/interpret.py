"""Read biology out of a trained model: differential programme activity and driver genes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def differential_activity(
    latent: np.ndarray, groups: pd.Series | np.ndarray, node_names: list[str]
) -> pd.DataFrame:
    """One-vs-rest comparison of every latent programme for every group.

    Returns a long table with Cohen's d (standardised mean difference), a
    two-sided Mann-Whitney U p-value and a BH-adjusted FDR per group.
    """
    groups = np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        inside = groups == g
        a, b = latent[inside], latent[~inside]
        pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-8
        d = (a.mean(0) - b.mean(0)) / pooled
        p = np.array([mannwhitneyu(a[:, k], b[:, k]).pvalue for k in range(latent.shape[1])])
        q = bh_fdr(p)
        for k, name in enumerate(node_names):
            rows.append({"group": g, "programme": name, "cohens_d": d[k], "pval": p[k], "fdr": q[k]})
    return pd.DataFrame(rows).sort_values(["group", "cohens_d"], ascending=[True, False])


def top_programmes(diff: pd.DataFrame, n: int = 3, fdr: float = 0.05) -> pd.DataFrame:
    """The n most up-regulated significant programmes per group."""
    sig = diff[(diff.fdr < fdr) & (diff.cohens_d > 0)]
    return sig.groupby("group", sort=False).head(n).reset_index(drop=True)


def driver_genes(weights: np.ndarray, genes: list[str], node_names: list[str], node: str, n: int = 10) -> pd.Series:
    """Genes with the largest decoder weight in one programme.

    Gene sets list every gene that *can* take part in a pathway; these weights
    show which members actually carry the signal in this dataset.
    """
    k = node_names.index(node)
    w = pd.Series(weights[k], index=genes)
    w = w[w != 0]
    return w.sort_values(ascending=False).head(n)
