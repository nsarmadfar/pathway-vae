"""Synthetic single-cell-like data with known ground-truth programme activities.

Real data never tells us the "true" activity of a pathway in each cell, so we
test the method where we do know it. The generator mimics the features that
make real gene sets hard to use:

* gene sets overlap (genes are shared between programmes),
* only part of each annotated set responds in the data,
* some programmes are cell-type specific, others vary continuously,
* some variation comes from programmes that are *not* in the prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimData:
    X: np.ndarray  # cells x genes, log-expression-like
    activity: np.ndarray  # cells x programmes, true activities of annotated programmes
    cell_type: np.ndarray  # cells
    gene_sets: dict[str, list[str]]  # annotated (prior) gene sets
    genes: list[str]


def simulate(
    n_cells: int = 2000,
    n_genes: int = 1500,
    n_sets: int = 30,
    set_size: tuple[int, int] = (20, 60),
    shared_pool: int = 150,
    shared_frac: float = 0.3,
    responsive_frac: float = 0.7,
    n_types: int = 6,
    programmes_per_type: int = 3,
    n_hidden: int = 3,
    noise: float = 0.5,
    seed: int = 0,
) -> SimData:
    rng = np.random.default_rng(seed)
    genes = [f"G{i:04d}" for i in range(n_genes)]
    pool = rng.choice(n_genes, shared_pool, replace=False)  # "hub" genes reused across sets
    rest = np.setdiff1d(np.arange(n_genes), pool)

    members, W = [], np.zeros((n_sets, n_genes))
    for k in range(n_sets):
        size = rng.integers(*set_size, endpoint=True)
        n_shared = int(round(size * shared_frac))
        idx = np.concatenate(
            [rng.choice(pool, n_shared, replace=False), rng.choice(rest, size - n_shared, replace=False)]
        )
        members.append(idx)
        responsive = rng.random(idx.size) < responsive_frac
        W[k, idx[responsive]] = rng.gamma(2.0, 0.5, responsive.sum())

    # Cell types switch on a few programmes each; every programme also varies continuously.
    cell_type = rng.integers(0, n_types, n_cells)
    on = np.zeros((n_types, n_sets))
    for t in range(n_types):
        on[t, rng.choice(n_sets, programmes_per_type, replace=False)] = 1.0
    activity = 1.5 * on[cell_type] + rng.normal(0, 0.5, (n_cells, n_sets))

    # Hidden programmes: real signal that the prior knowledge does not describe.
    W_hidden = np.zeros((n_hidden, n_genes))
    for h in range(n_hidden):
        idx = rng.choice(n_genes, 40, replace=False)
        W_hidden[h, idx] = rng.gamma(2.0, 0.5, idx.size)
    hidden = rng.normal(0, 1, (n_cells, n_hidden))

    baseline = rng.gamma(2.0, 0.5, n_genes)
    X = baseline + activity @ W + hidden @ W_hidden + rng.normal(0, noise, (n_cells, n_genes))
    gene_sets = {f"SET_{k:02d}": [genes[i] for i in members[k]] for k in range(n_sets)}
    return SimData(
        X=X.astype(np.float32),
        activity=activity.astype(np.float32),
        cell_type=np.array([f"type_{t}" for t in cell_type]),
        gene_sets=gene_sets,
        genes=genes,
    )


def corrupt_gene_sets(
    gene_sets: dict[str, list[str]], genes: list[str], frac: float, seed: int = 0
) -> dict[str, list[str]]:
    """Replace a fraction of each set's members with random non-members.

    Mimics imperfect pathway databases (missing and wrongly annotated genes).
    """
    rng = np.random.default_rng(seed)
    universe = np.array(genes)
    out = {}
    for name, members in gene_sets.items():
        members = list(members)
        n_swap = int(round(frac * len(members)))
        if n_swap:
            keep = list(rng.choice(members, len(members) - n_swap, replace=False))
            candidates = np.setdiff1d(universe, members)
            members = keep + list(rng.choice(candidates, n_swap, replace=False))
        out[name] = members
    return out
