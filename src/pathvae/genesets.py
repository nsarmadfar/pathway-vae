"""Gene set handling: read GMT files and turn gene sets into a binary decoder mask."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    """Read a GMT file into {set_name: [genes]}.

    Each line is: name <tab> description <tab> gene1 <tab> gene2 ...
    Lines starting with '#' are ignored.
    """
    sets: dict[str, list[str]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 3:
                continue
            name, genes = parts[0], [g for g in parts[2:] if g]
            sets[name] = list(dict.fromkeys(genes))  # dedupe, keep order
    return sets


@dataclass
class GeneSetMask:
    """Binary membership matrix linking latent nodes to genes.

    mask[k, g] == 1 means latent node k may use gene g in the decoder.
    The first ``n_sets`` rows are annotated gene sets; the remaining ``n_free``
    rows are unconstrained nodes that can use any gene (and are L1-penalised).
    """

    mask: np.ndarray
    set_names: list[str]
    genes: list[str]
    n_free: int = 0

    @property
    def n_sets(self) -> int:
        return len(self.set_names)

    @property
    def node_names(self) -> list[str]:
        return self.set_names + [f"free_{i}" for i in range(self.n_free)]

    def set_genes(self, name: str) -> list[str]:
        k = self.set_names.index(name)
        return [g for g, m in zip(self.genes, self.mask[k]) if m]


def build_mask(
    gene_sets: dict[str, Iterable[str]],
    genes: Iterable[str],
    min_genes: int = 10,
    max_genes: int = 500,
    n_free: int = 0,
) -> GeneSetMask:
    """Intersect gene sets with the measured genes and build the decoder mask.

    Sets with fewer than ``min_genes`` or more than ``max_genes`` measured genes
    are dropped, as is usual for gene set analyses.
    """
    genes = list(genes)
    index = {g: i for i, g in enumerate(genes)}
    rows, names = [], []
    for name, members in gene_sets.items():
        idx = sorted({index[g] for g in members if g in index})
        if min_genes <= len(idx) <= max_genes:
            row = np.zeros(len(genes), dtype=np.float32)
            row[idx] = 1.0
            rows.append(row)
            names.append(name)
    if not rows:
        raise ValueError("No gene set passed the size filter; check gene identifiers.")
    mask = np.vstack(rows)
    if n_free:
        mask = np.vstack([mask, np.ones((n_free, len(genes)), dtype=np.float32)])
    return GeneSetMask(mask=mask, set_names=names, genes=genes, n_free=n_free)


def genes_in_sets(gene_sets: dict[str, Iterable[str]]) -> set[str]:
    """Union of all genes across gene sets."""
    out: set[str] = set()
    for members in gene_sets.values():
        out.update(members)
    return out
