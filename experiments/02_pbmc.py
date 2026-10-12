"""Real data: interpretable programme activities in human PBMCs (10x PBMC 3k).

1. Train the pathway-masked VAE with MSigDB Hallmark pathways plus canonical
   immune cell-type marker programmes as prior knowledge.
2. Compare it with a dense VAE of the same size, PCA and gene set scoring on
   (a) how much cell-type information the representation keeps,
   (b) held-out reconstruction error, and
   (c) how reproducible each latent dimension is across random seeds.
3. Read biology straight from the latent space: which programmes are active in
   which cell type, and which genes drive them.

Run:  python scripts/download_data.py
      python experiments/02_pbmc.py            (full, ~20 min on a laptop CPU)
      python experiments/02_pbmc.py --quick    (smoke test)
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split

import _style
from _style import DIVERGING, plt
from pathvae import DenseVAE, PathwayVAE, TrainConfig, build_mask, fit, read_gmt, reconstruction_error
from pathvae.baselines import celltype_accuracy, geneset_scores, matched_correlation, pca_embedding, per_programme_correlation
from pathvae.interpret import differential_activity, driver_genes, top_programmes
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA, GS = ROOT / "data" / "raw" / "pbmc3k.h5ad", ROOT / "data" / "gene_sets"
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
N_FREE = 5


def load_pbmc(n_hvg: int = 2000):
    if not DATA.exists():
        raise SystemExit("PBMC data not found. Run `python scripts/download_data.py` first.")
    adata = sc.read_h5ad(DATA)
    raw = adata.raw.to_adata()  # raw layer is log1p(counts) for all 13,714 genes
    X = raw.X.toarray() if sparse.issparse(raw.X) else np.asarray(raw.X)
    counts = np.rint(np.expm1(X))
    ad = sc.AnnData(counts.astype(np.float32), obs=adata.obs[["louvain"]].copy(), var=raw.var[[]].copy())
    ad.obsm["X_umap"] = adata.obsm["X_umap"]
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg)

    gene_sets = {**read_gmt(GS / "h.all.v7.0.symbols.gmt"), **read_gmt(GS / "pbmc_cell_type_markers.gmt")}
    in_sets = {g for genes in gene_sets.values() for g in genes}
    keep = ad.var.highly_variable | ad.var_names.isin(in_sets)
    keep &= np.asarray((ad.X > 0).sum(0)).ravel() >= 3  # expressed in at least 3 cells
    ad = ad[:, keep].copy()
    return ad, gene_sets


UPPER = {"t": "T", "b": "B", "nk": "NK", "cd8": "CD8", "kras": "KRAS", "myc": "MYC", "wnt": "WNT",
         "dna": "DNA", "uv": "UV", "il2": "IL2", "il6": "IL6", "stat5": "STAT5", "stat3": "STAT3", "jak": "JAK",
         "tnfa": "TNFa", "nfkb": "NF-kB", "mtorc1": "mTORC1", "g2m": "G2M", "e2f": "E2F", "tgf": "TGF",
         "pi3k": "PI3K", "akt": "AKT", "mtor": "MTOR", "p53": "p53", "v1": "v1", "v2": "v2"}


def short(name: str) -> str:
    """HALLMARK_INTERFERON_ALPHA_RESPONSE -> 'Interferon alpha response (pathway)'."""
    for prefix, kind in (("HALLMARK_", "pathway"), ("MARKERS_", "marker")):
        if name.startswith(prefix):
            words = [UPPER.get(w, w) for w in name[len(prefix):].lower().split("_")]
            label = " ".join(words)
            return f"{label[0].upper()}{label[1:]} ({kind})"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved results")
    args = ap.parse_args()
    _style.apply()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    seeds = [0] if args.quick else [0, 1, 2]
    epochs = 10 if args.quick else 300

    if args.plot_only:
        ad, _ = load_pbmc()
        saved = np.load(ROOT / "results" / "pbmc_latent_seed0.npz", allow_pickle=True)
        diff = pd.read_csv(TABLES / "pbmc_differential_activity.csv")
        plot_heatmap(diff, pd.read_csv(TABLES / "pbmc_top_programmes.csv"))
        plot_umaps(ad, saved["Z"], list(saved["names"]), ad.obs.louvain.astype(str).values)
        return

    ad, gene_sets = load_pbmc()
    X = np.asarray(ad.X, dtype=np.float32)
    labels = ad.obs.louvain.astype(str).values
    mask = build_mask(gene_sets, ad.var_names, min_genes=5, max_genes=500, n_free=N_FREE)
    K = mask.mask.shape[0]
    print(f"{ad.n_obs} cells x {ad.n_vars} genes; {mask.n_sets} gene sets + {N_FREE} free nodes")

    idx_tr, idx_te = train_test_split(np.arange(ad.n_obs), test_size=0.2, stratify=labels, random_state=0)
    Xt = torch.tensor(X)

    rows, latents = [], {"Pathway VAE (this repo)": [], "Dense VAE": []}
    models = {}
    for seed in seeds:
        cfg = TrainConfig(epochs=epochs, seed=seed)
        pv = PathwayVAE(torch.tensor(mask.mask), n_free=N_FREE)
        fit(pv, X[idx_tr], cfg)
        dv = DenseVAE(ad.n_vars, K)
        fit(dv, X[idx_tr], cfg)
        pca = PCA(K, random_state=seed).fit(X[idx_tr])
        pca_mse = float(((pca.inverse_transform(pca.transform(X[idx_te])) - X[idx_te]) ** 2).mean())
        z_pv, z_dv = pv.latent(Xt).numpy(), dv.latent(Xt).numpy()
        latents["Pathway VAE (this repo)"].append(z_pv)
        latents["Dense VAE"].append(z_dv)
        models[seed] = pv
        for method, Z, mse in [
            ("Pathway VAE (this repo)", z_pv, reconstruction_error(pv, X[idx_te])),
            ("Dense VAE", z_dv, reconstruction_error(dv, X[idx_te])),
            ("PCA", pca.transform(X), pca_mse),
            ("Gene set score", geneset_scores(X, mask), np.nan),
        ]:
            rows.append({"seed": seed, "method": method, "celltype_balanced_acc": celltype_accuracy(Z, labels, seed),
                         "heldout_mse": mse, "interpretable_dims": method in ("Pathway VAE (this repo)", "Gene set score")})
        print(pd.DataFrame(rows[-4:]).round(4).to_string(index=False))

    perf = pd.DataFrame(rows)
    # Seed reproducibility of individual latent dimensions.
    stab = []
    for method, Zs in latents.items():
        for i, j in combinations(range(len(Zs)), 2):
            if method.startswith("Pathway"):
                r = per_programme_correlation(Zs[i][:, : mask.n_sets], Zs[j][:, : mask.n_sets])
            else:
                r = matched_correlation(Zs[i], Zs[j])
            stab.append({"method": method, "seeds": f"{i}-{j}", "median_r": float(np.median(r))})
    stab = pd.DataFrame(stab)

    if args.quick:
        print(perf.groupby("method").mean(numeric_only=True))
        print("quick run finished; results not saved")
        return

    summary = perf.groupby("method", sort=False).agg(
        celltype_balanced_acc=("celltype_balanced_acc", "mean"),
        celltype_balanced_acc_sd=("celltype_balanced_acc", "std"),
        heldout_mse=("heldout_mse", "mean"),
        heldout_mse_sd=("heldout_mse", "std"),
        interpretable_dims=("interpretable_dims", "first"),
    )
    summary = summary.join(stab.groupby("method").median_r.mean().rename("seed_reproducibility_r"))
    perf.to_csv(TABLES / "pbmc_performance_by_seed.csv", index=False)
    summary.to_csv(TABLES / "pbmc_performance_summary.csv")
    print(summary.round(4))

    # --- Interpretation with the seed-0 model --------------------------------------
    pv = models[0]
    Z = latents["Pathway VAE (this repo)"][0]
    names = mask.node_names
    np.savez_compressed(ROOT / "results" / "pbmc_latent_seed0.npz", Z=Z.astype(np.float32), names=np.array(names),
                        cells=np.array(ad.obs_names), cell_type=labels)
    diff = differential_activity(Z[:, : mask.n_sets], labels, mask.set_names)
    diff.to_csv(TABLES / "pbmc_differential_activity.csv", index=False)
    top = top_programmes(diff, n=3)
    top.to_csv(TABLES / "pbmc_top_programmes.csv", index=False)

    W = pv.gene_weights().numpy()
    drivers = []
    for prog in ["HALLMARK_INTERFERON_ALPHA_RESPONSE", "MARKERS_NK_CYTOTOXIC", "HALLMARK_INFLAMMATORY_RESPONSE"]:
        if prog in names:
            for gene, w in driver_genes(W, list(ad.var_names), names, prog, n=8).items():
                drivers.append({"programme": prog, "gene": gene, "weight": round(float(w), 4)})
    pd.DataFrame(drivers).to_csv(TABLES / "pbmc_driver_genes.csv", index=False)

    plot_heatmap(diff, top)
    plot_umaps(ad, Z, names, labels)


def plot_heatmap(diff, top):
    order = ["CD4 T cells", "CD8 T cells", "NK cells", "B cells", "CD14+ Monocytes",
             "FCGR3A+ Monocytes", "Dendritic cells", "Megakaryocytes"]
    progs = list(dict.fromkeys(top.set_index("group").loc[[g for g in order if g in set(top.group)]].programme))
    M = diff.pivot(index="programme", columns="group", values="cohens_d").loc[progs, order]
    vmax = float(np.nanmax(np.abs(M.values)))
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(progs) + 1.6))
    im = ax.imshow(M.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_yticks(range(len(progs)), [short(p) for p in progs])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if abs(v) >= 1:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * vmax else _style.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Cohen's d (cell type vs rest)")
    cb.outline.set_visible(False)
    ax.set_title("Top programmes per cell type, read directly from the latent space")
    fig.savefig(FIGS / "pbmc_programme_heatmap.png")
    plt.close(fig)


def plot_umaps(ad, Z, names, labels):
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list("blues", ["#e9eef5", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
    panels = ["MARKERS_B_CELL", "MARKERS_NK_CYTOTOXIC", "MARKERS_CLASSICAL_MONOCYTE",
              "HALLMARK_INTERFERON_ALPHA_RESPONSE", "MARKERS_PLATELET"]
    panels = [p for p in panels if p in names]
    U = ad.obsm["X_umap"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    axes = axes.ravel()
    ax = axes[0]
    ax.scatter(U[:, 0], U[:, 1], s=3, color="#c9c8c2", linewidths=0)
    nudge = {"Dendritic cells": (30, 22), "Megakaryocytes": (40, -12), "CD14+ Monocytes": (-10, 14)}
    for lab in pd.unique(labels):
        cx, cy = np.median(U[labels == lab], axis=0)
        dx, dy = nudge.get(lab, (0, 0))
        ax.annotate(lab, (cx, cy), xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                    fontsize=8, color=_style.INK,
                    arrowprops=dict(arrowstyle="-", color=_style.INK_2, lw=0.6) if (dx or dy) else None,
                    bbox=dict(boxstyle="round,pad=0.2", fc=_style.SURFACE, ec="none", alpha=0.85))
    ax.set_title("Cell types (reference labels)")
    for ax, prog in zip(axes[1:], panels):
        v = Z[:, names.index(prog)]
        lo, hi = np.percentile(v, [5, 99.5])
        o = np.argsort(v)
        sca = ax.scatter(U[o, 0], U[o, 1], c=v[o], s=3, cmap=blues, vmin=lo, vmax=hi, linewidths=0)
        ax.set_title(short(prog))
        cb = fig.colorbar(sca, ax=ax, shrink=0.7, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[1 + len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Latent node activity on the UMAP: each panel is one named node of the model", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "pbmc_programme_umaps.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
