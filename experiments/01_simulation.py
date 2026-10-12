"""In silico experiments: does the model recover known programme activities?

Three questions, each on synthetic data where the true activity of every
annotated programme in every cell is known:

  A. Benchmark: how well do different methods recover programme activity?
  B. Robustness: what happens when the prior knowledge is partly wrong?
  C. Overlap: what happens as gene sets share more genes?

Run:  python experiments/01_simulation.py            (full, ~15 min on a laptop CPU)
      python experiments/01_simulation.py --quick    (smoke test)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _style
from _style import METHOD_COLORS, plt
from pathvae import DenseVAE, PathwayVAE, TrainConfig, build_mask, fit
from pathvae.baselines import geneset_scores, matched_correlation, pca_embedding, per_programme_correlation
from pathvae.simulate import corrupt_gene_sets, simulate

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
N_FREE = 3


def pathway_vae_activity(X, gene_sets, genes, epochs, seed):
    mask = build_mask(gene_sets, genes, n_free=N_FREE)
    model = PathwayVAE(torch.tensor(mask.mask), n_free=N_FREE)
    fit(model, X, TrainConfig(epochs=epochs, seed=seed))
    z = model.latent(torch.tensor(X)).numpy()
    return z[:, : mask.n_sets], mask


def benchmark(seeds, epochs):
    rows = []
    for seed in seeds:
        d = simulate(seed=seed)
        z, mask = pathway_vae_activity(d.X, d.gene_sets, d.genes, epochs, seed)
        n_latent = mask.n_sets + N_FREE
        dense = DenseVAE(len(d.genes), n_latent)
        fit(dense, d.X, TrainConfig(epochs=epochs, seed=seed))
        results = {
            "Pathway VAE (this repo)": per_programme_correlation(z, d.activity),
            "Gene set score": per_programme_correlation(geneset_scores(d.X, mask), d.activity),
            "PCA (best match)": matched_correlation(pca_embedding(d.X, n_latent, seed), d.activity),
            "Dense VAE (best match)": matched_correlation(dense.latent(torch.tensor(d.X)).numpy(), d.activity),
        }
        for method, r in results.items():
            rows += [{"seed": seed, "method": method, "programme": k, "r": v} for k, v in enumerate(r)]
        print(f"[A] seed {seed}: " + ", ".join(f"{m}={np.median(r):.3f}" for m, r in results.items()))
    return pd.DataFrame(rows)


def robustness(seeds, epochs, fracs):
    rows = []
    for seed in seeds:
        d = simulate(seed=seed)
        for frac in fracs:
            noisy = corrupt_gene_sets(d.gene_sets, d.genes, frac, seed=seed)
            z, mask = pathway_vae_activity(d.X, noisy, d.genes, epochs, seed)
            for method, est in [("Pathway VAE (this repo)", z), ("Gene set score", geneset_scores(d.X, mask))]:
                rows.append({"seed": seed, "frac_wrong": frac, "method": method,
                             "median_r": float(np.median(per_programme_correlation(est, d.activity)))})
            print(f"[B] seed {seed} wrong={frac:.1f}: " + ", ".join(f"{r['method']}={r['median_r']:.3f}" for r in rows[-2:]))
    return pd.DataFrame(rows)


def overlap(seeds, epochs, shares):
    rows = []
    for seed in seeds:
        for share in shares:
            d = simulate(seed=seed, shared_frac=share)
            z, mask = pathway_vae_activity(d.X, d.gene_sets, d.genes, epochs, seed)
            for method, est in [("Pathway VAE (this repo)", z), ("Gene set score", geneset_scores(d.X, mask))]:
                rows.append({"seed": seed, "shared_frac": share, "method": method,
                             "median_r": float(np.median(per_programme_correlation(est, d.activity)))})
            print(f"[C] seed {seed} shared={share:.1f}: " + ", ".join(f"{r['method']}={r['median_r']:.3f}" for r in rows[-2:]))
    return pd.DataFrame(rows)


def plot_benchmark(df):
    methods = list(METHOD_COLORS)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    data = [df[df.method == m].r.values for m in methods]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=_style.INK, linewidth=2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set(facecolor=METHOD_COLORS[m], alpha=0.35, edgecolor=METHOD_COLORS[m], linewidth=1.5)
    rng = np.random.default_rng(0)
    for i, (vals, m) in enumerate(zip(data, methods), start=1):
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, vals.size), s=8, color=METHOD_COLORS[m], alpha=0.6, linewidths=0)
        ax.text(1.005, i, f"median {np.median(vals):.2f}", va="center", ha="left", color=_style.INK_2,
                transform=ax.get_yaxis_transform(), fontsize=9)
    ax.set_yticks(range(1, len(methods) + 1), methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=_style.GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Correlation with true programme activity (Pearson r, per programme)")
    ax.set_title("A. Recovery of known programme activities")
    fig.savefig(FIGS / "sim_A_benchmark.png")
    plt.close(fig)


def plot_sweep(df, xcol, xlabel, title, fname):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for method in ["Pathway VAE (this repo)", "Gene set score"]:
        g = df[df.method == method].groupby(xcol).median_r
        m, lo, hi = g.mean(), g.min(), g.max()
        c = METHOD_COLORS[method]
        ax.fill_between(m.index, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(m.index, m.values, color=c, marker="o", markersize=6)
        ax.annotate(method, (m.index[-1], m.values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=_style.INK, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median Pearson r with truth")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title)
    ax.margins(x=0.02)
    fig.savefig(FIGS / fname)
    plt.close(fig)


def plot_all(a, b, c):
    plot_benchmark(a)
    plot_sweep(b, "frac_wrong", "Fraction of each gene set replaced by wrong genes",
               "B. Robustness to errors in the prior", "sim_B_robustness.png")
    plot_sweep(c, "shared_frac", "Fraction of each gene set shared with other sets",
               "C. Disentangling overlapping gene sets", "sim_C_overlap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved tables")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    _style.apply()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        a, b, c = (pd.read_csv(TABLES / f) for f in ["sim_A_benchmark.csv", "sim_B_robustness.csv", "sim_C_overlap.csv"])
        plot_all(a, b, c)
        return

    seeds = [0] if args.quick else [0, 1, 2]
    epochs = 15 if args.quick else 100
    fracs = [0.0, 0.5] if args.quick else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    shares = [0.0, 0.6] if args.quick else [0.0, 0.2, 0.4, 0.6, 0.8]

    a = benchmark(seeds, epochs)
    b = robustness(seeds, epochs, fracs)
    c = overlap(seeds, epochs, shares)
    if args.quick:
        print("quick run finished; results not saved")
        return
    a.to_csv(TABLES / "sim_A_benchmark.csv", index=False)
    b.to_csv(TABLES / "sim_B_robustness.csv", index=False)
    c.to_csv(TABLES / "sim_C_overlap.csv", index=False)
    plot_all(a, b, c)
    print(a.groupby("method").r.median().round(3))


if __name__ == "__main__":
    main()
