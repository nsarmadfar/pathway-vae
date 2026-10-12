import numpy as np
import pytest
import torch

from pathvae import DenseVAE, PathwayVAE, TrainConfig, build_mask, fit, read_gmt
from pathvae.baselines import matched_correlation, per_programme_correlation
from pathvae.interpret import bh_fdr, differential_activity
from pathvae.simulate import corrupt_gene_sets, simulate


def test_read_gmt(tmp_path):
    p = tmp_path / "sets.gmt"
    p.write_text("# comment\nA\tdesc\tG1\tG2\tG2\nB\tdesc\tG3\n")
    sets = read_gmt(p)
    assert sets == {"A": ["G1", "G2"], "B": ["G3"]}


def test_build_mask_filters_and_free_nodes():
    sets = {"small": ["a"], "ok": ["a", "b", "c"], "unknown": ["x", "y", "z"]}
    m = build_mask(sets, ["a", "b", "c", "d"], min_genes=2, n_free=2)
    assert m.set_names == ["ok"]
    assert m.mask.shape == (3, 4)
    assert m.mask[0].tolist() == [1, 1, 1, 0]
    assert m.mask[1:].sum() == 8  # free nodes see every gene
    assert m.node_names == ["ok", "free_0", "free_1"]
    assert m.set_genes("ok") == ["a", "b", "c"]


def test_build_mask_raises_without_overlap():
    with pytest.raises(ValueError):
        build_mask({"s": ["x"] * 20}, ["a", "b"])


def test_decoder_respects_mask():
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float32)
    model = PathwayVAE(mask, hidden=8)
    W = model.gene_weights()
    assert torch.all(W[mask == 0] == 0)
    assert torch.all(W[mask == 1] > 0)  # nonnegative by default
    # gradients cannot leak into masked weights' contribution
    x = torch.randn(5, 4)
    recon, mu, logvar = model(x)
    loss, _, _ = model.loss(x, recon, mu, logvar)
    loss.backward()
    assert torch.all(model.decoder.weight.grad[mask == 0] == 0)


def test_dense_vae_shapes():
    model = DenseVAE(n_genes=6, n_latent=3, hidden=8)
    z = model.latent(torch.randn(4, 6))
    assert z.shape == (4, 3)


def test_simulation_shapes_and_corruption():
    d = simulate(n_cells=100, n_genes=200, n_sets=5, shared_pool=30, seed=1)
    assert d.X.shape == (100, 200) and d.activity.shape == (100, 5)
    noisy = corrupt_gene_sets(d.gene_sets, d.genes, frac=0.5, seed=0)
    for k in d.gene_sets:
        assert len(noisy[k]) == len(d.gene_sets[k])
        shared = len(set(noisy[k]) & set(d.gene_sets[k]))
        assert shared == len(d.gene_sets[k]) - round(0.5 * len(d.gene_sets[k]))


def test_model_recovers_simulated_programmes():
    """End-to-end: a short training run already recovers named programme activity."""
    d = simulate(n_cells=600, n_genes=400, n_sets=8, shared_pool=60, n_hidden=1, seed=0)
    m = build_mask(d.gene_sets, d.genes, n_free=1)
    model = PathwayVAE(torch.tensor(m.mask), n_free=1)
    fit(model, d.X, TrainConfig(epochs=60, seed=0))
    z = model.latent(torch.tensor(d.X)).numpy()[:, : m.n_sets]
    r = per_programme_correlation(z, d.activity)
    assert np.median(r) > 0.8


def test_metrics_and_stats():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(200, 3))
    assert np.allclose(per_programme_correlation(t, t), 1, atol=1e-4)
    shuffled = -t[:, [2, 0, 1]]
    assert np.allclose(matched_correlation(shuffled, t), 1, atol=1e-4)
    q = bh_fdr(np.array([0.01, 0.02, 0.5]))
    assert np.all(q >= [0.01, 0.02, 0.5]) and q.max() <= 1
    groups = np.array(["a"] * 100 + ["b"] * 100)
    lat = np.r_[rng.normal(2, 1, (100, 1)), rng.normal(0, 1, (100, 1))]
    diff = differential_activity(lat, groups, ["prog"])
    row = diff[diff.group == "a"].iloc[0]
    assert row.cohens_d > 1 and row.fdr < 1e-6
