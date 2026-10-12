"""Training loop shared by the pathway-masked model and the dense baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import PathwayVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta: float = 0.5  # weight of the KL term
    kl_warmup: int = 50  # epochs over which beta ramps from 0
    l1_free: float = 1e-3  # sparsity on free nodes' decoder weights
    seed: int = 0
    verbose: bool = False


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)


def fit(model: PathwayVAE, X: np.ndarray, cfg: TrainConfig | None = None) -> History:
    """Train ``model`` on a dense (cells x genes) float matrix ``X``."""
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    x_all = torch.as_tensor(np.asarray(X, dtype=np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist = History()
    n = x_all.shape[0]
    for epoch in range(cfg.epochs):
        model.train()
        beta = cfg.beta * min(1.0, (epoch + 1) / max(1, cfg.kl_warmup))
        perm = torch.randperm(n)
        tot = rec_sum = kl_sum = 0.0
        for start in range(0, n, cfg.batch_size):
            xb = x_all[perm[start : start + cfg.batch_size]]
            recon, mu, logvar = model(xb)
            loss, rec, kl = model.loss(xb, recon, mu, logvar, beta=beta, l1_free=cfg.l1_free)
            opt.zero_grad()
            loss.backward()
            opt.step()
            b = xb.shape[0]
            tot += loss.item() * b
            rec_sum += rec.item() * b
            kl_sum += kl.item() * b
        hist.loss.append(tot / n)
        hist.recon.append(rec_sum / n)
        hist.kl.append(kl_sum / n)
        if cfg.verbose and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
            print(f"epoch {epoch:4d}  loss {hist.loss[-1]:.2f}  recon {hist.recon[-1]:.2f}  kl {hist.kl[-1]:.2f}")
    model.eval()
    return hist


@torch.no_grad()
def reconstruction_error(model: PathwayVAE, X: np.ndarray) -> float:
    """Mean squared error per gene on held-out data, using the posterior mean."""
    model.eval()
    x = torch.as_tensor(np.asarray(X, dtype=np.float32))
    recon, _, _ = model(x)
    return float(((recon - x) ** 2).mean())
