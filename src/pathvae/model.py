"""Pathway-masked variational autoencoder.

The encoder is an ordinary neural network. The decoder is a single *linear*
layer whose weight matrix is multiplied by a fixed binary mask built from prior
knowledge (gene sets / pathways). Latent node k can therefore only reconstruct
the genes of gene set k, which ties every latent dimension to a named biological
programme: its value for a cell is that programme's inferred activity.

Because of this, no separate explanation method is needed to
read the latent space. The idea follows knowledge-primed models such as
VEGA (Seninge et al., 2021) and expiMap (Lotfollahi et al., 2023), in a compact,
self-contained implementation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Module):
    """Linear layer z -> x whose weights are zero outside a fixed mask.

    With ``nonnegative=True`` the weights go through a softplus, so a higher
    latent value always means higher expression of the set's genes and the sign
    of a programme's activity is directly readable.
    """

    def __init__(self, mask: torch.Tensor, nonnegative: bool = True):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.nonnegative = nonnegative
        init = torch.randn_like(self.mask) * 0.01
        if nonnegative:
            init = init - 3.0  # softplus(-3) ~ 0.05: small positive start
        self.weight = nn.Parameter(init)
        self.bias = nn.Parameter(torch.zeros(mask.shape[1]))

    def effective_weight(self) -> torch.Tensor:
        w = F.softplus(self.weight) if self.nonnegative else self.weight
        return w * self.mask

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.effective_weight() + self.bias


class PathwayVAE(nn.Module):
    """VAE with a nonlinear encoder and a gene-set-masked linear decoder.

    Parameters
    ----------
    mask:
        (n_nodes, n_genes) binary tensor. Rows are latent nodes (gene sets
        followed by any free nodes), columns are genes.
    n_free:
        How many of the last rows of ``mask`` are unconstrained "free" nodes.
        They soak up variation not covered by the prior (e.g. batch, cell
        cycle, unannotated programmes) and receive an L1 penalty so they stay
        sparse.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        n_free: int = 0,
        hidden: int = 256,
        dropout: float = 0.1,
        nonnegative: bool = True,
    ):
        super().__init__()
        n_nodes, n_genes = mask.shape
        self.n_nodes, self.n_genes, self.n_free = n_nodes, n_genes, n_free
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, n_nodes)
        self.logvar = nn.Linear(hidden, n_nodes)
        self.decoder = MaskedLinear(torch.as_tensor(mask), nonnegative=nonnegative)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decoder(z), mu, logvar

    def loss(self, x, recon, mu, logvar, beta: float = 1.0, l1_free: float = 0.0):
        rec = F.mse_loss(recon, x, reduction="none").sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
        total = rec + beta * kl
        if self.n_free and l1_free > 0:
            w_free = self.decoder.effective_weight()[-self.n_free :]
            total = total + l1_free * w_free.abs().sum()
        return total, rec.detach(), kl.detach()

    @torch.no_grad()
    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior mean activity of every latent node (cells x nodes)."""
        self.eval()
        return self.encode(x)[0]

    @torch.no_grad()
    def gene_weights(self) -> torch.Tensor:
        """Decoder weights (nodes x genes): how strongly each gene loads on each programme."""
        return self.decoder.effective_weight().detach()


class DenseVAE(PathwayVAE):
    """Black-box baseline: identical architecture but an all-ones mask.

    Same encoder, same latent size, same training, so any difference in
    performance is due to the prior-knowledge mask alone.
    """

    def __init__(self, n_genes: int, n_latent: int, **kwargs):
        kwargs.setdefault("nonnegative", False)
        super().__init__(torch.ones(n_latent, n_genes), n_free=0, **kwargs)
