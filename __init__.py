"""pathvae: a pathway-informed VAE for single-cell data, built from gene sets."""

from .genesets import GeneSetMask, build_mask, read_gmt
from .model import DenseVAE, PathwayVAE
from .train import TrainConfig, fit, reconstruction_error

__all__ = [
    "GeneSetMask",
    "build_mask",
    "read_gmt",
    "PathwayVAE",
    "DenseVAE",
    "TrainConfig",
    "fit",
    "reconstruction_error",
]
__version__ = "0.1.0"
