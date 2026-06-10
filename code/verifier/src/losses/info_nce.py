"""Symmetric InfoNCE loss + VICReg-style anti-collapse regularizers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """z_a, z_b: (B, D) L2-normalized. Returns scalar loss.

    Symmetric: average of A->B and B->A cross-entropy with diagonal targets.
    """
    logits_ab = z_a @ z_b.T / tau
    logits_ba = z_b @ z_a.T / tau
    target = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits_ab, target) + F.cross_entropy(logits_ba, target))


def vicreg_variance(h: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Penalize per-dim batch std below target_std. Apply to pre-L2-norm features.

    Without this, contrastive training with small batch / weakly-distinguishable
    features (XOF noise inputs, similar-looking frames) collapses to all-equal
    embeddings within ~1 epoch on this dataset (verified in run.log of the first
    launch attempt).
    """
    std = torch.sqrt(h.var(dim=0) + eps)
    return torch.relu(target_std - std).mean()


def vicreg_covariance(h: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal covariance, decorrelating embedding dimensions."""
    b, d = h.shape
    h_c = h - h.mean(dim=0, keepdim=True)
    cov = (h_c.T @ h_c) / max(b - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / d
