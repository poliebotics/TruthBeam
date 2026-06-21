"""Retrieval metrics for verification."""
from __future__ import annotations

import torch


def topk_retrieval(sim: torch.Tensor, k: int = 1) -> float:
    """Top-k accuracy with diagonal positives."""
    n = sim.shape[0]
    topk = sim.topk(k, dim=-1).indices
    target = torch.arange(n, device=sim.device).unsqueeze(-1)
    return (topk == target).any(dim=-1).float().mean().item()


def margin_stats(sim: torch.Tensor) -> dict[str, float]:
    """Diagonal vs best off-diagonal margin."""
    n = sim.shape[0]
    mask = torch.eye(n, dtype=torch.bool, device=sim.device)
    diag = sim.diag()
    sim_off = sim.masked_fill(mask, float("-inf"))
    best_off = sim_off.max(dim=-1).values
    margin = diag - best_off
    return {
        "diag_mean": diag.mean().item(),
        "best_off_mean": best_off.mean().item(),
        "margin_mean": margin.mean().item(),
        "margin_min": margin.min().item(),
        "margin_pct_positive": (margin > 0).float().mean().item(),
    }
