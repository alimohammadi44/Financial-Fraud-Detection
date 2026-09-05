from __future__ import annotations

import torch
import torch.nn.functional as F


def labeled_focal_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha_licit: float = 0.70,
    alpha_illicit: float = 0.29,
    reduction: str = "mean",
):
    """Focal loss over labeled licit/illicit nodes only.

    Elliptic unknown nodes use y=-1 and are deliberately excluded from the
    supervised objective while remaining available for graph propagation.
    """
    mask = y >= 0
    if not mask.any():
        # Preserve gradient connectivity to logits for empty-labeled snapshots.
        zero = logits.sum() * 0.0
        return zero, 0

    logits_l = logits[mask]
    y_l = y[mask].long()
    log_prob = F.log_softmax(logits_l, dim=-1)
    prob = log_prob.exp()
    row = torch.arange(y_l.numel(), device=y.device)
    log_pt = log_prob[row, y_l]
    pt = prob[row, y_l]
    alpha = torch.tensor([alpha_licit, alpha_illicit], device=y.device, dtype=logits.dtype)
    per_node = -alpha[y_l] * (1.0 - pt).pow(gamma) * log_pt

    if reduction == "sum":
        loss = per_node.sum()
    elif reduction == "mean":
        loss = per_node.mean()
    elif reduction == "none":
        loss = per_node
    else:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    return loss, int(y_l.numel())
