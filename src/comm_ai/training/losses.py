from __future__ import annotations

import torch
from torch import nn


def bce_logits_loss(logits: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    """Bit-wise BCE loss for coded bits represented as logits.

    Expected shapes are [batch, time, n_out] or [batch, n_bits].
    """

    return nn.BCEWithLogitsLoss()(logits, target_bits)
