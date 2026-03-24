from __future__ import annotations

import numpy as np
import torch
from torch import nn

from comm_ai.codes.trellis import Trellis
from comm_ai.decoders.bcjr import bcjr_decode


class LLRCalibrator(nn.Module):
    def __init__(self, n_out: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_out, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, llr_t: torch.Tensor) -> torch.Tensor:
        return self.scale * llr_t + self.net(llr_t)


class NeuralBCJRDecoder(nn.Module):
    def __init__(self, n_out: int, hidden: int = 16) -> None:
        super().__init__()
        self.calibrator = LLRCalibrator(n_out=n_out, hidden=hidden)

    def calibrate(self, llr: torch.Tensor) -> torch.Tensor:
        return self.calibrator(llr)


def neural_bcjr_decode(llr: np.ndarray, trellis: Trellis, model: NeuralBCJRDecoder) -> tuple[np.ndarray, np.ndarray]:
    n_out = trellis.out_bits.shape[-1]
    T = llr.shape[0] // n_out
    with torch.no_grad():
        x = torch.tensor(llr.reshape(T, n_out), dtype=torch.float32)
        x_adj = model.calibrate(x).cpu().numpy().reshape(-1)
    return bcjr_decode(x_adj, trellis)
