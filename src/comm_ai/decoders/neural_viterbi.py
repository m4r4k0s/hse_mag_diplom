from __future__ import annotations

import numpy as np
import torch
from torch import nn

from comm_ai.codes.trellis import Trellis
from comm_ai.decoders.viterbi import viterbi_decode


class BranchMetricNet(nn.Module):
    def __init__(self, n_out: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_out, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, llr_t: torch.Tensor) -> torch.Tensor:
        return llr_t + self.net(llr_t)


class NeuralViterbiDecoder(nn.Module):
    def __init__(self, n_out: int, hidden: int = 16, tau: float = 1.0) -> None:
        super().__init__()
        self.metric_net = BranchMetricNet(n_out=n_out, hidden=hidden)
        self.tau = nn.Parameter(torch.tensor(float(tau)))

    def calibrate(self, llr: torch.Tensor) -> torch.Tensor:
        return self.metric_net(llr)


def neural_viterbi_decode(llr: np.ndarray, trellis: Trellis, model: NeuralViterbiDecoder) -> np.ndarray:
    n_out = trellis.out_bits.shape[-1]
    T = llr.shape[0] // n_out
    with torch.no_grad():
        x = torch.tensor(llr.reshape(T, n_out), dtype=torch.float32)
        x_adj = model.calibrate(x).cpu().numpy().reshape(-1)
    return viterbi_decode(x_adj, trellis)
