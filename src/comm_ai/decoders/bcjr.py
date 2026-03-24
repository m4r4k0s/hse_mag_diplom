from __future__ import annotations

import numpy as np

from comm_ai.codes.trellis import Trellis


def _logsumexp(arr: np.ndarray) -> float:
    m = np.max(arr)
    return float(m + np.log(np.sum(np.exp(arr - m))))


def _gamma(llr_t: np.ndarray, out_bits: np.ndarray) -> float:
    symbols = 1.0 - 2.0 * out_bits
    return float(np.sum(llr_t * symbols))


def bcjr_decode(llr: np.ndarray, trellis: Trellis) -> tuple[np.ndarray, np.ndarray]:
    n_out = trellis.out_bits.shape[-1]
    T = llr.shape[0] // n_out
    llr_t = llr.reshape(T, n_out)
    S = trellis.num_states

    alpha = np.full((T + 1, S), -1e18)
    beta = np.full((T + 1, S), -1e18)
    alpha[0, 0] = 0.0
    beta[T, :] = 0.0

    gamma = np.zeros((T, S, 2), dtype=np.float64)
    for t in range(T):
        for s in range(S):
            for b in [0, 1]:
                gamma[t, s, b] = _gamma(llr_t[t], trellis.out_bits[s, b])

    for t in range(T):
        for ns in range(S):
            vals = []
            for s in range(S):
                for b in [0, 1]:
                    if trellis.next_state[s, b] == ns:
                        vals.append(alpha[t, s] + gamma[t, s, b])
            alpha[t + 1, ns] = _logsumexp(np.array(vals)) if vals else -1e18

    for t in range(T - 1, -1, -1):
        for s in range(S):
            vals = []
            for b in [0, 1]:
                ns = trellis.next_state[s, b]
                vals.append(gamma[t, s, b] + beta[t + 1, ns])
            beta[t, s] = _logsumexp(np.array(vals))

    llr_u = np.zeros(T, dtype=np.float64)
    for t in range(T):
        num = []
        den = []
        for s in range(S):
            for b in [0, 1]:
                ns = trellis.next_state[s, b]
                val = alpha[t, s] + gamma[t, s, b] + beta[t + 1, ns]
                (num if b == 0 else den).append(val)
        llr_u[t] = _logsumexp(np.array(num)) - _logsumexp(np.array(den))

    hard = (llr_u < 0).astype(np.int64)
    return llr_u, hard
