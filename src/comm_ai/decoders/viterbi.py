from __future__ import annotations

import numpy as np

from comm_ai.codes.trellis import Trellis


def _branch_metric(llr_t: np.ndarray, out_bits: np.ndarray) -> float:
    symbols = 1.0 - 2.0 * out_bits
    return float(np.sum(llr_t * symbols))


def viterbi_decode(llr: np.ndarray, trellis: Trellis) -> np.ndarray:
    n_out = trellis.out_bits.shape[-1]
    T = llr.shape[0] // n_out
    llr_t = llr.reshape(T, n_out)

    num_states = trellis.num_states
    pm = np.full(num_states, -1e18, dtype=np.float64)
    pm[0] = 0.0
    prev_state = np.zeros((T, num_states), dtype=np.int64)
    prev_bit = np.zeros((T, num_states), dtype=np.int64)

    for t in range(T):
        new_pm = np.full(num_states, -1e18, dtype=np.float64)
        for s in range(num_states):
            if pm[s] < -1e17:
                continue
            for b in [0, 1]:
                ns = trellis.next_state[s, b]
                m = pm[s] + _branch_metric(llr_t[t], trellis.out_bits[s, b])
                if m > new_pm[ns]:
                    new_pm[ns] = m
                    prev_state[t, ns] = s
                    prev_bit[t, ns] = b
        pm = new_pm

    state = int(np.argmax(pm))
    u_hat = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        b = prev_bit[t, state]
        u_hat[t] = b
        state = int(prev_state[t, state])
    return u_hat
