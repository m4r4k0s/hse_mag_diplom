from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trellis:
    num_states: int
    next_state: np.ndarray  # [state, input_bit]
    out_bits: np.ndarray  # [state, input_bit, n_outputs]


def octal_to_bits(poly: int, constraint_length: int) -> np.ndarray:
    dec = int(str(poly), 8)
    bits = np.array([(dec >> i) & 1 for i in range(constraint_length)], dtype=np.int64)
    return bits


def build_trellis(constraint_length: int, polynomials: tuple[int, ...]) -> Trellis:
    m = constraint_length - 1
    num_states = 2**m
    n_out = len(polynomials)
    next_state = np.zeros((num_states, 2), dtype=np.int64)
    out_bits = np.zeros((num_states, 2, n_out), dtype=np.int64)
    gens = [octal_to_bits(p, constraint_length) for p in polynomials]

    for state in range(num_states):
        state_bits = np.array([(state >> i) & 1 for i in range(m)], dtype=np.int64)
        for u in [0, 1]:
            shift = np.concatenate(([u], state_bits))
            ns_bits = shift[:-1]
            ns = int(sum(int(ns_bits[i]) << i for i in range(m)))
            next_state[state, u] = ns
            for j, g in enumerate(gens):
                out_bits[state, u, j] = int(np.sum(shift * g) % 2)
    return Trellis(num_states=num_states, next_state=next_state, out_bits=out_bits)
