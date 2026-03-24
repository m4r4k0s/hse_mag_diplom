from __future__ import annotations

import numpy as np

from comm_ai.codes.trellis import Trellis


def convolutional_encode(u: np.ndarray, trellis: Trellis, terminate: bool = False) -> np.ndarray:
    state = 0
    outputs = []
    for bit in u.astype(np.int64):
        out = trellis.out_bits[state, bit]
        outputs.append(out)
        state = int(trellis.next_state[state, bit])
    if terminate:
        m = int(np.log2(trellis.num_states))
        for _ in range(m):
            bit = 0
            out = trellis.out_bits[state, bit]
            outputs.append(out)
            state = int(trellis.next_state[state, bit])
    return np.concatenate(outputs)
