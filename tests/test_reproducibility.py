from pathlib import Path

import numpy as np

from comm_ai.datasets.signals_dataset import SignalsDataset, generate_and_save


def test_reproducibility_saved_signals(tmp_path: Path) -> None:
    cfg = {
        "code": {"constraint_length": 7, "polynomials": [171, 133], "rate": 0.5},
        "experiment": {"K": 16, "num_blocks": 3, "snr_db_list": [0], "seed": 77, "decoders": ["viterbi"]},
        "paths": {"outputs_root": "outputs/runs"},
    }
    p = tmp_path / "signals.npz"
    d1 = generate_and_save(cfg, p)
    d2 = SignalsDataset.load(p)
    assert np.array_equal(d1.u, d2.u)
    assert np.array_equal(d1.llr, d2.llr)
