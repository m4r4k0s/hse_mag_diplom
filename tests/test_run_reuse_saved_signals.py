import time
from pathlib import Path

from comm_ai.experiments.run_experiment import run
from comm_ai.utils.io import save_yaml


def test_run_reuses_saved_signals(tmp_path: Path) -> None:
    cfg = {
        "code": {"constraint_length": 7, "polynomials": [171, 133], "rate": 0.5},
        "experiment": {
            "K": 16,
            "num_blocks": 2,
            "snr_db_list": [0],
            "seed": 5,
            "batch_size": 2,
            "decoders": ["viterbi", "bcjr"],
            "run_name": "reuse_test",
            "reuse_saved_signals": False,
        },
        "paths": {"outputs_root": str(tmp_path)},
        "training": {"enabled": False, "reuse_checkpoints": False, "device": "cpu"},
        "neural": {"tau": 1.0},
        "checkpoint_paths": {},
    }

    cfg_path = tmp_path / "cfg.yaml"
    save_yaml(cfg, cfg_path)
    out = run(str(cfg_path))
    signals = out / "signals.npz"
    m1 = signals.stat().st_mtime

    time.sleep(0.1)
    cfg["experiment"]["reuse_saved_signals"] = True
    save_yaml(cfg, cfg_path)
    run(str(cfg_path))
    m2 = signals.stat().st_mtime

    assert m2 == m1
