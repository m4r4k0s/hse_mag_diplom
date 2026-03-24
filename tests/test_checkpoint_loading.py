from pathlib import Path

import torch

from comm_ai.datasets.signals_dataset import SignalsDataset
from comm_ai.decoders.neural_viterbi import NeuralViterbiDecoder
from comm_ai.training.train_neural_viterbi import train_neural_viterbi_model


def test_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    cfg = {
        "code": {"constraint_length": 7, "polynomials": [171, 133], "rate": 0.5},
        "experiment": {"K": 16, "num_blocks": 2, "snr_db_list": [0], "seed": 11, "decoders": ["viterbi"]},
        "paths": {"outputs_root": "outputs/runs"},
    }
    ds = SignalsDataset.generate(cfg)
    model = NeuralViterbiDecoder(n_out=2, hidden=8)
    ckpt = tmp_path / "nv.pt"
    train_neural_viterbi_model(model, ds, epochs=1, lr=1e-3, device="cpu", checkpoint_path=ckpt)

    restored = NeuralViterbiDecoder(n_out=2, hidden=8)
    restored.load_state_dict(torch.load(ckpt, map_location="cpu"))
    assert ckpt.exists()
