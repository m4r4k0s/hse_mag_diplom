from pathlib import Path

from comm_ai.datasets.signals_dataset import SignalsDataset
from comm_ai.decoders.neural_bcjr import NeuralBCJRDecoder
from comm_ai.decoders.neural_viterbi import NeuralViterbiDecoder
from comm_ai.training.train_neural_bcjr import train_neural_bcjr_model
from comm_ai.training.train_neural_viterbi import train_neural_viterbi_model


def _cfg() -> dict:
    return {
        "code": {"constraint_length": 7, "polynomials": [171, 133], "rate": 0.5},
        "experiment": {"K": 16, "num_blocks": 3, "snr_db_list": [0], "seed": 7, "decoders": ["viterbi"]},
        "paths": {"outputs_root": "outputs/runs"},
    }


def test_training_creates_checkpoint_files(tmp_path: Path) -> None:
    ds = SignalsDataset.generate(_cfg())
    nv = NeuralViterbiDecoder(n_out=2, hidden=8)
    nb = NeuralBCJRDecoder(n_out=2, hidden=8)

    nv_ckpt = tmp_path / "best_neural_viterbi.pt"
    nb_ckpt = tmp_path / "best_neural_bcjr.pt"

    train_neural_viterbi_model(nv, ds, epochs=1, lr=1e-3, device="cpu", checkpoint_path=nv_ckpt)
    train_neural_bcjr_model(nb, ds, epochs=1, lr=1e-3, device="cpu", checkpoint_path=nb_ckpt)

    assert nv_ckpt.exists()
    assert nb_ckpt.exists()
