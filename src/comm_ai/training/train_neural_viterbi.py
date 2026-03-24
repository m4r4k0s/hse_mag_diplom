from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import optim
from tqdm import tqdm

from comm_ai.datasets.signals_dataset import SignalsDataset
from comm_ai.decoders.neural_viterbi import NeuralViterbiDecoder
from comm_ai.training.losses import bce_logits_loss
from comm_ai.utils.io import load_yaml
from comm_ai.utils.seed import set_seed


def _select_training_subset(ds: SignalsDataset, train_snr_db_list: list[float] | None) -> tuple[torch.Tensor, torch.Tensor]:
    if train_snr_db_list:
        mask = torch.tensor([float(s) in train_snr_db_list for s in ds.snr_db], dtype=torch.bool)
    else:
        mask = torch.ones(ds.llr.shape[0], dtype=torch.bool)

    llr = torch.tensor(ds.llr[mask.numpy()], dtype=torch.float32)
    coded_bits = torch.tensor(ds.c[mask.numpy()], dtype=torch.float32)
    return llr, coded_bits


def train_neural_viterbi_model(
    model: NeuralViterbiDecoder,
    ds: SignalsDataset,
    epochs: int,
    lr: float,
    device: str,
    checkpoint_path: str | Path,
    train_snr_db_list: list[float] | None = None,
) -> list[float]:
    model.to(device)
    llr, coded_bits = _select_training_subset(ds, train_snr_db_list)
    n_out = model.metric_net.net[0].in_features
    llr = llr.view(llr.shape[0], -1, n_out).to(device)
    coded_bits = coded_bits.view(coded_bits.shape[0], -1, n_out).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    history: list[float] = []
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for _ in tqdm(range(epochs), desc="train_neural_viterbi"):
        model.train()
        opt.zero_grad()
        out = model.calibrate(llr)
        loss = bce_logits_loss(out, 1.0 - coded_bits)
        loss.backward()
        opt.step()
        value = float(loss.item())
        history.append(value)
        if value < best:
            best = value
            torch.save(model.state_dict(), checkpoint)

    return history




def train_from_config(cfg: dict[str, Any], dataset_path: str | Path | None = None) -> tuple[Path, list[float]]:
    """Notebook-friendly training API: train model from config and return checkpoint path + history."""
    set_seed(cfg["experiment"]["seed"])
    ds_path = Path(dataset_path) if dataset_path else Path(cfg["paths"]["outputs_root"]) / cfg["experiment"]["run_name"] / "signals.npz"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}. Run experiment generation first or pass dataset_path.")

    ds = SignalsDataset.load(ds_path)
    n_out = len(cfg["code"]["polynomials"])
    train_cfg = cfg.get("training", {})
    model = NeuralViterbiDecoder(n_out=n_out, hidden=train_cfg.get("hidden_dim", 16), tau=cfg.get("neural", {}).get("tau", 1.0))
    ckpt = Path(cfg["paths"]["outputs_root"]) / cfg["experiment"]["run_name"] / "checkpoints" / "best_neural_viterbi.pt"
    history = train_neural_viterbi_model(
        model=model,
        ds=ds,
        epochs=int(train_cfg.get("epochs", 5)),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        device=str(train_cfg.get("device", "cpu")),
        checkpoint_path=ckpt,
        train_snr_db_list=train_cfg.get("train_snr_db_list"),
    )
    return ckpt, history

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    cfg: dict[str, Any] = load_yaml(args.config)
    ckpt, history = train_from_config(cfg, dataset_path=args.dataset)
    print(f"Saved checkpoint: {ckpt}")
    print(f"Final loss: {history[-1]:.6f}")


if __name__ == "__main__":
    main()
