from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from comm_ai.datasets.signals_dataset import SignalsDataset, generate_and_save
from comm_ai.decoders.neural_bcjr import NeuralBCJRDecoder
from comm_ai.decoders.neural_viterbi import NeuralViterbiDecoder
from comm_ai.experiments.evaluate import evaluate_decoders
from comm_ai.training.train_neural_bcjr import train_neural_bcjr_model
from comm_ai.training.train_neural_viterbi import train_neural_viterbi_model
from comm_ai.utils.io import load_yaml, save_yaml
from comm_ai.utils.plotting import save_metric_plot
from comm_ai.utils.reporting import analysis_md, config_overview_md, metric_columns_description_md
from comm_ai.utils.seed import set_seed


def _default_checkpoint_paths(out_dir: Path) -> dict[str, Path]:
    ckpt_dir = out_dir / "checkpoints"
    return {
        "neural_viterbi": ckpt_dir / "best_neural_viterbi.pt",
        "neural_bcjr": ckpt_dir / "best_neural_bcjr.pt",
    }


def _resolve_checkpoint_paths(cfg: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    defaults = _default_checkpoint_paths(out_dir)
    user_ckpts = cfg.get("checkpoint_paths", {})
    return {
        "neural_viterbi": Path(user_ckpts.get("neural_viterbi", defaults["neural_viterbi"])),
        "neural_bcjr": Path(user_ckpts.get("neural_bcjr", defaults["neural_bcjr"])),
    }


def _ensure_neural_models(cfg: dict[str, Any], ds: SignalsDataset, out_dir: Path) -> tuple[NeuralViterbiDecoder | None, NeuralBCJRDecoder | None, bool, bool]:
    enabled = cfg["experiment"]["decoders"]
    train_cfg = cfg.get("training", {})
    should_train = bool(train_cfg.get("enabled", False))
    reuse_ckpt = bool(train_cfg.get("reuse_checkpoints", True))

    n_out = len(cfg["code"]["polynomials"])
    hidden_dim = int(train_cfg.get("hidden_dim", 16))
    tau = float(cfg.get("neural", {}).get("tau", 1.0))
    device = str(train_cfg.get("device", "cpu"))
    epochs = int(train_cfg.get("epochs", 5))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    train_snr = train_cfg.get("train_snr_db_list")

    ckpt_paths = _resolve_checkpoint_paths(cfg, out_dir)

    nv_model: NeuralViterbiDecoder | None = None
    nb_model: NeuralBCJRDecoder | None = None
    nv_trained = False
    nb_trained = False

    if "neural_viterbi" in enabled:
        nv_model = NeuralViterbiDecoder(n_out=n_out, hidden=hidden_dim, tau=tau)
        ckpt = ckpt_paths["neural_viterbi"]
        if reuse_ckpt and ckpt.exists():
            nv_model.load_state_dict(torch.load(ckpt, map_location=device))
            nv_trained = True
        elif should_train:
            train_neural_viterbi_model(nv_model, ds, epochs, lr, device, ckpt, train_snr)
            nv_model.load_state_dict(torch.load(ckpt, map_location=device))
            nv_trained = True
        else:
            print(
                "[WARNING] neural_viterbi checkpoint not found and training disabled. "
                "Set training.enabled=true or provide checkpoint_paths.neural_viterbi."
            )
            nv_model = None

    if "neural_bcjr" in enabled:
        nb_model = NeuralBCJRDecoder(n_out=n_out, hidden=hidden_dim)
        ckpt = ckpt_paths["neural_bcjr"]
        if reuse_ckpt and ckpt.exists():
            nb_model.load_state_dict(torch.load(ckpt, map_location=device))
            nb_trained = True
        elif should_train:
            train_neural_bcjr_model(nb_model, ds, epochs, lr, device, ckpt, train_snr)
            nb_model.load_state_dict(torch.load(ckpt, map_location=device))
            nb_trained = True
        else:
            print(
                "[WARNING] neural_bcjr checkpoint not found and training disabled. "
                "Set training.enabled=true or provide checkpoint_paths.neural_bcjr."
            )
            nb_model = None

    return nv_model, nb_model, nv_trained, nb_trained


def _run_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    commit = "unknown"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return {
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit,
        "device": cfg.get("training", {}).get("device", "cpu"),
        "versions": {
            "torch": torch.__version__,
            "pandas": pd.__version__,
        },
    }


def run(config_path: str) -> Path:
    cfg = load_yaml(config_path)
    set_seed(cfg["experiment"]["seed"])

    run_name = cfg["experiment"].get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["paths"]["outputs_root"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "signals.npz"
    reuse_saved = bool(cfg.get("experiment", {}).get("reuse_saved_signals", False))
    if reuse_saved and data_path.exists():
        ds = SignalsDataset.load(data_path)
    else:
        ds = generate_and_save(cfg, data_path)

    ds = SignalsDataset.load(data_path)
    nv_model, nb_model, nv_trained, nb_trained = _ensure_neural_models(cfg, ds, out_dir)

    df = evaluate_decoders(
        cfg,
        ds,
        neural_viterbi_model=nv_model,
        neural_bcjr_model=nb_model,
        neural_viterbi_trained=nv_trained,
        neural_bcjr_trained=nb_trained,
    )
    df.to_csv(out_dir / "results.csv", index=False)

    save_metric_plot(df, "ber", out_dir / "ber_plot.png")
    save_metric_plot(df, "fer", out_dir / "fer_plot.png")
    save_metric_plot(df, "decode_time_s", out_dir / "timing_plot.png")

    save_yaml(cfg, out_dir / "config_used.yaml")
    metadata = _run_metadata(cfg)
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Experiment summary\n\n")
        f.write("LLR-calibrated Neural Viterbi / Neural BCJR prototype.\n\n")
        f.write("## Artifacts\n")
        for item in [
            "signals.npz",
            "results.csv",
            "ber_plot.png",
            "fer_plot.png",
            "timing_plot.png",
            "config_used.yaml",
            "run_metadata.json",
            "checkpoints/best_neural_viterbi.pt",
            "checkpoints/best_neural_bcjr.pt",
        ]:
            f.write(f"- {item}\n")
        f.write("\n")
        f.write(config_overview_md(cfg) + "\n\n")
        f.write(metric_columns_description_md() + "\n")
        f.write("## Results table\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write(analysis_md(df) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    out = run(args.config)
    print(f"Saved run to: {out}")


if __name__ == "__main__":
    main()
