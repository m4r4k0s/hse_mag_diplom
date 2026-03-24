from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from comm_ai.channel.awgn import sigma2_from_snr_db
from comm_ai.channel.llr import bpsk_awgn_llr
from comm_ai.channel.modulation import bpsk_modulate
from comm_ai.codes.convolutional import convolutional_encode
from comm_ai.codes.trellis import build_trellis
from comm_ai.datasets.signals_dataset import SignalsDataset
from comm_ai.decoders.neural_bcjr import NeuralBCJRDecoder
from comm_ai.decoders.neural_viterbi import NeuralViterbiDecoder
from comm_ai.experiments.evaluate import evaluate_decoders
from comm_ai.training.train_neural_bcjr import train_neural_bcjr_model
from comm_ai.training.train_neural_viterbi import train_neural_viterbi_model
from comm_ai.utils.io import load_yaml, save_yaml
from comm_ai.utils.plotting import save_metric_plot
from comm_ai.utils.reporting import analysis_md, config_overview_md, metric_columns_description_md
from comm_ai.utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(__file__).resolve().parent
BACKUP_ROOT = DEMO_ROOT / "backups"
BACKUP_RESULTS = BACKUP_ROOT / "results"
BACKUP_DATASETS = BACKUP_ROOT / "datasets"
BACKUP_CHECKPOINTS = BACKUP_ROOT / "checkpoints"
BACKUP_FIGURES = BACKUP_ROOT / "figures"

for _p in [BACKUP_ROOT, BACKUP_RESULTS, BACKUP_DATASETS, BACKUP_CHECKPOINTS, BACKUP_FIGURES]:
    _p.mkdir(parents=True, exist_ok=True)


CHANNEL_DESCRIPTIONS: dict[str, str] = {
    "awgn": "BPSK + AWGN. Базовый учебный сценарий, в котором классические алгоритмы обычно наиболее сильны.",
    "rayleigh": "BPSK + Rayleigh fading + AWGN. Канал с замираниями, ближе к беспроводным сценариям.",
    "burst": "BPSK + AWGN с пакетными всплесками шума. Ошибки концентрируются на отдельных участках.",
    "noise_mismatch": "BPSK + AWGN, но декодер использует неверную оценку дисперсии шума. Проверка устойчивости к mismatch.",
    "amplitude_mismatch": "BPSK + изменение амплитуды сигнала при стандартной формуле LLR. Ещё один сценарий model mismatch.",
    "rayleigh_mismatch": "Rayleigh fading + AWGN, но при вычислении LLR игнорируется замирание канала. Тяжёлый случай model mismatch.",
    "impulsive_mismatch": "Импульсный шум поверх BPSK. Декодер использует обычную гауссову формулу LLR и не знает о редких сильных всплесках.",
}

LEVEL_DESCRIPTIONS: dict[str, str] = {
    "level_1": "Быстрый учебный уровень. Небольшой объём данных и простой канал, чтобы быстро показать весь pipeline.",
    "level_2": "Основной исследовательский уровень. Плотнее сетка SNR и больше блоков, уже похоже на содержательный эксперимент.",
    "level_3": "Усложнённые каналы. Здесь у neural-подходов больше шансов показать устойчивость и практическую полезность.",
    "level_4": "Самые жёсткие сценарии из демонстрационного набора. Здесь мы специально усиливаем mismatch и редкие тяжёлые искажения.",
}


@dataclass
class DemoPreset:
    name: str
    level: str
    scenario: str
    K: int
    num_blocks: int
    snr_db_list: list[float]
    seed: int = 42
    hidden_dim: int = 16
    epochs: int = 2
    learning_rate: float = 1e-3
    enable_training: bool = True
    tau: float = 1.0

    def to_config(self, decoders: list[str] | None = None, neural_only: bool = False) -> dict[str, Any]:
        decs = decoders or ["viterbi", "bcjr", "neural_viterbi", "neural_bcjr"]
        if neural_only:
            decs = [d for d in decs if d.startswith("neural")]
        run_name = self.name
        return {
            "code": {
                "constraint_length": 7,
                "polynomials": [171, 133],
                "rate": 0.5,
            },
            "experiment": {
                "K": self.K,
                "num_blocks": self.num_blocks,
                "snr_db_list": self.snr_db_list,
                "seed": self.seed,
                "batch_size": 16,
                "decoders": decs,
                "run_name": run_name,
                "reuse_saved_signals": True,
            },
            "paths": {
                "outputs_root": str(BACKUP_RESULTS),
            },
            "training": {
                "enabled": self.enable_training,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "hidden_dim": self.hidden_dim,
                "device": "cpu",
                "save_checkpoints": True,
                "reuse_checkpoints": True,
                "train_snr_db_list": list(self.snr_db_list),
            },
            "neural": {
                "tau": self.tau,
            },
            "checkpoint_paths": {
                "neural_viterbi": str(BACKUP_RESULTS / run_name / "checkpoints" / "best_neural_viterbi.pt"),
                "neural_bcjr": str(BACKUP_RESULTS / run_name / "checkpoints" / "best_neural_bcjr.pt"),
            },
            "demo": {
                "level": self.level,
                "scenario": self.scenario,
                "description": CHANNEL_DESCRIPTIONS[self.scenario],
            },
        }


PRESETS: dict[str, DemoPreset] = {
    "level_1_awgn_fast": DemoPreset(
        name="level_1_awgn_fast",
        level="level_1",
        scenario="awgn",
        K=32,
        num_blocks=10,
        snr_db_list=[0, 2, 4],
        epochs=1,
    ),
    "level_2_awgn_dense": DemoPreset(
        name="level_2_awgn_dense",
        level="level_2",
        scenario="awgn",
        K=64,
        num_blocks=20,
        snr_db_list=[0, 1, 2, 3, 4],
        epochs=2,
    ),
    "level_3_rayleigh": DemoPreset(
        name="level_3_rayleigh",
        level="level_3",
        scenario="rayleigh",
        K=64,
        num_blocks=16,
        snr_db_list=[0, 2, 4],
        epochs=2,
    ),
    "level_3_burst": DemoPreset(
        name="level_3_burst",
        level="level_3",
        scenario="burst",
        K=64,
        num_blocks=16,
        snr_db_list=[0, 2, 4],
        epochs=2,
    ),
    "level_3_noise_mismatch": DemoPreset(
        name="level_3_noise_mismatch",
        level="level_3",
        scenario="noise_mismatch",
        K=64,
        num_blocks=16,
        snr_db_list=[0, 2, 4],
        epochs=2,
    ),
    "level_3_amplitude_mismatch": DemoPreset(
        name="level_3_amplitude_mismatch",
        level="level_3",
        scenario="amplitude_mismatch",
        K=64,
        num_blocks=16,
        snr_db_list=[0, 2, 4],
        epochs=2,
    ),
    "level_4_rayleigh_mismatch": DemoPreset(
        name="level_4_rayleigh_mismatch",
        level="level_4",
        scenario="rayleigh_mismatch",
        K=96,
        num_blocks=20,
        snr_db_list=[-2, 0, 2, 4],
        epochs=3,
    ),
    "level_4_impulsive_mismatch": DemoPreset(
        name="level_4_impulsive_mismatch",
        level="level_4",
        scenario="impulsive_mismatch",
        K=96,
        num_blocks=20,
        snr_db_list=[-2, 0, 2, 4],
        epochs=3,
    ),
}


def presets_table() -> pd.DataFrame:
    rows = []
    for p in PRESETS.values():
        rows.append({
            "preset": p.name,
            "level": p.level,
            "scenario": p.scenario,
            "K": p.K,
            "num_blocks": p.num_blocks,
            "snr_db_list": str(p.snr_db_list),
            "epochs": p.epochs,
            "description": CHANNEL_DESCRIPTIONS[p.scenario],
        })
    return pd.DataFrame(rows)


def parameter_reference() -> pd.DataFrame:
    return pd.DataFrame([
        {"parameter": "K", "meaning": "Длина информационного блока в битах. Чем больше K, тем содержательнее эксперимент и тем дольше вычисления."},
        {"parameter": "num_blocks", "meaning": "Количество передаваемых блоков на каждую точку SNR. Больше блоков - надёжнее статистика BER/FER."},
        {"parameter": "snr_db_list", "meaning": "Сетка значений SNR в дБ. Чем больше точек, тем более гладкие и информативные кривые."},
        {"parameter": "seed", "meaning": "Фиксирует генератор случайных чисел и делает эксперимент воспроизводимым."},
        {"parameter": "decoders", "meaning": "Список декодеров, которые будут сравниваться: baseline и/или neural."},
        {"parameter": "epochs", "meaning": "Число эпох обучения neural-моделей. Больше эпох - дольше обучение и потенциально лучше адаптация."},
        {"parameter": "learning_rate", "meaning": "Шаг оптимизации для обучения neural-компонентов."},
        {"parameter": "hidden_dim", "meaning": "Ширина скрытого слоя в простой neural-модели калибровки LLR."},
        {"parameter": "scenario", "meaning": "Тип канала или тип model mismatch: AWGN, Rayleigh, burst, noise mismatch, amplitude mismatch и более тяжёлые сценарии."},
        {"parameter": "level", "meaning": "Готовый класс сложности. Level 1 - быстрый показ, Level 2 - основной режим, Level 3 - усложнённые каналы, Level 4 - самые тяжёлые mismatch-сценарии."},
        {"parameter": "reuse_saved_signals", "meaning": "Если True, повторно использует уже сохранённый сигнал и позволяет честно сравнить алгоритмы на одних и тех же данных."},
        {"parameter": "enable_training", "meaning": "Если True, обучает neural-модели. Если False, пытается загрузить уже сохранённые checkpoint-файлы."},
    ])


def build_custom_preset(
    name: str,
    level: str,
    scenario: str,
    K: int,
    num_blocks: int,
    snr_db_list: Iterable[float],
    seed: int = 42,
    epochs: int = 2,
    hidden_dim: int = 16,
    learning_rate: float = 1e-3,
    enable_training: bool = True,
    tau: float = 1.0,
) -> DemoPreset:
    return DemoPreset(
        name=name,
        level=level,
        scenario=scenario,
        K=K,
        num_blocks=num_blocks,
        snr_db_list=list(snr_db_list),
        seed=seed,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        enable_training=enable_training,
        tau=tau,
    )


def _rayleigh_channel(x: np.ndarray, sigma2: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = rng.rayleigh(scale=1.0 / np.sqrt(np.pi / 2.0), size=x.shape)
    n = rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
    y = h * x + n
    llr = 2.0 * h * y / sigma2
    local_sigma2 = np.full_like(x, sigma2, dtype=np.float64)
    return y, n, llr, local_sigma2


def _burst_channel(x: np.ndarray, sigma2: float, rng: np.random.Generator, burst_multiplier: float = 8.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_sigma2 = np.full_like(x, sigma2, dtype=np.float64)
    T = len(x)
    burst_len = max(4, T // 20)
    burst_count = max(1, T // (10 * burst_len))
    for _ in range(burst_count):
        start = int(rng.integers(0, max(1, T - burst_len)))
        local_sigma2[start:start + burst_len] = sigma2 * burst_multiplier
    n = rng.normal(0.0, np.sqrt(local_sigma2), size=x.shape)
    y = x + n
    llr = 2.0 * y / local_sigma2
    return y, n, llr, local_sigma2


def _noise_mismatch_channel(x: np.ndarray, sigma2: float, rng: np.random.Generator, mismatch_factor: float = 1.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
    y = x + n
    assumed_sigma2 = sigma2 * mismatch_factor
    llr = 2.0 * y / assumed_sigma2
    local_sigma2 = np.full_like(x, sigma2, dtype=np.float64)
    return y, n, llr, local_sigma2


def _amplitude_mismatch_channel(x: np.ndarray, sigma2: float, rng: np.random.Generator, amplitude: float = 0.65) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
    y = amplitude * x + n
    llr = 2.0 * y / sigma2
    local_sigma2 = np.full_like(x, sigma2, dtype=np.float64)
    return y, n, llr, local_sigma2


def _rayleigh_mismatch_channel(x: np.ndarray, sigma2: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = rng.rayleigh(scale=1.0 / np.sqrt(np.pi / 2.0), size=x.shape)
    n = rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
    y = h * x + n
    # Декодер использует упрощённую формулу и игнорирует мгновенное замирание h.
    llr = 2.0 * y / sigma2
    local_sigma2 = np.full_like(x, sigma2, dtype=np.float64)
    return y, n, llr, local_sigma2


def _impulsive_mismatch_channel(
    x: np.ndarray,
    sigma2: float,
    rng: np.random.Generator,
    impulse_probability: float = 0.08,
    impulse_multiplier: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_sigma2 = np.full_like(x, sigma2, dtype=np.float64)
    mask = rng.random(size=x.shape) < impulse_probability
    local_sigma2[mask] = sigma2 * impulse_multiplier
    n = rng.normal(0.0, np.sqrt(local_sigma2), size=x.shape)
    y = x + n
    # Формула LLR остаётся гауссовой и использует только номинальную дисперсию.
    llr = 2.0 * y / sigma2
    return y, n, llr, local_sigma2


def generate_dataset_for_scenario(cfg: dict[str, Any]) -> SignalsDataset:
    exp = cfg["experiment"]
    code = cfg["code"]
    scenario = cfg.get("demo", {}).get("scenario", "awgn")
    K = int(exp["K"])
    num_blocks = int(exp["num_blocks"])
    snrs = list(exp["snr_db_list"])
    seed = int(exp["seed"])
    rng = np.random.default_rng(seed)

    trellis = build_trellis(code["constraint_length"], tuple(code["polynomials"]))
    rows = []
    for snr in snrs:
        sigma2 = sigma2_from_snr_db(float(snr), rate=code["rate"])
        for _ in range(num_blocks):
            u = rng.integers(0, 2, size=K, dtype=np.int64)
            c = convolutional_encode(u, trellis)
            x = bpsk_modulate(c)
            if scenario == "awgn":
                n = rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
                y = x + n
                llr = bpsk_awgn_llr(y, sigma2)
                local_sigma2 = np.full_like(x, sigma2, dtype=np.float64)
            elif scenario == "rayleigh":
                y, n, llr, local_sigma2 = _rayleigh_channel(x, sigma2, rng)
            elif scenario == "burst":
                y, n, llr, local_sigma2 = _burst_channel(x, sigma2, rng)
            elif scenario == "noise_mismatch":
                y, n, llr, local_sigma2 = _noise_mismatch_channel(x, sigma2, rng)
            elif scenario == "amplitude_mismatch":
                y, n, llr, local_sigma2 = _amplitude_mismatch_channel(x, sigma2, rng)
            elif scenario == "rayleigh_mismatch":
                y, n, llr, local_sigma2 = _rayleigh_mismatch_channel(x, sigma2, rng)
            elif scenario == "impulsive_mismatch":
                y, n, llr, local_sigma2 = _impulsive_mismatch_channel(x, sigma2, rng)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
            rows.append((u, c, x, n, y, llr, float(snr), float(np.mean(local_sigma2))))

    u = np.stack([r[0] for r in rows])
    c = np.stack([r[1] for r in rows])
    x = np.stack([r[2] for r in rows])
    noise = np.stack([r[3] for r in rows])
    y = np.stack([r[4] for r in rows])
    llr = np.stack([r[5] for r in rows])
    snr_db = np.array([r[6] for r in rows], dtype=np.float64)
    sigma2 = np.array([r[7] for r in rows], dtype=np.float64)

    meta = {
        "config": cfg,
        "scenario": scenario,
        "channel_description": CHANNEL_DESCRIPTIONS[scenario],
        "created_at": datetime.now().isoformat(),
    }
    return SignalsDataset(u=u, c=c, x=x, noise=noise, y=y, llr=llr, snr_db=snr_db, sigma2=sigma2, seed=seed, meta=meta)


def save_dataset_backup(ds: SignalsDataset, run_name: str, out_dir: Path) -> Path:
    dataset_path = out_dir / "signals.npz"
    ds.save(dataset_path)
    shutil.copy2(dataset_path, BACKUP_DATASETS / f"{run_name}_signals.npz")
    return dataset_path


def _ensure_models(cfg: dict[str, Any], ds: SignalsDataset, out_dir: Path) -> tuple[NeuralViterbiDecoder | None, NeuralBCJRDecoder | None, bool, bool]:
    enabled = cfg["experiment"]["decoders"]
    train_cfg = cfg.get("training", {})
    should_train = bool(train_cfg.get("enabled", False))
    reuse_ckpt = bool(train_cfg.get("reuse_checkpoints", True))
    n_out = len(cfg["code"]["polynomials"])
    hidden_dim = int(train_cfg.get("hidden_dim", 16))
    tau = float(cfg.get("neural", {}).get("tau", 1.0))
    device = str(train_cfg.get("device", "cpu"))
    epochs = int(train_cfg.get("epochs", 2))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    train_snr = train_cfg.get("train_snr_db_list")

    nv_model: NeuralViterbiDecoder | None = None
    nb_model: NeuralBCJRDecoder | None = None
    nv_trained = False
    nb_trained = False

    if "neural_viterbi" in enabled:
        ckpt = Path(cfg["checkpoint_paths"]["neural_viterbi"])
        nv_model = NeuralViterbiDecoder(n_out=n_out, hidden=hidden_dim, tau=tau)
        if reuse_ckpt and ckpt.exists():
            nv_model.load_state_dict(torch.load(ckpt, map_location=device))
            nv_trained = True
        elif should_train:
            train_neural_viterbi_model(nv_model, ds, epochs, lr, device, ckpt, train_snr)
            nv_model.load_state_dict(torch.load(ckpt, map_location=device))
            nv_trained = True
        else:
            nv_model = None

    if "neural_bcjr" in enabled:
        ckpt = Path(cfg["checkpoint_paths"]["neural_bcjr"])
        nb_model = NeuralBCJRDecoder(n_out=n_out, hidden=hidden_dim)
        if reuse_ckpt and ckpt.exists():
            nb_model.load_state_dict(torch.load(ckpt, map_location=device))
            nb_trained = True
        elif should_train:
            train_neural_bcjr_model(nb_model, ds, epochs, lr, device, ckpt, train_snr)
            nb_model.load_state_dict(torch.load(ckpt, map_location=device))
            nb_trained = True
        else:
            nb_model = None

    return nv_model, nb_model, nv_trained, nb_trained


def _analysis_with_neural_focus(df: pd.DataFrame) -> str:
    if df.empty:
        return "Результаты отсутствуют."

    grouped = df.groupby("decoder", as_index=False).agg(
        ber=("ber", "mean"),
        fer=("fer", "mean"),
        decode_time_s=("decode_time_s", "mean"),
    )
    best_ber = grouped.loc[grouped["ber"].idxmin(), "decoder"]
    best_fer = grouped.loc[grouped["fer"].idxmin(), "decoder"]
    fastest = grouped.loc[grouped["decode_time_s"].idxmin(), "decoder"]
    neural = grouped[grouped["decoder"].str.startswith("neural")].copy()
    best_neural = None
    if not neural.empty:
        best_neural = neural.loc[neural["ber"].idxmin(), "decoder"]

    lines = [
        "## Автоматический вывод по выбранным параметрам",
        "",
        f"- Лучший алгоритм по BER: `{best_ber}`.",
        f"- Лучший алгоритм по FER: `{best_fer}`.",
        f"- Самый быстрый алгоритм: `{fastest}`.",
    ]
    if best_neural is not None:
        lines.append(f"- Лучший среди neural-моделей: `{best_neural}`.")
    if best_ber != fastest:
        lines.append("- Наблюдается компромисс: лучший по качеству алгоритм не совпадает с самым быстрым.")
    else:
        lines.append("- Один и тот же алгоритм лидирует и по качеству, и по скорости на выбранном срезе.")
    if best_neural is not None and best_neural == best_ber:
        lines.append("- На выбранном сценарии neural-подход оказался лидером по качеству, что указывает на потенциал обучаемой калибровки.")
    elif best_neural is not None:
        lines.append("- На выбранном сценарии классические алгоритмы остаются сильнее по качеству; это ожидаемо для простых и хорошо смоделированных каналов.")
    return "\n".join(lines)


def write_summary(out_dir: Path, cfg: dict[str, Any], df: pd.DataFrame) -> None:
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Summary for demo run\n\n")
        f.write("Ниже собран компактный итог для демонстрационного сценария.\n\n")
        f.write(config_overview_md(cfg) + "\n\n")
        f.write(metric_columns_description_md() + "\n\n")
        f.write("## Results table\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write(_analysis_with_neural_focus(df) + "\n")


def run_demo_experiment(
    cfg: dict[str, Any],
    force_regenerate: bool = False,
    force_retrain: bool = False,
) -> Path:
    set_seed(cfg["experiment"]["seed"])
    run_name = cfg["experiment"]["run_name"]
    out_dir = BACKUP_RESULTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["outputs_root"] = str(BACKUP_RESULTS)
    cfg["checkpoint_paths"] = {
        "neural_viterbi": str(out_dir / "checkpoints" / "best_neural_viterbi.pt"),
        "neural_bcjr": str(out_dir / "checkpoints" / "best_neural_bcjr.pt"),
    }

    dataset_path = out_dir / "signals.npz"
    if force_regenerate or not dataset_path.exists():
        ds = generate_dataset_for_scenario(cfg)
        save_dataset_backup(ds, run_name, out_dir)
    else:
        ds = SignalsDataset.load(dataset_path)

    if force_retrain:
        ckpt_dir = out_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    nv_model, nb_model, nv_trained, nb_trained = _ensure_models(cfg, ds, out_dir)
    print(f"Переходим к оценке декодеров. Результаты будут сохранены в: {out_dir}")
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
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "level": cfg.get("demo", {}).get("level"),
        "scenario": cfg.get("demo", {}).get("scenario"),
        "description": cfg.get("demo", {}).get("description"),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(out_dir, cfg, df)
    for img in ["ber_plot.png", "fer_plot.png", "timing_plot.png"]:
        shutil.copy2(out_dir / img, BACKUP_FIGURES / f"{run_name}_{img}")
    return out_dir


def aggregate_demo_results(results_root: Path | None = None) -> pd.DataFrame:
    results_root = results_root or BACKUP_RESULTS
    rows = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "results.csv"
        cfg_path = run_dir / "config_used.yaml"
        if not csv_path.exists() or not cfg_path.exists():
            continue
        cfg = load_yaml(cfg_path)
        df = pd.read_csv(csv_path)
        df["run_name"] = run_dir.name
        df["level"] = cfg.get("demo", {}).get("level", "unknown")
        df["scenario"] = cfg.get("demo", {}).get("scenario", "unknown")
        df["scenario_description"] = cfg.get("demo", {}).get("description", CHANNEL_DESCRIPTIONS.get(cfg.get("demo", {}).get("scenario", "awgn"), ""))
        df["K"] = cfg["experiment"]["K"]
        df["num_blocks"] = cfg["experiment"]["num_blocks"]
        df["snr_grid"] = str(cfg["experiment"]["snr_db_list"])
        df["training_enabled"] = cfg.get("training", {}).get("enabled", False)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def filter_results(
    df: pd.DataFrame,
    run_name: str | None = None,
    level: str | None = None,
    scenario: str | None = None,
    K: int | None = None,
    num_blocks: int | None = None,
    decoders: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if run_name:
        out = out[out["run_name"] == run_name]
    if level:
        out = out[out["level"] == level]
    if scenario:
        out = out[out["scenario"] == scenario]
    if K is not None:
        out = out[out["K"] == K]
    if num_blocks is not None:
        out = out[out["num_blocks"] == num_blocks]
    if decoders:
        out = out[out["decoder"].isin(decoders)]
    return out.reset_index(drop=True)


def automatic_commentary(df: pd.DataFrame) -> str:
    if df.empty:
        return "Для выбранного фильтра нет сохранённых результатов. Выберите другой набор параметров или сначала создайте бэкап нового прогона."
    return _analysis_with_neural_focus(df)


def list_available_runs() -> pd.DataFrame:
    df = aggregate_demo_results()
    if df.empty:
        return pd.DataFrame(columns=["run_name", "level", "scenario", "K", "num_blocks", "snr_grid"])
    return df[["run_name", "level", "scenario", "K", "num_blocks", "snr_grid"]].drop_duplicates().sort_values(["level", "scenario", "K", "num_blocks"])


def ensure_default_backups(preset_names: Iterable[str] | None = None) -> list[Path]:
    preset_names = list(preset_names) if preset_names is not None else list(PRESETS.keys())
    produced = []
    for name in preset_names:
        preset = PRESETS[name]
        cfg = preset.to_config()
        produced.append(run_demo_experiment(cfg, force_regenerate=False, force_retrain=False))
    return produced


def export_config_examples() -> None:
    cfg_dir = DEMO_ROOT / "configs"
    cfg_dir.mkdir(exist_ok=True)
    for p in PRESETS.values():
        save_yaml(p.to_config(), cfg_dir / f"{p.name}.yaml")


__all__ = [
    "BACKUP_ROOT",
    "BACKUP_RESULTS",
    "CHANNEL_DESCRIPTIONS",
    "DEMO_ROOT",
    "LEVEL_DESCRIPTIONS",
    "PRESETS",
    "PROJECT_ROOT",
    "DemoPreset",
    "aggregate_demo_results",
    "automatic_commentary",
    "build_custom_preset",
    "ensure_default_backups",
    "export_config_examples",
    "filter_results",
    "generate_dataset_for_scenario",
    "list_available_runs",
    "parameter_reference",
    "presets_table",
    "run_demo_experiment",
]
