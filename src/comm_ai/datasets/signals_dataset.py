from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from comm_ai.channel.awgn import awgn_channel, sigma2_from_snr_db
from comm_ai.channel.llr import bpsk_awgn_llr
from comm_ai.channel.modulation import bpsk_modulate
from comm_ai.codes.convolutional import convolutional_encode
from comm_ai.codes.trellis import build_trellis


@dataclass
class SignalsDataset:
    u: np.ndarray
    c: np.ndarray
    x: np.ndarray
    noise: np.ndarray
    y: np.ndarray
    llr: np.ndarray
    snr_db: np.ndarray
    sigma2: np.ndarray
    seed: int
    meta: dict[str, Any]

    @staticmethod
    def generate(config: dict[str, Any]) -> "SignalsDataset":
        exp = config["experiment"]
        code = config["code"]
        K = exp["K"]
        num_blocks = exp["num_blocks"]
        snrs = exp["snr_db_list"]
        seed = exp["seed"]
        rng = np.random.default_rng(seed)

        trellis = build_trellis(code["constraint_length"], tuple(code["polynomials"]))
        rows = []
        for snr in snrs:
            sigma2 = sigma2_from_snr_db(snr, rate=code["rate"])
            for _ in range(num_blocks):
                u = rng.integers(0, 2, size=K, dtype=np.int64)
                c = convolutional_encode(u, trellis)
                x = bpsk_modulate(c)
                y, n = awgn_channel(x, sigma2, rng)
                llr = bpsk_awgn_llr(y, sigma2)
                rows.append((u, c, x, n, y, llr, snr, sigma2))

        u = np.stack([r[0] for r in rows])
        c = np.stack([r[1] for r in rows])
        x = np.stack([r[2] for r in rows])
        noise = np.stack([r[3] for r in rows])
        y = np.stack([r[4] for r in rows])
        llr = np.stack([r[5] for r in rows])
        snr_db = np.array([r[6] for r in rows], dtype=np.float64)
        sigma2 = np.array([r[7] for r in rows], dtype=np.float64)
        return SignalsDataset(u, c, x, noise, y, llr, snr_db, sigma2, seed, {"config": config})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            u=self.u,
            c=self.c,
            x=self.x,
            noise=self.noise,
            y=self.y,
            llr=self.llr,
            snr_db=self.snr_db,
            sigma2=self.sigma2,
            seed=np.array([self.seed]),
            meta=np.array([json.dumps(self.meta)]),
        )

    @staticmethod
    def load(path: str | Path) -> "SignalsDataset":
        d = np.load(path, allow_pickle=True)
        return SignalsDataset(
            u=d["u"], c=d["c"], x=d["x"], noise=d["noise"], y=d["y"], llr=d["llr"],
            snr_db=d["snr_db"], sigma2=d["sigma2"], seed=int(d["seed"][0]), meta=json.loads(d["meta"][0])
        )


def generate_and_save(config: dict[str, Any], path: str | Path) -> SignalsDataset:
    ds = SignalsDataset.generate(config)
    ds.save(path)
    return ds
