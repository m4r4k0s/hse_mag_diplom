from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from comm_ai.codes.trellis import build_trellis
from comm_ai.datasets.signals_dataset import SignalsDataset
from comm_ai.decoders.bcjr import bcjr_decode
from comm_ai.decoders.neural_bcjr import NeuralBCJRDecoder, neural_bcjr_decode
from comm_ai.decoders.neural_viterbi import NeuralViterbiDecoder, neural_viterbi_decode
from comm_ai.decoders.viterbi import viterbi_decode
from comm_ai.utils.metrics import ber, fer


def evaluate_decoders(
    config: dict[str, Any],
    ds: SignalsDataset,
    neural_viterbi_model: NeuralViterbiDecoder | None = None,
    neural_bcjr_model: NeuralBCJRDecoder | None = None,
    neural_viterbi_trained: bool = False,
    neural_bcjr_trained: bool = False,
) -> pd.DataFrame:
    code = config["code"]
    enabled = config["experiment"]["decoders"]
    trellis = build_trellis(code["constraint_length"], tuple(code["polynomials"]))

    rows = []
    buckets: dict[tuple[str, float], dict[str, list[float]]] = defaultdict(lambda: {"ber": [], "fer": [], "time": []})

    for i in range(ds.u.shape[0]):
        u = ds.u[i]
        llr = ds.llr[i]
        snr = float(ds.snr_db[i])

        if "viterbi" in enabled:
            t0 = time.perf_counter()
            uh = viterbi_decode(llr, trellis)
            dt = time.perf_counter() - t0
            buckets[("viterbi", snr)]["ber"].append(ber(u, uh))
            buckets[("viterbi", snr)]["fer"].append(fer(u, uh))
            buckets[("viterbi", snr)]["time"].append(dt)
        if "bcjr" in enabled:
            t0 = time.perf_counter()
            _, uh = bcjr_decode(llr, trellis)
            dt = time.perf_counter() - t0
            buckets[("bcjr", snr)]["ber"].append(ber(u, uh))
            buckets[("bcjr", snr)]["fer"].append(fer(u, uh))
            buckets[("bcjr", snr)]["time"].append(dt)
        if "neural_viterbi" in enabled and neural_viterbi_model is not None:
            t0 = time.perf_counter()
            uh = neural_viterbi_decode(llr, trellis, neural_viterbi_model)
            dt = time.perf_counter() - t0
            buckets[("neural_viterbi", snr)]["ber"].append(ber(u, uh))
            buckets[("neural_viterbi", snr)]["fer"].append(fer(u, uh))
            buckets[("neural_viterbi", snr)]["time"].append(dt)
        if "neural_bcjr" in enabled and neural_bcjr_model is not None:
            t0 = time.perf_counter()
            _, uh = neural_bcjr_decode(llr, trellis, neural_bcjr_model)
            dt = time.perf_counter() - t0
            buckets[("neural_bcjr", snr)]["ber"].append(ber(u, uh))
            buckets[("neural_bcjr", snr)]["fer"].append(fer(u, uh))
            buckets[("neural_bcjr", snr)]["time"].append(dt)

    for (decoder, snr), vals in buckets.items():
        trained_used = False
        if decoder == "neural_viterbi":
            trained_used = neural_viterbi_trained
        if decoder == "neural_bcjr":
            trained_used = neural_bcjr_trained

        rows.append({
            "decoder": decoder,
            "snr_db": snr,
            "ber": float(np.mean(vals["ber"])),
            "fer": float(np.mean(vals["fer"])),
            "decode_time_s": float(np.mean(vals["time"])),
            "complexity_proxy": int(trellis.num_states * config["experiment"]["K"]),
            "trained_model_used": bool(trained_used),
        })
    return pd.DataFrame(rows).sort_values(["decoder", "snr_db"]).reset_index(drop=True)
