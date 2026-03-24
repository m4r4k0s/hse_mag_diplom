from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimerResult:
    elapsed_s: float


@contextmanager
def perf_timer() -> TimerResult:
    start = time.perf_counter()
    result = TimerResult(elapsed_s=0.0)
    yield result
    result.elapsed_s = time.perf_counter() - start
