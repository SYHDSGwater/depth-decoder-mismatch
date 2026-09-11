from __future__ import annotations

import math
import numpy as np


def head_regret(linear_loss: float, rich_loss: float) -> float:
    return float(linear_loss - rich_loss)


def interaction_2x2(r_lo_v_lo: float, r_hi_v_lo: float, r_lo_v_hi: float, r_hi_v_hi: float) -> float:
    return float((r_hi_v_hi - r_lo_v_hi) - (r_hi_v_lo - r_lo_v_lo))


def bits_per_byte(total_nll_nats: float, raw_bytes: int) -> float:
    if raw_bytes <= 0:
        raise ValueError("raw_bytes must be positive")
    return float(total_nll_nats / (math.log(2.0) * raw_bytes))


def ols_slope(depths, regrets) -> float:
    x = np.asarray(depths, dtype=float); y = np.asarray(regrets, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("depths and regrets must be same-length 1D arrays with >=2 points")
    x = x - x.mean(); denom = float(np.dot(x, x))
    if denom == 0: raise ValueError("depths must vary")
    return float(np.dot(x, y - y.mean()) / denom)
