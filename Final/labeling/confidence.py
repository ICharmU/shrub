from __future__ import annotations

import math
import numpy as np


def radial_confidence(distance_px: np.ndarray, radius_px: float, center: float = 1.0, edge: float = 0.35) -> np.ndarray:
    if radius_px <= 0:
        return np.zeros_like(distance_px, dtype=float)
    norm = np.clip(distance_px / radius_px, 0.0, 1.0)
    return center - (center - edge) * norm


def temporal_confidence(date_a, date_b, half_life_days: float = 365.0) -> float:
    if date_a is None or date_b is None:
        return float("nan")
    try:
        delta = abs((date_a - date_b).days)
    except Exception:
        return float("nan")
    return float(math.exp(-math.log(2) * (delta / max(half_life_days, 1e-6))))
