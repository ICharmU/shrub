from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATE_RE = re.compile(r"(20\d{6})")


def safe_stem(path_or_name: str | Path) -> str:
    return Path(path_or_name).stem


def extract_first_date_token(text: str) -> str | None:
    m = DATE_RE.search(str(text))
    return m.group(1) if m else None


def yyyymmdd_to_timestamp(value: str | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    try:
        return pd.to_datetime(value, format='%Y%m%d')
    except Exception:
        return pd.NaT


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def first_present(columns: Iterable[str], available: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in available}
    for c in columns:
        if c.lower() in lower:
            return lower[c.lower()]
    return None
