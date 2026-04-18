from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATE_RE = re.compile(r"(20\d{6})")


def setup_logging(
    *,
    name: str = "shrub",
    log_dir: str | Path | None = None,
    log_filename: str = "pipeline.log",
    level: int = logging.INFO,
    force: bool = False,
) -> logging.Logger:
    """
    Create a project-level logger with:
    - console handler
    - optional file handler

    Call this once near the top of the notebook.
    All module loggers should be children of this logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers and not force:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # clear old handlers if re-running in notebooks
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.debug("Logger initialized.")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Child logger under the project root logger namespace.
    Example: get_logger('labeling.sprint3_runner')
    """
    return logging.getLogger(f"shrub.{name}")


def tail_text_file(path: str | Path, n_lines: int = 40) -> str:
    path = Path(path)
    if not path.exists():
        return f"[missing file] {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def safe_stem(path_or_name: str | Path) -> str:
    return Path(path_or_name).stem


def extract_first_date_token(text: str) -> str | None:
    m = DATE_RE.search(str(text))
    return m.group(1) if m else None


def yyyymmdd_to_timestamp(value: str | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    try:
        return pd.to_datetime(value, format="%Y%m%d")
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

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

