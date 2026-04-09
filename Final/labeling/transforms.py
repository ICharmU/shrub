from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Final.labeling.io import parse_transform_txt
from Final.models import ShrubObjectColumns


COLS = ShrubObjectColumns()


def shrub_csv_to_transform_name(csv_name: str) -> str:
    stem = Path(csv_name).stem
    return f"{stem}toALS.txt"


def apply_homogeneous_transform_xy(M4x4: np.ndarray, x: np.ndarray, y: np.ndarray, z0: float = 0.0):
    pts = np.stack([x, y, np.full_like(x, z0, dtype=float), np.ones_like(x, dtype=float)], axis=0)
    out = M4x4 @ pts
    return out[0], out[1], out[2]


def get_tile_bounds_xy(meta: dict) -> tuple[float, float, float, float]:
    nb = meta.get("native_bounds")
    if not isinstance(nb, dict):
        raise ValueError(f"ALS metadata missing native_bounds for {meta.get('source_file')}")
    return float(nb["minx"]), float(nb["miny"]), float(nb["maxx"]), float(nb["maxy"])


def choose_best_als_tile(x_als: np.ndarray, y_als: np.ndarray, als_meta: list[dict]) -> dict:
    best = None
    best_count = -1
    for meta in als_meta:
        minx, miny, maxx, maxy = get_tile_bounds_xy(meta)
        inside = ((x_als >= minx) & (x_als <= maxx) & (y_als >= miny) & (y_als <= maxy))
        count = int(inside.sum())
        if count > best_count:
            best_count = count
            best = meta
    if best is None:
        raise RuntimeError("Could not choose an ALS tile for transformed shrubs.")
    return best


def transform_objects_to_als(
    df: pd.DataFrame,
    transform_path: str | Path,
    als_meta: list[dict],
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    M = parse_transform_txt(Path(transform_path))
    x = out[COLS.x_tls].to_numpy(dtype=float)
    y = out[COLS.y_tls].to_numpy(dtype=float)
    x_als, y_als, _ = apply_homogeneous_transform_xy(M, x, y)
    out[COLS.x_als] = x_als
    out[COLS.y_als] = y_als

    tile = choose_best_als_tile(x_als, y_als, als_meta)
    minx, miny, maxx, maxy = get_tile_bounds_xy(tile)
    inside = ((x_als >= minx) & (x_als <= maxx) & (y_als >= miny) & (y_als <= maxy))
    out[COLS.transform_confidence] = inside.astype(float)
    return out, tile
