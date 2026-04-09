from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as rio_transform

from Final.config import ProjectConfig
from Final.models import ShrubObjectColumns


COLS = ShrubObjectColumns()


@dataclass
class GridSpec:
    transform: object
    width: int
    height: int
    crs: CRS
    pixel_size_x: float
    pixel_size_y: float


def naip_pixel_size_x(src) -> float:
    return float(abs(src.transform.a))


def naip_pixel_size_y(src) -> float:
    return float(abs(src.transform.e))


def build_aligned_grid_from_points(x_img: np.ndarray, y_img: np.ndarray, src, pad_m: float) -> GridSpec:
    px = naip_pixel_size_x(src)
    py = naip_pixel_size_y(src)
    xmin = float(np.min(x_img) - pad_m)
    xmax = float(np.max(x_img) + pad_m)
    ymin = float(np.min(y_img) - pad_m)
    ymax = float(np.max(y_img) + pad_m)

    col0, row_top = (~src.transform) * (xmin, ymax)
    col1, row_bot = (~src.transform) * (xmax, ymin)

    col0 = math.floor(col0)
    row_top = math.floor(row_top)
    col1 = math.ceil(col1)
    row_bot = math.ceil(row_bot)

    width = int(col1 - col0)
    height = int(row_bot - row_top)
    if width <= 0 or height <= 0:
        raise ValueError("Computed output grid has non-positive dimensions.")

    x0, y0 = src.transform * (col0, row_top)
    dst_transform = from_origin(x0, y0, px, py)
    return GridSpec(dst_transform, width, height, src.crs, px, py)


def xy_to_rowcol(transform, x, y):
    inv = ~transform
    cols, rows = inv * (x, y)
    return np.asarray(rows), np.asarray(cols)


def align_objects_to_naip(
    df: pd.DataFrame,
    naip_path,
    tile_wkt: str,
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, GridSpec]:
    out = df.copy()
    with rasterio.open(naip_path) as src:
        x_img, y_img = rio_transform(CRS.from_wkt(tile_wkt), src.crs, list(out[COLS.x_als]), list(out[COLS.y_als]))
        x_img = np.asarray(x_img, dtype=float)
        y_img = np.asarray(y_img, dtype=float)
        grid = build_aligned_grid_from_points(x_img, y_img, src, cfg.raster.pad_m)
        rows, cols = xy_to_rowcol(grid.transform, x_img, y_img)

    out[COLS.x_naip] = x_img
    out[COLS.y_naip] = y_img
    out[COLS.row] = rows
    out[COLS.col] = cols
    return out, grid
