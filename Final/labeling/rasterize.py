from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from Final.labeling.alignment import GridSpec
from Final.config import ProjectConfig
from Final.labeling.confidence import boundary_confidence_from_mask, compose_pixel_confidence
from Final.labeling.subspace_reduction import filter_small_mask_components
from Final.models import ShrubObjectColumns


COLS = ShrubObjectColumns()


def draw_filled_circle(mask: np.ndarray, center_row: float, center_col: float, radius_px_y: float, radius_px_x: float, value: int = 1):
    h, w = mask.shape
    r0 = max(0, int(math.floor(center_row - radius_px_y)))
    r1 = min(h - 1, int(math.ceil(center_row + radius_px_y)))
    c0 = max(0, int(math.floor(center_col - radius_px_x)))
    c1 = min(w - 1, int(math.ceil(center_col + radius_px_x)))

    yy, xx = np.ogrid[r0:r1 + 1, c0:c1 + 1]
    rr = ((yy - center_row) / max(radius_px_y, 1e-6)) ** 2 + ((xx - center_col) / max(radius_px_x, 1e-6)) ** 2
    inside = rr <= 1.0
    mask[r0:r1 + 1, c0:c1 + 1][inside] = value
    return inside, (r0, r1, c0, c1)


def rasterize_objects(
    df: pd.DataFrame,
    grid: GridSpec,
    cfg: ProjectConfig,
    *,
    boundary_mode: str = "radial",
    apply_mask_subspace_reduction: bool = False,
    min_component_pixels: int = 4,
):
    binary = np.full((grid.height, grid.width), cfg.raster.background_value, dtype=np.uint8)
    confidence = np.full((grid.height, grid.width), cfg.raster.confidence_background, dtype=np.float32)
    object_id = np.zeros((grid.height, grid.width), dtype=np.int32)

    for _, row in df.iterrows():
        if not bool(row.get(COLS.valid_object, True)):
            continue

        r = float(row[COLS.row])
        c = float(row[COLS.col])
        rad_m = float(row[COLS.radius_m])
        rad_px_x = rad_m / max(grid.pixel_size_x, 1e-6)
        rad_px_y = rad_m / max(grid.pixel_size_y, 1e-6)

        temp_patch = np.zeros((grid.height, grid.width), dtype=np.uint8)
        _, bounds = draw_filled_circle(temp_patch, r, c, rad_px_y, rad_px_x, value=1)

        r0, r1, c0, c1 = bounds
        local_mask = temp_patch[r0:r1 + 1, c0:c1 + 1].astype(bool)

        binary[r0:r1 + 1, c0:c1 + 1][local_mask] = cfg.raster.shrub_value
        object_id[r0:r1 + 1, c0:c1 + 1][local_mask] = int(row[COLS.object_id])

        local_boundary_conf = boundary_confidence_from_mask(
            local_mask.astype(np.uint8),
            mode=boundary_mode,
            center=cfg.raster.confidence_center,
            edge=cfg.raster.confidence_edge,
        )

        local_conf = compose_pixel_confidence(
            local_boundary_conf,
            object_confidence=row.get(COLS.object_confidence, 1.0),
            temporal_confidence_value=row.get(COLS.temporal_confidence, np.nan),
            transform_confidence_value=row.get(COLS.transform_confidence, np.nan),
        )

        confidence[r0:r1 + 1, c0:c1 + 1][local_mask] = np.maximum(
            confidence[r0:r1 + 1, c0:c1 + 1][local_mask],
            local_conf[local_mask],
        )

    if apply_mask_subspace_reduction:
        binary, confidence, object_id = filter_small_mask_components(
            binary,
            confidence,
            object_id,
            min_component_pixels=min_component_pixels,
        )

    return binary, confidence, object_id


def write_single_band_geotiff(path: str | Path, array: np.ndarray, grid: GridSpec, dtype=None, nodata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = dtype or array.dtype
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(array.astype(dtype), 1)
    return path


def resample_single_band(array: np.ndarray, grid: GridSpec, target_resolution_m: float, resampling=Resampling.nearest):
    scale_x = target_resolution_m / grid.pixel_size_x
    scale_y = target_resolution_m / grid.pixel_size_y
    dst_width = max(1, int(round(grid.width / scale_x)))
    dst_height = max(1, int(round(grid.height / scale_y)))

    dst_transform = Affine(
        target_resolution_m, grid.transform.b, grid.transform.c,
        grid.transform.d, -target_resolution_m, grid.transform.f,
    )
    dst = np.zeros((dst_height, dst_width), dtype=array.dtype)
    reproject(
        source=array,
        destination=dst,
        src_transform=grid.transform,
        src_crs=grid.crs,
        dst_transform=dst_transform,
        dst_crs=grid.crs,
        resampling=resampling,
    )
    new_grid = GridSpec(dst_transform, dst_width, dst_height, grid.crs, target_resolution_m, target_resolution_m)
    return dst, new_grid
