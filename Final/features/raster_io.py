from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.enums import Resampling

from Final.features.models import CanonicalGrid, SourceRasterBundle


def validate_cached_raster(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False

    try:
        with rasterio.open(path) as src:
            h = max(1, min(16, src.height))
            w = max(1, min(16, src.width))
            src.read([1], window=Window(0, 0, w, h))
        return True
    except Exception:
        return False


def validate_optional_raster_asset(path: Any) -> bool:
    if path is None:
        return False
    try:
        return validate_cached_raster(Path(path))
    except Exception:
        return False


def infer_naip_band_names(path: str | Path) -> list[str]:
    path = Path(path)
    with rasterio.open(path) as src:
        count = src.count

    if count == 4:
        return ["red", "green", "blue", "nir"]
    if count == 3:
        return ["red", "green", "blue"]
    return [f"band_{i}" for i in range(1, count + 1)]


def read_raster_bundle(
    path: str | Path,
    *,
    site_id: str,
    source_name: str,
    band_names: list[str] | None = None,
) -> SourceRasterBundle:
    path = Path(path)

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        nodata = src.nodata

        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        if band_names is None:
            band_names = [f"{source_name}_band_{i}" for i in range(1, src.count + 1)]

        if len(band_names) != src.count:
            raise ValueError(
                f"Band name count mismatch for {path}: expected {src.count}, got {len(band_names)}"
            )

        arrays = {band_names[i]: arr[i] for i in range(src.count)}

        return SourceRasterBundle(
            site_id=site_id,
            source_name=source_name,
            arrays=arrays,
            transform=src.transform,
            crs=src.crs,
            nodata=nodata,
            metadata={
                "path": str(path),
                "width": src.width,
                "height": src.height,
                "count": src.count,
            },
        )


def reproject_band_to_grid(
    src_band: np.ndarray,
    *,
    src_transform,
    src_crs: Any,
    dst_grid: CanonicalGrid,
    resampling: Resampling = Resampling.bilinear,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    dst = np.full((dst_grid.height, dst_grid.width), dst_nodata, dtype=np.float32)

    reproject(
        source=src_band.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_grid.transform,
        dst_crs=dst_grid.crs,
        src_nodata=np.nan,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def align_bundle_to_grid(
    bundle: SourceRasterBundle,
    *,
    dst_grid: CanonicalGrid,
    resampling: Resampling,
) -> SourceRasterBundle:
    aligned = {}
    for name, band in bundle.arrays.items():
        aligned[name] = reproject_band_to_grid(
            band,
            src_transform=bundle.transform,
            src_crs=bundle.crs,
            dst_grid=dst_grid,
            resampling=resampling,
        )

    return SourceRasterBundle(
        site_id=bundle.site_id,
        source_name=bundle.source_name,
        arrays=aligned,
        transform=dst_grid.transform,
        crs=dst_grid.crs,
        nodata=np.nan,
        metadata={**bundle.metadata, "aligned_to": dst_grid.source_name},
    )


def slice_chunk_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> dict[str, np.ndarray]:
    out = {}
    for name, arr in arrays.items():
        out[name] = arr[row_start:row_end, col_start:col_end]
    return out


def crop_chunk_interior(
    expanded_arrays: dict[str, np.ndarray],
    *,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
) -> dict[str, np.ndarray]:
    cropped = {}
    for name, arr in expanded_arrays.items():
        cropped[name] = arr[row0:row1, col0:col1]
    return cropped