from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

from Final.features.models import CanonicalGrid, ChunkRecord, SourceRasterBundle
from Final.features.chunking import expanded_chunk_bounds


def validate_cached_raster(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False

    try:
        with rasterio.open(path) as src:
            if src.count < 1 or src.width <= 0 or src.height <= 0:
                return False

            windows = [
                Window(0, 0, min(32, src.width), min(32, src.height)),
                Window(
                    max(0, src.width // 2 - min(16, src.width // 2)),
                    max(0, src.height // 2 - min(16, src.height // 2)),
                    min(32, src.width),
                    min(32, src.height),
                ),
                Window(
                    max(0, src.width - min(32, src.width)),
                    max(0, src.height - min(32, src.height)),
                    min(32, src.width),
                    min(32, src.height),
                ),
            ]
            for w in windows:
                src.read([1], window=w)
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
    record: ChunkRecord,
    *,
    full_height: int,
    full_width: int,
    halo_px: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    bounds = expanded_chunk_bounds(
        record,
        height=full_height,
        width=full_width,
        halo_px=halo_px,
    )
    out = {}
    for name, arr in arrays.items():
        out[name] = arr[
            bounds["row_start"]:bounds["row_end"],
            bounds["col_start"]:bounds["col_end"],
        ]
    return out, bounds


def crop_chunk_interior(
    expanded_arrays: dict[str, np.ndarray],
    record: ChunkRecord,
    expanded_bounds: dict[str, int],
) -> dict[str, np.ndarray]:
    row0 = record.row_start - expanded_bounds["row_start"]
    row1 = row0 + record.height
    col0 = record.col_start - expanded_bounds["col_start"]
    col1 = col0 + record.width

    cropped = {}
    for name, arr in expanded_arrays.items():
        cropped[name] = arr[row0:row1, col0:col1]
    return cropped