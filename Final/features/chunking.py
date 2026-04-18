from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from rasterio.windows import Window

from Final.features.config import fe_cfg
from Final.features.models import CanonicalGrid, ChunkManifest, ChunkRecord
from Final.features.artifact_io import read_json, write_json


def chunk_manifest_frame(manifest: ChunkManifest) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in manifest.records])


def expanded_chunk_bounds(
    record: ChunkRecord,
    *,
    height: int,
    width: int,
    halo_px: int | None = None,
) -> dict[str, int]:
    halo = record.halo_px if halo_px is None else halo_px
    return {
        "row_start": max(0, record.row_start - halo),
        "row_end": min(height, record.row_end + halo),
        "col_start": max(0, record.col_start - halo),
        "col_end": min(width, record.col_end + halo),
    }


def chunk_window(record: ChunkRecord) -> Window:
    return Window(
        col_off=record.col_start,
        row_off=record.row_start,
        width=record.width,
        height=record.height,
    )


def expanded_chunk_window(
    record: ChunkRecord,
    *,
    height: int,
    width: int,
    halo_px: int | None = None,
) -> Window:
    b = expanded_chunk_bounds(record, height=height, width=width, halo_px=halo_px)
    return Window(
        col_off=b["col_start"],
        row_off=b["row_start"],
        width=b["col_end"] - b["col_start"],
        height=b["row_end"] - b["row_start"],
    )


def build_chunk_manifest(grid: CanonicalGrid) -> ChunkManifest:
    chunk_size = fe_cfg.chunking.chunk_size_px
    halo = fe_cfg.chunking.halo_px_default

    records: list[ChunkRecord] = []
    chunk_idx = 0

    for row_start in range(0, grid.height, chunk_size):
        for col_start in range(0, grid.width, chunk_size):
            row_end = min(row_start + chunk_size, grid.height)
            col_end = min(col_start + chunk_size, grid.width)
            records.append(
                ChunkRecord(
                    site_id=grid.site_id,
                    chunk_id=f"chunk_{chunk_idx:05d}",
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    halo_px=halo,
                )
            )
            chunk_idx += 1

    return ChunkManifest(
        site_id=grid.site_id,
        chunk_size_px=chunk_size,
        halo_px_default=halo,
        records=records,
    )