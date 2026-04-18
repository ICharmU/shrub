from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import pandas as pd
from rasterio.windows import Window

from Final.artifact_store import ArtifactStore
from Final.features.config import fe_cfg
from Final.features.models import CanonicalGrid, ChunkManifest, ChunkRecord
from Final.features.artifact_io import (
    read_json,
    write_json,
    render_artifact_rel_path,
    local_artifact_abs_path,
    remote_artifact_exists,
    try_load_json_artifact,
    persist_json_artifact,
)


def _stable_sig(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


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


def serialize_chunk_manifest(manifest: ChunkManifest) -> dict:
    return {
        "site_id": manifest.site_id,
        "chunk_size_px": manifest.chunk_size_px,
        "halo_px_default": manifest.halo_px_default,
        "records": [asdict(r) for r in manifest.records],
    }


def deserialize_chunk_manifest(payload: dict) -> ChunkManifest:
    return ChunkManifest(
        site_id=payload["site_id"],
        chunk_size_px=int(payload["chunk_size_px"]),
        halo_px_default=int(payload["halo_px_default"]),
        records=[ChunkRecord(**row) for row in payload["records"]],
    )


def chunk_manifest_cache_dir(site_id: str, data_signature: str, config_signature: str) -> Path:
    return fe_cfg.cache_root / "chunk_manifest" / site_id / data_signature / config_signature


def build_chunk_manifest(
    grid: CanonicalGrid,
    *,
    artifact_store: ArtifactStore,
    force_refresh: bool = False,
) -> ChunkManifest:
    chunk_size = fe_cfg.chunking.chunk_size_px
    halo = fe_cfg.chunking.halo_px_default

    data_sig = _stable_sig(
        {
            "site_id": grid.site_id,
            "width": grid.width,
            "height": grid.height,
            "transform": list(grid.transform)[:6],
            "crs": str(grid.crs),
            "canonical_grid_source": grid.source_name,
        }
    )
    config_sig = _stable_sig(
        {
            "chunk_size_px": chunk_size,
            "halo_px_default": halo,
            "version": fe_cfg.version,
        }
    )

    rel_path = render_artifact_rel_path(
        "chunk_manifest",
        site_id=grid.site_id,
        config_signature=config_sig,
    )

    cache_dir = chunk_manifest_cache_dir(grid.site_id, data_sig, config_sig)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_json = cache_dir / "chunk_manifest.json"

    if not force_refresh:
        if cache_json.exists():
            payload = read_json(cache_json)
            return deserialize_chunk_manifest(payload)

        payload = try_load_json_artifact(
            artifact_key="chunk_manifest",
            site_id=grid.site_id,
            artifact_store=artifact_store,
            config_sig=config_sig,
        )
        if payload is not None:
            write_json(cache_json, payload)
            return deserialize_chunk_manifest(payload)

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

    manifest = ChunkManifest(
        site_id=grid.site_id,
        chunk_size_px=chunk_size,
        halo_px_default=halo,
        records=records,
    )

    payload = serialize_chunk_manifest(manifest)
    write_json(cache_json, payload)

    persist_json_artifact(
        payload,
        artifact_key="chunk_manifest",
        site_id=grid.site_id,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )

    return manifest