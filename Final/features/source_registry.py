from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Any

import numpy as np

from Final.shared_utils import get_logger
from Final.artifact_store import ArtifactStore
from Final.features.config import fe_cfg
from Final.features.models import (
    CanonicalGrid,
    ChunkManifest,
    ChunkRecord,
    FeatureFamilySpec,
    RasterLayerRecord,
    RasterStackRegistry,
    SourceRasterBundle,
)
from Final.features.artifact_io import (
    current_fe_config_signature,
    persist_npz_artifact,
    render_artifact_rel_path,
    local_artifact_abs_path,
    remote_artifact_exists,
)
from Final.features.chunking import (
    expanded_chunk_bounds,
    crop_chunk_interior,
    slice_chunk_from_arrays,
)
from Final.features.stack_registry import append_layer_record, deduplicate_stack_registry, save_stack_registry
from Final.features.array_io import load_npz_dict

LOGGER = get_logger("features.source_registry")


FamilyComputeFn = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def enabled_family_specs(
    specs: dict[str, FeatureFamilySpec],
    *,
    enabled_keys: tuple[str, ...] | None = None,
) -> dict[str, FeatureFamilySpec]:
    if enabled_keys is None:
        return dict(specs)
    enabled = set(enabled_keys)
    return {k: v for k, v in specs.items() if k in enabled}


def _chunk_npz_rel_path(
    *,
    site_id: str,
    source_name: str,
    family_name: str,
    chunk_id: str,
) -> str:
    return render_artifact_rel_path(
        "family_chunk_npz",
        site_id=site_id,
        config_signature=current_fe_config_signature(),
        source_name=source_name,
        family_name=family_name,
        chunk_id=chunk_id,
    )


def _load_family_chunk_payload_if_present(
    *,
    artifact_store: ArtifactStore,
    site_id: str,
    source_name: str,
    family_name: str,
    chunk_id: str,
) -> dict[str, np.ndarray] | None:
    rel_path = _chunk_npz_rel_path(
        site_id=site_id,
        source_name=source_name,
        family_name=family_name,
        chunk_id=chunk_id,
    )
    local_path = local_artifact_abs_path(rel_path)

    if local_path.exists():
        LOGGER.info(
            "FAMILY CHUNK CACHE HIT (local) | site=%s | source=%s | family=%s | chunk=%s",
            site_id, source_name, family_name, chunk_id,
        )
        return load_npz_dict(local_path)

    if remote_artifact_exists(rel_path):
        LOGGER.info(
            "FAMILY CHUNK CACHE HIT (remote) | site=%s | source=%s | family=%s | chunk=%s",
            site_id, source_name, family_name, chunk_id,
        )
        pulled = artifact_store.pull(rel_path, local_path=local_path)
        payload = load_npz_dict(pulled)
        try:
            if Path(pulled).exists():
                Path(pulled).unlink()
        except Exception:
            pass
        return payload

    return None


def register_family_chunk_outputs(
    registry: RasterStackRegistry,
    *,
    persisted_rec,
    site_id: str,
    source_name: str,
    family_name: str,
    chunk_id: str,
    arrays: dict[str, np.ndarray],
) -> RasterStackRegistry:
    for layer_name, arr in arrays.items():
        rec = RasterLayerRecord(
            site_id=site_id,
            source_name=source_name,
            family_name=family_name,
            layer_name=f"{layer_name}::{chunk_id}",
            chunk_id=chunk_id,
            shape=tuple(arr.shape),
            dtype=str(arr.dtype),
            rel_path=persisted_rec.rel_path,
            local_path=persisted_rec.local_path,
            remote_ref=persisted_rec.remote_ref,
        )
        append_layer_record(registry, rec)
    return registry


def compute_family_chunk(
    *,
    raw_bundle: SourceRasterBundle,
    record: ChunkRecord,
    canonical_grid: CanonicalGrid,
    family_spec: FeatureFamilySpec,
    compute_fn: FamilyComputeFn,
) -> dict[str, np.ndarray]:
    expanded_arrays, expanded_bounds = slice_chunk_from_arrays(
        raw_bundle.arrays,
        record,
        full_height=canonical_grid.height,
        full_width=canonical_grid.width,
        halo_px=family_spec.required_halo_px,
    )
    expanded_outputs = compute_fn(expanded_arrays)

    row0 = record.row_start - expanded_bounds["row_start"]
    row1 = row0 + record.height
    col0 = record.col_start - expanded_bounds["col_start"]
    col1 = col0 + record.width

    return crop_chunk_interior(
        expanded_outputs,
        row0=row0,
        row1=row1,
        col0=col0,
        col1=col1,
    )


def run_source_chunked_pipeline(
    *,
    artifact_store: ArtifactStore,
    site_id: str,
    source_name: str,
    raw_bundle: SourceRasterBundle,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    family_specs: dict[str, FeatureFamilySpec],
    family_compute_fns: dict[str, FamilyComputeFn],
    family_cfg_payloads: dict[str, dict[str, Any]] | None = None,
    registry: RasterStackRegistry,
) -> RasterStackRegistry:
    family_cfg_payloads = family_cfg_payloads or {}

    LOGGER.info(
        "SOURCE CHUNKED RUN | site=%s | source=%s | families=%s | n_chunks=%d",
        site_id,
        source_name,
        tuple(family_specs.keys()),
        len(chunk_manifest.records),
    )

    for family_name, family_spec in family_specs.items():
        LOGGER.info(
            "SOURCE FAMILY START | site=%s | source=%s | family=%s",
            site_id,
            source_name,
            family_name,
        )
        compute_fn = family_compute_fns[family_name]

        for record in chunk_manifest.records:
            cached = _load_family_chunk_payload_if_present(
                artifact_store=artifact_store,
                site_id=site_id,
                source_name=source_name,
                family_name=family_name,
                chunk_id=record.chunk_id,
            )
            if cached is not None:
                registry = register_family_chunk_outputs(
                    registry,
                    persisted_rec=type("Tmp", (), {
                        "rel_path": _chunk_npz_rel_path(
                            site_id=site_id,
                            source_name=source_name,
                            family_name=family_name,
                            chunk_id=record.chunk_id,
                        ),
                        "local_path": None,
                        "remote_ref": None,
                    })(),
                    site_id=site_id,
                    source_name=source_name,
                    family_name=family_name,
                    chunk_id=record.chunk_id,
                    arrays=cached,
                )
                continue

            arrays = compute_family_chunk(
                raw_bundle=raw_bundle,
                record=record,
                canonical_grid=canonical_grid,
                family_spec=family_spec,
                compute_fn=compute_fn,
            )

            persisted = persist_npz_artifact(
                arrays,
                artifact_key="family_chunk_npz",
                site_id=site_id,
                artifact_store=artifact_store,
                config_sig=current_fe_config_signature(),
                source_name=source_name,
                family_name=family_name,
                chunk_id=record.chunk_id,
            )

            registry = register_family_chunk_outputs(
                registry,
                persisted_rec=persisted,
                site_id=site_id,
                source_name=source_name,
                family_name=family_name,
                chunk_id=record.chunk_id,
                arrays=arrays,
            )

        LOGGER.info(
            "SOURCE FAMILY DONE | site=%s | source=%s | family=%s",
            site_id,
            source_name,
            family_name,
        )

    registry = deduplicate_stack_registry(registry)
    save_stack_registry(registry, artifact_store=artifact_store)

    LOGGER.info(
        "SOURCE CHUNKED RUN COMPLETE | site=%s | source=%s | registered_layers=%d",
        site_id,
        source_name,
        len(registry.layers),
    )
    return registry