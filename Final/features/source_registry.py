from __future__ import annotations

from typing import Any, Callable

import numpy as np

from Final.artifact_store import ArtifactStore
from Final.shared_utils import get_logger

from Final.features.artifact_io import (
    current_fe_config_signature,
    load_npz_dict,
    local_artifact_abs_path,
    persist_npz_artifact,
    remote_artifact_exists,
    render_artifact_rel_path,
)
from Final.features.models import (
    CanonicalGrid,
    ChunkManifest,
    FeatureFamilySpec,
    RasterLayerRecord,
    RasterStackRegistry,
    SourceRasterBundle,
)
from Final.features.raster_io import (
    align_bundle_to_grid,
    crop_chunk_interior,
    slice_chunk_from_arrays,
)
from Final.features.source_specs import SOURCE_SPECS
from Final.features.stack_registry import (
    append_layer_record,
    deduplicate_stack_registry,
    load_or_init_stack_registry,
    save_stack_registry,
)

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
    config_signature: str,
) -> str:
    return render_artifact_rel_path(
        "family_chunk_npz",
        site_id=site_id,
        config_signature=config_signature,
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
    config_signature: str,
) -> tuple[dict[str, np.ndarray] | None, str]:
    rel_path = _chunk_npz_rel_path(
        site_id=site_id,
        source_name=source_name,
        family_name=family_name,
        chunk_id=chunk_id,
        config_signature=config_signature,
    )
    local_path = local_artifact_abs_path(rel_path, artifact_store=artifact_store)

    if local_path.exists():
        LOGGER.info(
            "FAMILY CHUNK CACHE HIT (local) | site=%s | source=%s | family=%s | chunk=%s",
            site_id, source_name, family_name, chunk_id,
        )
        return load_npz_dict(local_path), rel_path

    if remote_artifact_exists(rel_path, artifact_store=artifact_store):
        LOGGER.info(
            "FAMILY CHUNK CACHE HIT (remote) | site=%s | source=%s | family=%s | chunk=%s",
            site_id, source_name, family_name, chunk_id,
        )
        pulled = artifact_store.pull(rel_path, local_path=local_path)
        try:
            return load_npz_dict(pulled), rel_path
        finally:
            try:
                pulled = local_path
                if pulled.exists():
                    pulled.unlink()
            except Exception:
                pass

    return None, rel_path


def register_family_chunk_outputs(
    registry: RasterStackRegistry,
    *,
    site_id: str,
    source_name: str,
    family_name: str,
    chunk_id: str,
    arrays: dict[str, np.ndarray],
    rel_path: str,
    local_path: str | None = None,
    remote_ref: str | None = None,
    notes: list[str] | None = None,
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
            rel_path=rel_path,
            local_path=local_path,
            remote_ref=remote_ref,
            notes=list(notes or []),
        )
        append_layer_record(registry, rec)
    return registry


def compute_family_chunk(
    *,
    raw_bundle: SourceRasterBundle,
    canonical_grid: CanonicalGrid,
    record,
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
    return crop_chunk_interior(expanded_outputs, record, expanded_bounds)


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
    registry: RasterStackRegistry | None = None,
    rebuild_registry: bool = False,
) -> RasterStackRegistry:
    config_sig = current_fe_config_signature()
    family_cfg_payloads = family_cfg_payloads or {}

    aligned_bundle = align_bundle_to_grid(
        raw_bundle,
        dst_grid=canonical_grid,
        resampling=SOURCE_SPECS[source_name].default_resampling,
    )

    if registry is None or rebuild_registry:
        registry = load_or_init_stack_registry(
            site_id,
            artifact_store=artifact_store,
            config_signature=config_sig,
        )

    LOGGER.info(
        "SOURCE CHUNKED RUN | site=%s | source=%s | families=%s | n_chunks=%d",
        site_id,
        source_name,
        tuple(family_specs.keys()),
        len(chunk_manifest.records),
    )

    for family_name, family_spec in family_specs.items():
        compute_fn = family_compute_fns[family_name]

        LOGGER.info(
            "SOURCE FAMILY START | site=%s | source=%s | family=%s",
            site_id,
            source_name,
            family_name,
        )

        for record in chunk_manifest.records:
            cached, rel_path = _load_family_chunk_payload_if_present(
                artifact_store=artifact_store,
                site_id=site_id,
                source_name=source_name,
                family_name=family_name,
                chunk_id=record.chunk_id,
                config_signature=config_sig,
            )

            if cached is not None:
                registry = register_family_chunk_outputs(
                    registry,
                    site_id=site_id,
                    source_name=source_name,
                    family_name=family_name,
                    chunk_id=record.chunk_id,
                    arrays=cached,
                    rel_path=rel_path,
                    local_path=None,
                    remote_ref=rel_path,
                    notes=[record.chunk_id, "cache-hit"],
                )
                continue

            arrays = compute_family_chunk(
                raw_bundle=aligned_bundle,
                canonical_grid=canonical_grid,
                record=record,
                family_spec=family_spec,
                compute_fn=compute_fn,
            )

            persisted = persist_npz_artifact(
                arrays,
                artifact_key="family_chunk_npz",
                site_id=site_id,
                artifact_store=artifact_store,
                config_sig=config_sig,
                source_name=source_name,
                family_name=family_name,
                chunk_id=record.chunk_id,
            )

            registry = register_family_chunk_outputs(
                registry,
                site_id=site_id,
                source_name=source_name,
                family_name=family_name,
                chunk_id=record.chunk_id,
                arrays=arrays,
                rel_path=persisted.rel_path,
                local_path=persisted.local_path,
                remote_ref=persisted.remote_ref,
                notes=[record.chunk_id] + list(persisted.notes),
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