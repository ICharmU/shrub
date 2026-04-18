from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from Final.artifact_store import ArtifactStore
from Final.shared_utils import get_logger

from Final.features.artifact_io import (
    current_fe_config_signature,
    persist_json_artifact,
    try_load_json_artifact,
)
from Final.features.models import RasterLayerRecord, RasterStackRegistry

LOGGER = get_logger("features.stack_registry")


def stack_registry_frame(registry: RasterStackRegistry) -> pd.DataFrame:
    if not registry.layers:
        return pd.DataFrame()
    return pd.DataFrame([asdict(x) for x in registry.layers])


def init_stack_registry(
    site_id: str,
    *,
    config_signature: str | None = None,
) -> RasterStackRegistry:
    return RasterStackRegistry(
        site_id=site_id,
        config_signature=config_signature or current_fe_config_signature(),
        layers=[],
    )


def append_layer_record(
    registry: RasterStackRegistry,
    record: RasterLayerRecord,
) -> RasterStackRegistry:
    registry.layers.append(record)
    return registry


def extend_layer_records(
    registry: RasterStackRegistry,
    records: Iterable[RasterLayerRecord],
) -> RasterStackRegistry:
    registry.layers.extend(list(records))
    return registry


def deduplicate_stack_registry(registry: RasterStackRegistry) -> RasterStackRegistry:
    seen: set[tuple[str, str, str, str, str | None]] = set()
    deduped: list[RasterLayerRecord] = []

    for row in registry.layers:
        key = (
            row.source_name,
            row.family_name,
            row.layer_name,
            row.chunk_id or "",
            row.rel_path,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    registry.layers = deduped
    return registry


def save_stack_registry(
    registry: RasterStackRegistry,
    *,
    artifact_store: ArtifactStore,
):
    payload = {
        "site_id": registry.site_id,
        "config_signature": registry.config_signature,
        "layers": [asdict(x) for x in registry.layers],
    }
    return persist_json_artifact(
        payload,
        artifact_key="stack_registry",
        site_id=registry.site_id,
        artifact_store=artifact_store,
        config_sig=registry.config_signature,
    )


def load_stack_registry(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    config_signature: str | None = None,
) -> RasterStackRegistry | None:
    payload = try_load_json_artifact(
        artifact_key="stack_registry",
        site_id=site_id,
        artifact_store=artifact_store,
        config_sig=config_signature or current_fe_config_signature(),
    )
    if payload is None:
        return None

    return RasterStackRegistry(
        site_id=payload["site_id"],
        config_signature=payload["config_signature"],
        layers=[RasterLayerRecord(**row) for row in payload.get("layers", [])],
    )


def load_or_init_stack_registry(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    config_signature: str | None = None,
) -> RasterStackRegistry:
    existing = load_stack_registry(
        site_id,
        artifact_store=artifact_store,
        config_signature=config_signature,
    )
    if existing is not None:
        LOGGER.info(
            "Using existing stack registry | site=%s | layers=%d",
            site_id,
            len(existing.layers),
        )
        return deduplicate_stack_registry(existing)

    return init_stack_registry(
        site_id,
        config_signature=config_signature,
    )