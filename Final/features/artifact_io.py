from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from Final.artifact_store import (
    ArtifactStore,
    LocalArtifactStore,
    HybridArtifactStore,
    DriveRegistryArtifactStore,
)
from Final.models import ArtifactSpec, StorageTier
from Final.shared_utils import ensure_dir

from Final.features.config import fe_cfg
from Final.features.artifact_specs import FE_ARTIFACT_SPECS


def current_fe_config_signature() -> str:
    payload = {
        "version": fe_cfg.version,
        "sources": fe_cfg.enabled_sources,
        "source_order": fe_cfg.source_order,
        "canonical_grid_source": fe_cfg.canonical_grid_source,
        "naip": fe_cfg.naip.__dict__,
        "als": fe_cfg.als.__dict__,
        "terrain": fe_cfg.terrain.__dict__,
        "rap": fe_cfg.rap.__dict__,
        "chunking": fe_cfg.chunking.__dict__,
        "object_agg": fe_cfg.object_agg.__dict__,
    }
    return str(abs(hash(json.dumps(payload, sort_keys=True, default=str))))[:16]


def artifact_spec_for_key(artifact_key: str) -> ArtifactSpec:
    return FE_ARTIFACT_SPECS[artifact_key]


def render_artifact_rel_path(
    artifact_key: str,
    *,
    site_id: str,
    config_signature: str | None = None,
    source_name: str | None = None,
    family_name: str | None = None,
    chunk_id: str | None = None,
    filename: str | None = None,
) -> str:
    spec = artifact_spec_for_key(artifact_key)
    config_signature = config_signature or current_fe_config_signature()
    return spec.rel_path_template.format(
        site_id=site_id,
        config_signature=config_signature,
        source_name=source_name or "unknown_source",
        family_name=family_name or "unknown_family",
        chunk_id=chunk_id or "whole",
        filename=filename or "artifact.bin",
    )


def local_artifact_abs_path(rel_path: str, *, artifact_store: ArtifactStore) -> Path:
    if isinstance(artifact_store, LocalArtifactStore):
        return artifact_store.storage_root / rel_path
    if isinstance(artifact_store, HybridArtifactStore):
        return ensure_dir(fe_cfg.cache_root / "_artifact_staging") / rel_path
    return ensure_dir(fe_cfg.cache_root / "_remote_stage") / rel_path


@dataclass
class PersistedArtifactRecord:
    artifact_key: str
    rel_path: str
    local_path: str | None = None
    remote_ref: str | None = None
    storage_tier: str | None = None
    exists_local: bool = False
    exists_remote: bool = False
    pruned_local: bool = False
    notes: list[str] = field(default_factory=list)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def remote_artifact_exists(rel_path: str, *, artifact_store: ArtifactStore) -> bool:
    try:
        if isinstance(artifact_store, LocalArtifactStore):
            return False
        if isinstance(artifact_store, HybridArtifactStore):
            return artifact_store.remote_store.exists(rel_path) if artifact_store.remote_store else False
        return artifact_store.exists(rel_path)
    except Exception:
        return False


def push_artifact_if_needed(
    local_path: str | Path,
    *,
    artifact_key: str,
    rel_path: str,
    artifact_store: ArtifactStore,
) -> PersistedArtifactRecord:
    spec = artifact_spec_for_key(artifact_key)
    local_path = Path(local_path)

    rec = PersistedArtifactRecord(
        artifact_key=artifact_key,
        rel_path=rel_path,
        local_path=str(local_path),
        storage_tier=spec.storage_tier.value,
        exists_local=local_path.exists(),
    )

    if not local_path.exists():
        rec.notes.append("Local artifact missing before push.")
        return rec

    should_push_remote = (
        fe_cfg.persistence.push_large_artifacts_to_remote
        and spec.storage_tier in {StorageTier.LOCAL_THEN_REMOTE, StorageTier.REMOTE_ONLY}
        and isinstance(artifact_store, (HybridArtifactStore, DriveRegistryArtifactStore))
    )

    if should_push_remote:
        rec.remote_ref = artifact_store.push(local_path, rel_path=rel_path)
        rec.exists_remote = True
    else:
        rec.notes.append("Remote push skipped by policy or store type.")

    return rec


def prune_local_artifact_if_allowed(
    rec: PersistedArtifactRecord,
    *,
    local_path: str | Path,
    artifact_store: ArtifactStore,
) -> PersistedArtifactRecord:
    spec = artifact_spec_for_key(rec.artifact_key)
    local_path = Path(local_path)

    if not fe_cfg.persistence.prune_local_after_remote_push:
        rec.notes.append("Local pruning disabled by config.")
        return rec

    if not spec.prune_local_after_push:
        rec.notes.append("Artifact spec does not allow local prune.")
        return rec

    if fe_cfg.persistence.verify_remote_before_prune:
        if not remote_artifact_exists(rec.rel_path, artifact_store=artifact_store):
            rec.notes.append("Remote artifact not verified; skipping prune.")
            return rec

    if local_path.exists():
        local_path.unlink()
        rec.pruned_local = True
        rec.exists_local = False
        rec.notes.append("Local artifact pruned after verified remote push.")

    return rec


def persist_json_artifact(
    payload: dict[str, Any],
    *,
    artifact_key: str,
    site_id: str,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
    source_name: str | None = None,
    family_name: str | None = None,
    chunk_id: str | None = None,
) -> PersistedArtifactRecord:
    rel_path = render_artifact_rel_path(
        artifact_key,
        site_id=site_id,
        config_signature=config_sig,
        source_name=source_name,
        family_name=family_name,
        chunk_id=chunk_id,
    )
    local_path = local_artifact_abs_path(rel_path, artifact_store=artifact_store)
    write_json(local_path, payload)

    rec = push_artifact_if_needed(local_path, artifact_key=artifact_key, rel_path=rel_path, artifact_store=artifact_store)
    return prune_local_artifact_if_allowed(rec, local_path=local_path, artifact_store=artifact_store)


def persist_npz_artifact(
    arrays: dict[str, np.ndarray],
    *,
    artifact_key: str,
    site_id: str,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
    source_name: str,
    family_name: str,
    chunk_id: str,
) -> PersistedArtifactRecord:
    rel_path = render_artifact_rel_path(
        artifact_key,
        site_id=site_id,
        config_signature=config_sig,
        source_name=source_name,
        family_name=family_name,
        chunk_id=chunk_id,
    )
    local_path = local_artifact_abs_path(rel_path, artifact_store=artifact_store)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(local_path, **arrays)

    rec = push_artifact_if_needed(local_path, artifact_key=artifact_key, rel_path=rel_path, artifact_store=artifact_store)
    return prune_local_artifact_if_allowed(rec, local_path=local_path, artifact_store=artifact_store)


def persist_existing_file_artifact(
    local_path: str | Path,
    *,
    artifact_key: str,
    site_id: str,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
    source_name: str | None = None,
    family_name: str | None = None,
    chunk_id: str | None = None,
    filename: str | None = None,
) -> PersistedArtifactRecord:
    local_path = Path(local_path)
    rel_path = render_artifact_rel_path(
        artifact_key,
        site_id=site_id,
        config_signature=config_sig,
        source_name=source_name,
        family_name=family_name,
        chunk_id=chunk_id,
        filename=filename or local_path.name,
    )
    rec = push_artifact_if_needed(local_path, artifact_key=artifact_key, rel_path=rel_path, artifact_store=artifact_store)
    return prune_local_artifact_if_allowed(rec, local_path=local_path, artifact_store=artifact_store)


def try_load_json_artifact(
    *,
    artifact_key: str,
    site_id: str,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
    source_name: str | None = None,
    family_name: str | None = None,
    chunk_id: str | None = None,
) -> dict[str, Any] | None:
    rel_path = render_artifact_rel_path(
        artifact_key,
        site_id=site_id,
        config_signature=config_sig,
        source_name=source_name,
        family_name=family_name,
        chunk_id=chunk_id,
    )
    local_path = local_artifact_abs_path(rel_path, artifact_store=artifact_store)

    if local_path.exists():
        return read_json(local_path)

    if remote_artifact_exists(rel_path, artifact_store=artifact_store):
        pulled = artifact_store.pull(rel_path, local_path=local_path)
        payload = read_json(pulled)

        if isinstance(artifact_store, HybridArtifactStore) and local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass

        return payload

    return None