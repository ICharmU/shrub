from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

from Final.models import ModuleVariant, StageCacheRecord


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json_dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def hash_payload(payload: dict) -> str:
    text = stable_json_dumps(payload)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_module_variants(module_variants: list[ModuleVariant]) -> list[dict]:
    rows = [asdict(mv) for mv in module_variants]
    rows = sorted(rows, key=lambda x: (x["module_name"], x["variant_name"]))
    return rows


def stage_manifest_path(stage_cache_dir: str | Path) -> Path:
    stage_cache_dir = Path(stage_cache_dir)
    stage_cache_dir.mkdir(parents=True, exist_ok=True)
    return stage_cache_dir / "stage_cache_manifest.json"


def write_stage_cache_manifest(
    *,
    stage_cache_dir: str | Path,
    stage_name: str,
    data_signature: str,
    config_signature: str,
    module_variants: list[ModuleVariant],
    artifact_paths: dict,
    success: bool,
    notes: list[str] | None = None,
) -> Path:
    record = StageCacheRecord(
        stage_name=stage_name,
        data_signature=data_signature,
        config_signature=config_signature,
        module_variants=normalize_module_variants(module_variants),
        artifact_paths=artifact_paths,
        success=success,
        created_at=utc_now_iso(),
        notes=notes or [],
    )
    path = stage_manifest_path(stage_cache_dir)
    path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
    return path


def read_stage_cache_manifest(stage_cache_dir: str | Path) -> StageCacheRecord | None:
    path = stage_manifest_path(stage_cache_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StageCacheRecord(**payload)


def is_valid_stage_cache(
    *,
    stage_cache_dir: str | Path,
    expected_stage_name: str,
    expected_data_signature: str,
    expected_config_signature: str,
) -> bool:
    record = read_stage_cache_manifest(stage_cache_dir)
    if record is None:
        return False
    if not record.success:
        return False
    if record.stage_name != expected_stage_name:
        return False
    if record.data_signature != expected_data_signature:
        return False
    if record.config_signature != expected_config_signature:
        return False
    return True