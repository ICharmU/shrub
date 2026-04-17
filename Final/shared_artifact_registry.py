from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from Final.models import SharedArtifactRecord, SharedArtifactStatus


class SharedArtifactRegistry:
    def __init__(self, artifact_store, *, root_prefix: str = "shared_artifacts"):
        self.artifact_store = artifact_store
        self.root_prefix = root_prefix

    def _rel_path(self, artifact_family: str, shared_signature: str) -> str:
        return f"{self.root_prefix}/{artifact_family}/{shared_signature}.json"

    def load(self, *, artifact_family: str, shared_signature: str) -> SharedArtifactRecord | None:
        rel_path = self._rel_path(artifact_family, shared_signature)
        try:
            local = self.artifact_store.pull(rel_path)
            payload = json.loads(Path(local).read_text(encoding="utf-8"))
            return SharedArtifactRecord(**payload)
        except Exception:
            return None

    def save(self, record: SharedArtifactRecord) -> str | None:
        rel_path = self._rel_path(record.artifact_family, record.shared_signature)
        tmp = Path.cwd() / ".shared_artifact_registry_tmp.json"
        tmp.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
        try:
            return self.artifact_store.push(tmp, rel_path=rel_path)
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass

    def upsert_requirement(
        self,
        *,
        artifact_family: str,
        shared_signature: str,
        producer_pipeline: str,
        trial_id: str,
    ) -> SharedArtifactRecord:
        rec = self.load(artifact_family=artifact_family, shared_signature=shared_signature)
        if rec is None:
            rec = SharedArtifactRecord(
                artifact_family=artifact_family,
                shared_signature=shared_signature,
                producer_pipeline=producer_pipeline,
                status=SharedArtifactStatus.MISSING,
            )
        if trial_id not in rec.required_by_trials:
            rec.required_by_trials.append(trial_id)
        self.save(rec)
        return rec

    def mark_available(
        self,
        *,
        artifact_family: str,
        shared_signature: str,
        producer_pipeline: str,
        rel_path: str | None = None,
        local_path: str | None = None,
        remote_ref: str | None = None,
        source_trial: str | None = None,
        metadata: dict | None = None,
        status: SharedArtifactStatus = SharedArtifactStatus.VALID,
    ) -> SharedArtifactRecord:
        rec = self.load(artifact_family=artifact_family, shared_signature=shared_signature)
        if rec is None:
            rec = SharedArtifactRecord(
                artifact_family=artifact_family,
                shared_signature=shared_signature,
                producer_pipeline=producer_pipeline,
            )
        rec.rel_path = rel_path or rec.rel_path
        rec.local_path = local_path or rec.local_path
        rec.remote_ref = remote_ref or rec.remote_ref
        rec.status = status
        if source_trial and source_trial not in rec.source_trials:
            rec.source_trials.append(source_trial)
        if metadata:
            rec.metadata.update(metadata)
        self.save(rec)
        return rec