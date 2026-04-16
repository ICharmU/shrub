from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Final.artifact_store import ArtifactStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class WorkClaim:
    work_key: str
    owner_id: str
    status: str
    claimed_at: str
    heartbeat_at: str
    ttl_sec: int
    payload: dict[str, Any] = field(default_factory=dict)


class CoordinationManager:
    def __init__(self, artifact_store: ArtifactStore, *, root_prefix: str = "coordination"):
        self.artifact_store = artifact_store
        self.root_prefix = root_prefix

    @staticmethod
    def default_owner_id() -> str:
        return socket.gethostname()

    def _rel_path(self, *parts: str) -> str:
        return "/".join([self.root_prefix, *parts])

    def sync(self) -> None:
        self.artifact_store.sync_registry()

    def load_json(self, rel_path: str) -> dict | None:
        try:
            local = self.artifact_store.pull(rel_path)
            return json.loads(Path(local).read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_json(self, rel_path: str, payload: dict) -> str | None:
        tmp = Path.cwd() / ".coordination_tmp.json"
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        try:
            return self.artifact_store.push(tmp, rel_path=rel_path)
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass

    def claim_work(
        self,
        *,
        experiment_name: str,
        trial_id: str,
        pipeline_name: str,
        stage_name: str,
        work_key: str,
        owner_id: str | None = None,
        ttl_sec: int = 900,
        payload: dict | None = None,
    ) -> tuple[bool, dict]:
        owner_id = owner_id or self.default_owner_id()
        rel_path = self._rel_path(experiment_name, trial_id, pipeline_name, stage_name, f"{work_key}.json")

        existing = self.load_json(rel_path)
        now = utc_now_iso()

        if existing is not None and existing.get("status") in {"claimed", "running"}:
            return False, existing

        claim = WorkClaim(
            work_key=work_key,
            owner_id=owner_id,
            status="claimed",
            claimed_at=now,
            heartbeat_at=now,
            ttl_sec=ttl_sec,
            payload=payload or {},
        )
        self.save_json(rel_path, asdict(claim))
        return True, asdict(claim)

    def heartbeat(
        self,
        *,
        experiment_name: str,
        trial_id: str,
        pipeline_name: str,
        stage_name: str,
        work_key: str,
        owner_id: str | None = None,
    ) -> None:
        owner_id = owner_id or self.default_owner_id()
        rel_path = self._rel_path(experiment_name, trial_id, pipeline_name, stage_name, f"{work_key}.json")
        payload = self.load_json(rel_path) or {}
        payload["owner_id"] = owner_id
        payload["status"] = payload.get("status", "running")
        payload["heartbeat_at"] = utc_now_iso()
        self.save_json(rel_path, payload)

    def mark_complete(
        self,
        *,
        experiment_name: str,
        trial_id: str,
        pipeline_name: str,
        stage_name: str,
        work_key: str,
        owner_id: str | None = None,
        payload_update: dict | None = None,
    ) -> None:
        owner_id = owner_id or self.default_owner_id()
        rel_path = self._rel_path(experiment_name, trial_id, pipeline_name, stage_name, f"{work_key}.json")
        payload = self.load_json(rel_path) or {}
        payload["owner_id"] = owner_id
        payload["status"] = "complete"
        payload["heartbeat_at"] = utc_now_iso()
        payload["finished_at"] = utc_now_iso()
        if payload_update:
            payload.setdefault("payload", {}).update(payload_update)
        self.save_json(rel_path, payload)