from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from Final.models import ExperimentState, PipelineRunResult


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SectionRunRecord:
    section_name: str
    config_signature: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    qa_outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class TrialRecord:
    trial_id: str
    created_at: str
    status: str
    section_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    section_runs: dict[str, SectionRunRecord] = field(default_factory=dict)
    score_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    work_units: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolution: dict[str, Any] = field(default_factory=dict)
    scheduler_meta: dict[str, Any] = field(default_factory=dict)

class ExperimentController:
    def __init__(self, experiment_root: str | Path, experiment_name: str):
        self.experiment_root = Path(experiment_root)
        self.experiment_name = experiment_name
        self.root = self.experiment_root / experiment_name
        self.root.mkdir(parents=True, exist_ok=True)

        self.trials_dir = self.root / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.root / "experiment_state.json"

    def create_state(self) -> ExperimentState:
        state = ExperimentState(
            experiment_name=self.experiment_name,
            active_modules=[],
            notes="",
        )
        self.save_state(state)
        return state

    def load_state(self) -> ExperimentState:
        if not self.state_path.exists():
            return self.create_state()
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return ExperimentState(**payload)

    def save_state(self, state: ExperimentState) -> Path:
        self.state_path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")
        return self.state_path

    def trial_path(self, trial_id: str) -> Path:
        return self.trials_dir / f"{trial_id}.json"

    def create_trial(self, trial_id: str | None = None) -> TrialRecord:
        trial_id = trial_id or f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trial = TrialRecord(
            trial_id=trial_id,
            created_at=utc_now_iso(),
            status="created",
        )
        self.save_trial(trial)
        return trial

    def load_trial(self, trial_id: str) -> TrialRecord:
        path = self.trial_path(trial_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["section_runs"] = {
            k: SectionRunRecord(**v) for k, v in payload.get("section_runs", {}).items()
        }
        return TrialRecord(**payload)

    def resolve_trial_status(self, trial: TrialRecord) -> str:
        work_units = trial.work_units or {}
        if not work_units:
            if trial.section_runs:
                # legacy fallback
                if all(rec.status == "success" for rec in trial.section_runs.values()):
                    return "success"
                if any(str(rec.status).startswith("section_failed") for rec in trial.section_runs.values()):
                    return "failed"
                return "in_progress"
            return "created"

        statuses = [u.get("status") for u in work_units.values()]
        if statuses and all(s == "complete" for s in statuses):
            return "success"
        if any(s in {"claimed", "running"} for s in statuses):
            return "running"
        if any(s == "blocked" for s in statuses):
            return "waiting"
        if any(s in {"pending"} for s in statuses):
            return "runnable"
        if any(s == "complete" for s in statuses):
            return "partial"
        if any(s == "failed" for s in statuses):
            return "partial"
        return "created"

    def update_trial_resolution(self, trial: TrialRecord) -> None:
        trial.status = self.resolve_trial_status(trial)
        self.trial_resolution_summary(trial)

    def save_trial(self, trial: TrialRecord) -> Path:
        path = self.trial_path(trial.trial_id)
        payload = asdict(trial)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def set_section_config(self, trial: TrialRecord, section_name: str, config_dict: dict) -> None:
        trial.section_configs[section_name] = dict(config_dict)

    def section_is_complete(self, trial: TrialRecord, section_name: str) -> bool:
        rec = trial.section_runs.get(section_name)
        return rec is not None and rec.status == "success"

    def record_section_result(
        self,
        trial: TrialRecord,
        *,
        section_name: str,
        config_signature: str,
        result: PipelineRunResult,
    ) -> None:
        trial.section_runs[section_name] = SectionRunRecord(
            section_name=section_name,
            config_signature=config_signature,
            status=result.status,
            started_at=None,
            finished_at=utc_now_iso(),
            metrics=result.metrics,
            qa_outputs=result.qa_outputs,
            notes=result.notes,
        )

        if not result.success and not str(result.status).startswith("partial"):
            trial.status = f"section_failed:{section_name}"
        else:
            self.update_trial_resolution(trial)

    def next_pending_section(self, trial: TrialRecord, section_order: list[str]) -> str | None:
        for section_name in section_order:
            if not self.section_is_complete(trial, section_name):
                return section_name
        return None

    def upsert_work_unit(self, trial: TrialRecord, unit: dict) -> None:
        unit_id = unit["unit_id"]
        trial.work_units[unit_id] = dict(unit)
        self.update_trial_resolution(trial)

    def upsert_work_units(self, trial: TrialRecord, units: list[dict]) -> None:
        for unit in units:
            self.upsert_work_unit(trial, unit)
        self.update_trial_resolution(trial)

    def update_work_unit_status(
        self,
        trial: TrialRecord,
        *,
        unit_id: str,
        status: str,
        owner_id: str | None = None,
        note: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if unit_id not in trial.work_units:
            raise KeyError(f"Unknown unit_id={unit_id} for trial={trial.trial_id}")

        unit = trial.work_units[unit_id]
        unit["status"] = status
        if owner_id is not None:
            unit["owner_id"] = owner_id
        if extra:
            unit.update(extra)
        if note:
            unit.setdefault("notes", []).append(note)

        self.update_trial_resolution(trial)

    def trial_resolution_summary(self, trial: TrialRecord) -> dict[str, Any]:
        work_units = trial.work_units or {}
        statuses = [u.get("status", "pending") for u in work_units.values()]

        summary = {
            "trial_id": trial.trial_id,
            "status": self.resolve_trial_status(trial),
            "total_units": len(work_units),
            "complete_units": sum(s == "complete" for s in statuses),
            "runnable_units": sum(s == "pending" for s in statuses),
            "blocked_units": sum(s == "blocked" for s in statuses),
            "failed_units": sum(s == "failed" for s in statuses),
            "ineligible_units": sum(s == "ineligible" for s in statuses),
            "claimed_units": sum(s == "claimed" for s in statuses),
            "running_units": sum(s == "running" for s in statuses),
        }
        trial.resolution = dict(summary)
        return summary

    def trials_frame(self) -> list[dict]:
        rows = []
        for path in sorted(self.trials_dir.glob("*.json")):
            trial = self.load_trial(path.stem)
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "status": trial.status,
                    "n_section_runs": len(trial.section_runs),
                    "n_work_units": len(trial.work_units),
                    **(trial.resolution or {}),
                }
            )
        return rows

    def completed_trials_frame(self) -> list[dict]:
        rows = self.trials_frame()
        return [r for r in rows if r.get("status") == "success"]
    
    def trial_health_frame(self, trial: TrialRecord) -> list[dict]:
        rows = []
        for unit in (trial.work_units or {}).values():
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "unit_id": unit.get("unit_id"),
                    "pipeline_name": unit.get("pipeline_name"),
                    "stage_name": unit.get("stage_name"),
                    "scope": unit.get("scope"),
                    "status": unit.get("status"),
                    "runtime_eligible": unit.get("runtime_eligible"),
                    "dependencies": unit.get("dependencies", []),
                    "dependency_reasons": unit.get("dependency_reasons", []),
                    "priority": unit.get("priority"),
                    "site_id": unit.get("site_id"),
                    "plot_id": unit.get("plot_id"),
                    "source_version": unit.get("source_version"),
                }
            )
        return rows