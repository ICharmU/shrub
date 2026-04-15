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

        if result.success:
            trial.status = "in_progress"
        else:
            trial.status = f"section_failed:{section_name}"

    def next_pending_section(self, trial: TrialRecord, section_order: list[str]) -> str | None:
        for section_name in section_order:
            if not self.section_is_complete(trial, section_name):
                return section_name
        return None