from __future__ import annotations

from dataclasses import asdict
from typing import Any
import time

from Final.coordination import CoordinationManager
from Final.experiment_controller import ExperimentController, TrialRecord


class WorkUnitScheduler:
    def __init__(self, controller: ExperimentController, coordination: CoordinationManager):
        self.controller = controller
        self.coordination = coordination

    def effective_priority(self, unit: dict, trial_units: list[dict]) -> tuple[int, int, str]:
        base = int(unit.get("priority", 100))
        unlocked = 0

        for other in trial_units:
            deps = other.get("dependencies") or []
            if unit.get("unit_id") in deps:
                unlocked += 1
            elif unit.get("work_key") in deps:
                unlocked += 1
            elif unit.get("stage_name") in deps:
                unlocked += 1

        return (base, -unlocked, unit.get("unit_id"))

    def refresh_trial_units(self, trial: TrialRecord, pipeline, runtime_report=None) -> TrialRecord:
        runtime_report = runtime_report or pipeline.runtime_report()
        units = pipeline.enumerate_work_units(
            trial_id=trial.trial_id,
            config_signature=pipeline.config_signature(),
            runtime_report=runtime_report,
        )
        self.controller.upsert_work_units(trial, units)
        self.controller.save_trial(trial)
        return trial

    def select_next_runnable_unit(self, trial_units: list[dict]) -> dict | None:
        runnable = [
            u for u in trial_units
            if u.get("status") == "pending"
            and u.get("runtime_eligible", True)
            and not u.get("dependencies")
        ]
        if not runnable:
            return None

        runnable = sorted(
            runnable,
            key=lambda u: self.effective_priority(u, trial_units),
        )
        return runnable[0]

    def select_next_job(self, *, trials: list[TrialRecord], pipelines: dict[str, Any]):
        candidates = []

        for trial in trials:
            for pipeline_name, pipeline in pipelines.items():
                if pipeline_name not in trial.section_configs:
                    continue

                runtime_report = pipeline.runtime_report()
                trial = self.refresh_trial_units(trial, pipeline, runtime_report=runtime_report)

                unit = self.select_next_runnable_unit(list(trial.work_units.values()))
                if unit is not None:
                    candidates.append((trial, pipeline, unit))

        if not candidates:
            return None

        candidates = sorted(
            candidates,
            key=lambda x: (
                x[2].get("priority", 100),
                x[0].trial_id,
                x[2].get("unit_id"),
            ),
        )
        return candidates[0]

    def claim_unit(self, *, trial: TrialRecord, pipeline, unit: dict) -> tuple[bool, dict]:
        owner_id = self.coordination.default_owner_id()

        ok, payload = self.coordination.claim_work_if_available(
            experiment_name=self.controller.experiment_name,
            trial_id=trial.trial_id,
            pipeline_name=pipeline.pipeline_name,
            stage_name=unit["stage_name"],
            work_key=unit["work_key"],
            owner_id=owner_id,
            ttl_sec=900,
            payload={
                "unit_id": unit["unit_id"],
                "config_signature": unit["config_signature"],
                "scope": unit["scope"],
                "site_id": unit.get("site_id"),
                "plot_id": unit.get("plot_id"),
                "source_version": unit.get("source_version"),
            },
        )
        if ok:
            self.controller.update_work_unit_status(
                trial,
                unit_id=unit["unit_id"],
                status="claimed",
                owner_id=owner_id,
                extra={"claimed_at": payload.get("claimed_at"), "heartbeat_at": payload.get("heartbeat_at")},
            )
            self.controller.save_trial(trial)
        return ok, payload

    def run_claimed_job(self, *, trial: TrialRecord, pipeline, unit: dict, state):
        owner_id = self.coordination.default_owner_id()

        self.controller.update_work_unit_status(
            trial,
            unit_id=unit["unit_id"],
            status="running",
            owner_id=owner_id,
        )
        self.controller.save_trial(trial)

        self.coordination.heartbeat(
            experiment_name=self.controller.experiment_name,
            trial_id=trial.trial_id,
            pipeline_name=pipeline.pipeline_name,
            stage_name=unit["stage_name"],
            work_key=unit["work_key"],
            owner_id=owner_id,
        )

        try:
            result = pipeline.run_work_unit(unit, trial_id=trial.trial_id, state=state)

            self.coordination.mark_complete(
                experiment_name=self.controller.experiment_name,
                trial_id=trial.trial_id,
                pipeline_name=pipeline.pipeline_name,
                stage_name=unit["stage_name"],
                work_key=unit["work_key"],
                owner_id=owner_id,
                payload_update={"result_type": str(type(result).__name__)},
            )

            self.controller.update_work_unit_status(
                trial,
                unit_id=unit["unit_id"],
                status="complete",
                owner_id=owner_id,
            )
            self.controller.save_trial(trial)
            return result, state

        except Exception as e:
            self.coordination.mark_failed(
                experiment_name=self.controller.experiment_name,
                trial_id=trial.trial_id,
                pipeline_name=pipeline.pipeline_name,
                stage_name=unit["stage_name"],
                work_key=unit["work_key"],
                owner_id=owner_id,
                error=str(e),
            )
            self.controller.update_work_unit_status(
                trial,
                unit_id=unit["unit_id"],
                status="failed",
                owner_id=owner_id,
                note=str(e),
            )
            self.controller.save_trial(trial)
            raise

    def run_labeling_grid_search(
        self,
        *,
        trials: list[TrialRecord],
        pipelines: dict[str, Any],
        state,
        max_idle_polls: int = 20,
        idle_sleep_sec: float = 15.0,
        progress_callback=None,
    ):
        idle_polls = 0

        while True:
            next_job = self.select_next_job(trials=trials, pipelines=pipelines)

            if next_job is None:
                idle_polls += 1
                if progress_callback is not None:
                    progress_callback(trials=trials, pipelines=pipelines, active_job=None)

                if idle_polls >= max_idle_polls:
                    break

                time.sleep(idle_sleep_sec)
                continue

            idle_polls = 0
            trial, pipeline, unit = next_job

            claimed, _ = self.claim_unit(trial=trial, pipeline=pipeline, unit=unit)
            if not claimed:
                continue

            if progress_callback is not None:
                progress_callback(trials=trials, pipelines=pipelines, active_job=(trial, pipeline, unit))

            self.run_claimed_job(trial=trial, pipeline=pipeline, unit=unit, state=state)

        return trials, state