from __future__ import annotations

from dataclasses import asdict
from typing import Any
import time
import pandas as pd

from Final.coordination import CoordinationManager
from Final.experiment_controller import ExperimentController, TrialRecord


class WorkUnitScheduler:
    def __init__(self, controller: ExperimentController, coordination: CoordinationManager):
        self.controller = controller
        self.coordination = coordination

    def scheduler_snapshot_rows(self, *, trials: list[TrialRecord]):
        rows = []
        for trial in trials:
            summary = self.controller.trial_resolution_summary(trial)
            rows.append(summary)
        return rows

    def scheduler_snapshot_frame(self, *, trials: list[TrialRecord]):
        return pd.DataFrame(self.scheduler_snapshot_rows(trials=trials))

    def current_trial_frame(self, *, trial: TrialRecord):
        rows = self.controller.trial_health_frame(trial)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        cols = [
            "trial_id", "pipeline_name", "stage_name", "scope", "status",
            "runtime_eligible", "priority", "site_id", "plot_id",
            "source_version", "dependencies", "dependency_reasons",
        ]
        keep = [c for c in cols if c in df.columns]
        return df[keep].sort_values(
            ["status", "stage_name", "priority", "site_id", "plot_id"],
            na_position="last",
        ).reset_index(drop=True)

    def candidate_frame(self, *, candidates: list[dict], top_n: int = 10):
        rows = []
        for c in candidates[:top_n]:
            unit = c["unit"]
            rows.append(
                {
                    "trial_id": c["trial"].trial_id,
                    "pipeline_name": c["pipeline"].pipeline_name,
                    "unit_id": unit.get("unit_id"),
                    "stage_name": unit.get("stage_name"),
                    "scope": unit.get("scope"),
                    "priority": unit.get("priority"),
                    "effective_priority": c.get("effective_priority"),
                    "site_id": unit.get("site_id"),
                    "plot_id": unit.get("plot_id"),
                    "source_version": unit.get("source_version"),
                }
            )
        return pd.DataFrame(rows)

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

    def refresh_trial_units(
        self,
        trial: TrialRecord,
        pipeline,
        runtime_report=None,
        force: bool = False,
    ) -> TrialRecord:
        runtime_report = runtime_report or pipeline.runtime_report()
        config_signature = pipeline.config_signature() if hasattr(pipeline, "config_signature") else ""
        new_fp = pipeline.work_unit_refresh_fingerprint(
            trial_id=trial.trial_id,
            config_signature=config_signature,
            runtime_report=runtime_report,
        )
    
        pipeline_meta = (trial.scheduler_meta or {}).get(pipeline.pipeline_name, {})
        old_fp = pipeline_meta.get("work_unit_fingerprint")
    
        has_live_units = any(
            u.get("status") in {"claimed", "running"}
            for u in (trial.work_units or {}).values()
            if u.get("pipeline_name") == pipeline.pipeline_name
        )
    
        if (not force) and old_fp == new_fp and trial.work_units and not has_live_units:
            pipeline.logger.info(
                "SCHEDULER REFRESH SKIP | trial=%s | pipeline=%s | fingerprint_unchanged=%s",
                trial.trial_id,
                pipeline.pipeline_name,
                new_fp,
            )
            return trial
    
        pipeline.logger.info(
            "SCHEDULER REFRESH RUN  | trial=%s | pipeline=%s | force=%s | old_fp=%s | new_fp=%s",
            trial.trial_id,
            pipeline.pipeline_name,
            force,
            old_fp,
            new_fp,
        )
    
        units = pipeline.enumerate_work_units(
            trial_id=trial.trial_id,
            config_signature=config_signature,
            runtime_report=runtime_report,
            register_shared_requirements=(old_fp is None),
        )
        self.controller.upsert_work_units(trial, units)
    
        trial.scheduler_meta.setdefault(pipeline.pipeline_name, {})
        trial.scheduler_meta[pipeline.pipeline_name]["work_unit_fingerprint"] = new_fp
        trial.scheduler_meta[pipeline.pipeline_name]["last_refreshed_at"] = time.time()
    
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

    # def collect_candidate_jobs(self, *, trials: list[TrialRecord], pipelines: dict[str, Any]):
    #     candidates = []
    #     total_pairs = sum(
    #         1
    #         for trial in trials
    #         for pipeline_name in pipelines.keys()
    #         if pipeline_name in trial.section_configs
    #     )
    #     done = 0
    
    #     for trial in trials:
    #         for pipeline_name, pipeline in pipelines.items():
    #             if pipeline_name not in trial.section_configs:
    #                 continue
    
    #             done += 1
    #             pipeline.logger.info(
    #                 "SCHEDULER SCAN | %d/%d | trial=%s | pipeline=%s",
    #                 done, total_pairs, trial.trial_id, pipeline_name
    #             )
    
    #             runtime_report = pipeline.runtime_report()
    #             trial = self.refresh_trial_units(trial, pipeline, runtime_report=runtime_report)
    
    #             trial_units = list(trial.work_units.values())
    #             runnable = [
    #                 u for u in trial_units
    #                 if u.get("status") == "pending"
    #                 and u.get("runtime_eligible", True)
    #                 and not u.get("dependencies")
    #             ]
    
    #             pipeline.logger.info(
    #                 "SCHEDULER SCAN DONE | trial=%s | pipeline=%s | runnable=%d | total_units=%d",
    #                 trial.trial_id,
    #                 pipeline_name,
    #                 len(runnable),
    #                 len(trial_units),
    #             )
    
    #             for unit in runnable:
    #                 eff = self.effective_priority(unit, trial_units)
    #                 candidates.append(
    #                     {
    #                         "trial": trial,
    #                         "pipeline": pipeline,
    #                         "unit": unit,
    #                         "effective_priority": eff,
    #                         "base_priority": unit.get("priority", 100),
    #                     }
    #                 )
    
    #     candidates = sorted(
    #         candidates,
    #         key=lambda x: (
    #             x["effective_priority"],
    #             x["trial"].trial_id,
    #             x["unit"].get("unit_id"),
    #         ),
    #     )
    
    #     if candidates:
    #         top = candidates[0]
    #         top_unit = top["unit"]
    #         top["pipeline"].logger.info(
    #             "SCHEDULER PICK | trial=%s | unit=%s | stage=%s | scope=%s | eff=%s | total_candidates=%d",
    #             top["trial"].trial_id,
    #             top_unit.get("unit_id"),
    #             top_unit.get("stage_name"),
    #             top_unit.get("scope"),
    #             top["effective_priority"],
    #             len(candidates),
    #         )
    #     else:
    #         # use any pipeline logger if available
    #         if pipelines:
    #             next(iter(pipelines.values())).logger.info("SCHEDULER PICK | no runnable candidates")
    
    #     return candidates

    def collect_candidate_jobs(self, *, trials, pipelines, force_refresh: bool = False):
        candidates = []
    
        for trial in trials:
            for pipeline_name, pipeline in pipelines.items():
                if pipeline_name not in trial.section_configs:
                    continue
    
                runtime_report = pipeline.runtime_report()
                trial = self.refresh_trial_units(
                    trial,
                    pipeline,
                    runtime_report=runtime_report,
                    force=force_refresh,
                )
    
                trial_units = [
                    u for u in trial.work_units.values()
                    if u.get("pipeline_name") == pipeline.pipeline_name
                ]
    
                runnable = [
                    u for u in trial_units
                    if u.get("status") == "pending"
                    and u.get("runtime_eligible", True)
                    and not u.get("dependencies")
                ]
    
                for unit in runnable:
                    eff = self.effective_priority(unit, trial_units)
                    candidates.append(
                        {
                            "trial": trial,
                            "pipeline": pipeline,
                            "unit": unit,
                            "effective_priority": eff,
                            "base_priority": unit.get("priority", 100),
                        }
                    )
    
        candidates.sort(
            key=lambda x: (
                x["effective_priority"],
                x["trial"].trial_id,
                x["unit"].get("unit_id"),
            )
        )
        return candidates

    def select_next_job(self, *, trials: list[TrialRecord], pipelines: dict[str, Any]):
        candidates = self.collect_candidate_jobs(trials=trials, pipelines=pipelines)
        if not candidates:
            return None

        top = candidates[0]
        return top["trial"], top["pipeline"], top["unit"]

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