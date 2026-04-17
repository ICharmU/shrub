from __future__ import annotations

from dataclasses import asdict
from typing import Any
import pandas as pd
from time import perf_counter
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
    
from Final.experiment_controller import ExperimentController, TrialRecord
from Final.pipeline_base import BasePipeline


class GridSearchController:
    def __init__(self, controller: ExperimentController):
        self.controller = controller
        self._last_preflight_cache = None

    def build_preflight_cache(
        self,
        *,
        trials: list[TrialRecord],
        pipelines_by_trial: dict[str, dict[str, BasePipeline]],
        show_progress: bool = True,
        log_details: bool = True,
    ) -> dict[tuple[str, str], dict]:
        """
        Returns:
            {
                (trial_id, pipeline_name): {
                    "trial": trial,
                    "pipeline": pipeline,
                    "snapshot": PipelinePreflightSnapshot,
                }
            }
        """
        cache = {}

        jobs = []
        for trial in trials:
            for pipeline_name, pipeline in pipelines_by_trial.get(trial.trial_id, {}).items():
                jobs.append((trial, pipeline_name, pipeline))

        iterator = jobs
        if show_progress and tqdm is not None:
            iterator = tqdm(jobs, desc="Building preflight cache", unit="pipeline")

        t0 = perf_counter()
        n_done = 0

        for trial, pipeline_name, pipeline in iterator:
            key = (trial.trial_id, pipeline_name)

            if log_details:
                pipeline.logger.info(
                    "GRID PREFLIGHT START | trial=%s | pipeline=%s",
                    trial.trial_id,
                    pipeline_name,
                )

            snap_t0 = perf_counter()
            snapshot = pipeline.build_preflight_snapshot(
                trial_id=trial.trial_id,
                config_signature=pipeline.config_signature() if hasattr(pipeline, "config_signature") else "",
                runtime_report=pipeline.runtime_report(),
                log_prefix="GRID ",
            )
            snap_t1 = perf_counter()

            cache[key] = {
                "trial": trial,
                "pipeline": pipeline,
                "snapshot": snapshot,
            }

            n_done += 1
            elapsed = snap_t1 - t0
            avg = elapsed / max(n_done, 1)
            remaining = avg * (len(jobs) - n_done)

            if log_details:
                pipeline.logger.info(
                    "GRID PREFLIGHT DONE  | trial=%s | pipeline=%s | idx=%d/%d | dt=%.2fs | elapsed=%.2fs | est_remaining=%.2fs",
                    trial.trial_id,
                    pipeline_name,
                    n_done,
                    len(jobs),
                    snap_t1 - snap_t0,
                    elapsed,
                    remaining,
                )

            if show_progress and tqdm is not None:
                iterator.set_postfix_str(f"elapsed={elapsed:.1f}s eta={remaining:.1f}s")

        self._last_preflight_cache = cache
        return cache

    def section_space_frame(self, pipeline: BasePipeline) -> pd.DataFrame:
        return pipeline.config_space_frame()

    def pipeline_module_state_frame(self, pipeline: BasePipeline) -> pd.DataFrame:
        rows = []
        for stage_name, variants in pipeline.module_state().items():
            for mv in variants:
                rows.append(
                    {
                        "pipeline": pipeline.pipeline_name,
                        "stage_name": stage_name,
                        "module_name": mv.module_name,
                        "enabled": mv.enabled,
                        "variant_name": mv.variant_name,
                        "params": mv.params,
                    }
                )
        return pd.DataFrame(rows)
    
    def trial_runtime_health_frame(
        self,
        *,
        trials: list[TrialRecord],
        pipelines_by_trial: dict[str, dict[str, BasePipeline]],
        show_progress: bool = True,
        log_details: bool = True,
    ) -> pd.DataFrame:
        cache = self.build_preflight_cache(
            trials=trials,
            pipelines_by_trial=pipelines_by_trial,
            show_progress=show_progress,
            log_details=log_details,
        )
        return self.trial_runtime_health_frame_from_cache(preflight_cache=cache)

    def trial_runtime_health_frame_from_cache(
        self,
        *,
        preflight_cache: dict[tuple[str, str], dict],
    ) -> pd.DataFrame:
        rows = []
        for (trial_id, pipeline_name), bundle in preflight_cache.items():
            snapshot = bundle["snapshot"]
            pipeline = bundle["pipeline"]

            rr_payload = snapshot.runtime_report
            runtime_report = pipeline.runtime_report()
            runtime_report.detected_image_key = rr_payload.get("detected_image_key")
            runtime_report.detected_image_alias = rr_payload.get("detected_image_alias")
            runtime_report.detected_conda_env = rr_payload.get("detected_conda_env")
            runtime_report.capabilities = rr_payload.get("capabilities", [])
            runtime_report.available_executables = rr_payload.get("available_executables", [])
            runtime_report.available_python_modules = rr_payload.get("available_python_modules", [])
            runtime_report.marker_files_found = rr_payload.get("marker_files_found", [])
            runtime_report.marker_env_matches = rr_payload.get("marker_env_matches", {})
            runtime_report.notes = rr_payload.get("notes", [])

            elig_df = pipeline.pipeline_runtime_summary_frame(runtime_report=runtime_report)
            if not elig_df.empty:
                elig_df = elig_df.copy()
                elig_df["trial_id"] = trial_id
                rows.extend(elig_df.to_dict(orient="records"))

        return pd.DataFrame(rows)
    
    def trial_dependency_health_frame(
        self,
        *,
        trials: list[TrialRecord],
        pipelines_by_trial: dict[str, dict[str, BasePipeline]],
        show_progress: bool = True,
        log_details: bool = True,
    ) -> pd.DataFrame:
        cache = self.build_preflight_cache(
            trials=trials,
            pipelines_by_trial=pipelines_by_trial,
            show_progress=show_progress,
            log_details=log_details,
        )
        return self.trial_dependency_health_frame_from_cache(preflight_cache=cache)

    def trial_dependency_health_frame_from_cache(
        self,
        *,
        preflight_cache: dict[tuple[str, str], dict],
    ) -> pd.DataFrame:
        rows = []
        for (trial_id, pipeline_name), bundle in preflight_cache.items():
            snapshot = bundle["snapshot"]
            for unit in snapshot.work_units:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "pipeline_name": pipeline_name,
                        "config_signature": snapshot.config_signature,
                        "unit_id": unit.get("unit_id"),
                        "stage_name": unit.get("stage_name"),
                        "scope": unit.get("scope"),
                        "status": unit.get("status"),
                        "runtime_eligible": unit.get("runtime_eligible"),
                        "dependency_ready": not bool(unit.get("dependencies")),
                        "dependencies": unit.get("dependencies", []),
                        "dependency_reasons": unit.get("dependency_reasons", []),
                        "priority": unit.get("priority"),
                        "site_id": unit.get("site_id"),
                        "plot_id": unit.get("plot_id"),
                        "source_version": unit.get("source_version"),
                    }
                )
        return pd.DataFrame(rows)
    
    def grid_preflight_frame(
        self,
        *,
        trials: list[TrialRecord],
        pipelines_by_trial: dict[str, dict[str, BasePipeline]],
        show_progress: bool = True,
        log_details: bool = True,
    ) -> pd.DataFrame:
        cache = self.build_preflight_cache(
            trials=trials,
            pipelines_by_trial=pipelines_by_trial,
            show_progress=show_progress,
            log_details=log_details,
        )
        return self.grid_preflight_frame_from_cache(preflight_cache=cache)

    def grid_preflight_frame_from_cache(
        self,
        *,
        preflight_cache: dict[tuple[str, str], dict],
    ) -> pd.DataFrame:
        rows = []
        for (trial_id, pipeline_name), bundle in preflight_cache.items():
            snapshot = bundle["snapshot"]
            runtime_image_key = snapshot.runtime_report.get("detected_image_key")

            for stage in snapshot.stage_health_rows:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "pipeline_name": pipeline_name,
                        "config_signature": snapshot.config_signature,
                        "runtime_image_key": runtime_image_key,
                        "stage_name": stage.get("stage_name"),
                        "runtime_eligible": stage.get("runtime_eligible"),
                        "dependency_ready": stage.get("dependency_ready"),
                        "status": stage.get("status"),
                        "missing_capabilities": stage.get("missing_capabilities", []),
                        "blocking_dependencies": stage.get("blocking_dependencies", []),
                        "n_total_units": stage.get("n_total_units", 0),
                        "n_complete_units": stage.get("n_complete_units", 0),
                        "n_pending_units": stage.get("n_pending_units", 0),
                        "n_blocked_units": stage.get("n_blocked_units", 0),
                        "n_failed_units": stage.get("n_failed_units", 0),
                        "n_ineligible_units": stage.get("n_ineligible_units", 0),
                    }
                )
        return pd.DataFrame(rows)

    def preflight_cache_summary_frame(
        self,
        *,
        preflight_cache: dict[tuple[str, str], dict],
    ) -> pd.DataFrame:
        rows = []
        for (trial_id, pipeline_name), bundle in preflight_cache.items():
            snapshot = bundle["snapshot"]
            rows.append(
                {
                    "trial_id": trial_id,
                    "pipeline_name": pipeline_name,
                    "config_signature": snapshot.config_signature,
                    "runtime_image_key": snapshot.runtime_report.get("detected_image_key"),
                    "n_work_units": len(snapshot.work_units),
                    "n_stage_rows": len(snapshot.stage_health_rows),
                }
            )
        return pd.DataFrame(rows)

    def set_trial_section_config(
        self,
        trial: TrialRecord,
        pipeline: BasePipeline,
        config_dict: dict,
    ) -> None:
        self.controller.set_section_config(trial, pipeline.pipeline_name, config_dict)

    def run_section_once(
        self,
        trial: TrialRecord,
        pipeline: BasePipeline,
        *,
        state,
    ):
        result = pipeline.run()
        state = pipeline.apply_to_experiment_state(state, result)
        config_signature = pipeline.config_signature() if hasattr(pipeline, "config_signature") else ""
        self.controller.record_section_result(
            trial,
            section_name=pipeline.pipeline_name,
            config_signature=config_signature,
            result=result,
        )
        self.controller.save_trial(trial)
        self.controller.save_state(state)
        return result, state

    def run_trial_until(
        self,
        trial: TrialRecord,
        pipelines: dict[str, BasePipeline],
        section_order: list[str],
        *,
        state,
    ):
        while True:
            next_section = self.controller.next_pending_section(trial, section_order)
            if next_section is None:
                trial.status = "success"
                self.controller.save_trial(trial)
                break

            pipeline = pipelines[next_section]
            result, state = self.run_section_once(trial, pipeline, state=state)
            if not result.success:
                break

        return trial, state

    def run_trials_work_queue(
        self,
        *,
        trials: list[TrialRecord],
        pipelines: dict[str, BasePipeline],
        scheduler,
        state,
    ):
        while True:
            next_job = scheduler.select_next_job(trials=trials, pipelines=pipelines)
            if next_job is None:
                break
    
            trial, pipeline, unit = next_job
            scheduler.run_claimed_job(trial=trial, pipeline=pipeline, unit=unit, state=state)

    def trials_frame(self) -> pd.DataFrame:
        rows = self.controller.trials_frame()
        return pd.DataFrame(rows)

    def completed_trials_frame(self) -> pd.DataFrame:
        df = self.trials_frame()
        if df.empty:
            return df
        return df[df["status"] == "success"].reset_index(drop=True)

    def get_or_create_trial_for_config(
        self,
        *,
        pipeline: BasePipeline,
        config_dict: dict,
        trial_prefix: str | None = None,
    ) -> TrialRecord:
        config_signature = pipeline.config_signature(config_dict) if hasattr(pipeline, "config_signature") else ""
        trial_prefix = trial_prefix or pipeline.pipeline_name
        trial_id = f"{trial_prefix}_{config_signature}"

        try:
            trial = self.controller.load_trial(trial_id)
        except Exception:
            trial = self.controller.create_trial(trial_id=trial_id)

        self.set_trial_section_config(trial, pipeline, config_dict)
        self.controller.save_trial(trial)
        return trial