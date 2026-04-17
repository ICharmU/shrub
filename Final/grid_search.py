from __future__ import annotations

from dataclasses import asdict
from typing import Any
import pandas as pd

from Final.experiment_controller import ExperimentController, TrialRecord
from Final.pipeline_base import BasePipeline


class GridSearchController:
    def __init__(self, controller: ExperimentController):
        self.controller = controller

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
    ) -> pd.DataFrame:
        rows = []
        for trial in trials:
            for pipeline_name, pipeline in pipelines_by_trial.get(trial.trial_id, {}).items():
                df = pipeline.pipeline_runtime_summary_frame()
                if not df.empty:
                    df = df.copy()
                    df["trial_id"] = trial.trial_id
                    rows.extend(df.to_dict(orient="records"))
        return pd.DataFrame(rows)
    
    def trial_dependency_health_frame(
        self,
        *,
        trials: list[TrialRecord],
        pipelines_by_trial: dict[str, dict[str, BasePipeline]],
    ) -> pd.DataFrame:
        rows = []
        for trial in trials:
            for pipeline_name, pipeline in pipelines_by_trial.get(trial.trial_id, {}).items():
                config_signature = pipeline.config_signature() if hasattr(pipeline, "config_signature") else ""
                units = pipeline.enumerate_work_units(
                    trial_id=trial.trial_id,
                    config_signature=config_signature,
                    runtime_report=pipeline.runtime_report(),
                )
                for unit in units:
                    rows.append(
                        {
                            "trial_id": trial.trial_id,
                            "pipeline_name": pipeline_name,
                            "config_signature": config_signature,
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
    ) -> pd.DataFrame:
        rows = []
        for trial in trials:
            for pipeline_name, pipeline in pipelines_by_trial.get(trial.trial_id, {}).items():
                report = pipeline.build_trial_health_report(
                    trial_id=trial.trial_id,
                    config_signature=pipeline.config_signature() if hasattr(pipeline, "config_signature") else "",
                    runtime_report=pipeline.runtime_report(),
                )
                for stage in report.stages:
                    rows.append(
                        {
                            "trial_id": report.trial_id,
                            "pipeline_name": report.pipeline_name,
                            "config_signature": report.config_signature,
                            "runtime_image_key": report.runtime_image_key,
                            "stage_name": stage.stage_name,
                            "runtime_eligible": stage.runtime_eligible,
                            "dependency_ready": stage.dependency_ready,
                            "status": stage.status,
                            "missing_capabilities": stage.missing_capabilities,
                            "blocking_dependencies": stage.blocking_dependencies,
                            "n_total_units": stage.n_total_units,
                            "n_complete_units": stage.n_complete_units,
                            "n_pending_units": stage.n_pending_units,
                            "n_blocked_units": stage.n_blocked_units,
                            "n_failed_units": stage.n_failed_units,
                            "n_ineligible_units": stage.n_ineligible_units,
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