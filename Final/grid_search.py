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