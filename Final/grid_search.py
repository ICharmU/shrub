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

    def completed_trials_frame(self) -> pd.DataFrame:
        rows = []
        for path in sorted(self.controller.trials_dir.glob("*.json")):
            payload = pd.read_json(path, typ="series").to_dict()
            rows.append(
                {
                    "trial_id": payload.get("trial_id"),
                    "status": payload.get("status"),
                    "n_section_runs": len(payload.get("section_runs", {})),
                }
            )
        return pd.DataFrame(rows)