from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from itertools import product
import pandas as pd

from Final.shared_utils import get_logger
from Final.models import (
    ExperimentState,
    ModuleVariant,
    PipelineRunResult,
    PipelineSpec,
    PipelineStateUpdate,
    ExecutionEligibility, 
    ExecutionEligibilityStatus, 
    RuntimeRequirementMode
)
from Final.pipeline_caching import hash_payload, is_valid_stage_cache, prune_stage_artifacts, write_stage_cache_manifest
from Final.runtime_detection import detect_runtime_capabilities

class BasePipeline(ABC):
    """
    Generic pipeline abstraction for section pipelines:
    labeling, features, modeling, postprocessing.

    Concrete subclasses should implement:
    - build_pipeline_spec()
    - run()
    """

    def __init__(self, cfg, *, pipeline_name: str, output_root: str | Path):
        self.cfg = cfg
        self.pipeline_name = pipeline_name
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(f"pipeline.{pipeline_name}")
        self._pipeline_spec_cache: PipelineSpec | None = None

    @abstractmethod
    def build_pipeline_spec(self) -> PipelineSpec:
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs) -> PipelineRunResult:
        raise NotImplementedError

    @property
    def pipeline_spec(self) -> PipelineSpec:
        if self._pipeline_spec_cache is None:
            self._pipeline_spec_cache = self.build_pipeline_spec()
        return self._pipeline_spec_cache
    
    

    def config_dict(self) -> dict:
        if hasattr(self, "pipeline_config"):
            value = getattr(self, "pipeline_config")
            if is_dataclass(value):
                return asdict(value)
        return {}

    def stage_spec(self, stage_name: str):
        for stage in self.pipeline_spec.stages:
            if stage.name == stage_name:
                return stage
        raise KeyError(f"Unknown stage_name={stage_name}")

    def module_spec(self, module_key: str):
        if module_key not in self.pipeline_spec.modules:
            raise KeyError(f"Unknown module_key={module_key}")
        return self.pipeline_spec.modules[module_key]

    def resolve_module_variant(
        self,
        module_key: str,
        config_payload: dict | None = None,
    ) -> ModuleVariant:
        config_payload = config_payload or self.config_dict()
        spec = self.module_spec(module_key)

        enabled = True
        if spec.enabled_key is not None:
            enabled = bool(config_payload.get(spec.enabled_key, False))

        if not enabled:
            variant_name = "disabled"
        elif spec.variant_key is not None:
            variant_name = str(config_payload.get(spec.variant_key, "default"))
        elif spec.enabled_key is not None:
            variant_name = "enabled"
        else:
            variant_name = "default"

        params = {k: config_payload.get(k) for k in spec.param_keys}

        return ModuleVariant(
            module_name=module_key,
            enabled=enabled,
            variant_name=variant_name,
            params=params,
        )

    def module_state(self, config_payload: dict | None = None):
        config_payload = config_payload or self.config_dict()
        return {
            stage.name: [
                self.resolve_module_variant(module_key, config_payload=config_payload)
                for module_key in stage.module_keys
            ]
            for stage in self.pipeline_spec.stages
        }

    def stage_module_variants(self, stage_name: str, config_payload: dict | None = None) -> list[ModuleVariant]:
        return self.module_state(config_payload=config_payload).get(stage_name, [])

    def stage_config_signature(self, stage_name: str, config_payload: dict | None = None) -> str:
        payload = {
            "stage_name": stage_name,
            "module_variants": [asdict(mv) for mv in self.stage_module_variants(stage_name, config_payload=config_payload)],
        }
        return hash_payload(payload)

    def enumerate_config_space(self, base_config: dict | None = None) -> list[dict]:
        base = dict(base_config or self.config_dict())
        axes = self.pipeline_spec.search_axes

        if not axes:
            return [base]

        keys = [axis.key for axis in axes]
        values_list = [axis.values for axis in axes]

        variants = []
        for values in product(*values_list):
            cfg = dict(base)
            for k, v in zip(keys, values):
                cfg[k] = v
            variants.append(cfg)

        unique = []
        seen = set()
        for cfg in variants:
            sig = hash_payload(cfg)
            if sig not in seen:
                seen.add(sig)
                unique.append(cfg)
        return unique

    def config_space_frame(self):
        rows = []
        for cfg in self.enumerate_config_space():
            row = dict(cfg)
            row["config_signature"] = hash_payload(cfg)
            rows.append(row)
        return pd.DataFrame(rows)

    def validate_stage_cache(
        self,
        *,
        stage_name: str,
        stage_cache_dir: str | Path,
        expected_data_signature: str,
        expected_config_signature: str,
    ) -> bool:
        policy = self.stage_spec(stage_name).cache_policy
        return is_valid_stage_cache(
            stage_cache_dir=stage_cache_dir,
            expected_stage_name=stage_name,
            expected_data_signature=expected_data_signature,
            expected_config_signature=expected_config_signature,
            cache_policy=policy,
        )

    def write_stage_cache(
        self,
        *,
        stage_name: str,
        stage_cache_dir: str | Path,
        data_signature: str,
        config_signature: str,
        artifact_paths: dict,
        success: bool,
        notes: list[str] | None = None,
        config_payload: dict | None = None,
    ) -> Path:
        manifest_path = write_stage_cache_manifest(
            stage_cache_dir=stage_cache_dir,
            stage_name=stage_name,
            data_signature=data_signature,
            config_signature=config_signature,
            module_variants=self.stage_module_variants(stage_name, config_payload=config_payload),
            artifact_paths=artifact_paths,
            success=success,
            notes=notes,
        )

        policy = self.stage_spec(stage_name).cache_policy
        if success and policy.prune_after_success:
            deleted = prune_stage_artifacts(
                artifact_paths=artifact_paths,
                cache_policy=policy,
            )
            if deleted:
                self.logger.info("Pruned %d artifact(s) for stage=%s", len(deleted), stage_name)

        return manifest_path

    def make_state_update(self, result: PipelineRunResult) -> PipelineStateUpdate:
        active_modules = []
        for stage in self.pipeline_spec.stages:
            for mv in self.stage_module_variants(stage.name):
                if mv.enabled and self.module_spec(mv.module_name).include_in_state:
                    active_modules.append(f"{mv.module_name}:{mv.variant_name}")

        return PipelineStateUpdate(
            section_name=self.pipeline_name,
            status=result.status,
            active_modules=active_modules,
            qa_outputs=result.qa_outputs,
            metrics=result.metrics,
            notes=result.notes,
        )

    def apply_to_experiment_state(self, state: ExperimentState, result: PipelineRunResult) -> ExperimentState:
        update = self.make_state_update(result)

        state.section_status[self.pipeline_name] = update.status
        state.qa_outputs[self.pipeline_name] = update.qa_outputs

        for mod in update.active_modules:
            if mod not in state.active_modules:
                state.active_modules.append(mod)

        if result.raster_outputs is not None:
            state.raster_outputs = result.raster_outputs
        if result.object_outputs is not None:
            state.object_outputs = result.object_outputs

        return state

    def save_run_result(self, result: PipelineRunResult, filename: str | None = None, subdir: str | None = None) -> Path:
        root = self.output_root / subdir if subdir else self.output_root
        root.mkdir(parents=True, exist_ok=True)

        filename = filename or f"{self.pipeline_name}_run_result.json"
        path = root / filename
        path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        self.logger.info("Saved pipeline run result to %s", path)
        return path
    
    def runtime_report(self):
        return detect_runtime_capabilities(self.cfg)

    def module_runtime_eligibility(self, module_key: str, runtime_report=None) -> ExecutionEligibility:
        runtime_report = runtime_report or self.runtime_report()
        spec = self.module_spec(module_key)
        req = getattr(spec, "runtime_requirement", None)

        if req is None:
            return ExecutionEligibility(
                status=ExecutionEligibilityStatus.ELIGIBLE,
                satisfied_capabilities=list(runtime_report.capabilities),
                detected_image_key=runtime_report.detected_image_key,
                reason="No runtime requirement declared.",
            )

        caps = set(runtime_report.capabilities)
        required = set(req.required_capabilities or ())
        allowed_images = set(req.allowed_images or ())

        image_ok = True
        if allowed_images:
            image_ok = runtime_report.detected_image_key in allowed_images

        if req.mode == RuntimeRequirementMode.ANY:
            cap_ok = (not required) or bool(caps.intersection(required))
        else:
            cap_ok = required.issubset(caps)

        ok = image_ok and cap_ok
        return ExecutionEligibility(
            status=ExecutionEligibilityStatus.ELIGIBLE if ok else ExecutionEligibilityStatus.INELIGIBLE,
            satisfied_capabilities=sorted(caps.intersection(required)),
            missing_capabilities=sorted(required - caps),
            detected_image_key=runtime_report.detected_image_key,
            reason="eligible" if ok else "runtime requirement not satisfied",
        )

    def stage_runtime_eligibility(self, stage_name: str, runtime_report=None) -> dict[str, ExecutionEligibility]:
        runtime_report = runtime_report or self.runtime_report()
        return {
            module_key: self.module_runtime_eligibility(module_key, runtime_report=runtime_report)
            for module_key in self.stage_spec(stage_name).module_keys
        }

    def sync_artifact_registry_if_available(self) -> None:
        store = getattr(self, "artifact_store", None)
        if store is not None:
            try:
                store.sync_registry()
            except Exception as e:
                self.logger.warning("Artifact registry sync failed: %s", e)