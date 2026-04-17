from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from itertools import product
import pandas as pd
from time import perf_counter

from Final.shared_utils import get_logger
from Final.models import (
    ExperimentState,
    ModuleVariant,
    PipelineRunResult,
    PipelineSpec,
    PipelineStateUpdate,
    ExecutionEligibility, 
    ExecutionEligibilityStatus, 
    RuntimeRequirementMode,
    WorkUnitRecord,
    PipelineStageHealth,
    TrialHealthReport,
    PipelinePreflightSnapshot,
)
from Final.pipeline_caching import hash_payload, is_valid_stage_cache, \
    prune_stage_artifacts, write_stage_cache_manifest, read_stage_cache_manifest
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

    def stage_cache_exists(self, *, stage_name: str, stage_cache_dir: str | Path, expected_data_signature: str, expected_config_signature: str) -> bool:
        return self.validate_stage_cache(
            stage_name=stage_name,
            stage_cache_dir=stage_cache_dir,
            expected_data_signature=expected_data_signature,
            expected_config_signature=expected_config_signature,
        )

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

    def read_stage_cache_artifact_paths(self, stage_cache_dir: str | Path) -> dict:
        rec = read_stage_cache_manifest(stage_cache_dir)
        return rec.artifact_paths if rec is not None else {}

    def reconcile_stage_artifacts_with_storage_policy(
        self,
        *,
        stage_name: str,
        artifact_paths: dict,
    ) -> dict[str, dict]:
        results = {}
        for key, value in (artifact_paths or {}).items():
            results[key] = self.reconcile_artifact_reference(key=key, value=value)

        self.logger.info(
            "STORAGE POLICY RECONCILE DONE | stage=%s | n_artifacts=%d | statuses=%s",
            stage_name,
            len(results),
            {k: v.get("status") for k, v in results.items()},
        )
        return results


    def reconcile_artifact_reference(self, *, key: str, value):
        """
        Base implementation is intentionally minimal.
        Concrete pipelines can override to interpret rel paths vs local paths.
        """
        return {"status": "noop", "value": value}
    
    def hydrate_artifact(self, *, rel_path: str, local_path: str | Path | None = None, reason: str = "") -> Path | None:
        store = getattr(self, "artifact_store", None)
        if store is None:
            self.logger.info("HYDRATE SKIP | no artifact store | rel_path=%s", rel_path)
            return None

        try:
            self.logger.info("HYDRATE START | rel_path=%s | reason=%s", rel_path, reason)
            pulled = store.pull(rel_path, local_path=local_path)
            self.logger.info("HYDRATE DONE  | rel_path=%s | local_path=%s", rel_path, pulled)
            return Path(pulled)
        except Exception as e:
            self.logger.warning("HYDRATE FAIL | rel_path=%s | reason=%s | error=%s", rel_path, reason, e)
            return None
        
    def validate_hydrated_artifact(self, *, rel_path: str, local_path: Path, artifact_key: str | None = None) -> bool:
        """
        Default existence-only validation. Concrete pipelines should override for
        TIFF/CSV/JSON-specific validation.
        """
        ok = local_path.exists() and local_path.is_file()
        self.logger.info(
            "HYDRATE VALIDATE | rel_path=%s | artifact_key=%s | ok=%s",
            rel_path, artifact_key, ok
        )
        return ok
    
    def hydrate_and_validate_artifact(
        self,
        *,
        rel_path: str,
        local_path: str | Path | None = None,
        artifact_key: str | None = None,
        reason: str = "",
    ) -> Path | None:
        pulled = self.hydrate_artifact(rel_path=rel_path, local_path=local_path, reason=reason)
        if pulled is None:
            return None
        ok = self.validate_hydrated_artifact(rel_path=rel_path, local_path=pulled, artifact_key=artifact_key)
        if not ok:
            self.logger.warning("HYDRATE INVALID | rel_path=%s | artifact_key=%s", rel_path, artifact_key)
            return None
        return pulled
    
    def enforce_storage_policy_for_stage_cache(self, *, stage_name: str, stage_cache_dir: str | Path) -> None:
        artifact_paths = self.read_stage_cache_artifact_paths(stage_cache_dir)
        self.logger.info(
            "STORAGE POLICY RECONCILE | stage=%s | cache_dir=%s | n_artifacts=%d",
            stage_name, stage_cache_dir, len(artifact_paths)
        )
        self.reconcile_stage_artifacts_with_storage_policy(
            stage_name=stage_name,
            artifact_paths=artifact_paths,
        )

    def work_unit_refresh_fingerprint(
        self,
        *,
        trial_id: str,
        config_signature: str | None = None,
        runtime_report=None,
    ) -> str:
        runtime_report = runtime_report or self.runtime_report()
        payload = {
            "pipeline_name": self.pipeline_name,
            "trial_id": trial_id,
            "config_signature": config_signature or "",
            "runtime_image": getattr(runtime_report, "detected_image_key", None),
            "runtime_caps": sorted(getattr(runtime_report, "capabilities", []) or []),
        }
        return hash_payload(payload)

    def enumerate_work_units(
        self,
        *,
        trial_id: str,
        config_signature: str | None = None,
        runtime_report=None,
        register_shared_requirements: bool = False,
    ) -> list[dict]:
        """
        Concrete pipelines should override this and return serializable work-unit dicts.

        register_shared_requirements=False keeps preflight/reporting side-effect free.
        """
        return []

    def run_work_unit(self, unit: dict, *, trial_id: str, state=None):
        """
        Concrete pipelines should override this.
        """
        raise NotImplementedError(f"{self.pipeline_name} does not implement run_work_unit()")

    def stage_required_capabilities(self, stage_name: str, runtime_report=None) -> list[str]:
        runtime_report = runtime_report or self.runtime_report()
        elig = self.stage_runtime_eligibility(stage_name, runtime_report=runtime_report)
        caps = []
        for info in elig.values():
            caps.extend(info.missing_capabilities)
        return sorted(set(caps))
    
    def shared_signature_for_stage(self, stage_name: str, **kwargs) -> str | None:
        """
        Override in concrete pipelines when a stage has a reusable cross-trial identity.
        """
        return None

    def shared_artifact_family_for_stage(self, stage_name: str) -> str | None:
        """
        Override in concrete pipelines when a stage produces reusable shared artifacts.
        """
        return None
    
    def pipeline_runtime_summary_frame(self, runtime_report=None) -> pd.DataFrame:
        runtime_report = runtime_report or self.runtime_report()
        rows = []

        for stage in self.pipeline_spec.stages:
            elig = self.stage_runtime_eligibility(stage.name, runtime_report=runtime_report)
            for module_key, info in elig.items():
                mv = self.resolve_module_variant(module_key)
                rows.append(
                    {
                        "pipeline_name": self.pipeline_name,
                        "stage_name": stage.name,
                        "module_key": module_key,
                        "enabled": mv.enabled,
                        "variant_name": mv.variant_name,
                        "runtime_eligible": info.status == ExecutionEligibilityStatus.ELIGIBLE,
                        "missing_capabilities": info.missing_capabilities,
                        "reason": info.reason,
                        "detected_image_key": info.detected_image_key,
                    }
                )
        return pd.DataFrame(rows)
    
    def summarize_stage_health_from_units(
        self,
        *,
        trial_id: str,
        config_signature: str,
        units: list[dict],
        runtime_report=None,
    ) -> list[PipelineStageHealth]:
        runtime_report = runtime_report or self.runtime_report()
        rows = []

        for stage in self.pipeline_spec.stages:
            stage_units = [u for u in units if u.get("stage_name") == stage.name]
            elig = self.stage_runtime_eligibility(stage.name, runtime_report=runtime_report)

            missing_caps = sorted(
                {
                    cap
                    for info in elig.values()
                    for cap in info.missing_capabilities
                }
            )

            blocking_deps = sorted(
                {
                    dep
                    for u in stage_units
                    for dep in (u.get("dependencies") or [])
                }
            )

            rows.append(
                PipelineStageHealth(
                    pipeline_name=self.pipeline_name,
                    config_signature=config_signature,
                    stage_name=stage.name,
                    runtime_eligible=all(
                        info.status == ExecutionEligibilityStatus.ELIGIBLE
                        for info in elig.values()
                    ),
                    dependency_ready=all(not (u.get("dependencies") or []) for u in stage_units) if stage_units else True,
                    status="complete" if stage_units and all(u.get("status") == "complete" for u in stage_units)
                    else "failed" if any(u.get("status") == "failed" for u in stage_units)
                    else "blocked" if any(u.get("status") == "blocked" for u in stage_units)
                    else "pending",
                    missing_capabilities=missing_caps,
                    blocking_dependencies=blocking_deps,
                    n_total_units=len(stage_units),
                    n_complete_units=sum(u.get("status") == "complete" for u in stage_units),
                    n_pending_units=sum(u.get("status") == "pending" for u in stage_units),
                    n_blocked_units=sum(u.get("status") == "blocked" for u in stage_units),
                    n_failed_units=sum(u.get("status") == "failed" for u in stage_units),
                    n_ineligible_units=sum(u.get("status") == "ineligible" for u in stage_units),
                )
            )

        return rows
    
    def build_trial_health_report(
        self,
        *,
        trial_id: str,
        config_signature: str | None = None,
        runtime_report=None,
    ) -> TrialHealthReport:
        config_signature = config_signature or (
            self.config_signature() if hasattr(self, "config_signature") else ""
        )
        runtime_report = runtime_report or self.runtime_report()

        units = self.enumerate_work_units(
            trial_id=trial_id,
            config_signature=config_signature,
            runtime_report=runtime_report,
            register_shared_requirements=False,
        )
        stages = self.summarize_stage_health_from_units(
            trial_id=trial_id,
            config_signature=config_signature,
            units=units,
            runtime_report=runtime_report,
        )

        return TrialHealthReport(
            trial_id=trial_id,
            pipeline_name=self.pipeline_name,
            config_signature=config_signature,
            runtime_image_key=runtime_report.detected_image_key,
            n_total_units=len(units),
            n_complete_units=sum(u.get("status") == "complete" for u in units),
            n_pending_units=sum(u.get("status") == "pending" for u in units),
            n_blocked_units=sum(u.get("status") == "blocked" for u in units),
            n_failed_units=sum(u.get("status") == "failed" for u in units),
            n_ineligible_units=sum(u.get("status") == "ineligible" for u in units),
            stages=stages,
        )

    def build_preflight_snapshot(
        self,
        *,
        trial_id: str,
        config_signature: str | None = None,
        runtime_report=None,
        log_prefix: str = "",
    ) -> PipelinePreflightSnapshot:
        config_signature = config_signature or (
            self.config_signature() if hasattr(self, "config_signature") else ""
        )

        t0 = perf_counter()
        runtime_report = runtime_report or self.runtime_report()
        t1 = perf_counter()

        self.logger.info(
            "%sPREFLIGHT SNAPSHOT | pipeline=%s | trial=%s | step=runtime_report | dt=%.2fs",
            log_prefix,
            self.pipeline_name,
            trial_id,
            t1 - t0,
        )

        units = self.enumerate_work_units(
            trial_id=trial_id,
            config_signature=config_signature,
            runtime_report=runtime_report,
            register_shared_requirements=False,
        )
        t2 = perf_counter()

        self.logger.info(
            "%sPREFLIGHT SNAPSHOT | pipeline=%s | trial=%s | step=enumerate_work_units | n_units=%d | dt=%.2fs",
            log_prefix,
            self.pipeline_name,
            trial_id,
            len(units),
            t2 - t1,
        )

        stage_health = self.summarize_stage_health_from_units(
            trial_id=trial_id,
            config_signature=config_signature,
            units=units,
            runtime_report=runtime_report,
        )
        t3 = perf_counter()

        self.logger.info(
            "%sPREFLIGHT SNAPSHOT | pipeline=%s | trial=%s | step=stage_health | n_stages=%d | dt=%.2fs | total=%.2fs",
            log_prefix,
            self.pipeline_name,
            trial_id,
            len(stage_health),
            t3 - t2,
            t3 - t0,
        )

        return PipelinePreflightSnapshot(
            trial_id=trial_id,
            pipeline_name=self.pipeline_name,
            config_signature=config_signature,
            runtime_report=asdict(runtime_report),
            work_units=units,
            stage_health_rows=[asdict(x) for x in stage_health],
        )