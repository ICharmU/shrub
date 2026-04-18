from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable
import json
import hashlib
import tempfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from Final.pipeline_base import BasePipeline
from Final.models import (
    PipelineDomain,
    PipelineRunResult,
    CanonicalRasterOutputs,
    CanonicalObjectOutputs,
    PipelineSpec,
    StageSpec,
    ModuleSpec,
    SearchAxis,
    CachePolicy,
    CacheRetentionMode,
    ArtifactSpec,
    StorageTier,
    RuntimeRequirement,
    RuntimeRequirementMode,
    WorkUnitScope,
    WorkUnitStatus,
    SharedArtifactStatus,
)
from Final.artifact_store import LocalArtifactStore, DriveRegistryArtifactStore, HybridArtifactStore
from Final.shared_artifact_registry import SharedArtifactRegistry
from Final.coordination import CoordinationManager
from Final.shared_utils import get_logger
from Final.pipeline_caching import hash_payload


@dataclass
class ModelingPipelineConfig:
    """Configuration for cross-site modeling.

    This class intentionally separates *representation choice* from *model choice*.
    The features pipeline is responsible for assembling aligned raster/object spaces;
    the modeling pipeline is responsible for selecting from those spaces, training,
    inference, calibration, and export.
    """

    features_config_signature: str = ""
    labeling_config_signature: str = ""

    feature_profile: str = "all_sources"  # naip | naip_als | naip_als_3dep | all_sources
    use_raster_features: bool = True
    use_object_features: bool = False
    use_confidence_weights: bool = True
    use_hard_negative_mining: bool = False

    model_family: str = "custom_unet"  # pixel_logreg | small_cnn | custom_unet | library_unet | object_tabular
    patch_size: int = 128
    patch_stride: int = 64
    patch_halo: int = 16
    batch_size: int = 16
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss_mode: str = "dice"  # dice | bce | bce_dice | focal_dice

    fold_strategy: str = "leave_one_site_out"  # leave_one_site_out | grouped_kfold | fixed_holdout
    holdout_sites: tuple[str, ...] = ()
    grouped_kfold_n_splits: int = 3

    threshold_mode: str = "validation_tuned"  # fixed | validation_tuned | percentile
    fixed_threshold: float = 0.5
    min_positive_fraction: float = 0.001
    max_positive_fraction: float = 0.75

    export_binary_predictions: bool = True
    export_probability_predictions: bool = True
    export_uncertainty: bool = False

    force_refresh_dataset_index: bool = False
    force_refresh_sampling: bool = False
    force_refresh_training: bool = False
    force_refresh_inference: bool = False
    force_refresh_calibration: bool = False


@dataclass
class ModelingPipelineOps:
    """Repo-specific compute hooks.

    The orchestration in this file is concrete; these hooks isolate the parts that
    depend on your exact stack-registry schema, notebook-side dataset loaders, and
    preferred model-training code.
    """

    build_site_dataset: Callable[..., dict[str, Any]]
    train_model: Callable[..., dict[str, Any]]
    run_inference: Callable[..., dict[str, Any]]
    calibrate_predictions: Callable[..., dict[str, Any]]
    export_prediction_bundle: Callable[..., dict[str, Any]]

    clear_artifact_staging_dir: Callable[[], None] | None = None


class ModelingPipeline(BasePipeline):
    def __init__(
        self,
        cfg,
        *,
        ops: ModelingPipelineOps,
        pipeline_config: ModelingPipelineConfig | None = None,
    ):
        super().__init__(
            cfg,
            pipeline_name="modeling",
            output_root=cfg.output.modeling_root / "pipeline_runs",
        )
        self.logger = get_logger("modeling.pipeline")
        self.ops = ops
        self.pipeline_config = pipeline_config or ModelingPipelineConfig()
        self._enumeration_cache: dict[tuple, list[dict[str, Any]]] = {}

        self.artifact_store = self._build_artifact_store()
        self.coordination = CoordinationManager(
            self.artifact_store,
            root_prefix=getattr(getattr(cfg, "coordination", object()), "root_prefix", "coordination"),
        )
        self.shared_registry = SharedArtifactRegistry(
            self.artifact_store,
            root_prefix=getattr(getattr(cfg, "shared_artifacts", object()), "registry_prefix", "shared_artifacts"),
        )

    # ------------------------------------------------------------------
    # Config/runtime helpers
    # ------------------------------------------------------------------

    def config_dict(self) -> dict:
        return asdict(self.pipeline_config)

    def config_signature(self, config_dict: dict | None = None) -> str:
        payload = config_dict if config_dict is not None else self.config_dict()
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def _runtime_caps(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        runtime_cfg = getattr(self.cfg, "modeling_runtime", None)
        return tuple(getattr(runtime_cfg, name, default))

    def _storage_cfg(self):
        runtime_cfg = getattr(self.cfg, "modeling_runtime", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "storage"):
            return runtime_cfg.storage
        return type("Storage", (), {
            "enable_local_store": True,
            "enable_drive_store": False,
            "use_hybrid_store": False,
            "fail_if_drive_missing": False,
        })()

    def _storage_policy_cfg(self):
        runtime_cfg = getattr(self.cfg, "modeling_runtime", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "storage_policy"):
            return runtime_cfg.storage_policy
        return type("StoragePolicy", (), {
            "push_large_artifacts_to_remote": False,
            "prune_local_after_remote_push": False,
            "verify_remote_before_prune": True,
        })()

    # ------------------------------------------------------------------
    # Artifact specs / storage
    # ------------------------------------------------------------------

    def artifact_specs(self) -> dict[str, ArtifactSpec]:
        return {
            "dataset_manifest": ArtifactSpec(
                key="dataset_manifest",
                rel_path_template="modeling/{config_signature}/datasets/{site}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "sample_manifest": ArtifactSpec(
                key="sample_manifest",
                rel_path_template="modeling/{config_signature}/samples/{site}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "model_bundle": ArtifactSpec(
                key="model_bundle",
                rel_path_template="modeling/{config_signature}/models/{site}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "calibration_json": ArtifactSpec(
                key="calibration_json",
                rel_path_template="modeling/{config_signature}/calibration/{site}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "probability_raster": ArtifactSpec(
                key="probability_raster",
                rel_path_template="modeling/{config_signature}/predictions/{site}/probability.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "binary_raster": ArtifactSpec(
                key="binary_raster",
                rel_path_template="modeling/{config_signature}/predictions/{site}/binary.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=False,
                prune_local_after_push=True,
            ),
            "uncertainty_raster": ArtifactSpec(
                key="uncertainty_raster",
                rel_path_template="modeling/{config_signature}/predictions/{site}/uncertainty.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=False,
                prune_local_after_push=True,
            ),
            "site_metrics": ArtifactSpec(
                key="site_metrics",
                rel_path_template="modeling/{config_signature}/metrics/{site}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "run_result": ArtifactSpec(
                key="run_result",
                rel_path_template="modeling/{config_signature}/modeling_run_result.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
        }

    def _build_artifact_store(self):
        storage = self._storage_cfg()
        global_store = self.cfg.artifact_store

        local_store = LocalArtifactStore(
            repo_root=self.cfg.data.project_root,
            storage_root=Path(global_store.local_storage_root),
        )

        drive_store = None
        if storage.enable_drive_store:
            missing = [
                str(p) for p in [
                    global_store.drive_client_secrets_path,
                    global_store.drive_config_path,
                ] if not Path(p).exists()
            ]
            if missing:
                msg = f"Missing Drive config files: {missing}"
                if storage.fail_if_drive_missing:
                    raise FileNotFoundError(msg)
                self.logger.warning(msg)
            else:
                drive_store = DriveRegistryArtifactStore(
                    repo_root=self.cfg.data.project_root,
                    registry_path=Path(global_store.drive_registry_path),
                    drive_config_path=Path(global_store.drive_config_path),
                    client_secrets_path=Path(global_store.drive_client_secrets_path),
                    credentials_path=Path(global_store.drive_credentials_path),
                )

        if storage.use_hybrid_store and drive_store is not None:
            self.logger.info("Using HybridArtifactStore for modeling pipeline")
            return HybridArtifactStore(local_store=local_store, remote_store=drive_store)

        if drive_store is not None and not storage.enable_local_store:
            self.logger.info("Using DriveRegistryArtifactStore only for modeling pipeline")
            return drive_store

        self.logger.info("Using LocalArtifactStore only for modeling pipeline")
        return local_store

    def _artifact_spec(self, key: str) -> ArtifactSpec:
        return self.artifact_specs()[key]

    def _render_rel_path(self, artifact_key: str, **kwargs) -> str:
        return self._artifact_spec(artifact_key).rel_path_template.format(**kwargs)

    def _local_artifact_path(self, rel_path: str) -> Path:
        if isinstance(self.artifact_store, LocalArtifactStore):
            return self.artifact_store.storage_root / rel_path
        if isinstance(self.artifact_store, HybridArtifactStore):
            return self.artifact_store.local_store.storage_root / rel_path
        return self.cfg.output.modeling_root / "_remote_stage" / rel_path

    def _remote_exists(self, rel_path: str) -> bool:
        try:
            return self.artifact_store.exists(rel_path)
        except Exception:
            return False

    def _push_if_needed(self, local_path: Path, artifact_key: str, rel_path: str) -> str | None:
        spec = self._artifact_spec(artifact_key)
        policy = self._storage_policy_cfg()
        should_push = (
            policy.push_large_artifacts_to_remote
            and spec.storage_tier in {StorageTier.LOCAL_THEN_REMOTE, StorageTier.REMOTE_ONLY}
            and isinstance(self.artifact_store, (HybridArtifactStore, DriveRegistryArtifactStore))
        )
        if not should_push:
            return None
        return self.artifact_store.push(local_path, rel_path=rel_path)

    def _prune_if_allowed(self, local_path: Path, artifact_key: str, rel_path: str) -> None:
        spec = self._artifact_spec(artifact_key)
        policy = self._storage_policy_cfg()
        if not policy.prune_local_after_remote_push:
            return
        if not spec.prune_local_after_push:
            return
        if policy.verify_remote_before_prune and not self._remote_exists(rel_path):
            return
        if local_path.exists():
            local_path.unlink()

    def _persist_json_artifact(self, payload: dict, artifact_key: str, **fmt) -> tuple[Path, str, str | None]:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        remote_ref = self._push_if_needed(local_path, artifact_key, rel_path)
        self._prune_if_allowed(local_path, artifact_key, rel_path)
        return local_path, rel_path, remote_ref

    def validate_hydrated_artifact(self, *, rel_path: str, local_path: Path, artifact_key: str | None = None) -> bool:
        try:
            if artifact_key in {"dataset_manifest", "sample_manifest", "model_bundle", "calibration_json", "site_metrics", "run_result"}:
                json.loads(local_path.read_text(encoding="utf-8"))
                return True
            if artifact_key in {"probability_raster", "binary_raster", "uncertainty_raster"}:
                with rasterio.open(local_path) as src:
                    src.read(1, window=Window(0, 0, min(16, src.width), min(16, src.height)))
                return True
            return super().validate_hydrated_artifact(rel_path=rel_path, local_path=local_path, artifact_key=artifact_key)
        except Exception as e:
            self.logger.warning("HYDRATE VALIDATE FAIL | rel_path=%s | artifact_key=%s | error=%s", rel_path, artifact_key, e)
            return False

    # ------------------------------------------------------------------
    # Shared-artifact helpers
    # ------------------------------------------------------------------

    def shared_signature_dataset_index(self, site: str) -> str:
        payload = {
            "site": site,
            "features_config_signature": self.pipeline_config.features_config_signature,
            "labeling_config_signature": self.pipeline_config.labeling_config_signature,
            "feature_profile": self.pipeline_config.feature_profile,
            "use_raster_features": self.pipeline_config.use_raster_features,
            "use_object_features": self.pipeline_config.use_object_features,
        }
        return hash_payload(payload)

    def shared_signature_sampling(self, site: str) -> str:
        payload = {
            "site": site,
            "dataset_sig": self.shared_signature_dataset_index(site),
            "patch_size": self.pipeline_config.patch_size,
            "patch_stride": self.pipeline_config.patch_stride,
            "patch_halo": self.pipeline_config.patch_halo,
            "fold_strategy": self.pipeline_config.fold_strategy,
            "holdout_sites": self.pipeline_config.holdout_sites,
        }
        return hash_payload(payload)

    def shared_signature_training(self, site: str) -> str:
        payload = {
            "site": site,
            "sampling_sig": self.shared_signature_sampling(site),
            "model_family": self.pipeline_config.model_family,
            "loss_mode": self.pipeline_config.loss_mode,
            "batch_size": self.pipeline_config.batch_size,
            "max_epochs": self.pipeline_config.max_epochs,
            "learning_rate": self.pipeline_config.learning_rate,
            "weight_decay": self.pipeline_config.weight_decay,
            "use_confidence_weights": self.pipeline_config.use_confidence_weights,
            "use_hard_negative_mining": self.pipeline_config.use_hard_negative_mining,
        }
        return hash_payload(payload)

    def shared_signature_inference(self, site: str) -> str:
        payload = {
            "site": site,
            "training_sig": self.shared_signature_training(site),
            "export_probability_predictions": self.pipeline_config.export_probability_predictions,
            "export_binary_predictions": self.pipeline_config.export_binary_predictions,
            "export_uncertainty": self.pipeline_config.export_uncertainty,
        }
        return hash_payload(payload)

    def shared_signature_calibration(self, site: str) -> str:
        payload = {
            "site": site,
            "inference_sig": self.shared_signature_inference(site),
            "threshold_mode": self.pipeline_config.threshold_mode,
            "fixed_threshold": self.pipeline_config.fixed_threshold,
            "min_positive_fraction": self.pipeline_config.min_positive_fraction,
            "max_positive_fraction": self.pipeline_config.max_positive_fraction,
        }
        return hash_payload(payload)

    def shared_artifact_is_valid(self, *, artifact_family: str, shared_signature: str) -> bool:
        if not getattr(getattr(self.cfg, "shared_artifacts", object()), "enable_shared_artifact_registry", False):
            return False
        rec = self.shared_registry.load(artifact_family=artifact_family, shared_signature=shared_signature)
        return rec is not None and str(rec.status).lower().endswith("valid")

    def register_shared_requirement(self, *, artifact_family: str, shared_signature: str, trial_id: str, metadata: dict | None = None) -> None:
        if not getattr(getattr(self.cfg, "shared_artifacts", object()), "enable_shared_artifact_registry", False):
            return
        rec = self.shared_registry.upsert_requirement(
            artifact_family=artifact_family,
            shared_signature=shared_signature,
            producer_pipeline=self.pipeline_name,
            trial_id=trial_id,
        )
        if metadata:
            rec.metadata.update(metadata)
            self.shared_registry.save(rec)

    def _mark_shared_available(
        self,
        *,
        artifact_family: str,
        shared_signature: str,
        rel_path: str | None = None,
        local_path: str | None = None,
        remote_ref: str | None = None,
        metadata: dict | None = None,
        trial_id: str | None = None,
    ) -> None:
        if not getattr(getattr(self.cfg, "shared_artifacts", object()), "enable_shared_artifact_registry", False):
            return
        self.shared_registry.mark_available(
            artifact_family=artifact_family,
            shared_signature=shared_signature,
            producer_pipeline=self.pipeline_name,
            rel_path=rel_path,
            local_path=local_path,
            remote_ref=remote_ref,
            source_trial=trial_id,
            metadata=metadata or {},
            status=SharedArtifactStatus.VALID,
        )

    # ------------------------------------------------------------------
    # Pipeline spec
    # ------------------------------------------------------------------

    def build_pipeline_spec(self) -> PipelineSpec:
        modules = {
            "modeling.dataset_index.base": ModuleSpec(
                key="modeling.dataset_index.base",
                stage_name="dataset_index",
                param_keys=("features_config_signature", "labeling_config_signature", "feature_profile"),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self._runtime_caps("require_capability_dataset_index", ("runtime:modeling",)),
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "modeling.sampling.base": ModuleSpec(
                key="modeling.sampling.base",
                stage_name="sampling",
                param_keys=("patch_size", "patch_stride", "patch_halo", "fold_strategy", "holdout_sites"),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self._runtime_caps("require_capability_sampling", ("runtime:modeling",)),
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "modeling.train.logreg": ModuleSpec(
                key="modeling.train.logreg",
                stage_name="train",
                enabled_key=None,
                variant_key="model_family",
                param_keys=("model_family", "loss_mode", "batch_size", "max_epochs", "learning_rate", "weight_decay", "use_confidence_weights", "use_hard_negative_mining"),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self._runtime_caps("require_capability_train", ("runtime:modeling",)),
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "modeling.infer.base": ModuleSpec(
                key="modeling.infer.base",
                stage_name="infer",
                param_keys=("export_probability_predictions", "export_binary_predictions", "export_uncertainty"),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self._runtime_caps("require_capability_infer", ("runtime:modeling",)),
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "modeling.calibrate.base": ModuleSpec(
                key="modeling.calibrate.base",
                stage_name="calibrate",
                param_keys=("threshold_mode", "fixed_threshold", "min_positive_fraction", "max_positive_fraction"),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self._runtime_caps("require_capability_calibrate", ("runtime:modeling",)),
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
        }

        stages = [
            StageSpec(name="dataset_index", module_keys=["modeling.dataset_index.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="sampling", module_keys=["modeling.sampling.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="train", module_keys=["modeling.train.logreg"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="infer", module_keys=["modeling.infer.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="calibrate", module_keys=["modeling.calibrate.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
        ]

        search_axes = [
            SearchAxis(key="feature_profile", values=["naip", "naip_als", "naip_als_3dep", "all_sources"], stage_name="dataset_index", module_key="modeling.dataset_index.base"),
            SearchAxis(key="model_family", values=["pixel_logreg", "small_cnn", "custom_unet"], stage_name="train", module_key="modeling.train.logreg"),
            SearchAxis(key="loss_mode", values=["dice", "bce", "bce_dice"], stage_name="train", module_key="modeling.train.logreg"),
            SearchAxis(key="use_confidence_weights", values=[False, True], stage_name="train", module_key="modeling.train.logreg"),
            SearchAxis(key="threshold_mode", values=["fixed", "validation_tuned"], stage_name="calibrate", module_key="modeling.calibrate.base"),
        ]

        return PipelineSpec(
            pipeline_name="modeling",
            domain=PipelineDomain.MODELING,
            stages=stages,
            modules=modules,
            search_axes=search_axes,
        )

    # ------------------------------------------------------------------
    # Work-unit enumeration
    # ------------------------------------------------------------------

    def enumerate_work_units(
        self,
        *,
        trial_id: str,
        config_signature: str | None = None,
        runtime_report=None,
        register_shared_requirements: bool = False,
    ) -> list[dict]:
        config_signature = config_signature or self.config_signature()
        runtime_report = runtime_report or self.runtime_report()

        enum_cache_key = (
            trial_id,
            config_signature,
            getattr(runtime_report, "detected_image_key", None),
            register_shared_requirements,
            self.work_unit_refresh_fingerprint(trial_id=trial_id, config_signature=config_signature, runtime_report=runtime_report),
        )
        if enum_cache_key in self._enumeration_cache:
            return [dict(u) for u in self._enumeration_cache[enum_cache_key]]

        units: list[dict[str, Any]] = []

        dataset_ok = all(info.status.value == "eligible" for info in self.stage_runtime_eligibility("dataset_index", runtime_report).values())
        sampling_ok = all(info.status.value == "eligible" for info in self.stage_runtime_eligibility("sampling", runtime_report).values())
        train_ok = all(info.status.value == "eligible" for info in self.stage_runtime_eligibility("train", runtime_report).values())
        infer_ok = all(info.status.value == "eligible" for info in self.stage_runtime_eligibility("infer", runtime_report).values())
        calibrate_ok = all(info.status.value == "eligible" for info in self.stage_runtime_eligibility("calibrate", runtime_report).values())

        dataset_complete: dict[str, bool] = {}
        sampling_complete: dict[str, bool] = {}
        train_complete: dict[str, bool] = {}
        infer_complete: dict[str, bool] = {}

        for site in self.cfg.sites:
            ds_sig = self.shared_signature_dataset_index(site)
            if register_shared_requirements:
                self.register_shared_requirement(artifact_family="modeling.dataset_index", shared_signature=ds_sig, trial_id=trial_id, metadata={"site": site})
            ds_complete = self.shared_artifact_is_valid(artifact_family="modeling.dataset_index", shared_signature=ds_sig)
            dataset_complete[site] = ds_complete
            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:dataset_index:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "dataset_index",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if ds_complete else (WorkUnitStatus.PENDING.value if dataset_ok else WorkUnitStatus.INELIGIBLE.value),
                "dependencies": [],
                "dependency_reasons": [],
                "runtime_required_capabilities": list(self._runtime_caps("require_capability_dataset_index", ("runtime:modeling",))),
                "runtime_eligible": dataset_ok,
                "priority": 100,
                "site_id": site,
                "shared_artifact_family": "modeling.dataset_index",
                "shared_signature": ds_sig,
            })

        for site in self.cfg.sites:
            samp_sig = self.shared_signature_sampling(site)
            if register_shared_requirements:
                self.register_shared_requirement(artifact_family="modeling.sampling", shared_signature=samp_sig, trial_id=trial_id, metadata={"site": site})
            deps = [] if dataset_complete.get(site, False) else ["dataset_index"]
            dep_reasons = [] if dataset_complete.get(site, False) else ["Dataset manifest is not ready."]
            samp_complete = self.shared_artifact_is_valid(artifact_family="modeling.sampling", shared_signature=samp_sig)
            sampling_complete[site] = samp_complete
            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:sampling:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "sampling",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if samp_complete else (
                    WorkUnitStatus.PENDING.value if (sampling_ok and not deps) else (
                        WorkUnitStatus.BLOCKED.value if sampling_ok else WorkUnitStatus.INELIGIBLE.value
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self._runtime_caps("require_capability_sampling", ("runtime:modeling",))),
                "runtime_eligible": sampling_ok,
                "priority": 200,
                "site_id": site,
                "shared_artifact_family": "modeling.sampling",
                "shared_signature": samp_sig,
            })

        for site in self.cfg.sites:
            train_sig = self.shared_signature_training(site)
            if register_shared_requirements:
                self.register_shared_requirement(artifact_family="modeling.train", shared_signature=train_sig, trial_id=trial_id, metadata={"site": site})
            deps = [] if sampling_complete.get(site, False) else ["sampling"]
            dep_reasons = [] if sampling_complete.get(site, False) else ["Sampling manifest is not ready."]
            tr_complete = self.shared_artifact_is_valid(artifact_family="modeling.train", shared_signature=train_sig)
            train_complete[site] = tr_complete
            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:train:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "train",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if tr_complete else (
                    WorkUnitStatus.PENDING.value if (train_ok and not deps) else (
                        WorkUnitStatus.BLOCKED.value if train_ok else WorkUnitStatus.INELIGIBLE.value
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self._runtime_caps("require_capability_train", ("runtime:modeling",))),
                "runtime_eligible": train_ok,
                "priority": 300,
                "site_id": site,
                "shared_artifact_family": "modeling.train",
                "shared_signature": train_sig,
            })

        for site in self.cfg.sites:
            infer_sig = self.shared_signature_inference(site)
            if register_shared_requirements:
                self.register_shared_requirement(artifact_family="modeling.infer", shared_signature=infer_sig, trial_id=trial_id, metadata={"site": site})
            deps = [] if train_complete.get(site, False) else ["train"]
            dep_reasons = [] if train_complete.get(site, False) else ["Model bundle is not ready."]
            inf_complete = self.shared_artifact_is_valid(artifact_family="modeling.infer", shared_signature=infer_sig)
            infer_complete[site] = inf_complete
            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:infer:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "infer",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if inf_complete else (
                    WorkUnitStatus.PENDING.value if (infer_ok and not deps) else (
                        WorkUnitStatus.BLOCKED.value if infer_ok else WorkUnitStatus.INELIGIBLE.value
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self._runtime_caps("require_capability_infer", ("runtime:modeling",))),
                "runtime_eligible": infer_ok,
                "priority": 400,
                "site_id": site,
                "shared_artifact_family": "modeling.infer",
                "shared_signature": infer_sig,
            })

        for site in self.cfg.sites:
            cal_sig = self.shared_signature_calibration(site)
            if register_shared_requirements:
                self.register_shared_requirement(artifact_family="modeling.calibrate", shared_signature=cal_sig, trial_id=trial_id, metadata={"site": site})
            deps = [] if infer_complete.get(site, False) else ["infer"]
            dep_reasons = [] if infer_complete.get(site, False) else ["Inference bundle is not ready."]
            cal_complete = self.shared_artifact_is_valid(artifact_family="modeling.calibrate", shared_signature=cal_sig)
            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:calibrate:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "calibrate",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if cal_complete else (
                    WorkUnitStatus.PENDING.value if (calibrate_ok and not deps) else (
                        WorkUnitStatus.BLOCKED.value if calibrate_ok else WorkUnitStatus.INELIGIBLE.value
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self._runtime_caps("require_capability_calibrate", ("runtime:modeling",))),
                "runtime_eligible": calibrate_ok,
                "priority": 500,
                "site_id": site,
                "shared_artifact_family": "modeling.calibrate",
                "shared_signature": cal_sig,
            })

        self._enumeration_cache[enum_cache_key] = [dict(u) for u in units]
        return units

    # ------------------------------------------------------------------
    # Work-unit execution
    # ------------------------------------------------------------------

    def run_work_unit(self, unit: dict, *, trial_id: str, state=None):
        stage_name = unit["stage_name"]
        site_id = unit.get("site_id")
        cfg_sig = self.config_signature()

        if stage_name == "dataset_index":
            dataset_bundle = self.ops.build_site_dataset(
                site_id=site_id,
                artifact_store=self.artifact_store,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            _, rel_path, remote_ref = self._persist_json_artifact(dataset_bundle, "dataset_manifest", config_signature=cfg_sig, site=site_id)
            self._mark_shared_available(
                artifact_family="modeling.dataset_index",
                shared_signature=self.shared_signature_dataset_index(site_id),
                rel_path=rel_path,
                local_path=str(self._local_artifact_path(rel_path)),
                remote_ref=remote_ref,
                metadata={"site": site_id, "keys": sorted(dataset_bundle.keys())},
                trial_id=trial_id,
            )
            return {"stage_name": stage_name, "site_id": site_id, "keys": sorted(dataset_bundle.keys())}

        if stage_name == "sampling":
            dataset_bundle = self.ops.build_site_dataset(
                site_id=site_id,
                artifact_store=self.artifact_store,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            sampling_bundle = {
                "site_id": site_id,
                "fold_strategy": self.pipeline_config.fold_strategy,
                "patch_size": self.pipeline_config.patch_size,
                "patch_stride": self.pipeline_config.patch_stride,
                "patch_halo": self.pipeline_config.patch_halo,
                "dataset_summary": {k: v for k, v in dataset_bundle.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
            }
            _, rel_path, remote_ref = self._persist_json_artifact(sampling_bundle, "sample_manifest", config_signature=cfg_sig, site=site_id)
            self._mark_shared_available(
                artifact_family="modeling.sampling",
                shared_signature=self.shared_signature_sampling(site_id),
                rel_path=rel_path,
                local_path=str(self._local_artifact_path(rel_path)),
                remote_ref=remote_ref,
                metadata={"site": site_id, "patch_size": self.pipeline_config.patch_size},
                trial_id=trial_id,
            )
            return {"stage_name": stage_name, "site_id": site_id}

        if stage_name == "train":
            dataset_bundle = self.ops.build_site_dataset(
                site_id=site_id,
                artifact_store=self.artifact_store,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            model_bundle = self.ops.train_model(
                site_id=site_id,
                dataset_bundle=dataset_bundle,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            _, rel_path, remote_ref = self._persist_json_artifact(model_bundle, "model_bundle", config_signature=cfg_sig, site=site_id)
            self._mark_shared_available(
                artifact_family="modeling.train",
                shared_signature=self.shared_signature_training(site_id),
                rel_path=rel_path,
                local_path=str(self._local_artifact_path(rel_path)),
                remote_ref=remote_ref,
                metadata={"site": site_id, "model_family": self.pipeline_config.model_family},
                trial_id=trial_id,
            )
            return {"stage_name": stage_name, "site_id": site_id, "model_family": self.pipeline_config.model_family}

        if stage_name == "infer":
            dataset_bundle = self.ops.build_site_dataset(
                site_id=site_id,
                artifact_store=self.artifact_store,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            model_bundle = self.ops.train_model(
                site_id=site_id,
                dataset_bundle=dataset_bundle,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            prediction_bundle = self.ops.run_inference(
                site_id=site_id,
                dataset_bundle=dataset_bundle,
                model_bundle=model_bundle,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            export_bundle = self.ops.export_prediction_bundle(
                site_id=site_id,
                prediction_bundle=prediction_bundle,
                artifact_store=self.artifact_store,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
                config_signature=cfg_sig,
                render_rel_path=self._render_rel_path,
                local_artifact_path=self._local_artifact_path,
                push_if_needed=self._push_if_needed,
                prune_if_allowed=self._prune_if_allowed,
            )
            self._mark_shared_available(
                artifact_family="modeling.infer",
                shared_signature=self.shared_signature_inference(site_id),
                rel_path=export_bundle.get("probability_rel_path"),
                local_path=export_bundle.get("probability_local_path"),
                remote_ref=export_bundle.get("probability_remote_ref"),
                metadata={"site": site_id, "export_keys": sorted(export_bundle.keys())},
                trial_id=trial_id,
            )
            return {"stage_name": stage_name, "site_id": site_id, "export_keys": sorted(export_bundle.keys())}

        if stage_name == "calibrate":
            dataset_bundle = self.ops.build_site_dataset(
                site_id=site_id,
                artifact_store=self.artifact_store,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            model_bundle = self.ops.train_model(
                site_id=site_id,
                dataset_bundle=dataset_bundle,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            prediction_bundle = self.ops.run_inference(
                site_id=site_id,
                dataset_bundle=dataset_bundle,
                model_bundle=model_bundle,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            calibration_bundle = self.ops.calibrate_predictions(
                site_id=site_id,
                dataset_bundle=dataset_bundle,
                model_bundle=model_bundle,
                prediction_bundle=prediction_bundle,
                modeling_config=self.pipeline_config,
                cfg=self.cfg,
            )
            _, rel_path, remote_ref = self._persist_json_artifact(calibration_bundle, "calibration_json", config_signature=cfg_sig, site=site_id)
            metrics_payload = calibration_bundle.get("metrics", calibration_bundle)
            self._persist_json_artifact(metrics_payload, "site_metrics", config_signature=cfg_sig, site=site_id)
            self._mark_shared_available(
                artifact_family="modeling.calibrate",
                shared_signature=self.shared_signature_calibration(site_id),
                rel_path=rel_path,
                local_path=str(self._local_artifact_path(rel_path)),
                remote_ref=remote_ref,
                metadata={"site": site_id, "threshold_mode": self.pipeline_config.threshold_mode},
                trial_id=trial_id,
            )
            return {"stage_name": stage_name, "site_id": site_id}

        raise NotImplementedError(f"Unknown modeling work unit stage={stage_name}")

    # ------------------------------------------------------------------
    # End-to-end runner
    # ------------------------------------------------------------------

    def run(self, *, trial_id: str = "adhoc_modeling", state=None, **kwargs) -> PipelineRunResult:
        runtime_report = self.runtime_report()
        executed_units: list[str] = []
        loop_guard = 0

        while True:
            loop_guard += 1
            if loop_guard > 50000:
                raise RuntimeError("ModelingPipeline debug run exceeded loop guard.")

            units = self.enumerate_work_units(
                trial_id=trial_id,
                config_signature=self.config_signature(),
                runtime_report=runtime_report,
                register_shared_requirements=False,
            )
            runnable = [u for u in units if u["status"] == WorkUnitStatus.PENDING.value and not u.get("dependencies")]
            if not runnable:
                break
            runnable = sorted(runnable, key=lambda u: (u.get("priority", 100), u["unit_id"]))
            unit = runnable[0]
            self.run_work_unit(unit, trial_id=trial_id, state=state)
            executed_units.append(unit["unit_id"])
            self._enumeration_cache.clear()
            if self.ops.clear_artifact_staging_dir is not None:
                try:
                    self.ops.clear_artifact_staging_dir()
                except Exception:
                    pass

        final_units = self.enumerate_work_units(
            trial_id=trial_id,
            config_signature=self.config_signature(),
            runtime_report=runtime_report,
            register_shared_requirements=False,
        )
        success = all(u["status"] == WorkUnitStatus.COMPLETE.value for u in final_units)
        status = "success" if success else "partial"

        site_metric_rows = []
        prediction_rows = []
        cfg_sig = self.config_signature()
        for site in self.cfg.sites:
            metric_path = self._local_artifact_path(self._render_rel_path("site_metrics", config_signature=cfg_sig, site=site))
            if metric_path.exists():
                try:
                    site_metric_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            prob_path = self._local_artifact_path(self._render_rel_path("probability_raster", config_signature=cfg_sig, site=site))
            if prob_path.exists():
                prediction_rows.append({"site_id": site, "probability_path": str(prob_path)})

        metrics_df = pd.DataFrame(site_metric_rows)
        preds_df = pd.DataFrame(prediction_rows)

        result = PipelineRunResult(
            pipeline_name=self.pipeline_name,
            success=success,
            status=status,
            raster_outputs=CanonicalRasterOutputs(
                features=pd.DataFrame({"site_id": list(self.cfg.sites), "feature_profile": self.pipeline_config.feature_profile}),
                predictions=preds_df if not preds_df.empty else None,
            ),
            object_outputs=CanonicalObjectOutputs(),
            qa_outputs={
                "executed_units": executed_units,
                "n_total_units": len(final_units),
                "runtime_report": asdict(runtime_report),
            },
            metrics={
                "n_total_units": len(final_units),
                "n_complete_units": sum(u["status"] == WorkUnitStatus.COMPLETE.value for u in final_units),
                "n_pending_units": sum(u["status"] == WorkUnitStatus.PENDING.value for u in final_units),
                "n_blocked_units": sum(u["status"] == WorkUnitStatus.BLOCKED.value for u in final_units),
                "n_ineligible_units": sum(u["status"] == WorkUnitStatus.INELIGIBLE.value for u in final_units),
                "n_sites_with_metrics": int(metrics_df["site_id"].nunique()) if "site_id" in metrics_df.columns and not metrics_df.empty else 0,
            },
            notes=[
                "Cross-site modeling pipeline execution through dataset indexing, sampling, training, inference, and calibration.",
                "Domain-specific dataset loading/training/inference/export are delegated to ModelingPipelineOps hooks so they can be wired to your current FE registry and notebook-side trainers.",
            ],
        )
        self.save_run_result(result, subdir=cfg_sig)
        self._persist_json_artifact(asdict(result), "run_result", config_signature=cfg_sig)
        return result


# ----------------------------------------------------------------------
# Optional default ops scaffold
# ----------------------------------------------------------------------


def build_modeling_pipeline_ops() -> ModelingPipelineOps:
    """Return explicit hooks that force repo-specific wiring.

    This keeps the pipeline code concrete without guessing the exact schema of
    your stack registry / dataset tensors. Replace these closures with imports
    from your modeling notebook/module once you wire the training path.
    """

    def _not_wired(*args, **kwargs):
        raise NotImplementedError(
            "ModelingPipelineOps is not wired yet. Provide repo-specific functions for build_site_dataset, train_model, run_inference, calibrate_predictions, and export_prediction_bundle."
        )

    return ModelingPipelineOps(
        build_site_dataset=_not_wired,
        train_model=_not_wired,
        run_inference=_not_wired,
        calibrate_predictions=_not_wired,
        export_prediction_bundle=_not_wired,
    )
