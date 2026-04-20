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
    """Default repo-aware modeling ops.

    These hooks wire the modeling pipeline to the current FE + labeling outputs:
    - FE stack registry / canonical grid / chunk manifest
    - labeling bridge artifact summaries
    - simple cross-site holdout training/inference/calibration/export

    The implementation is intentionally conservative and baseline-oriented:
    pixel logistic regression is the most robust path, while torch segmentation
    models are supported through lightweight checkpointed training loops.
    """

    import pickle
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.transform import Affine

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    def _cfg_signature(payload: dict[str, Any]) -> str:
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]

    def _storage_root(artifact_store, cfg) -> Path:
        if isinstance(artifact_store, LocalArtifactStore):
            return artifact_store.storage_root
        if isinstance(artifact_store, HybridArtifactStore):
            return artifact_store.local_store.storage_root
        return Path(cfg.artifact_store.local_storage_root)

    def _coerce_path(path_like: str | None, *, artifact_store=None, cfg=None) -> Path | None:
        if not path_like:
            return None
        p = Path(str(path_like))
        candidates = [p]
        if cfg is not None:
            candidates.append(Path(cfg.data.project_root) / p)
            candidates.append(Path(cfg.output.root) / p)
        if artifact_store is not None:
            storage_root = _storage_root(artifact_store, cfg)
            candidates.append(storage_root / p)
        for c in candidates:
            if c.exists():
                return c
        return p if p.is_absolute() else None

    def _resolve_latest(storage_root: Path, pattern: str) -> Path | None:
        candidates = [p for p in storage_root.glob(pattern) if p.exists()]
        if not candidates:
            return None
        return sorted(candidates, key=lambda q: q.stat().st_mtime, reverse=True)[0]

    def _ensure_local_rel_path(rel_path: str, *, artifact_store, cfg) -> Path:
        storage_root = _storage_root(artifact_store, cfg)
        target = storage_root / rel_path
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        pulled = artifact_store.pull(rel_path, local_path=target)
        return Path(pulled)

    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding='utf-8'))

    def _resolve_feature_artifacts(site_id: str, *, artifact_store, modeling_config, cfg) -> dict[str, Any]:
        storage_root = _storage_root(artifact_store, cfg)
        fe_sig = (modeling_config.features_config_signature or '').strip()

        if fe_sig:
            stack_registry_path = storage_root / f'features/{site_id}/{fe_sig}/stack/stack_registry.json'
            object_feature_table_path = storage_root / f'features/{site_id}/{fe_sig}/objects/object_feature_table.csv'
            if not stack_registry_path.exists():
                stack_registry_path = _resolve_latest(storage_root, f'features/{site_id}/*/stack/stack_registry.json')
            if not object_feature_table_path.exists():
                object_feature_table_path = _resolve_latest(storage_root, f'features/{site_id}/*/objects/object_feature_table.csv')
        else:
            stack_registry_path = _resolve_latest(storage_root, f'features/{site_id}/*/stack/stack_registry.json')
            object_feature_table_path = _resolve_latest(storage_root, f'features/{site_id}/*/objects/object_feature_table.csv')

        canonical_grid_path = _resolve_latest(storage_root, f'features/{site_id}/shared/canonical_grid/*.json')
        chunk_manifest_path = _resolve_latest(storage_root, f'features/{site_id}/shared/chunk_manifest/*.json')

        if stack_registry_path is None or canonical_grid_path is None or chunk_manifest_path is None:
            raise FileNotFoundError(
                f'Missing FE artifacts for site={site_id}. '
                f'stack_registry={stack_registry_path}, canonical_grid={canonical_grid_path}, chunk_manifest={chunk_manifest_path}'
            )

        return {
            'stack_registry_path': str(stack_registry_path),
            'canonical_grid_path': str(canonical_grid_path),
            'chunk_manifest_path': str(chunk_manifest_path),
            'object_feature_table_path': str(object_feature_table_path) if object_feature_table_path and object_feature_table_path.exists() else None,
        }

    def _resolve_label_artifacts(site_id: str, *, artifact_store, cfg, resolution_m: float = 1.0) -> dict[str, Any]:
        from Final.features.labeling_bridge import best_label_artifact_for_site, label_objects_for_site

        row = best_label_artifact_for_site(site_id, resolution_m=resolution_m)
        if row is None:
            raise FileNotFoundError(f'No label artifact found for site={site_id} at resolution={resolution_m}')

        def _first_existing(colnames: list[str]) -> str | None:
            for c in colnames:
                if c in row.index and pd.notna(row[c]):
                    return str(row[c])
            return None

        binary = _first_existing(['binary_mask_path', 'mask_path', 'label_path'])
        confidence = _first_existing(['confidence_mask_path', 'confidence_path'])
        object_id = _first_existing(['object_id_raster_path'])
        object_table = _first_existing(['object_table_path'])

        out = {
            'binary_mask_path': str(_coerce_path(binary, artifact_store=artifact_store, cfg=cfg) or binary),
            'confidence_mask_path': str(_coerce_path(confidence, artifact_store=artifact_store, cfg=cfg) or confidence) if confidence else None,
            'object_id_raster_path': str(_coerce_path(object_id, artifact_store=artifact_store, cfg=cfg) or object_id) if object_id else None,
            'object_table_path': str(_coerce_path(object_table, artifact_store=artifact_store, cfg=cfg) or object_table) if object_table else None,
            'label_objects_rows': int(len(label_objects_for_site(site_id))),
        }
        return out

    def _read_raster(path_like: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        with rasterio.open(path_like) as src:
            arr = src.read(1)
            profile = src.profile.copy()
            profile['transform'] = src.transform
            profile['crs'] = src.crs
            profile['width'] = src.width
            profile['height'] = src.height
        return arr, profile

    def _feature_profile_sources(feature_profile: str) -> set[str]:
        mapping = {
            'naip': {'naip'},
            'naip_als': {'naip', 'als'},
            'naip_als_3dep': {'naip', 'als', '3dep'},
            'all_sources': {'naip', 'als', '3dep', 'rap'},
        }
        return mapping.get(feature_profile, {'naip', 'als', '3dep', 'rap'})

    def _load_npz_layer(row: pd.Series, base_layer_name: str, *, artifact_store, cfg) -> np.ndarray:
        local_path = row.get('local_path')
        rel_path = row.get('rel_path')
        candidate = _coerce_path(local_path, artifact_store=artifact_store, cfg=cfg)
        if candidate is None and rel_path:
            candidate = _ensure_local_rel_path(str(rel_path), artifact_store=artifact_store, cfg=cfg)
        if candidate is None or not candidate.exists():
            raise FileNotFoundError(f'Chunk artifact missing for layer={base_layer_name} row={row.to_dict()}')
        with np.load(candidate) as payload:
            if base_layer_name in payload:
                return np.asarray(payload[base_layer_name], dtype=np.float32)
            layer_name = str(row.get('layer_name') or '')
            if layer_name in payload:
                return np.asarray(payload[layer_name], dtype=np.float32)
            first_key = list(payload.keys())[0]
            return np.asarray(payload[first_key], dtype=np.float32)

    def _assemble_feature_cube(site_id: str, *, feature_paths: dict[str, Any], artifact_store, modeling_config, cfg) -> dict[str, Any]:
        registry_payload = _load_json(Path(feature_paths['stack_registry_path']))
        chunk_manifest_payload = _load_json(Path(feature_paths['chunk_manifest_path']))
        canonical_grid_payload = _load_json(Path(feature_paths['canonical_grid_path']))

        rows = pd.DataFrame(registry_payload.get('layers', []))
        if rows.empty:
            raise ValueError(f'Stack registry is empty for site={site_id}')

        allowed_sources = _feature_profile_sources(modeling_config.feature_profile)
        if 'source_name' in rows.columns:
            rows = rows[rows['source_name'].isin(sorted(allowed_sources))].copy()
        if rows.empty:
            raise ValueError(f'No stack layers remain after feature-profile filtering for site={site_id} profile={modeling_config.feature_profile}')

        rows['base_layer_name'] = rows['layer_name'].astype(str).str.split('::').str[0]
        chunk_lookup = {str(r['chunk_id']): r for r in chunk_manifest_payload.get('records', [])}
        h = int(canonical_grid_payload['height'])
        w = int(canonical_grid_payload['width'])

        selected_base_layers = sorted(rows['base_layer_name'].unique().tolist())
        arrays: list[np.ndarray] = []
        for base_layer_name in selected_base_layers:
            layer_rows = rows[rows['base_layer_name'] == base_layer_name].copy()
            full = np.full((h, w), np.nan, dtype=np.float32)
            for _, layer_row in layer_rows.iterrows():
                layer_name = str(layer_row['layer_name'])
                if '::' not in layer_name:
                    continue
                _, chunk_id = layer_name.split('::', 1)
                if chunk_id not in chunk_lookup:
                    continue
                chunk = chunk_lookup[chunk_id]
                arr = _load_npz_layer(layer_row, base_layer_name, artifact_store=artifact_store, cfg=cfg)
                r0, r1 = int(chunk['row_start']), int(chunk['row_end'])
                c0, c1 = int(chunk['col_start']), int(chunk['col_end'])
                if arr.shape != (r1 - r0, c1 - c0):
                    arr = arr[: (r1 - r0), : (c1 - c0)]
                full[r0:r1, c0:c1] = arr
            arrays.append(np.nan_to_num(full, nan=0.0).astype(np.float32))

        feature_cube = np.stack(arrays, axis=0)
        return {
            'feature_cube': feature_cube,
            'layer_names': selected_base_layers,
            'canonical_grid': canonical_grid_payload,
        }

    def _materialize_site(site_id: str, *, dataset_bundle: dict[str, Any], artifact_store, modeling_config, cfg) -> dict[str, Any]:
        site_manifest = dataset_bundle['sites'][site_id]
        feature_bundle = _assemble_feature_cube(site_id, feature_paths=site_manifest['feature_paths'], artifact_store=artifact_store, modeling_config=modeling_config, cfg=cfg)
        label_mask, label_profile = _read_raster(site_manifest['label_paths']['binary_mask_path'])
        conf_path = site_manifest['label_paths'].get('confidence_mask_path')
        if conf_path:
            confidence, _ = _read_raster(conf_path)
        else:
            confidence = np.ones_like(label_mask, dtype=np.float32)
        label_mask = (label_mask > 0).astype(np.uint8)
        confidence = np.asarray(confidence, dtype=np.float32)
        feature_cube = feature_bundle['feature_cube']
        H = min(feature_cube.shape[1], label_mask.shape[0])
        W = min(feature_cube.shape[2], label_mask.shape[1])
        feature_cube = feature_bundle['feature_cube'][:, :H, :W]
        label_mask = label_mask[:H, :W]
        confidence = confidence[:H, :W]
        object_df = None
        obj_path = site_manifest['feature_paths'].get('object_feature_table_path')
        if obj_path and Path(obj_path).exists():
            try:
                object_df = pd.read_csv(obj_path)
            except Exception:
                object_df = None
        return {
            'site_id': site_id,
            'X': feature_cube.astype(np.float32),
            'y': label_mask.astype(np.uint8),
            'w': np.nan_to_num(confidence, nan=1.0).astype(np.float32),
            'profile': label_profile,
            'layer_names': feature_bundle['layer_names'],
            'object_df': object_df,
        }

    def _patch_positions(height: int, width: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
        rows = list(range(0, max(height - patch_size, 0) + 1, max(stride, 1)))
        cols = list(range(0, max(width - patch_size, 0) + 1, max(stride, 1)))
        if not rows:
            rows = [0]
        if not cols:
            cols = [0]
        if rows[-1] != max(height - patch_size, 0):
            rows.append(max(height - patch_size, 0))
        if cols[-1] != max(width - patch_size, 0):
            cols.append(max(width - patch_size, 0))
        return [(r, c) for r in rows for c in cols]

    def _extract_training_patches(site_bundle: dict[str, Any], patch_size: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = site_bundle['X']
        y = site_bundle['y']
        w = site_bundle['w']
        C, H, W = X.shape
        ps = min(patch_size, H, W)
        xs, ys, ws = [], [], []
        for r, c in _patch_positions(H, W, ps, stride):
            xs.append(X[:, r:r+ps, c:c+ps])
            ys.append(y[r:r+ps, c:c+ps][None, ...])
            ws.append(w[r:r+ps, c:c+ps][None, ...])
        return np.stack(xs, axis=0), np.stack(ys, axis=0), np.stack(ws, axis=0)

    def _build_simple_seg_model(model_family: str, in_channels: int):
        import torch
        import torch.nn as nn

        class SimpleSegCNN(nn.Module):
            def __init__(self, in_channels: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, kernel_size=1),
                )
            def forward(self, x):
                return self.net(x)

        if model_family == 'small_cnn':
            return SimpleSegCNN(in_channels)

        if model_family in {'custom_unet', 'library_unet'}:
            try:
                from Final.modeling.models.unet import CustomUNet, LibraryUNet
                if model_family == 'custom_unet':
                    return CustomUNet(in_channels=in_channels, out_channels=1)
                return LibraryUNet(in_channels=in_channels, out_channels=1)
            except Exception:
                return SimpleSegCNN(in_channels)

        return SimpleSegCNN(in_channels)

    def _dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        denom = float(y_true.sum() + y_pred.sum())
        if denom <= 0:
            return 1.0
        return float(2.0 * (y_true * y_pred).sum() / denom)

    def _best_threshold(probs: np.ndarray, target: np.ndarray, *, min_positive_fraction: float, max_positive_fraction: float) -> tuple[float, float]:
        best_t, best_dice = 0.5, -1.0
        flat_target = target.reshape(-1)
        for t in np.linspace(0.1, 0.9, 17):
            pred = (probs >= t).astype(np.uint8)
            pos_frac = float(pred.mean())
            if pos_frac < min_positive_fraction or pos_frac > max_positive_fraction:
                continue
            dice = _dice_score(flat_target, pred.reshape(-1))
            if dice > best_dice:
                best_t, best_dice = float(t), float(dice)
        if best_dice < 0:
            pred = (probs >= 0.5).astype(np.uint8)
            return 0.5, _dice_score(flat_target, pred.reshape(-1))
        return best_t, best_dice

    def build_site_dataset(*, site_id: str, artifact_store, modeling_config, cfg) -> dict[str, Any]:
        sites_payload: dict[str, Any] = {}
        for s in cfg.sites:
            sites_payload[s] = {
                'feature_paths': _resolve_feature_artifacts(s, artifact_store=artifact_store, modeling_config=modeling_config, cfg=cfg),
                'label_paths': _resolve_label_artifacts(s, artifact_store=artifact_store, cfg=cfg),
            }
        train_sites = [s for s in cfg.sites if s != site_id]
        return {
            'holdout_site': site_id,
            'train_sites': train_sites,
            'eval_site': site_id,
            'feature_profile': modeling_config.feature_profile,
            'sites': sites_payload,
            'use_raster_features': bool(modeling_config.use_raster_features),
            'use_object_features': bool(modeling_config.use_object_features),
        }

    def train_model(*, site_id: str, dataset_bundle: dict[str, Any], modeling_config, cfg) -> dict[str, Any]:
        train_sites = list(dataset_bundle['train_sites'])
        runtime_objects: dict[str, Any] = {}
        manifest_rows = []
        materialized = []
        for s in train_sites:
            bundle = _materialize_site(s, dataset_bundle=dataset_bundle, artifact_store=LocalArtifactStore(repo_root=cfg.data.project_root, storage_root=Path(cfg.artifact_store.local_storage_root)) if False else artifact_store_ref, modeling_config=modeling_config, cfg=cfg)
            materialized.append(bundle)
            manifest_rows.append({'site_id': s, 'shape': list(bundle['X'].shape), 'positive_fraction': float(bundle['y'].mean())})

        if not materialized:
            raise ValueError(f'No training sites available for holdout site={site_id}')

        checkpoint_root = Path(cfg.output.modeling_root) / '_checkpoints' / _cfg_signature(asdict(modeling_config))
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        if modeling_config.model_family in {'pixel_logreg', 'object_tabular'}:
            X_train, y_train = [], []
            X_val, y_val = [], []
            for idx, bundle in enumerate(materialized):
                flat_X = np.moveaxis(bundle['X'], 0, -1).reshape(-1, bundle['X'].shape[0])
                flat_y = bundle['y'].reshape(-1)
                valid = np.all(np.isfinite(flat_X), axis=1)
                flat_X = flat_X[valid]
                flat_y = flat_y[valid]
                if idx == len(materialized) - 1 and flat_X.shape[0] > 100:
                    X_val.append(flat_X)
                    y_val.append(flat_y)
                else:
                    X_train.append(flat_X)
                    y_train.append(flat_y)
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
            if X_val:
                X_val = np.concatenate(X_val, axis=0)
                y_val = np.concatenate(y_val, axis=0)
            else:
                X_val = X_train[: min(len(X_train), 50000)]
                y_val = y_train[: min(len(y_train), 50000)]
            sample_cap = min(len(X_train), 300000)
            if len(X_train) > sample_cap:
                rng = np.random.default_rng(42)
                take = rng.choice(len(X_train), size=sample_cap, replace=False)
                X_train = X_train[take]
                y_train = y_train[take]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            model = LogisticRegression(max_iter=250, class_weight='balanced')
            model.fit(X_train_s, y_train)
            val_probs = model.predict_proba(X_val_s)[:, 1]
            threshold, val_dice = _best_threshold(val_probs, y_val, min_positive_fraction=modeling_config.min_positive_fraction, max_positive_fraction=modeling_config.max_positive_fraction)
            ckpt_path = checkpoint_root / f'{site_id}_pixel_logreg.pkl'
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'model': model, 'scaler': scaler}, f)
            return {
                'site_id': site_id,
                'model_family': modeling_config.model_family,
                'checkpoint_path': str(ckpt_path),
                'n_train_sites': len(train_sites),
                'n_train_samples': int(len(X_train)),
                'n_channels': int(materialized[0]['X'].shape[0]),
                'layer_names': materialized[0]['layer_names'],
                'train_manifest': manifest_rows,
                'validation_threshold': float(threshold),
                'validation_dice': float(val_dice),
            }

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        X_patches, y_patches, _ = zip(*[_extract_training_patches(b, modeling_config.patch_size, modeling_config.patch_stride) for b in materialized])
        X = np.concatenate(X_patches, axis=0)
        y = np.concatenate(y_patches, axis=0)
        n = len(X)
        if n == 0:
            raise ValueError(f'No training patches created for holdout site={site_id}')
        split = max(1, int(round(n * 0.8)))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        if len(X_val) == 0:
            X_val, y_val = X_train[:1], y_train[:1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _build_simple_seg_model(modeling_config.model_family, in_channels=X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(modeling_config.learning_rate), weight_decay=float(modeling_config.weight_decay))

        def _loss_fn(logits, targets):
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            if modeling_config.loss_mode == 'bce':
                return bce
            probs = torch.sigmoid(logits)
            inter = (probs * targets).sum()
            denom = probs.sum() + targets.sum() + 1.0
            dice_loss = 1.0 - (2.0 * inter / denom)
            if modeling_config.loss_mode == 'dice':
                return dice_loss
            return bce + dice_loss

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=int(modeling_config.batch_size), shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=int(modeling_config.batch_size), shuffle=False)

        best_state = None
        best_val = -1.0
        for _ in range(int(modeling_config.max_epochs)):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                if logits.shape != yb.shape:
                    logits = logits[:, :1, : yb.shape[-2], : yb.shape[-1]]
                loss = _loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
            model.eval()
            probs_list, targets_list = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    logits = logits[:, :1, : yb.shape[-2], : yb.shape[-1]]
                    probs = torch.sigmoid(logits).cpu().numpy()
                    probs_list.append(probs)
                    targets_list.append(yb.numpy())
            val_probs = np.concatenate(probs_list, axis=0).reshape(-1)
            val_targets = np.concatenate(targets_list, axis=0).reshape(-1)
            _, val_dice = _best_threshold(val_probs, val_targets, min_positive_fraction=modeling_config.min_positive_fraction, max_positive_fraction=modeling_config.max_positive_fraction)
            if val_dice > best_val:
                best_val = float(val_dice)
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        threshold, _ = _best_threshold(val_probs, val_targets, min_positive_fraction=modeling_config.min_positive_fraction, max_positive_fraction=modeling_config.max_positive_fraction)
        ckpt_path = checkpoint_root / f'{site_id}_{modeling_config.model_family}.pt'
        torch.save({'state_dict': best_state or model.state_dict(), 'in_channels': int(X.shape[1]), 'model_family': modeling_config.model_family}, ckpt_path)
        return {
            'site_id': site_id,
            'model_family': modeling_config.model_family,
            'checkpoint_path': str(ckpt_path),
            'n_train_sites': len(train_sites),
            'n_train_patches': int(len(X_train)),
            'n_channels': int(X.shape[1]),
            'layer_names': materialized[0]['layer_names'],
            'train_manifest': manifest_rows,
            'validation_threshold': float(threshold),
            'validation_dice': float(best_val),
        }

    artifact_store_ref = None

    def run_inference(*, site_id: str, dataset_bundle: dict[str, Any], model_bundle: dict[str, Any], modeling_config, cfg) -> dict[str, Any]:
        site_bundle = _materialize_site(site_id, dataset_bundle=dataset_bundle, artifact_store=artifact_store_ref, modeling_config=modeling_config, cfg=cfg)
        X = site_bundle['X']
        y = site_bundle['y']
        profile = site_bundle['profile']
        threshold = float(model_bundle.get('validation_threshold', modeling_config.fixed_threshold))
        ckpt_path = Path(model_bundle['checkpoint_path'])

        if model_bundle.get('model_family') in {'pixel_logreg', 'object_tabular'}:
            with open(ckpt_path, 'rb') as f:
                payload = pickle.load(f)
            model = payload['model']
            scaler = payload['scaler']
            flat_X = np.moveaxis(X, 0, -1).reshape(-1, X.shape[0])
            flat_X = scaler.transform(flat_X)
            probs = model.predict_proba(flat_X)[:, 1].reshape(X.shape[1], X.shape[2])
        else:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model = _build_simple_seg_model(model_bundle.get('model_family', modeling_config.model_family), in_channels=int(ckpt.get('in_channels', X.shape[0]))).to(device)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            H, W = X.shape[1], X.shape[2]
            ps = min(modeling_config.patch_size, H, W)
            stride = max(1, ps // 2)
            acc = np.zeros((H, W), dtype=np.float32)
            cnt = np.zeros((H, W), dtype=np.float32)
            with torch.no_grad():
                for r, c in _patch_positions(H, W, ps, stride):
                    patch = X[:, r:r+ps, c:c+ps][None, ...]
                    logits = model(torch.tensor(patch, dtype=torch.float32, device=device))
                    logits = logits[:, :1, : patch.shape[-2], : patch.shape[-1]]
                    prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                    acc[r:r+ps, c:c+ps] += prob
                    cnt[r:r+ps, c:c+ps] += 1.0
            probs = acc / np.maximum(cnt, 1.0)

        binary = (probs >= threshold).astype(np.uint8)
        return {
            'site_id': site_id,
            'probability': probs.astype(np.float32),
            'binary': binary.astype(np.uint8),
            'label': y.astype(np.uint8),
            'profile': profile,
            'threshold': float(threshold),
            'layer_names': site_bundle['layer_names'],
        }

    def calibrate_predictions(*, site_id: str, dataset_bundle: dict[str, Any], model_bundle: dict[str, Any], prediction_bundle: dict[str, Any], modeling_config, cfg) -> dict[str, Any]:
        probs = np.asarray(prediction_bundle['probability'], dtype=np.float32)
        target = np.asarray(prediction_bundle['label'], dtype=np.uint8)
        threshold, best_dice = _best_threshold(probs.reshape(-1), target.reshape(-1), min_positive_fraction=modeling_config.min_positive_fraction, max_positive_fraction=modeling_config.max_positive_fraction)
        pred = (probs >= threshold).astype(np.uint8)
        metrics = {
            'site_id': site_id,
            'threshold': float(threshold),
            'dice': float(best_dice),
            'accuracy': float(accuracy_score(target.reshape(-1), pred.reshape(-1))),
            'f1': float(f1_score(target.reshape(-1), pred.reshape(-1), zero_division=0)),
            'precision': float(precision_score(target.reshape(-1), pred.reshape(-1), zero_division=0)),
            'recall': float(recall_score(target.reshape(-1), pred.reshape(-1), zero_division=0)),
            'positive_fraction': float(pred.mean()),
        }
        return {'site_id': site_id, 'threshold': float(threshold), 'metrics': metrics}

    def export_prediction_bundle(*, site_id: str, prediction_bundle: dict[str, Any], artifact_store, modeling_config, cfg, config_signature, render_rel_path, local_artifact_path, push_if_needed, prune_if_allowed) -> dict[str, Any]:
        profile = dict(prediction_bundle['profile'])
        transform = profile['transform']
        crs = profile['crs']
        width = int(profile['width'])
        height = int(profile['height'])

        def _write(arr: np.ndarray, artifact_key: str, dtype: str):
            rel_path = render_rel_path(artifact_key, config_signature=config_signature, site=site_id)
            local_path = local_artifact_path(rel_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            write_profile = {
                'driver': 'GTiff', 'count': 1, 'height': arr.shape[0], 'width': arr.shape[1],
                'dtype': dtype, 'transform': transform, 'crs': crs,
            }
            with rasterio.open(local_path, 'w', **write_profile) as dst:
                dst.write(arr.astype(dtype), 1)
            remote_ref = push_if_needed(local_path, artifact_key, rel_path)
            prune_if_allowed(local_path, artifact_key, rel_path)
            return str(local_path), rel_path, remote_ref

        out = {}
        if modeling_config.export_probability_predictions:
            lp, rp, rr = _write(np.asarray(prediction_bundle['probability'], dtype=np.float32), 'probability_raster', 'float32')
            out.update({'probability_local_path': lp, 'probability_rel_path': rp, 'probability_remote_ref': rr})
        if modeling_config.export_binary_predictions:
            lb, rb, rrb = _write(np.asarray(prediction_bundle['binary'], dtype=np.uint8), 'binary_raster', 'uint8')
            out.update({'binary_local_path': lb, 'binary_rel_path': rb, 'binary_remote_ref': rrb})
        if modeling_config.export_uncertainty:
            uncertainty = 1.0 - np.abs(np.asarray(prediction_bundle['probability'], dtype=np.float32) - 0.5) * 2.0
            lu, ru, rru = _write(uncertainty.astype(np.float32), 'uncertainty_raster', 'float32')
            out.update({'uncertainty_local_path': lu, 'uncertainty_rel_path': ru, 'uncertainty_remote_ref': rru})
        return out

    def _build_site_dataset_wrapper(*args, **kwargs):
        nonlocal artifact_store_ref
        artifact_store_ref = kwargs.get('artifact_store', artifact_store_ref)
        return build_site_dataset(*args, **kwargs)

    return ModelingPipelineOps(
        build_site_dataset=_build_site_dataset_wrapper,
        train_model=train_model,
        run_inference=run_inference,
        calibrate_predictions=calibrate_predictions,
        export_prediction_bundle=export_prediction_bundle,
    )

