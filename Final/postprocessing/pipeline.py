from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable
import json
import hashlib

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.windows import Window
from rasterio.features import shapes
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.segmentation import watershed

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
class PostprocessingPipelineConfig:
    modeling_config_signature: str = ""
    features_config_signature: str = ""

    threshold_mode: str = "validation_tuned"  # fixed | validation_tuned
    fixed_threshold: float = 0.5

    cleanup_mode: str = "standard"  # none | standard | aggressive
    min_component_pixels: int = 4
    fill_holes: bool = True
    opening_radius: int = 1
    closing_radius: int = 1

    split_mode: str = "watershed"  # connected_components | watershed
    watershed_min_distance: int = 3

    enable_object_enrichment: bool = True
    enable_object_filtering: bool = True
    object_filter_min_area: float = 1.0
    object_filter_max_area: float = 1000.0
    object_filter_min_confidence: float = 0.0

    compute_site_summaries: bool = True
    predict_heights: bool = False


@dataclass
class PostprocessingPipelineOps:
    """Optional repo-specific hooks for loading modeling outputs or enriching objects."""

    load_site_prediction_bundle: Callable[..., dict[str, Any]]
    load_site_feature_bundle: Callable[..., dict[str, Any]] | None = None
    enrich_object_table: Callable[..., pd.DataFrame] | None = None
    filter_object_table: Callable[..., pd.DataFrame] | None = None
    clear_artifact_staging_dir: Callable[[], None] | None = None


class PostprocessingPipeline(BasePipeline):
    def __init__(self, cfg, *, ops: PostprocessingPipelineOps, pipeline_config: PostprocessingPipelineConfig | None = None):
        super().__init__(cfg, pipeline_name="postprocessing", output_root=cfg.output.postprocessing_root / "pipeline_runs")
        self.logger = get_logger("postprocessing.pipeline")
        self.ops = ops
        self.pipeline_config = pipeline_config or PostprocessingPipelineConfig()
        self._enumeration_cache: dict[tuple, list[dict[str, Any]]] = {}

        self.artifact_store = self._build_artifact_store()
        self.coordination = CoordinationManager(self.artifact_store, root_prefix=getattr(getattr(cfg, "coordination", object()), "root_prefix", "coordination"))
        self.shared_registry = SharedArtifactRegistry(self.artifact_store, root_prefix=getattr(getattr(cfg, "shared_artifacts", object()), "registry_prefix", "shared_artifacts"))

    def config_dict(self) -> dict:
        return asdict(self.pipeline_config)

    def config_signature(self, config_dict: dict | None = None) -> str:
        payload = config_dict if config_dict is not None else self.config_dict()
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def _runtime_caps(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        runtime_cfg = getattr(self.cfg, "postprocessing_runtime", None)
        return tuple(getattr(runtime_cfg, name, default))

    def _storage_cfg(self):
        runtime_cfg = getattr(self.cfg, "postprocessing_runtime", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "storage"):
            return runtime_cfg.storage
        return type("Storage", (), {
            "enable_local_store": True,
            "enable_drive_store": False,
            "use_hybrid_store": False,
            "fail_if_drive_missing": False,
        })()

    def _storage_policy_cfg(self):
        runtime_cfg = getattr(self.cfg, "postprocessing_runtime", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "storage_policy"):
            return runtime_cfg.storage_policy
        return type("StoragePolicy", (), {
            "push_large_artifacts_to_remote": False,
            "prune_local_after_remote_push": False,
            "verify_remote_before_prune": True,
        })()

    def artifact_specs(self) -> dict[str, ArtifactSpec]:
        return {
            "binary_raster": ArtifactSpec(
                key="binary_raster",
                rel_path_template="postprocessing/{config_signature}/{site}/binary.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "object_id_raster": ArtifactSpec(
                key="object_id_raster",
                rel_path_template="postprocessing/{config_signature}/{site}/object_id.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "predicted_objects": ArtifactSpec(
                key="predicted_objects",
                rel_path_template="postprocessing/{config_signature}/{site}/predicted_objects.csv",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "site_summary": ArtifactSpec(
                key="site_summary",
                rel_path_template="postprocessing/{config_signature}/{site}/site_summary.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "run_result": ArtifactSpec(
                key="run_result",
                rel_path_template="postprocessing/{config_signature}/postprocessing_run_result.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
        }

    def _build_artifact_store(self):
        storage = self._storage_cfg()
        global_store = self.cfg.artifact_store
        local_store = LocalArtifactStore(repo_root=self.cfg.data.project_root, storage_root=Path(global_store.local_storage_root))

        drive_store = None
        if storage.enable_drive_store:
            missing = [str(p) for p in [global_store.drive_client_secrets_path, global_store.drive_config_path] if not Path(p).exists()]
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
            return HybridArtifactStore(local_store=local_store, remote_store=drive_store)
        if drive_store is not None and not storage.enable_local_store:
            return drive_store
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
        return self.cfg.output.postprocessing_root / "_remote_stage" / rel_path

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
        if not policy.prune_local_after_remote_push or not spec.prune_local_after_push:
            return
        if policy.verify_remote_before_prune and not self._remote_exists(rel_path):
            return
        if local_path.exists():
            local_path.unlink()

    def validate_hydrated_artifact(self, *, rel_path: str, local_path: Path, artifact_key: str | None = None) -> bool:
        try:
            if artifact_key in {"binary_raster", "object_id_raster"}:
                with rasterio.open(local_path) as src:
                    src.read(1, window=Window(0, 0, min(16, src.width), min(16, src.height)))
                return True
            if artifact_key == "predicted_objects":
                pd.read_csv(local_path, nrows=5)
                return True
            if artifact_key in {"site_summary", "run_result"}:
                json.loads(local_path.read_text(encoding="utf-8"))
                return True
            return super().validate_hydrated_artifact(rel_path=rel_path, local_path=local_path, artifact_key=artifact_key)
        except Exception as e:
            self.logger.warning("HYDRATE VALIDATE FAIL | rel_path=%s | artifact_key=%s | error=%s", rel_path, artifact_key, e)
            return False

    def _persist_json(self, payload: dict, artifact_key: str, **fmt) -> tuple[Path, str, str | None]:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        remote_ref = self._push_if_needed(local_path, artifact_key, rel_path)
        self._prune_if_allowed(local_path, artifact_key, rel_path)
        return local_path, rel_path, remote_ref

    def _persist_df(self, df: pd.DataFrame, artifact_key: str, **fmt) -> tuple[Path, str, str | None]:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
        remote_ref = self._push_if_needed(local_path, artifact_key, rel_path)
        self._prune_if_allowed(local_path, artifact_key, rel_path)
        return local_path, rel_path, remote_ref

    def _write_raster_like(self, arr: np.ndarray, profile: dict, artifact_key: str, **fmt) -> tuple[Path, str, str | None]:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        out_profile = dict(profile)
        out_profile.update(count=1, dtype=str(arr.dtype), compress="deflate")
        with rasterio.open(local_path, "w", **out_profile) as dst:
            dst.write(arr, 1)
        remote_ref = self._push_if_needed(local_path, artifact_key, rel_path)
        self._prune_if_allowed(local_path, artifact_key, rel_path)
        return local_path, rel_path, remote_ref

    def build_pipeline_spec(self) -> PipelineSpec:
        modules = {
            "postprocessing.threshold.base": ModuleSpec(
                key="postprocessing.threshold.base",
                stage_name="threshold",
                param_keys=("threshold_mode", "fixed_threshold"),
                runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_threshold", ("runtime:modeling",)), mode=RuntimeRequirementMode.ALL),
            ),
            "postprocessing.mask_cleanup.base": ModuleSpec(
                key="postprocessing.mask_cleanup.base",
                stage_name="mask_cleanup",
                param_keys=("cleanup_mode", "min_component_pixels", "fill_holes", "opening_radius", "closing_radius"),
                runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_mask_cleanup", ("runtime:modeling",)), mode=RuntimeRequirementMode.ALL),
            ),
            "postprocessing.instance_split.base": ModuleSpec(
                key="postprocessing.instance_split.base",
                stage_name="instance_split",
                param_keys=("split_mode", "watershed_min_distance"),
                runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_instance_split", ("runtime:modeling",)), mode=RuntimeRequirementMode.ALL),
            ),
            "postprocessing.object_extract.base": ModuleSpec(
                key="postprocessing.object_extract.base",
                stage_name="object_extract",
                runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_object_extract", ("runtime:modeling",)), mode=RuntimeRequirementMode.ALL),
            ),
            "postprocessing.object_enrich.base": ModuleSpec(
                key="postprocessing.object_enrich.base",
                stage_name="object_enrich",
                enabled_key="enable_object_enrichment",
                runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_object_enrich", ("runtime:modeling",)), mode=RuntimeRequirementMode.ALL),
            ),
            "postprocessing.object_filter.base": ModuleSpec(
                key="postprocessing.object_filter.base",
                stage_name="object_filter",
                enabled_key="enable_object_filtering",
                param_keys=("object_filter_min_area", "object_filter_max_area", "object_filter_min_confidence"),
                runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_object_filter", ("runtime:modeling",)), mode=RuntimeRequirementMode.ALL),
            ),
            "postprocessing.site_summary.base": ModuleSpec(
                key="postprocessing.site_summary.base",
                stage_name="site_summary",
                enabled_key="compute_site_summaries",
                runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_site_summary", ("runtime:modeling",)), mode=RuntimeRequirementMode.ALL),
            ),
        }
        stages = [
            StageSpec(name="threshold", module_keys=["postprocessing.threshold.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="mask_cleanup", module_keys=["postprocessing.mask_cleanup.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="instance_split", module_keys=["postprocessing.instance_split.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="object_extract", module_keys=["postprocessing.object_extract.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="object_enrich", module_keys=["postprocessing.object_enrich.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="object_filter", module_keys=["postprocessing.object_filter.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="site_summary", module_keys=["postprocessing.site_summary.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
        ]
        search_axes = [
            SearchAxis(key="threshold_mode", values=["fixed", "validation_tuned"], stage_name="threshold", module_key="postprocessing.threshold.base"),
            SearchAxis(key="cleanup_mode", values=["standard", "aggressive"], stage_name="mask_cleanup", module_key="postprocessing.mask_cleanup.base"),
            SearchAxis(key="split_mode", values=["connected_components", "watershed"], stage_name="instance_split", module_key="postprocessing.instance_split.base"),
            SearchAxis(key="enable_object_filtering", values=[False, True], stage_name="object_filter", module_key="postprocessing.object_filter.base"),
        ]
        return PipelineSpec(pipeline_name="postprocessing", domain=PipelineDomain.POSTPROCESSING, stages=stages, modules=modules, search_axes=search_axes)

    # ------------------------------------------------------------------
    # Shared signatures / work units
    # ------------------------------------------------------------------

    def _shared_sig(self, stage: str, site: str) -> str:
        payload = {"stage": stage, "site": site, "pipeline_config": asdict(self.pipeline_config)}
        return hash_payload(payload)

    def shared_artifact_is_valid(self, *, artifact_family: str, shared_signature: str) -> bool:
        if not getattr(getattr(self.cfg, "shared_artifacts", object()), "enable_shared_artifact_registry", False):
            return False
        rec = self.shared_registry.load(artifact_family=artifact_family, shared_signature=shared_signature)
        return rec is not None and str(rec.status).lower().endswith("valid")

    def register_shared_requirement(self, *, artifact_family: str, shared_signature: str, trial_id: str, metadata: dict | None = None) -> None:
        if not getattr(getattr(self.cfg, "shared_artifacts", object()), "enable_shared_artifact_registry", False):
            return
        rec = self.shared_registry.upsert_requirement(artifact_family=artifact_family, shared_signature=shared_signature, producer_pipeline=self.pipeline_name, trial_id=trial_id)
        if metadata:
            rec.metadata.update(metadata)
            self.shared_registry.save(rec)

    def _mark_shared_available(self, *, artifact_family: str, shared_signature: str, rel_path: str | None = None, local_path: str | None = None, remote_ref: str | None = None, metadata: dict | None = None, trial_id: str | None = None) -> None:
        if not getattr(getattr(self.cfg, "shared_artifacts", object()), "enable_shared_artifact_registry", False):
            return
        self.shared_registry.mark_available(artifact_family=artifact_family, shared_signature=shared_signature, producer_pipeline=self.pipeline_name, rel_path=rel_path, local_path=local_path, remote_ref=remote_ref, source_trial=trial_id, metadata=metadata or {}, status=SharedArtifactStatus.VALID)

    def enumerate_work_units(self, *, trial_id: str, config_signature: str | None = None, runtime_report=None, register_shared_requirements: bool = False) -> list[dict]:
        config_signature = config_signature or self.config_signature()
        runtime_report = runtime_report or self.runtime_report()
        enum_cache_key = (trial_id, config_signature, getattr(runtime_report, "detected_image_key", None), register_shared_requirements, self.work_unit_refresh_fingerprint(trial_id=trial_id, config_signature=config_signature, runtime_report=runtime_report))
        if enum_cache_key in self._enumeration_cache:
            return [dict(u) for u in self._enumeration_cache[enum_cache_key]]

        stage_ok = {
            stage.name: all(info.status.value == "eligible" for info in self.stage_runtime_eligibility(stage.name, runtime_report).values())
            for stage in self.pipeline_spec.stages
        }
        stage_order = [s.name for s in self.pipeline_spec.stages]
        stage_complete: dict[tuple[str, str], bool] = {}
        units: list[dict[str, Any]] = []

        for site in self.cfg.sites:
            for idx, stage_name in enumerate(stage_order):
                shared_sig = self._shared_sig(stage_name, site)
                artifact_family = f"postprocessing.{stage_name}"
                if register_shared_requirements:
                    self.register_shared_requirement(artifact_family=artifact_family, shared_signature=shared_sig, trial_id=trial_id, metadata={"site": site, "stage": stage_name})
                complete = self.shared_artifact_is_valid(artifact_family=artifact_family, shared_signature=shared_sig)
                stage_complete[(site, stage_name)] = complete

                deps, dep_reasons = [], []
                if idx > 0 and not stage_complete.get((site, stage_order[idx - 1]), False):
                    deps.append(stage_order[idx - 1])
                    dep_reasons.append(f"Previous stage {stage_order[idx - 1]} is not ready.")

                units.append({
                    "unit_id": f"{trial_id}:{self.pipeline_name}:{stage_name}:{site}",
                    "trial_id": trial_id,
                    "pipeline_name": self.pipeline_name,
                    "config_signature": config_signature,
                    "stage_name": stage_name,
                    "work_key": site,
                    "scope": WorkUnitScope.SITE.value,
                    "status": WorkUnitStatus.COMPLETE.value if complete else (
                        WorkUnitStatus.PENDING.value if (stage_ok[stage_name] and not deps) else (
                            WorkUnitStatus.BLOCKED.value if stage_ok[stage_name] else WorkUnitStatus.INELIGIBLE.value
                        )
                    ),
                    "dependencies": deps,
                    "dependency_reasons": dep_reasons,
                    "runtime_required_capabilities": list(self._runtime_caps(f"require_capability_{stage_name}", ("runtime:modeling",))),
                    "runtime_eligible": stage_ok[stage_name],
                    "priority": 100 * (idx + 1),
                    "site_id": site,
                    "shared_artifact_family": artifact_family,
                    "shared_signature": shared_sig,
                })

        self._enumeration_cache[enum_cache_key] = [dict(u) for u in units]
        return units

    # ------------------------------------------------------------------
    # Core image/object logic
    # ------------------------------------------------------------------

    def _resolve_threshold(self, bundle: dict[str, Any]) -> float:
        if self.pipeline_config.threshold_mode == "fixed":
            return float(self.pipeline_config.fixed_threshold)
        if "calibration" in bundle and isinstance(bundle["calibration"], dict):
            tuned = bundle["calibration"].get("threshold")
            if tuned is not None:
                return float(tuned)
        return float(self.pipeline_config.fixed_threshold)

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool)
        if self.pipeline_config.cleanup_mode == "none":
            return mask.astype(np.uint8)
        if self.pipeline_config.fill_holes:
            mask = ndi.binary_fill_holes(mask)
        if self.pipeline_config.opening_radius > 0:
            mask = morphology.binary_opening(mask, morphology.disk(self.pipeline_config.opening_radius))
        if self.pipeline_config.closing_radius > 0:
            mask = morphology.binary_closing(mask, morphology.disk(self.pipeline_config.closing_radius))
        mask = morphology.remove_small_objects(mask, min_size=max(1, int(self.pipeline_config.min_component_pixels)))
        if self.pipeline_config.cleanup_mode == "aggressive":
            mask = morphology.binary_opening(mask, morphology.disk(max(1, self.pipeline_config.opening_radius + 1)))
            mask = morphology.binary_closing(mask, morphology.disk(max(1, self.pipeline_config.closing_radius + 1)))
            mask = morphology.remove_small_objects(mask, min_size=max(1, int(self.pipeline_config.min_component_pixels * 2)))
        return mask.astype(np.uint8)

    def _split_instances(self, binary_mask: np.ndarray) -> np.ndarray:
        if self.pipeline_config.split_mode == "connected_components":
            labeled, _ = ndi.label(binary_mask > 0)
            return labeled.astype(np.int32)
        dist = ndi.distance_transform_edt(binary_mask > 0)
        footprint = morphology.disk(max(1, int(self.pipeline_config.watershed_min_distance)))
        local_max = morphology.local_maxima(dist, footprint=footprint)
        markers, _ = ndi.label(local_max)
        labels = watershed(-dist, markers=markers, mask=binary_mask > 0)
        return labels.astype(np.int32)

    def _extract_object_table(self, labels: np.ndarray, probability: np.ndarray, profile: dict) -> pd.DataFrame:
        transform = profile["transform"]
        rows: list[dict[str, Any]] = []
        for region in measure.regionprops(labels, intensity_image=probability):
            rr, cc = region.coords[:, 0], region.coords[:, 1]
            centroid_row, centroid_col = region.centroid
            x_center, y_center = xy(transform, centroid_row, centroid_col)
            rows.append({
                "object_id": int(region.label),
                "row": float(centroid_row),
                "col": float(centroid_col),
                "x": float(x_center),
                "y": float(y_center),
                "pixel_area": int(region.area),
                "bbox_min_row": int(region.bbox[0]),
                "bbox_min_col": int(region.bbox[1]),
                "bbox_max_row": int(region.bbox[2]),
                "bbox_max_col": int(region.bbox[3]),
                "perimeter": float(region.perimeter),
                "eccentricity": float(getattr(region, "eccentricity", 0.0)),
                "solidity": float(getattr(region, "solidity", 0.0)),
                "mean_probability": float(region.mean_intensity),
                "max_probability": float(np.max(probability[rr, cc])) if len(rr) else 0.0,
            })
        return pd.DataFrame(rows)

    def _site_summary(self, site_id: str, objects_df: pd.DataFrame, binary_mask: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
        n_objects = int(len(objects_df))
        positive_pixels = int((binary_mask > 0).sum())
        total_pixels = int(binary_mask.size)
        summary = {
            "site_id": site_id,
            "n_objects": n_objects,
            "positive_pixels": positive_pixels,
            "total_pixels": total_pixels,
            "cover_fraction": float(positive_pixels / total_pixels) if total_pixels else 0.0,
            "count_density_per_10k_px": float(n_objects / max(total_pixels, 1) * 10000.0),
            "mean_object_area_px": float(objects_df["pixel_area"].mean()) if not objects_df.empty else 0.0,
            "median_object_area_px": float(objects_df["pixel_area"].median()) if not objects_df.empty else 0.0,
            "mean_object_probability": float(objects_df["mean_probability"].mean()) if not objects_df.empty else 0.0,
            "mean_probability_over_mask": float(probability[binary_mask > 0].mean()) if positive_pixels else 0.0,
        }
        return summary

    def run_work_unit(self, unit: dict, *, trial_id: str, state=None):
        stage_name = unit["stage_name"]
        site_id = unit.get("site_id")
        cfg_sig = self.config_signature()

        bundle = self.ops.load_site_prediction_bundle(
            site_id=site_id,
            artifact_store=self.artifact_store,
            postprocessing_config=self.pipeline_config,
            cfg=self.cfg,
        )
        feature_bundle = None
        if self.ops.load_site_feature_bundle is not None:
            feature_bundle = self.ops.load_site_feature_bundle(
                site_id=site_id,
                artifact_store=self.artifact_store,
                postprocessing_config=self.pipeline_config,
                cfg=self.cfg,
            )

        probability = np.asarray(bundle["probability"], dtype=np.float32)
        profile = dict(bundle["profile"])

        threshold = self._resolve_threshold(bundle)
        binary = (probability >= threshold).astype(np.uint8)
        cleaned = self._cleanup_mask(binary)
        labels = self._split_instances(cleaned)
        objects_df = self._extract_object_table(labels, probability, profile)

        if stage_name in {"object_enrich", "object_filter", "site_summary"} and self.pipeline_config.enable_object_enrichment:
            if self.ops.enrich_object_table is not None:
                objects_df = self.ops.enrich_object_table(
                    site_id=site_id,
                    objects_df=objects_df,
                    feature_bundle=feature_bundle,
                    probability=probability,
                    binary_mask=cleaned,
                    labels=labels,
                    postprocessing_config=self.pipeline_config,
                    cfg=self.cfg,
                )

        if stage_name in {"object_filter", "site_summary"} and self.pipeline_config.enable_object_filtering:
            if self.ops.filter_object_table is not None:
                objects_df = self.ops.filter_object_table(
                    site_id=site_id,
                    objects_df=objects_df,
                    probability=probability,
                    binary_mask=cleaned,
                    labels=labels,
                    postprocessing_config=self.pipeline_config,
                    cfg=self.cfg,
                )
            else:
                keep = np.ones(len(objects_df), dtype=bool)
                if not objects_df.empty and "pixel_area" in objects_df.columns:
                    keep &= objects_df["pixel_area"].between(self.pipeline_config.object_filter_min_area, self.pipeline_config.object_filter_max_area).to_numpy()
                if not objects_df.empty and "mean_probability" in objects_df.columns:
                    keep &= (objects_df["mean_probability"] >= self.pipeline_config.object_filter_min_confidence).to_numpy()
                objects_df = objects_df.loc[keep].reset_index(drop=True)

        summary = self._site_summary(site_id, objects_df, cleaned, probability)

        binary_path, binary_rel, binary_ref = self._write_raster_like(cleaned.astype(np.uint8), profile, "binary_raster", config_signature=cfg_sig, site=site_id)
        object_id_path, object_rel, object_ref = self._write_raster_like(labels.astype(np.int32), profile, "object_id_raster", config_signature=cfg_sig, site=site_id)
        obj_path, obj_rel, obj_ref = self._persist_df(objects_df, "predicted_objects", config_signature=cfg_sig, site=site_id)
        summary_path, summary_rel, summary_ref = self._persist_json(summary, "site_summary", config_signature=cfg_sig, site=site_id)

        self._mark_shared_available(
            artifact_family=f"postprocessing.{stage_name}",
            shared_signature=self._shared_sig(stage_name, site_id),
            rel_path=summary_rel if stage_name == "site_summary" else obj_rel,
            local_path=str(summary_path if stage_name == "site_summary" else obj_path),
            remote_ref=summary_ref if stage_name == "site_summary" else obj_ref,
            metadata={"site": site_id, "stage": stage_name, "n_objects": int(len(objects_df))},
            trial_id=trial_id,
        )
        return {"stage_name": stage_name, "site_id": site_id, "n_objects": int(len(objects_df))}

    def run(self, *, trial_id: str = "adhoc_postprocessing", state=None, **kwargs) -> PipelineRunResult:
        runtime_report = self.runtime_report()
        executed_units: list[str] = []
        loop_guard = 0
        while True:
            loop_guard += 1
            if loop_guard > 50000:
                raise RuntimeError("PostprocessingPipeline debug run exceeded loop guard.")
            units = self.enumerate_work_units(trial_id=trial_id, config_signature=self.config_signature(), runtime_report=runtime_report, register_shared_requirements=False)
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

        final_units = self.enumerate_work_units(trial_id=trial_id, config_signature=self.config_signature(), runtime_report=runtime_report, register_shared_requirements=False)
        success = all(u["status"] == WorkUnitStatus.COMPLETE.value for u in final_units)
        status = "success" if success else "partial"

        cfg_sig = self.config_signature()
        obj_frames = []
        summary_rows = []
        raster_rows = []
        for site in self.cfg.sites:
            obj_path = self._local_artifact_path(self._render_rel_path("predicted_objects", config_signature=cfg_sig, site=site))
            sum_path = self._local_artifact_path(self._render_rel_path("site_summary", config_signature=cfg_sig, site=site))
            bin_path = self._local_artifact_path(self._render_rel_path("binary_raster", config_signature=cfg_sig, site=site))
            oid_path = self._local_artifact_path(self._render_rel_path("object_id_raster", config_signature=cfg_sig, site=site))
            if obj_path.exists():
                try:
                    df = pd.read_csv(obj_path)
                    if not df.empty:
                        df["site_id"] = site
                        obj_frames.append(df)
                except Exception:
                    pass
            if sum_path.exists():
                try:
                    summary_rows.append(json.loads(sum_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            if bin_path.exists() or oid_path.exists():
                raster_rows.append({"site_id": site, "binary_path": str(bin_path), "object_id_path": str(oid_path)})

        objects_df = pd.concat(obj_frames, ignore_index=True) if obj_frames else pd.DataFrame()
        summaries_df = pd.DataFrame(summary_rows)
        rasters_df = pd.DataFrame(raster_rows)

        result = PipelineRunResult(
            pipeline_name=self.pipeline_name,
            success=success,
            status=status,
            raster_outputs=CanonicalRasterOutputs(predictions=rasters_df if not rasters_df.empty else None),
            object_outputs=CanonicalObjectOutputs(predicted_objects=objects_df if not objects_df.empty else None, quality_flags=objects_df[[c for c in ["object_id", "mean_probability", "pixel_area"] if c in objects_df.columns]].copy() if not objects_df.empty else None),
            qa_outputs={"executed_units": executed_units, "n_total_units": len(final_units), "runtime_report": asdict(runtime_report)},
            metrics={
                "n_total_units": len(final_units),
                "n_complete_units": sum(u["status"] == WorkUnitStatus.COMPLETE.value for u in final_units),
                "n_sites_with_objects": int(objects_df["site_id"].nunique()) if "site_id" in objects_df.columns and not objects_df.empty else 0,
                "n_predicted_objects": int(len(objects_df)),
                "mean_cover_fraction": float(summaries_df["cover_fraction"].mean()) if "cover_fraction" in summaries_df.columns and not summaries_df.empty else 0.0,
            },
            notes=[
                "Postprocessing converts shrub probability maps into binary masks, instance labels, predicted shrub objects, and site summaries.",
                "Default implementation performs thresholding, cleanup, watershed/CC instance splitting, object extraction, optional enrichment/filtering, and summary export.",
            ],
        )
        self.save_run_result(result, subdir=cfg_sig)
        self._persist_json(asdict(result), "run_result", config_signature=cfg_sig)
        return result


def build_postprocessing_pipeline_ops() -> PostprocessingPipelineOps:
    def _not_wired(*args, **kwargs):
        raise NotImplementedError("PostprocessingPipelineOps.load_site_prediction_bundle must be wired to your modeling outputs.")
    return PostprocessingPipelineOps(load_site_prediction_bundle=_not_wired)
