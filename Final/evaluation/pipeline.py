from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable
import json
import hashlib

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi
from skimage import measure
from skimage.transform import resize

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
class EvaluationPipelineConfig:
    labeling_config_signature: str = ""
    postprocessing_config_signature: str = ""

    object_iou_match_threshold: float = 0.2
    centroid_distance_gate_px: float | None = None
    multiresolutions_m: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    raster_weight: float = 0.35
    object_weight: float = 0.30
    multires_weight: float = 0.15
    site_summary_weight: float = 0.20


@dataclass
class EvaluationPipelineOps:
    load_label_bundle: Callable[..., dict[str, Any]]
    load_prediction_bundle: Callable[..., dict[str, Any]]
    load_site_summary_bundle: Callable[..., dict[str, Any]] | None = None
    clear_artifact_staging_dir: Callable[[], None] | None = None


class EvaluationPipeline(BasePipeline):
    def __init__(self, cfg, *, ops: EvaluationPipelineOps, pipeline_config: EvaluationPipelineConfig | None = None):
        super().__init__(cfg, pipeline_name="evaluation", output_root=(getattr(cfg.output, "evaluation_root", cfg.output.root / "evaluation") / "pipeline_runs"))
        self.logger = get_logger("evaluation.pipeline")
        self.ops = ops
        self.pipeline_config = pipeline_config or EvaluationPipelineConfig()
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
        runtime_cfg = getattr(self.cfg, "evaluation_runtime", None)
        return tuple(getattr(runtime_cfg, name, default))

    def _storage_cfg(self):
        runtime_cfg = getattr(self.cfg, "evaluation_runtime", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "storage"):
            return runtime_cfg.storage
        return type("Storage", (), {
            "enable_local_store": True,
            "enable_drive_store": False,
            "use_hybrid_store": False,
            "fail_if_drive_missing": False,
        })()

    def _storage_policy_cfg(self):
        runtime_cfg = getattr(self.cfg, "evaluation_runtime", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "storage_policy"):
            return runtime_cfg.storage_policy
        return type("StoragePolicy", (), {
            "push_large_artifacts_to_remote": False,
            "prune_local_after_remote_push": False,
            "verify_remote_before_prune": True,
        })()

    def artifact_specs(self) -> dict[str, ArtifactSpec]:
        return {
            "site_metrics": ArtifactSpec(
                key="site_metrics",
                rel_path_template="evaluation/{config_signature}/{site}/metrics.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "composite_scores": ArtifactSpec(
                key="composite_scores",
                rel_path_template="evaluation/{config_signature}/composite_scores.csv",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "run_result": ArtifactSpec(
                key="run_result",
                rel_path_template="evaluation/{config_signature}/evaluation_run_result.json",
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
        return getattr(self.cfg.output, "evaluation_root", self.cfg.output.root / "evaluation") / "_remote_stage" / rel_path

    def _persist_json(self, payload: dict, artifact_key: str, **fmt) -> tuple[Path, str, str | None]:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return local_path, rel_path, None

    def _persist_df(self, df: pd.DataFrame, artifact_key: str, **fmt) -> tuple[Path, str, str | None]:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
        return local_path, rel_path, None

    def validate_hydrated_artifact(self, *, rel_path: str, local_path: Path, artifact_key: str | None = None) -> bool:
        try:
            if artifact_key in {"site_metrics", "run_result"}:
                json.loads(local_path.read_text(encoding="utf-8"))
                return True
            if artifact_key == "composite_scores":
                pd.read_csv(local_path, nrows=5)
                return True
            return super().validate_hydrated_artifact(rel_path=rel_path, local_path=local_path, artifact_key=artifact_key)
        except Exception as e:
            self.logger.warning("HYDRATE VALIDATE FAIL | rel_path=%s | artifact_key=%s | error=%s", rel_path, artifact_key, e)
            return False

    def build_pipeline_spec(self) -> PipelineSpec:
        modules = {
            "evaluation.alignment_check.base": ModuleSpec(key="evaluation.alignment_check.base", stage_name="alignment_check", runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_alignment_check", ("runtime:python",)), mode=RuntimeRequirementMode.ALL)),
            "evaluation.raster_metrics.base": ModuleSpec(key="evaluation.raster_metrics.base", stage_name="raster_metrics", runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_raster_metrics", ("runtime:python",)), mode=RuntimeRequirementMode.ALL)),
            "evaluation.objectization.base": ModuleSpec(key="evaluation.objectization.base", stage_name="objectization", runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_objectization", ("runtime:python",)), mode=RuntimeRequirementMode.ALL)),
            "evaluation.object_match.base": ModuleSpec(key="evaluation.object_match.base", stage_name="object_match", param_keys=("object_iou_match_threshold", "centroid_distance_gate_px"), runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_object_match", ("runtime:python",)), mode=RuntimeRequirementMode.ALL)),
            "evaluation.multires.base": ModuleSpec(key="evaluation.multires.base", stage_name="multires_metrics", param_keys=("multiresolutions_m",), runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_multires", ("runtime:python",)), mode=RuntimeRequirementMode.ALL)),
            "evaluation.site_summary.base": ModuleSpec(key="evaluation.site_summary.base", stage_name="site_summary_metrics", runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_site_summary", ("runtime:python",)), mode=RuntimeRequirementMode.ALL)),
            "evaluation.composite.base": ModuleSpec(key="evaluation.composite.base", stage_name="composite_score", param_keys=("raster_weight", "object_weight", "multires_weight", "site_summary_weight"), runtime_requirement=RuntimeRequirement(required_capabilities=self._runtime_caps("require_capability_composite", ("runtime:python",)), mode=RuntimeRequirementMode.ALL)),
        }
        stages = [
            StageSpec(name="alignment_check", module_keys=["evaluation.alignment_check.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="raster_metrics", module_keys=["evaluation.raster_metrics.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="objectization", module_keys=["evaluation.objectization.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="object_match", module_keys=["evaluation.object_match.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="multires_metrics", module_keys=["evaluation.multires.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="site_summary_metrics", module_keys=["evaluation.site_summary.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
            StageSpec(name="composite_score", module_keys=["evaluation.composite.base"], cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN)),
        ]
        return PipelineSpec(pipeline_name="evaluation", domain=PipelineDomain.SHARED, stages=stages, modules=modules, search_axes=[])

    def _sig(self, site: str) -> str:
        return hash_payload({"site": site, "config": asdict(self.pipeline_config)})

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
        units: list[dict[str, Any]] = []
        for site in self.cfg.sites:
            for idx, stage_name in enumerate(stage_order):
                deps = [] if idx == 0 else [stage_order[idx - 1]]
                dep_reasons = [] if idx == 0 else [f"Previous stage {stage_order[idx - 1]} must complete first."]
                units.append({
                    "unit_id": f"{trial_id}:{self.pipeline_name}:{stage_name}:{site}",
                    "trial_id": trial_id,
                    "pipeline_name": self.pipeline_name,
                    "config_signature": config_signature,
                    "stage_name": stage_name,
                    "work_key": site,
                    "scope": WorkUnitScope.SITE.value,
                    "status": WorkUnitStatus.PENDING.value if (stage_ok[stage_name] and idx == 0) else (WorkUnitStatus.BLOCKED.value if stage_ok[stage_name] else WorkUnitStatus.INELIGIBLE.value),
                    "dependencies": deps,
                    "dependency_reasons": dep_reasons,
                    "runtime_required_capabilities": list(self._runtime_caps(f"require_capability_{stage_name}", ("runtime:python",))),
                    "runtime_eligible": stage_ok[stage_name],
                    "priority": 100 * (idx + 1),
                    "site_id": site,
                })
        self._enumeration_cache[enum_cache_key] = [dict(u) for u in units]
        return units

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        inter = float(np.logical_and(y_true > 0, y_pred > 0).sum())
        denom = float((y_true > 0).sum() + (y_pred > 0).sum())
        return (2.0 * inter / denom) if denom else 1.0

    @staticmethod
    def _iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        inter = float(np.logical_and(y_true > 0, y_pred > 0).sum())
        union = float(np.logical_or(y_true > 0, y_pred > 0).sum())
        return (inter / union) if union else 1.0

    @staticmethod
    def _precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        tp = float(np.logical_and(y_true > 0, y_pred > 0).sum())
        fp = float(np.logical_and(y_true == 0, y_pred > 0).sum())
        fn = float(np.logical_and(y_true > 0, y_pred == 0).sum())
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        return precision, recall

    @staticmethod
    def _label_objects(mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
        labels, _ = ndi.label(mask > 0)
        rows = []
        for region in measure.regionprops(labels):
            rows.append({
                "object_id": int(region.label),
                "centroid_row": float(region.centroid[0]),
                "centroid_col": float(region.centroid[1]),
                "area": int(region.area),
                "bbox_min_row": int(region.bbox[0]),
                "bbox_min_col": int(region.bbox[1]),
                "bbox_max_row": int(region.bbox[2]),
                "bbox_max_col": int(region.bbox[3]),
            })
        return labels.astype(np.int32), pd.DataFrame(rows)

    def _object_matches(self, label_labels: np.ndarray, pred_labels: np.ndarray, label_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        matches: list[dict[str, Any]] = []
        used_preds: set[int] = set()
        used_labels: set[int] = set()

        label_ids = sorted(int(v) for v in np.unique(label_labels) if v != 0)
        pred_ids = sorted(int(v) for v in np.unique(pred_labels) if v != 0)

        for lid in label_ids:
            lmask = (label_labels == lid)
            best_pid = None
            best_iou = -1.0
            for pid in pred_ids:
                if pid in used_preds:
                    continue
                pmask = (pred_labels == pid)
                iou = self._iou(lmask.astype(np.uint8), pmask.astype(np.uint8))
                if iou > best_iou:
                    best_iou = iou
                    best_pid = pid
            if best_pid is None or best_iou < self.pipeline_config.object_iou_match_threshold:
                continue
            used_labels.add(lid)
            used_preds.add(best_pid)
            lrow = label_df.loc[label_df["object_id"] == lid].iloc[0] if not label_df.empty else None
            prow = pred_df.loc[pred_df["object_id"] == best_pid].iloc[0] if not pred_df.empty else None
            centroid_error = None
            area_error = None
            if lrow is not None and prow is not None:
                centroid_error = float(np.sqrt((lrow["centroid_row"] - prow["centroid_row"]) ** 2 + (lrow["centroid_col"] - prow["centroid_col"]) ** 2))
                area_error = float(abs(lrow["area"] - prow["area"]))
            matches.append({
                "label_object_id": lid,
                "pred_object_id": best_pid,
                "iou": float(best_iou),
                "centroid_error_px": centroid_error,
                "area_error_px": area_error,
            })

        tp = len(matches)
        fp = max(0, len(pred_ids) - tp)
        fn = max(0, len(label_ids) - tp)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics = {
            "object_precision": float(precision),
            "object_recall": float(recall),
            "object_f1": float(f1),
            "count_difference": float(len(label_ids) - len(pred_ids)),
            "mean_centroid_error_px": float(np.mean([m["centroid_error_px"] for m in matches if m["centroid_error_px"] is not None])) if matches else 0.0,
            "mean_area_error_px": float(np.mean([m["area_error_px"] for m in matches if m["area_error_px"] is not None])) if matches else 0.0,
        }
        return pd.DataFrame(matches), metrics

    def _downsample_mask(self, mask: np.ndarray, scale_factor: float) -> np.ndarray:
        if scale_factor <= 1.0:
            return mask.astype(np.uint8)
        out_h = max(1, int(round(mask.shape[0] / scale_factor)))
        out_w = max(1, int(round(mask.shape[1] / scale_factor)))
        resized = resize(mask.astype(float), (out_h, out_w), order=0, preserve_range=True, anti_aliasing=False)
        return (resized >= 0.5).astype(np.uint8)

    def _composite(self, raster_metrics: dict[str, float], object_metrics: dict[str, float], multires_metrics: dict[str, float], site_summary_metrics: dict[str, float]) -> dict[str, float]:
        raster_score = np.mean([raster_metrics.get("dice", 0.0), raster_metrics.get("iou", 0.0)])
        object_score = np.mean([
            object_metrics.get("object_f1", 0.0),
            max(0.0, 1.0 - abs(object_metrics.get("count_difference", 0.0)) / max(1.0, object_metrics.get("count_difference_scale", 1.0))),
        ])
        multires_score = multires_metrics.get("mean_multires_iou", 0.0)
        site_score = site_summary_metrics.get("site_summary_score", 0.0)
        composite = (
            self.pipeline_config.raster_weight * raster_score
            + self.pipeline_config.object_weight * object_score
            + self.pipeline_config.multires_weight * multires_score
            + self.pipeline_config.site_summary_weight * site_score
        )
        return {
            "raster_score": float(raster_score),
            "object_score": float(object_score),
            "multires_score": float(multires_score),
            "site_summary_score": float(site_score),
            "composite_score": float(composite),
        }

    def run_work_unit(self, unit: dict, *, trial_id: str, state=None):
        # The per-stage work units are used mainly for scheduler visibility. We compute
        # the full site evaluation bundle in one pass and then let later stages consume
        # the already-written metrics artifact in repeated refresh cycles.
        stage_name = unit["stage_name"]
        site_id = unit.get("site_id")
        cfg_sig = self.config_signature()

        label_bundle = self.ops.load_label_bundle(site_id=site_id, artifact_store=self.artifact_store, evaluation_config=self.pipeline_config, cfg=self.cfg)
        pred_bundle = self.ops.load_prediction_bundle(site_id=site_id, artifact_store=self.artifact_store, evaluation_config=self.pipeline_config, cfg=self.cfg)
        summary_bundle = None
        if self.ops.load_site_summary_bundle is not None:
            summary_bundle = self.ops.load_site_summary_bundle(site_id=site_id, artifact_store=self.artifact_store, evaluation_config=self.pipeline_config, cfg=self.cfg)

        label_mask = np.asarray(label_bundle["mask"], dtype=np.uint8)
        pred_mask = np.asarray(pred_bundle["mask"], dtype=np.uint8)

        raster_metrics = {}
        object_metrics = {}
        multires_metrics = {}
        site_summary_metrics = {"site_summary_score": 0.0}

        aligned = label_mask.shape == pred_mask.shape
        if not aligned:
            raise ValueError(f"Alignment check failed for site={site_id}: label shape {label_mask.shape} != pred shape {pred_mask.shape}")

        precision, recall = self._precision_recall(label_mask, pred_mask)
        raster_metrics.update({
            "aligned": True,
            "dice": self._dice(label_mask, pred_mask),
            "iou": self._iou(label_mask, pred_mask),
            "precision": precision,
            "recall": recall,
        })

        label_labels, label_df = self._label_objects(label_mask)
        pred_labels, pred_df = self._label_objects(pred_mask)
        _, object_metrics = self._object_matches(label_labels, pred_labels, label_df, pred_df)
        object_metrics["count_difference_scale"] = float(max(len(label_df), 1))

        multires_rows = []
        for res in self.pipeline_config.multiresolutions_m:
            if res <= 1.0:
                continue
            l_low = self._downsample_mask(label_mask, res)
            p_low = self._downsample_mask(pred_mask, res)
            multires_rows.append({"resolution_m": float(res), "iou": self._iou(l_low, p_low), "dice": self._dice(l_low, p_low)})
        multires_df = pd.DataFrame(multires_rows)
        multires_metrics = {
            "mean_multires_iou": float(multires_df["iou"].mean()) if not multires_df.empty else raster_metrics["iou"],
            "mean_multires_dice": float(multires_df["dice"].mean()) if not multires_df.empty else raster_metrics["dice"],
        }

        if summary_bundle is not None:
            label_summary = summary_bundle.get("label_summary", {})
            pred_summary = summary_bundle.get("pred_summary", {})
            diffs = []
            for key in ["cover_fraction", "count_density_per_10k_px", "mean_object_area_px", "median_object_area_px"]:
                if key in label_summary and key in pred_summary:
                    scale = max(abs(float(label_summary[key])), 1.0)
                    diffs.append(max(0.0, 1.0 - abs(float(label_summary[key]) - float(pred_summary[key])) / scale))
            site_summary_metrics = {"site_summary_score": float(np.mean(diffs)) if diffs else 0.0}

        composite = self._composite(raster_metrics, object_metrics, multires_metrics, site_summary_metrics)
        metrics_payload = {
            "site_id": site_id,
            "raster_metrics": raster_metrics,
            "object_metrics": object_metrics,
            "multires_metrics": multires_metrics,
            "site_summary_metrics": site_summary_metrics,
            "composite": composite,
        }
        self._persist_json(metrics_payload, "site_metrics", config_signature=cfg_sig, site=site_id)
        return {"stage_name": stage_name, "site_id": site_id, "composite_score": composite["composite_score"]}

    def run(self, *, trial_id: str = "adhoc_evaluation", state=None, **kwargs) -> PipelineRunResult:
        runtime_report = self.runtime_report()
        executed_units: list[str] = []
        # For evaluation, we intentionally execute one unit per site/stage order using the pipeline
        # scheduler contract, but every site-stage call recomputes the site bundle deterministically.
        # That keeps the work-unit view informative while avoiding fragile partial metric artifacts.
        units = self.enumerate_work_units(trial_id=trial_id, config_signature=self.config_signature(), runtime_report=runtime_report, register_shared_requirements=False)
        for unit in sorted(units, key=lambda u: (u.get("site_id", ""), u.get("priority", 100))):
            if unit["runtime_eligible"]:
                self.run_work_unit(unit, trial_id=trial_id, state=state)
                executed_units.append(unit["unit_id"])

        cfg_sig = self.config_signature()
        rows = []
        for site in self.cfg.sites:
            path = self._local_artifact_path(self._render_rel_path("site_metrics", config_signature=cfg_sig, site=site))
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        site_df = pd.DataFrame([
            {
                "site_id": row["site_id"],
                "raster_score": row["composite"]["raster_score"],
                "object_score": row["composite"]["object_score"],
                "multires_score": row["composite"]["multires_score"],
                "site_summary_score": row["composite"]["site_summary_score"],
                "composite_score": row["composite"]["composite_score"],
            }
            for row in rows
        ])
        if not site_df.empty:
            self._persist_df(site_df, "composite_scores", config_signature=cfg_sig)

        result = PipelineRunResult(
            pipeline_name=self.pipeline_name,
            success=not site_df.empty,
            status="success" if not site_df.empty else "failed",
            raster_outputs=CanonicalRasterOutputs(labels=None, predictions=site_df if not site_df.empty else None),
            object_outputs=CanonicalObjectOutputs(objects=None, predicted_objects=None),
            qa_outputs={"executed_units": executed_units, "runtime_report": asdict(runtime_report)},
            metrics={
                "n_sites": int(len(site_df)),
                "mean_composite_score": float(site_df["composite_score"].mean()) if not site_df.empty else 0.0,
                "mean_raster_score": float(site_df["raster_score"].mean()) if not site_df.empty else 0.0,
                "mean_object_score": float(site_df["object_score"].mean()) if not site_df.empty else 0.0,
                "mean_multires_score": float(site_df["multires_score"].mean()) if not site_df.empty else 0.0,
                "mean_site_summary_score": float(site_df["site_summary_score"].mean()) if not site_df.empty else 0.0,
            },
            notes=[
                "Evaluation pipeline scores raster overlap, object quality, multi-resolution consistency, and site-summary fidelity, then combines them into a composite score.",
                "Current implementation uses per-site deterministic metric computation and outputs a site-level composite score table suitable for grid-search comparison.",
            ],
        )
        self.save_run_result(result, subdir=cfg_sig)
        self._persist_json(asdict(result), "run_result", config_signature=cfg_sig)
        return result


def build_evaluation_pipeline_ops() -> EvaluationPipelineOps:
    """Default repo-aware evaluation ops.

    Labels come from the labeling bridge. Predictions prefer postprocessing
    outputs when a postprocessing config signature is supplied, and otherwise
    fall back to modeling binary/probability outputs.
    """

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import rasterio

    def _storage_root(artifact_store, cfg) -> Path:
        if isinstance(artifact_store, LocalArtifactStore):
            return artifact_store.storage_root
        if isinstance(artifact_store, HybridArtifactStore):
            return artifact_store.local_store.storage_root
        return Path(cfg.artifact_store.local_storage_root)

    def _resolve_latest(storage_root: Path, pattern: str) -> Path | None:
        candidates = [p for p in storage_root.glob(pattern) if p.exists()]
        if not candidates:
            return None
        return sorted(candidates, key=lambda q: q.stat().st_mtime, reverse=True)[0]

    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding='utf-8'))

    def _read_raster(path_like: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        with rasterio.open(path_like) as src:
            arr = src.read(1)
            profile = src.profile.copy()
            profile['transform'] = src.transform
            profile['crs'] = src.crs
            profile['width'] = src.width
            profile['height'] = src.height
        return arr, profile

    def _summary_from_mask_and_objects(site_id: str, mask: np.ndarray, objects_df: pd.DataFrame | None) -> dict[str, Any]:
        objects_df = objects_df if objects_df is not None else pd.DataFrame()
        n_objects = int(len(objects_df)) if not objects_df.empty else int((mask > 0).sum() > 0)
        total_pixels = int(mask.size)
        positive_pixels = int((mask > 0).sum())
        area_col = 'pixel_area' if 'pixel_area' in objects_df.columns else None
        return {
            'site_id': site_id,
            'cover_fraction': float(positive_pixels / max(total_pixels, 1)),
            'count_density_per_10k_px': float(n_objects / max(total_pixels, 1) * 10000.0),
            'mean_object_area_px': float(objects_df[area_col].mean()) if (area_col and not objects_df.empty) else 0.0,
            'median_object_area_px': float(objects_df[area_col].median()) if (area_col and not objects_df.empty) else 0.0,
        }

    def load_label_bundle(*, site_id: str, artifact_store, evaluation_config, cfg) -> dict[str, Any]:
        from Final.features.labeling_bridge import best_label_artifact_for_site, label_objects_for_site
        row = best_label_artifact_for_site(site_id, resolution_m=1.0)
        if row is None:
            raise FileNotFoundError(f'No labeling artifact found for site={site_id}')
        mask_path = None
        for col in ['binary_mask_path', 'mask_path', 'label_path']:
            if col in row.index and pd.notna(row[col]):
                mask_path = Path(str(row[col]))
                break
        if mask_path is None or not mask_path.exists():
            raise FileNotFoundError(f'Label mask path missing or unreadable for site={site_id}: {mask_path}')
        mask, profile = _read_raster(mask_path)
        mask = (mask > 0).astype(np.uint8)
        objects_df = label_objects_for_site(site_id)
        return {'site_id': site_id, 'mask': mask, 'profile': profile, 'objects_df': objects_df}

    def load_prediction_bundle(*, site_id: str, artifact_store, evaluation_config, cfg) -> dict[str, Any]:
        storage_root = _storage_root(artifact_store, cfg)
        post_sig = (evaluation_config.postprocessing_config_signature or '').strip()
        if post_sig:
            bin_path = storage_root / f'postprocessing/{post_sig}/{site_id}/binary.tif'
            obj_path = storage_root / f'postprocessing/{post_sig}/{site_id}/predicted_objects.csv'
            if not bin_path.exists():
                bin_path = _resolve_latest(storage_root, f'postprocessing/*/{site_id}/binary.tif')
            if obj_path is None or not obj_path.exists():
                obj_path = _resolve_latest(storage_root, f'postprocessing/*/{site_id}/predicted_objects.csv')
        else:
            bin_path = _resolve_latest(storage_root, f'postprocessing/*/{site_id}/binary.tif')
            obj_path = _resolve_latest(storage_root, f'postprocessing/*/{site_id}/predicted_objects.csv')
        if bin_path is not None and bin_path.exists():
            mask, profile = _read_raster(bin_path)
            objects_df = pd.read_csv(obj_path) if obj_path is not None and obj_path.exists() else pd.DataFrame()
            return {'site_id': site_id, 'mask': (mask > 0).astype(np.uint8), 'profile': profile, 'objects_df': objects_df}

        # Fallback to modeling outputs.
        prob_path = _resolve_latest(storage_root, f'modeling/*/predictions/{site_id}/probability.tif')
        if prob_path is None or not prob_path.exists():
            raise FileNotFoundError(f'No postprocessing/modeling prediction bundle found for site={site_id}')
        probs, profile = _read_raster(prob_path)
        mask = (probs >= 0.5).astype(np.uint8)
        return {'site_id': site_id, 'mask': mask, 'profile': profile, 'objects_df': pd.DataFrame()}

    def load_site_summary_bundle(*, site_id: str, artifact_store, evaluation_config, cfg) -> dict[str, Any]:
        label_bundle = load_label_bundle(site_id=site_id, artifact_store=artifact_store, evaluation_config=evaluation_config, cfg=cfg)
        pred_bundle = load_prediction_bundle(site_id=site_id, artifact_store=artifact_store, evaluation_config=evaluation_config, cfg=cfg)
        storage_root = _storage_root(artifact_store, cfg)
        post_sig = (evaluation_config.postprocessing_config_signature or '').strip()
        pred_summary = None
        if post_sig:
            sp = storage_root / f'postprocessing/{post_sig}/{site_id}/site_summary.json'
            if sp.exists():
                pred_summary = _load_json(sp)
        if pred_summary is None:
            sp = _resolve_latest(storage_root, f'postprocessing/*/{site_id}/site_summary.json')
            if sp is not None and sp.exists():
                pred_summary = _load_json(sp)
        if pred_summary is None:
            pred_summary = _summary_from_mask_and_objects(site_id, pred_bundle['mask'], pred_bundle.get('objects_df'))
        label_summary = _summary_from_mask_and_objects(site_id, label_bundle['mask'], label_bundle.get('objects_df'))
        return {'label_summary': label_summary, 'pred_summary': pred_summary}

    return EvaluationPipelineOps(
        load_label_bundle=load_label_bundle,
        load_prediction_bundle=load_prediction_bundle,
        load_site_summary_bundle=load_site_summary_bundle,
    )

