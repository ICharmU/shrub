from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import tempfile
from time import perf_counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import pandas as pd
import rasterio
from rasterio.windows import Window

from Final.pipeline_base import BasePipeline
from Final.models import (
    ModuleCard,
    PipelineDomain,
    RepresentationTarget,
    SpatialScope,
    ResolutionScope,
    AvailabilityTier,
    RuntimeTier,
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
    ExecutionEligibilityStatus,
    WorkUnitScope, 
    WorkUnitStatus,
    SharedArtifactStatus
)
from Final.artifact_store import (
    LocalArtifactStore,
    DriveRegistryArtifactStore,
    HybridArtifactStore,
)
from Final.pipeline_caching import hash_payload, read_stage_cache_manifest, write_stage_cache_manifest
from Final.coordination import CoordinationManager
from Final.gating import (
    QACheckSpec,
    QACheckResult,
    ModuleQAProfile,
    ModuleQAEvaluation,
)
from Final.shared_artifact_registry import SharedArtifactRegistry
from Final.shared_utils import get_logger


from Final.labeling.sprint3_runner import (
    summarize_ptx_entries_by_site,
    select_ptx_entries,
    cleanup_stale_ptx_cache,
    download_ptx_with_cache,
    run_sprint3_for_ptx,
    append_results_manifest,
)
from Final.labeling.sprint3_standardize import standardize_sprint3_manifest
from Final.labeling.object_refinement import refine_shrub_objects
from Final.labeling.transforms import transform_objects_to_als, shrub_csv_to_transform_name
from Final.labeling.alignment import align_objects_to_naip
from Final.labeling.subspace_reduction import SubspaceReductionConfig
from Final.labeling.rasterize import (
    rasterize_objects,
    resample_single_band,
    write_single_band_geotiff,
)
from Final.labeling.dedup import deduplicate_artifact_table, extract_plot_key
from Final.labeling.qa import create_overlay_figure
from Final.labeling.export import export_table
from Final.labeling.io import download_file, extract_als_metadata
from Final.labeling.manifests import (
    list_files_with_suffix,
    site_to_remote_base,
    site_to_tif_name,
)

@dataclass
class LabelingPipelineConfig:
    sprint3_variant: str = "revised"
    use_shape_descriptors: bool = True
    use_temporal_confidence: bool = False
    use_boundary_confidence: bool = True
    use_transform_confidence: bool = False
    use_object_subspace_filter: bool = False
    rasterization_mode: str = "circle"
    multires: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    force_rerun_sprint4: bool = False
    force_refresh_site_assets: bool = False
    nonfatal_qa_overlay: bool = True

    # Sprint 3 execution / caching
    run_sprint3: bool = True
    sprint3_variants: tuple[str, ...] = ("original", "revised")
    max_ptx_per_site: int | None = 1
    force_rerun_sprint3: bool = False
    require_success_artifacts_sprint3: bool = True
    cleanup_ptx_after_all_variants: bool = True
    cleanup_stale_ptx_before_run: bool = True
    stale_ptx_days: int = 2

    boundary_confidence_mode: str = "radial"   # "radial" or "universal"
    site_reference_dates: dict[str, str] = field(default_factory=dict)

    subspace_min_component_pixels: int = 4
    subspace_min_object_confidence: float = 0.55
    subspace_min_transform_confidence: float = 0.50
    subspace_min_temporal_confidence: float = 0.40
    subspace_max_height_m: float = 3.5

    allow_adopt_global_outputs: bool = False


@dataclass
class LabelingModuleSpec:
    key: str
    stage: str
    description: str
    enabled: bool = True
    submodules: list[str] = field(default_factory=list)


class LabelingPipeline(BasePipeline):
    def __init__(self, cfg, pipeline_config: LabelingPipelineConfig | None = None):
        super().__init__(
            cfg,
            pipeline_name="labeling",
            output_root=cfg.output.labeling_root / "pipeline_runs",
        )
        self.logger = get_logger("labeling.pipeline")
        self.pipeline_config = pipeline_config or LabelingPipelineConfig()

        storage = self.cfg.labeling_runtime.storage
        global_store = self.cfg.artifact_store

        if global_store.local_storage_root is None:
            global_store.local_storage_root = cfg.output.labeling_root / "artifact_store_local"

        self.artifact_store = self._build_artifact_store()
        self._enumeration_cache = {}
        self.coordination = CoordinationManager(
            self.artifact_store,
            root_prefix=self.cfg.coordination.root_prefix,
        )
        self.shared_registry = SharedArtifactRegistry(
            self.artifact_store,
            root_prefix=self.cfg.shared_artifacts.registry_prefix,
        )

    def artifact_specs(self) -> dict[str, ArtifactSpec]:
        return {
            "site_metadata_manifest": ArtifactSpec(
                key="site_metadata_manifest",
                rel_path_template="labeling/{site}/{config_signature}/site_assets/site_metadata_manifest.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "source_inventory": ArtifactSpec(
                key="source_inventory",
                rel_path_template="labeling/{site}/{config_signature}/site_assets/source_inventory.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "als_metadata_json": ArtifactSpec(
                key="als_metadata_json",
                rel_path_template="labeling/{site}/{config_signature}/site_assets/als_metadata.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "transform_index": ArtifactSpec(
                key="transform_index",
                rel_path_template="labeling/{site}/{config_signature}/transforms/transform_index.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "transform_txt": ArtifactSpec(
                key="transform_txt",
                rel_path_template="labeling/{site}/{config_signature}/transforms/{plot_id}.txt",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "binary_mask": ArtifactSpec(
                key="binary_mask",
                rel_path_template="labeling/{site}/{config_signature}/{source_version}/{plot_id}/mask.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "confidence_mask": ArtifactSpec(
                key="confidence_mask",
                rel_path_template="labeling/{site}/{config_signature}/{source_version}/{plot_id}/confidence.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
             "site_naip_raster": ArtifactSpec(
                key="site_naip_raster",
                rel_path_template="labeling/{site}/shared/site_assets/naip/{filename}",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "site_als_metadata_json": ArtifactSpec(
                key="site_als_metadata_json",
                rel_path_template="labeling/{site}/shared/site_assets/als_metadata/als_metadata.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "object_id_raster": ArtifactSpec(
                key="object_id_raster",
                rel_path_template="labeling/{site}/{config_signature}/{source_version}/{plot_id}/object_id.tif",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "object_table": ArtifactSpec(
                key="object_table",
                rel_path_template="labeling/{site}/{config_signature}/{source_version}/{plot_id}/objects.csv",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "qa_overlay": ArtifactSpec(
                key="qa_overlay",
                rel_path_template="labeling/{site}/{config_signature}/{source_version}/{plot_id}/overlay.png",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=False,
                prune_local_after_push=True,
            ),
            "objects_summary": ArtifactSpec(
                key="objects_summary",
                rel_path_template="labeling/{config_signature}/summaries/objects_all.csv",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "artifacts_summary": ArtifactSpec(
                key="artifacts_summary",
                rel_path_template="labeling/{config_signature}/summaries/artifacts_all.csv",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "run_manifest": ArtifactSpec(
                key="run_manifest",
                rel_path_template="labeling/{config_signature}/run_manifest.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "run_result": ArtifactSpec(
                key="run_result",
                rel_path_template="labeling/{config_signature}/labeling_run_result.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
        }

    def build_pipeline_spec(self) -> PipelineSpec:
        modules = {
            "labeling.sprint3.execution": ModuleSpec(
                key="labeling.sprint3.execution",
                stage_name="sprint3",
                enabled_key=None,
                variant_key="sprint3_variants",
                param_keys=("max_ptx_per_site", "force_rerun_sprint3", "require_success_artifacts_sprint3"),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_sprint3,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.standardize.base": ModuleSpec(
                key="labeling.standardize.base",
                stage_name="standardize",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_standardize,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.refine.shape_descriptors": ModuleSpec(
                key="labeling.refine.shape_descriptors",
                stage_name="refine",
                enabled_key="use_shape_descriptors",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_refine,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.refine.temporal_confidence": ModuleSpec(
                key="labeling.refine.temporal_confidence",
                stage_name="refine",
                enabled_key="use_temporal_confidence",
                param_keys=("site_reference_dates",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_refine,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.refine.object_subspace_filter": ModuleSpec(
                key="labeling.refine.object_subspace_filter",
                stage_name="refine",
                enabled_key="use_object_subspace_filter",
                param_keys=(
                    "subspace_min_object_confidence",
                    "subspace_min_transform_confidence",
                    "subspace_min_temporal_confidence",
                    "subspace_max_height_m",
                ),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_refine,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.transfer.base": ModuleSpec(
                key="labeling.transfer.base",
                stage_name="transfer",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_transfer,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.rasterize.mode": ModuleSpec(
                key="labeling.rasterize.mode",
                stage_name="rasterize",
                variant_key="rasterization_mode",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_rasterize,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.boundary_confidence": ModuleSpec(
                key="labeling.boundary_confidence",
                stage_name="rasterize",
                enabled_key="use_boundary_confidence",
                variant_key="boundary_confidence_mode",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_rasterize,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.mask_subspace_reduction": ModuleSpec(
                key="labeling.mask_subspace_reduction",
                stage_name="rasterize",
                enabled_key="use_object_subspace_filter",
                param_keys=("subspace_min_component_pixels",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_rasterize,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "labeling.multires_export": ModuleSpec(
                key="labeling.multires_export",
                stage_name="rasterize",
                param_keys=("multires",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.labeling_runtime.require_capability_rasterize,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
        }

        stages = [
            StageSpec(
                name="sprint3",
                module_keys=["labeling.sprint3.execution"],
                cache_policy=CachePolicy(
                    require_manifest=False,   # sprint3 runner has its own scoped cache already
                    allow_legacy_reuse=False,
                    retention_mode=CacheRetentionMode.LEAN,
                ),
            ),
            StageSpec(
                name="standardize",
                module_keys=["labeling.standardize.base"],
                cache_policy=CachePolicy(
                    require_manifest=True,
                    allow_legacy_reuse=False,
                    retention_mode=CacheRetentionMode.LEAN,
                ),
            ),
            StageSpec(
                name="refine",
                module_keys=[
                    "labeling.refine.shape_descriptors",
                    "labeling.refine.temporal_confidence",
                    "labeling.refine.object_subspace_filter",
                ],
                cache_policy=CachePolicy(
                    require_manifest=True,
                    allow_legacy_reuse=False,
                    retention_mode=CacheRetentionMode.LEAN,
                ),
            ),
            StageSpec(
                name="transfer",
                module_keys=["labeling.transfer.base"],
                cache_policy=CachePolicy(
                    require_manifest=True,
                    allow_legacy_reuse=False,
                    retention_mode=CacheRetentionMode.LEAN,
                ),
            ),
            StageSpec(
                name="rasterize",
                module_keys=[
                    "labeling.rasterize.mode",
                    "labeling.boundary_confidence",
                    "labeling.mask_subspace_reduction",
                    "labeling.multires_export",
                ],
                cache_policy=CachePolicy(
                    require_manifest=True,
                    allow_legacy_reuse=False,
                    retention_mode=CacheRetentionMode.LEAN,
                    # later, if disk gets too tight, you can add artifact_keys_to_prune=("qa_path",)
                    prune_after_success=False,
                ),
            ),
        ]

        search_axes = [
            SearchAxis(
                key="sprint3_variants",
                values=[("revised",), ("original", "revised")],
                stage_name="sprint3",
                module_key="labeling.sprint3.execution",
            ),
            SearchAxis(
                key="use_temporal_confidence",
                values=[False, True],
                stage_name="refine",
                module_key="labeling.refine.temporal_confidence",
            ),
            SearchAxis(
                key="boundary_confidence_mode",
                values=["radial", "universal"],
                stage_name="rasterize",
                module_key="labeling.boundary_confidence",
            ),
            SearchAxis(
                key="use_object_subspace_filter",
                values=[False, True],
                stage_name="refine",
                module_key="labeling.refine.object_subspace_filter",
            ),
            SearchAxis(
                key="max_ptx_per_site",
                values=[1],
                stage_name="sprint3",
                module_key="labeling.sprint3.execution",
            ),
        ]

        return PipelineSpec(
            pipeline_name="labeling",
            domain=PipelineDomain.LABELING,
            stages=stages,
            modules=modules,
            search_axes=search_axes,
        )

    def _build_artifact_store(self):
        storage = self.cfg.labeling_runtime.storage
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
                ]
                if not Path(p).exists()
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
            self.logger.info("Using HybridArtifactStore for labeling pipeline")
            return HybridArtifactStore(local_store=local_store, remote_store=drive_store)
    
        if drive_store is not None and not storage.enable_local_store:
            self.logger.info("Using DriveRegistryArtifactStore only for labeling pipeline")
            return drive_store
    
        self.logger.info("Using LocalArtifactStore only for labeling pipeline")
        return local_store

    def _artifact_spec(self, key: str) -> ArtifactSpec:
        return self.artifact_specs()[key]

    def _render_rel_path(self, artifact_key: str, **kwargs) -> str:
        spec = self._artifact_spec(artifact_key)
        return spec.rel_path_template.format(**kwargs)

    def _local_artifact_path(self, rel_path: str) -> Path:
        if isinstance(self.artifact_store, LocalArtifactStore):
            return self.artifact_store.storage_root / rel_path
        if isinstance(self.artifact_store, HybridArtifactStore):
            return self.artifact_store.local_store.storage_root / rel_path
        return self.cfg.output.labeling_root / "_remote_stage" / rel_path

    def stage_is_eligible(self, stage_name: str, runtime_report=None) -> tuple[bool, dict]:
        runtime_report = runtime_report or self.runtime_report()
        elig = self.stage_runtime_eligibility(stage_name, runtime_report=runtime_report)
        all_ok = all(v.status == ExecutionEligibilityStatus.ELIGIBLE for v in elig.values())
        return all_ok, elig
    
    
    def stage_block_reason(self, stage_name: str, runtime_report=None) -> str:
        ok, elig = self.stage_is_eligible(stage_name, runtime_report=runtime_report)
        if ok:
            return ""
        msgs = []
        for module_key, info in elig.items():
            if info.status != ExecutionEligibilityStatus.ELIGIBLE:
                msgs.append(f"{module_key}: missing={info.missing_capabilities}")
        return "; ".join(msgs)

    def _remote_exists(self, rel_path: str) -> bool:
        try:
            return self.artifact_store.exists(rel_path)
        except Exception:
            return False

    def _push_if_needed(self, local_path: Path, artifact_key: str, rel_path: str) -> str | None:
        spec = self._artifact_spec(artifact_key)
        storage = self.cfg.labeling_runtime.storage
        policy = self.cfg.labeling_runtime.storage_policy
    
        should_push = (
            policy.push_large_artifacts_to_remote
            and spec.storage_tier in {StorageTier.LOCAL_THEN_REMOTE, StorageTier.REMOTE_ONLY}
            and isinstance(self.artifact_store, (HybridArtifactStore, DriveRegistryArtifactStore))
        )
        if not should_push:
            return None
    
        self.logger.info("PUSH ARTIFACT | key=%s | rel_path=%s", artifact_key, rel_path)
        return self.artifact_store.push(local_path, rel_path=rel_path)

    def _prune_if_allowed(self, local_path: Path, artifact_key: str, rel_path: str) -> None:
        spec = self._artifact_spec(artifact_key)
        policy = self.cfg.labeling_runtime.storage_policy
    
        if not policy.prune_local_after_remote_push:
            return
        if not spec.prune_local_after_push:
            return
        if policy.verify_remote_before_prune and not self._remote_exists(rel_path):
            return
        if local_path.exists():
            local_path.unlink()
            self.logger.info("PRUNED LOCAL ARTIFACT | rel_path=%s", rel_path)

    def _persist_file_artifact(self, local_path: Path, artifact_key: str, **fmt) -> tuple[str, str | None]:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        remote_ref = self._push_if_needed(local_path, artifact_key, rel_path)
        self._prune_if_allowed(local_path, artifact_key, rel_path)
        return rel_path, remote_ref

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
            if artifact_key in {"binary_mask", "confidence_mask", "object_id_raster", "site_naip_raster"}:
                with rasterio.open(local_path) as src:
                    src.read(1, window=Window(0, 0, min(16, src.width), min(16, src.height)))
                return True
    
            if artifact_key in {
                "objects_summary",
                "artifacts_summary",
                "object_table",
            }:
                pd.read_csv(local_path, nrows=5)
                return True
    
            if artifact_key in {
                "run_manifest",
                "site_metadata_manifest",
                "source_inventory",
                "als_metadata_json",
                "site_als_metadata_json",
                "transform_index",
            }:
                json.loads(local_path.read_text(encoding="utf-8"))
                return True
    
            if artifact_key == "qa_overlay":
                return local_path.exists() and local_path.stat().st_size > 0
    
            if artifact_key == "transform_txt":
                return local_path.exists() and local_path.stat().st_size > 0
    
            return super().validate_hydrated_artifact(
                rel_path=rel_path,
                local_path=local_path,
                artifact_key=artifact_key,
            )
        except Exception as e:
            self.logger.warning(
                "HYDRATE VALIDATE FAIL | rel_path=%s | artifact_key=%s | error=%s",
                rel_path, artifact_key, e
            )
            return False

    def _try_load_json_artifact(self, artifact_key: str, **fmt) -> dict | None:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)

        if local_path.exists():
            try:
                if self.validate_hydrated_artifact(
                    rel_path=rel_path,
                    local_path=local_path,
                    artifact_key=artifact_key,
                ):
                    self.logger.info(
                        "JSON ARTIFACT HIT | key=%s | rel_path=%s | source=local",
                        artifact_key, rel_path
                    )
                    return json.loads(local_path.read_text(encoding="utf-8"))
                else:
                    self.logger.warning(
                        "JSON ARTIFACT INVALID | key=%s | rel_path=%s | source=local | deleting local copy",
                        artifact_key, rel_path
                    )
                    local_path.unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(
                    "JSON ARTIFACT LOCAL READ FAIL | key=%s | rel_path=%s | error=%s",
                    artifact_key, rel_path, e
                )
                local_path.unlink(missing_ok=True)

        if self._remote_exists(rel_path):
            pulled = self.hydrate_and_validate_artifact(
                rel_path=rel_path,
                local_path=local_path,
                artifact_key=artifact_key,
                reason=f"load_json_artifact:{artifact_key}",
            )
            if pulled is not None:
                self.logger.info(
                    "JSON ARTIFACT HIT | key=%s | rel_path=%s | source=remote",
                    artifact_key, rel_path
                )
                return json.loads(Path(pulled).read_text(encoding="utf-8"))

        self.logger.info("JSON ARTIFACT MISS | key=%s | rel_path=%s", artifact_key, rel_path)
        return None

    def standardize_stage_data_signature(self, runs_df: pd.DataFrame) -> str:
        payload = {
            "rows": runs_df[
                [c for c in ["site_id", "variant", "ptx_name", "input_ptx", "shrub_csv", "returncode"] if c in runs_df.columns]
            ].fillna("").astype(str).to_dict(orient="records")
        }
        return hash_payload(payload)

    def refine_stage_data_signature(self, objects_df: pd.DataFrame) -> str:
        payload = {
            "rows": objects_df[
                [c for c in ["site_id", "plot_id", "object_id", "source_version", "variant", "ptx_date_token"] if c in objects_df.columns]
            ].fillna("").astype(str).to_dict(orient="records")
        }
        return hash_payload(payload)

    def transfer_stage_data_signature(
        self,
        *,
        site: str,
        plot_id: str,
        source_version: str,
        objects_group: pd.DataFrame,
    ) -> str:
        payload = {
            "site": site,
            "plot_id": plot_id,
            "source_version": source_version,
            "rows": objects_group[
                [c for c in ["object_id", "x_tls", "y_tls", "radius_m", "object_confidence", "temporal_confidence", "transform_confidence", "valid_object"] if c in objects_group.columns]
            ].fillna("").astype(str).to_dict(orient="records"),
        }
        return hash_payload(payload)

    def _mark_site_assets_shared_valid(
        self,
        *,
        site: str,
        naip_local: Path,
        als_meta: list[dict],
        trial_id: str | None = None,
    ) -> None:
        if not self.cfg.shared_artifacts.enable_shared_artifact_registry:
            return
    
        shared_sig = self.shared_signature_site_assets(site)
        self.shared_registry.mark_available(
            artifact_family="labeling.site_assets",
            shared_signature=shared_sig,
            producer_pipeline=self.pipeline_name,
            source_trial=trial_id or getattr(self, "_active_trial_id", None),
            metadata={
                "site": site,
                "naip_local": str(naip_local),
                "als_metadata_rows": len(als_meta or []),
            },
            status=SharedArtifactStatus.VALID,
        )

    # -------------------------------------------------------------------------
    # Paths / caches
    # -------------------------------------------------------------------------

    @property
    def summary_dir(self) -> Path:
        return self.cfg.output.labeling_root / "summaries"

    @property
    def sprint3_manifest_csv(self) -> Path:
        return self.cfg.output.labeling_root / "manifests" / self.cfg.labeling.sprint3_manifest_name

    @property
    def sprint4_manifest_csv(self) -> Path:
        return self.cfg.output.labeling_root / "manifests" / "sprint4_artifacts.csv"

    @property
    def site_cache_root(self) -> Path:
        root = self.cfg.output.labeling_root / "site_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_cache_root(self, site: str) -> Path:
        root = self.site_cache_root / site
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_transform_cache_root(self, site: str) -> Path:
        root = self._site_cache_root(site) / "transforms"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_transform_index_path(self, site: str) -> Path:
        return self._site_transform_cache_root(site) / "transform_index.json"

    def _site_naip_cache_root(self, site: str) -> Path:
        root = self._site_cache_root(site) / "naip"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_als_cache_root(self, site: str) -> Path:
        root = self._site_cache_root(site) / "als_metadata"
        root.mkdir(parents=True, exist_ok=True)
        return root
    
    @property
    def ptx_cache_root(self) -> Path:
        root = self.cfg.output.labeling_root / "ptx_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def sprint3_output_root(self) -> Path:
        return self.cfg.output.labeling_root

    def stage_cache_root(self, stage_name: str) -> Path:
        root = self.cfg.output.labeling_root / "stage_cache" / stage_name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def standardize_stage_cache_dir(self, data_signature: str, config_signature: str) -> Path:
        d = self.stage_cache_root("standardize") / f"{data_signature}__{config_signature}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def refine_stage_cache_dir(self, data_signature: str, config_signature: str) -> Path:
        d = self.stage_cache_root("refine") / f"{data_signature}__{config_signature}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def transfer_stage_cache_dir(self, site: str, plot_id: str, source_version: str, data_signature: str, config_signature: str) -> Path:
        d = self.stage_cache_root("transfer") / site / source_version / plot_id / f"{data_signature}__{config_signature}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def has_valid_transfer_stage_cache(
        self,
        *,
        site: str,
        plot_id: str,
        source_version: str,
        objects_group: pd.DataFrame,
    ) -> bool:
        data_sig = self.transfer_stage_data_signature(
            site=site,
            plot_id=plot_id,
            source_version=source_version,
            objects_group=objects_group,
        )
        config_sig = self.stage_config_signature("transfer") + "__" + self.stage_config_signature("rasterize")
        cache_dir = self.transfer_stage_cache_dir(site, plot_id, source_version, data_sig, config_sig)

        cache_object_csv = cache_dir / "objects_transferred.csv"
        cache_artifacts_csv = cache_dir / "artifacts.csv"

        return (
            self.validate_stage_cache(
                stage_name="transfer",
                stage_cache_dir=cache_dir,
                expected_data_signature=data_sig,
                expected_config_signature=config_sig,
            )
            and cache_object_csv.exists()
            and cache_artifacts_csv.exists()
        )
    
    def _is_rel_path_string(self, value: str) -> bool:
        return isinstance(value, str) and value.startswith("labeling/")

    def reconcile_artifact_reference(self, *, key: str, value):
        if value is None:
            return {"status": "missing"}

        artifact_key = None
        if key.endswith("_rel_path"):
            artifact_key = key.replace("_rel_path", "")
        elif key in self.artifact_specs():
            artifact_key = key

        # rel-path case
        if isinstance(value, str) and self._is_rel_path_string(value):
            rel_path = value
            local_path = self._local_artifact_path(rel_path)

            hydrated = None
            if (not local_path.exists()) and self._remote_exists(rel_path):
                hydrated = self.hydrate_and_validate_artifact(
                    rel_path=rel_path,
                    local_path=local_path,
                    artifact_key=artifact_key,
                    reason=f"reconcile:{key}",
                )
            elif local_path.exists():
                ok = self.validate_hydrated_artifact(
                    rel_path=rel_path,
                    local_path=local_path,
                    artifact_key=artifact_key,
                )
                if not ok and self._remote_exists(rel_path):
                    self.logger.warning(
                        "RECONCILE LOCAL INVALID | key=%s | rel_path=%s | attempting rehydrate",
                        key, rel_path
                    )
                    local_path.unlink(missing_ok=True)
                    hydrated = self.hydrate_and_validate_artifact(
                        rel_path=rel_path,
                        local_path=local_path,
                        artifact_key=artifact_key,
                        reason=f"reconcile_invalid_local:{key}",
                    )

            if artifact_key in self.artifact_specs():
                if local_path.exists():
                    self._prune_if_allowed(local_path, artifact_key, rel_path)

            return {
                "status": "reconciled_rel_path",
                "rel_path": rel_path,
                "artifact_key": artifact_key,
                "exists_local": local_path.exists(),
                "exists_remote": self._remote_exists(rel_path),
                "hydrated_now": hydrated is not None,
            }

        # local file case
        p = Path(value) if isinstance(value, str) else None
        if p is not None and p.exists() and p.is_file():
            try:
                rel_path = str(p.relative_to(self.cfg.data.project_root))
            except Exception:
                return {"status": "local_only", "path": str(p)}

            if not self._remote_exists(rel_path):
                self.logger.info("RECONCILE PUSH | key=%s | rel_path=%s", key, rel_path)
                self.artifact_store.push(p, rel_path=rel_path)

            return {
                "status": "reconciled_local_path",
                "rel_path": rel_path,
                "artifact_key": artifact_key,
                "exists_local": p.exists(),
                "exists_remote": self._remote_exists(rel_path),
            }

        return {"status": "unhandled", "value": value}
        

    def _current_config_signature(self) -> str:
        return self.config_signature()
    
    
    def _site_metadata_manifest_cache_payload(self, *, site: str, naip_local: Path, als_meta: list[dict]) -> dict:
        return {
            "site": site,
            "naip_local": str(naip_local),
            "als_metadata_count": len(als_meta),
            "als_metadata": als_meta,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
    
    
    def _build_source_inventory(self, site: str) -> dict:
        site_base = site_to_remote_base(self.cfg, site)
    
        naip_name = site_to_tif_name(site)
        naip_url = f"{site_base}/{self.cfg.data.naip_3dep_dir}/{naip_name}"
        als_url = f"{site_base}/{self.cfg.data.als_dir}"
    
        inventory = {
            "site": site,
            "site_base": site_base,
            "naip": {
                "expected_name": naip_name,
                "remote_url": naip_url,
            },
            "als": {
                "remote_url": als_url,
                "files": [],
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
    
        try:
            als_files = list_files_with_suffix(als_url, (".laz", ".las", ".copc.laz"))
            inventory["als"]["files"] = als_files
        except Exception as e:
            inventory["als"]["error"] = str(e)
    
        return inventory
    
    
    def _load_source_inventory(self, site: str) -> dict | None:
        payload = self._try_load_json_artifact(
            "source_inventory",
            site=site,
            config_signature=self._current_config_signature(),
        )
        return payload
    
    
    def _persist_source_inventory(self, site: str, inventory: dict) -> None:
        self._persist_json_artifact(
            inventory,
            "source_inventory",
            site=site,
            config_signature=self._current_config_signature(),
        )
    
    
    def _load_site_metadata_manifest(self, site: str) -> dict | None:
        payload = self._try_load_json_artifact(
            "site_metadata_manifest",
            site=site,
            config_signature=self._current_config_signature(),
        )
        return payload
    
    
    def _persist_site_metadata_manifest(self, site: str, *, naip_local: Path, als_meta: list[dict]) -> None:
        payload = self._site_metadata_manifest_cache_payload(
            site=site,
            naip_local=naip_local,
            als_meta=als_meta,
        )
        self._persist_json_artifact(
            payload,
            "site_metadata_manifest",
            site=site,
            config_signature=self._current_config_signature(),
        )
    
    
    def _load_cached_als_metadata_artifact(self, site: str) -> list[dict] | None:
        payload = self._try_load_json_artifact(
            "als_metadata_json",
            site=site,
            config_signature=self._current_config_signature(),
        )
        if payload is None:
            return None
        return payload.get("rows", [])
    
    
    def _persist_als_metadata_artifact(self, site: str, rows: list[dict]) -> None:
        self._persist_json_artifact(
            {"site": site, "rows": rows},
            "als_metadata_json",
            site=site,
            config_signature=self._current_config_signature(),
        )

    # -------------------------------------------------------------------------
    # Config signatures / per-run persistence / config-space enumeration
    # -------------------------------------------------------------------------

    def config_dict(self) -> dict:
        return asdict(self.pipeline_config)

    def config_signature(self, config_dict: dict | None = None) -> str:
        payload = config_dict if config_dict is not None else self.config_dict()
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def run_dir_for_signature(self, signature: str | None = None) -> Path:
        signature = signature or self.config_signature()
        run_dir = self.output_root / signature
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run_manifest_path(self, signature: str | None = None) -> Path:
        return self.run_dir_for_signature(signature) / "run_manifest.json"

    def run_summary_dir(self, signature: str | None = None) -> Path:
        d = self.run_dir_for_signature(signature) / "summaries"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def config_space_frame(self) -> pd.DataFrame:
        rows = []
        for cfg in self.enumerate_config_space():
            rows.append(
                {
                    "config_signature": self.config_signature(cfg),
                    "sprint3_variants": ",".join(cfg["sprint3_variants"]),
                    "use_temporal_confidence": cfg["use_temporal_confidence"],
                    "boundary_confidence_mode": cfg["boundary_confidence_mode"],
                    "use_object_subspace_filter": cfg["use_object_subspace_filter"],
                    "max_ptx_per_site": cfg["max_ptx_per_site"],
                }
            )
        return pd.DataFrame(rows).sort_values("config_signature").reset_index(drop=True)
    
    def shared_artifact_family_for_stage(self, stage_name: str) -> str | None:
        mapping = {
            "site_assets": "labeling.site_assets",
            "sprint3": "labeling.sprint3.outputs",
            "standardize": "labeling.standardize.outputs",
            "refine": "labeling.refine.outputs",
            "transfer": "labeling.transfer.outputs",
            "rasterize": "labeling.transfer.outputs",
        }
        return mapping.get(stage_name)
    
    def shared_signature_site_assets(self, site: str) -> str:
        payload = {
            "site": site,
            "naip_name": site_to_tif_name(site),
            "als_dir": self.cfg.data.als_dir,
            "naip_dir": self.cfg.data.naip_3dep_dir,
        }
        return hash_payload(payload)
    
    def shared_signature_sprint3(self, *, site: str, variant: str, ptx_name: str | None, ptx_url: str | None) -> str:
        payload = {
            "site": site,
            "variant": variant,
            "ptx_name": ptx_name,
            "ptx_url": ptx_url,
            "max_ptx_per_site": self.pipeline_config.max_ptx_per_site,
            "require_success_artifacts_sprint3": self.pipeline_config.require_success_artifacts_sprint3,
        }
        return hash_payload(payload)
    
    def shared_signature_standardize(self, runs_df: pd.DataFrame) -> str:
        return self.standardize_stage_data_signature(runs_df)

    def shared_signature_refine(self, objects_df: pd.DataFrame) -> str:
        return self.refine_stage_data_signature(objects_df)

    def shared_signature_transfer(
        self,
        *,
        site: str,
        plot_id: str,
        source_version: str,
        objects_group: pd.DataFrame,
    ) -> str:
        return self.transfer_stage_data_signature(
            site=site,
            plot_id=plot_id,
            source_version=source_version,
            objects_group=objects_group,
        )
    
    def shared_signature_for_stage(self, stage_name: str, **kwargs) -> str | None:
        if stage_name == "site_assets":
            site = kwargs["site"]
            return self.shared_signature_site_assets(site)
        if stage_name == "sprint3":
            return self.shared_signature_sprint3(
                site=kwargs["site"],
                variant=kwargs["variant"],
                ptx_name=kwargs.get("ptx_name"),
                ptx_url=kwargs.get("ptx_url"),
            )
        return None
    
    def shared_artifact_is_valid(
        self,
        *,
        artifact_family: str,
        shared_signature: str,
    ) -> bool:
        if not self.cfg.shared_artifacts.enable_shared_artifact_registry:
            return False

        rec = self.shared_registry.load(
            artifact_family=artifact_family,
            shared_signature=shared_signature,
        )
        if rec is None:
            return False

        return str(rec.status).lower().endswith("valid")
    
    def register_shared_requirement(
        self,
        *,
        artifact_family: str,
        shared_signature: str,
        trial_id: str,
        metadata: dict | None = None,
    ):
        if not self.cfg.shared_artifacts.enable_shared_artifact_registry:
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
    
    def repair_or_rehydrate_artifact(
        self,
        *,
        artifact_key: str,
        rel_path: str,
        local_path: Path,
        validator_key: str | None = None,
        recompute_fn=None,
    ) -> Path | None:
        policy = self.cfg.artifact_repair

        if local_path.exists():
            ok = self.validate_hydrated_artifact(
                rel_path=rel_path,
                local_path=local_path,
                artifact_key=validator_key or artifact_key,
            )
            if ok:
                return local_path

            self.logger.warning(
                "LOCAL ARTIFACT INVALID | key=%s | rel_path=%s | local_path=%s",
                artifact_key, rel_path, local_path,
            )
            if policy.repair_invalid_local_assets:
                local_path.unlink(missing_ok=True)

        if policy.prefer_remote_hydration_for_invalid_local_assets and self._remote_exists(rel_path):
            pulled = self.hydrate_and_validate_artifact(
                rel_path=rel_path,
                local_path=local_path,
                artifact_key=validator_key or artifact_key,
                reason=f"repair_or_rehydrate:{artifact_key}",
            )
            if pulled is not None:
                return pulled

        if recompute_fn is not None and policy.recompute_only_if_local_and_remote_invalid:
            recompute_fn()
            if local_path.exists():
                return local_path

        return None
    
    def enforce_storage_policy_for_existing_run(
        self,
        *,
        signature: str | None = None,
        existing: PipelineRunResult | None = None,
    ) -> dict[str, dict]:
        signature = signature or self.config_signature()
        existing = existing or self.try_load_existing_run(signature=signature)

        report = {
            "signature": signature,
            "run_level": {},
            "stage_caches": {},
        }

        # Run-level artifacts
        run_rel_paths = {
            "objects_summary": self._render_rel_path("objects_summary", config_signature=signature),
            "artifacts_summary": self._render_rel_path("artifacts_summary", config_signature=signature),
            "run_manifest": self._render_rel_path("run_manifest", config_signature=signature),
            "run_result": self._render_rel_path("run_result", config_signature=signature),
        }

        for artifact_key, rel_path in run_rel_paths.items():
            report["run_level"][artifact_key] = self.reconcile_artifact_reference(
                key=artifact_key,
                value=rel_path,
            )

        # Stage cache directories
        stage_cache_root = self.cfg.output.labeling_root / "stage_cache"
        if stage_cache_root.exists():
            for manifest_path in stage_cache_root.rglob("stage_cache_manifest.json"):
                stage_cache_dir = manifest_path.parent
                try:
                    rec = read_stage_cache_manifest(stage_cache_dir)
                    if rec is None:
                        continue
                    stage_name = rec.stage_name
                    self.enforce_storage_policy_for_stage_cache(
                        stage_name=stage_name,
                        stage_cache_dir=stage_cache_dir,
                    )
                    report["stage_caches"][str(stage_cache_dir)] = {
                        "stage_name": stage_name,
                        "status": "reconciled",
                    }
                except Exception as e:
                    self.logger.warning(
                        "STORAGE POLICY RECONCILE FAIL | stage_cache_dir=%s | error=%s",
                        stage_cache_dir, e
                    )
                    report["stage_caches"][str(stage_cache_dir)] = {
                        "status": "error",
                        "error": str(e),
                    }

        self.logger.info(
            "EXISTING RUN STORAGE POLICY RECONCILE DONE | signature=%s | n_stage_caches=%d",
            signature,
            len(report["stage_caches"]),
        )
        return report

    def try_load_existing_run(self, signature: str | None = None) -> PipelineRunResult | None:
        """
        Prefer config-specific pipeline-run outputs. If absent, optionally adopt
        existing global summary outputs as a one-time bridge for notebook-era runs.
        """
        signature = signature or self.config_signature()
        manifest_path = self.run_manifest_path(signature)
    
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.logger.info("Using existing labeling run manifest for signature=%s", signature)
            return PipelineRunResult(**payload)
    
        payload = self._try_load_json_artifact(
            "run_manifest",
            config_signature=signature,
        )
        if payload is not None:
            self.logger.info("Using remote-backed labeling run manifest for signature=%s", signature)
            return PipelineRunResult(**payload)
    
        # bridge mode remains the same below
        global_objects = self.summary_dir / "objects_all.csv"
        global_artifacts = self.summary_dir / "artifacts_all.csv"
    
        if (
            self.pipeline_config.allow_adopt_global_outputs
            and global_objects.exists()
            and global_artifacts.exists()
        ):
            self.logger.info(
                "No config-specific run manifest for signature=%s, but found existing global labeling summaries. Adopting them.",
                signature,
            )
    
            objects_df = pd.read_csv(global_objects)
            artifacts_df = pd.read_csv(global_artifacts)
    
            adopted_objects, adopted_artifacts = self.finalize_outputs(
                objects_df,
                artifacts_df,
                signature=signature,
                write_global=False,
            )
    
            result = PipelineRunResult(
                pipeline_name=self.pipeline_name,
                success=(not objects_df.empty) and (not artifacts_df.empty),
                status="adopted_existing_outputs",
                raster_outputs=CanonicalRasterOutputs(
                    labels=artifacts_df,
                    qa_overlays=artifacts_df[["site_id", "plot_id", "qa_overlay_path"]].copy()
                    if not artifacts_df.empty and "qa_overlay_path" in artifacts_df.columns
                    else None,
                ),
                object_outputs=CanonicalObjectOutputs(
                    objects=objects_df,
                    source_provenance=objects_df[
                        [c for c in ["site_id", "plot_id", "source_version", "source_file"] if c in objects_df.columns]
                    ].copy()
                    if not objects_df.empty
                    else None,
                ),
                qa_outputs={
                    "objects_csv": str(adopted_objects),
                    "artifacts_csv": str(adopted_artifacts),
                    "artifacts_manifest_csv": str(self.sprint4_manifest_csv),
                    "adopted_from_global_outputs": True,
                    "config_signature": signature,
                },
                metrics={
                    "n_object_rows": int(len(objects_df)),
                    "n_artifact_rows": int(len(artifacts_df)),
                    "n_sites": int(objects_df["site_id"].nunique()) if "site_id" in objects_df.columns and not objects_df.empty else 0,
                },
                notes=["Adopted existing global labeling outputs into config-specific run directory."],
            )
            self.save_pipeline_run_manifest(result, signature=signature)
            return result
    
        return None

    def save_pipeline_run_manifest(self, result: PipelineRunResult, signature: str | None = None) -> Path:
        signature = signature or self.config_signature()
        path = self.run_manifest_path(signature)
        path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    
        self._persist_file_artifact(
            path,
            "run_manifest",
            config_signature=signature,
        )
    
        self.logger.info("Saved labeling run manifest to %s", path)
        return path

    def executed_stages_from_result(self, result: PipelineRunResult) -> set[str]:
        return set((result.qa_outputs or {}).get("executed_stages", []) or [])
    
    
    def skipped_stages_from_result(self, result: PipelineRunResult) -> dict[str, str]:
        return dict((result.qa_outputs or {}).get("skipped_stages", {}) or {})
    
    
    def runnable_stages_for_runtime(self, runtime_report=None) -> set[str]:
        runtime_report = runtime_report or self.runtime_report()
        runnable = set()
        for stage in self.pipeline_spec.stages:
            ok, _ = self.stage_is_eligible(stage.name, runtime_report=runtime_report)
            if ok:
                runnable.add(stage.name)
        return runnable
    
    
    def additional_runnable_stages_remaining(
        self,
        existing: PipelineRunResult,
        *,
        runtime_report=None,
    ) -> set[str]:
        runtime_report = runtime_report or self.runtime_report()
        already_done = self.executed_stages_from_result(existing)
        runnable_now = self.runnable_stages_for_runtime(runtime_report=runtime_report)
        return runnable_now - already_done
    
    
    def should_reuse_existing_run(
        self,
        existing: PipelineRunResult,
        *,
        runtime_report=None,
    ) -> tuple[bool, str]:
        runtime_report = runtime_report or self.runtime_report()
        policy = self.cfg.labeling_runtime
    
        if existing.success:
            if policy.reuse_successful_runs:
                return True, "existing run is successful"
            return False, "successful reuse disabled by config"
    
        # partial/incomplete run
        if not policy.resume_partial_runs:
            return True, "partial runs are configured to be reused, not resumed"
    
        remaining = self.additional_runnable_stages_remaining(existing, runtime_report=runtime_report)
        if remaining:
            return False, f"partial run can be resumed; remaining runnable stages={sorted(remaining)}"
    
        if policy.reuse_partial_runs_when_no_new_stages_are_eligible:
            return True, "partial run reused because no new stages are eligible on this runtime"
    
        return False, "partial run not reused by policy"

    def _path_state(self, path: Path) -> dict:
        if not path.exists():
            return {"exists": False}
        stat = path.stat()
        return {
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    
    def _latest_stage_manifest_mtime_ns(self, stage_name: str) -> int | None:
        root = self.stage_cache_root(stage_name)
        manifests = list(root.rglob("stage_cache_manifest.json"))
        if not manifests:
            return None
        return max(int(p.stat().st_mtime_ns) for p in manifests if p.exists())
    
    def work_unit_refresh_fingerprint(
        self,
        *,
        trial_id: str,
        config_signature: str | None = None,
        runtime_report=None,
    ) -> str:
        runtime_report = runtime_report or self.runtime_report()
        config_signature = config_signature or self.config_signature()
    
        shared_site_asset_states = {}
        for site in self.cfg.sites:
            shared_sig = self.shared_signature_site_assets(site)
            rec = None
            if self.cfg.shared_artifacts.enable_shared_artifact_registry:
                rec = self.shared_registry.load(
                    artifact_family="labeling.site_assets",
                    shared_signature=shared_sig,
                )
            shared_site_asset_states[site] = {
                "shared_signature": shared_sig,
                "status": getattr(rec, "status", None) if rec is not None else None,
                "updated_at": getattr(rec, "updated_at", None) if rec is not None else None,
            }
    
        payload = {
            "pipeline_name": self.pipeline_name,
            "trial_id": trial_id,
            "config_signature": config_signature,
            "runtime_image": getattr(runtime_report, "detected_image_key", None),
            "runtime_caps": sorted(getattr(runtime_report, "capabilities", []) or []),
            "sprint3_manifest": self._path_state(self.sprint3_manifest_csv),
            "standardize_manifest_mtime_ns": self._latest_stage_manifest_mtime_ns("standardize"),
            "refine_manifest_mtime_ns": self._latest_stage_manifest_mtime_ns("refine"),
            "transfer_manifest_mtime_ns": self._latest_stage_manifest_mtime_ns("transfer"),
            "shared_registry_enabled": bool(self.cfg.shared_artifacts.enable_shared_artifact_registry),
            "shared_site_asset_states": shared_site_asset_states,
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
        config_signature = config_signature or self.config_signature()
        runtime_report = runtime_report or self.runtime_report()

        enum_cache_key = (
            trial_id,
            config_signature,
            getattr(runtime_report, "detected_image_key", None),
            register_shared_requirements,
            self.work_unit_refresh_fingerprint(
                trial_id=trial_id,
                config_signature=config_signature,
                runtime_report=runtime_report,
            ),
        )
        if enum_cache_key in self._enumeration_cache:
            self.logger.info(
                "ENUM WORK UNITS CACHE HIT | trial=%s | pipeline=%s",
                trial_id, self.pipeline_name
            )
            return [dict(u) for u in self._enumeration_cache[enum_cache_key]]

        units = []
        manifest_csv = self.sprint3_manifest_csv

        site_asset_unit_idx_by_site: dict[str, int] = {}
        transfer_statuses_by_site: dict[str, list[str]] = {}

        enum_t0 = perf_counter()

        self.logger.info(
            "ENUM WORK UNITS START | trial=%s | pipeline=%s | config=%s | register_shared_requirements=%s | runtime_image=%s",
            trial_id,
            self.pipeline_name,
            config_signature,
            register_shared_requirements,
            getattr(runtime_report, "detected_image_key", None),
        )

        # ------------------------------------------------------------------
        # Stage: site_assets (one per site, shared across trials)
        # ------------------------------------------------------------------
        for site in self.cfg.sites:
            site_assets_ok = True  # asset prep itself is allowed wherever transfer-side python runs
            site_assets_shared_sig = self.shared_signature_site_assets(site)

            if register_shared_requirements and self.cfg.shared_artifacts.enable_shared_artifact_registry:
                self.register_shared_requirement(
                    artifact_family="labeling.site_assets",
                    shared_signature=site_assets_shared_sig,
                    trial_id=trial_id,
                    metadata={"site": site},
                )

            site_assets_complete = self.shared_artifact_is_valid(
                artifact_family="labeling.site_assets",
                shared_signature=site_assets_shared_sig,
            )

            unit = {
                "unit_id": f"{trial_id}:{self.pipeline_name}:site_assets:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "site_assets",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if site_assets_complete else (
                    WorkUnitStatus.PENDING.value if site_assets_ok else WorkUnitStatus.INELIGIBLE.value
                ),
                "dependencies": [],
                "dependency_reasons": [],
                "runtime_required_capabilities": list(self.cfg.labeling_runtime.require_capability_transfer),
                "runtime_eligible": site_assets_ok,
                "priority": 5,
                "site_id": site,
                "shared_artifact_family": "labeling.site_assets",
                "shared_signature": site_assets_shared_sig,
            }
            site_asset_unit_idx_by_site[site] = len(units)
            units.append(unit)

        self.logger.info(
            "ENUM WORK UNITS | trial=%s | stage=site_assets | n_units=%d",
            trial_id,
            sum(1 for u in units if u.get("stage_name") == "site_assets"),
        )

        # ------------------------------------------------------------------
        # Stage: sprint3 (shared across trials)
        # ------------------------------------------------------------------
        sprint3_ok, _ = self.stage_is_eligible("sprint3", runtime_report=runtime_report)

        ptx_summary_df = pd.DataFrame()
        try:
            if sprint3_ok:
                ptx_summary_df = summarize_ptx_entries_by_site(
                    cfg=self.cfg,
                    site_ids=self.cfg.sites,
                )
                selected_ptx_df = select_ptx_entries(
                    ptx_summary_df,
                    max_ptx_per_site=self.pipeline_config.max_ptx_per_site,
                )
            else:
                selected_ptx_df = pd.DataFrame()
        except Exception:
            selected_ptx_df = pd.DataFrame()

        sprint3_shared_complete = False
        if not selected_ptx_df.empty:
            all_shared = True
            for _, row in selected_ptx_df.iterrows():
                site = row["site_id"]
                ptx_name = row["ptx_name"]
                ptx_url = row.get("ptx_url")
                for variant in self.pipeline_config.sprint3_variants:
                    shared_sig = self.shared_signature_sprint3(
                        site=site,
                        variant=variant,
                        ptx_name=ptx_name,
                        ptx_url=ptx_url,
                    )
                    if register_shared_requirements and self.cfg.shared_artifacts.enable_shared_artifact_registry:
                        self.register_shared_requirement(
                            artifact_family="labeling.sprint3.outputs",
                            shared_signature=shared_sig,
                            trial_id=trial_id,
                            metadata={"site": site, "variant": variant, "ptx_name": ptx_name},
                        )
                    if not self.shared_artifact_is_valid(
                        artifact_family="labeling.sprint3.outputs",
                        shared_signature=shared_sig,
                    ):
                        all_shared = False
            sprint3_shared_complete = all_shared

        # fallback legacy/local manifest check
        sprint3_complete = sprint3_shared_complete or manifest_csv.exists()

        units.append({
            "unit_id": f"{trial_id}:{self.pipeline_name}:sprint3",
            "trial_id": trial_id,
            "pipeline_name": self.pipeline_name,
            "config_signature": config_signature,
            "stage_name": "sprint3",
            "work_key": "sprint3",
            "scope": WorkUnitScope.STAGE.value,
            "status": WorkUnitStatus.COMPLETE.value if sprint3_complete else (
                WorkUnitStatus.PENDING.value if sprint3_ok else WorkUnitStatus.INELIGIBLE.value
            ),
            "dependencies": [],
            "dependency_reasons": [],
            "runtime_required_capabilities": list(self.cfg.labeling_runtime.require_capability_sprint3),
            "runtime_eligible": sprint3_ok,
            "priority": 10,
        })

        self.logger.info(
            "ENUM WORK UNITS | trial=%s | stage=sprint3 | sprint3_complete=%s | sprint3_ok=%s",
            trial_id,
            sprint3_complete,
            sprint3_ok,
        )

        # ------------------------------------------------------------------
        # Stage: standardize
        # ------------------------------------------------------------------
        standardize_ok, _ = self.stage_is_eligible("standardize", runtime_report=runtime_report)

        std_deps = []
        std_dep_reasons = []
        if not sprint3_complete:
            std_deps.append("sprint3")
            std_dep_reasons.append("Sprint 3 shared outputs/manifest missing.")

        std_complete = False
        runs_df = pd.DataFrame()
        try:
            if manifest_csv.exists():
                runs_df = pd.read_csv(manifest_csv)
                if "returncode" in runs_df.columns:
                    runs_df = runs_df[runs_df["returncode"] == 0].copy()
                if "variant" in runs_df.columns and self.pipeline_config.sprint3_variants:
                    runs_df = runs_df[runs_df["variant"].isin(self.pipeline_config.sprint3_variants)].copy()

                if not runs_df.empty:
                    std_shared_sig = self.shared_signature_standardize(runs_df)
                    if register_shared_requirements and self.cfg.shared_artifacts.enable_shared_artifact_registry:
                        self.register_shared_requirement(
                            artifact_family="labeling.standardize.outputs",
                            shared_signature=std_shared_sig,
                            trial_id=trial_id,
                        )

                    std_complete = self.shared_artifact_is_valid(
                        artifact_family="labeling.standardize.outputs",
                        shared_signature=std_shared_sig,
                    )

                    if not std_complete:
                        data_sig = self.standardize_stage_data_signature(runs_df)
                        config_sig = self.stage_config_signature("standardize")
                        cache_dir = self.standardize_stage_cache_dir(data_sig, config_sig)
                        cache_csv = cache_dir / "objects_standardized.csv"
                        std_complete = (
                            self.validate_stage_cache(
                                stage_name="standardize",
                                stage_cache_dir=cache_dir,
                                expected_data_signature=data_sig,
                                expected_config_signature=config_sig,
                            )
                            and cache_csv.exists()
                        )
        except Exception:
            std_complete = False

        units.append({
            "unit_id": f"{trial_id}:{self.pipeline_name}:standardize",
            "trial_id": trial_id,
            "pipeline_name": self.pipeline_name,
            "config_signature": config_signature,
            "stage_name": "standardize",
            "work_key": "standardize",
            "scope": WorkUnitScope.STAGE.value,
            "status": WorkUnitStatus.COMPLETE.value if std_complete else (
                WorkUnitStatus.PENDING.value if (standardize_ok and not std_deps) else (
                    WorkUnitStatus.BLOCKED.value if standardize_ok else WorkUnitStatus.INELIGIBLE.value
                )
            ),
            "dependencies": std_deps,
            "dependency_reasons": std_dep_reasons,
            "runtime_required_capabilities": list(self.cfg.labeling_runtime.require_capability_standardize),
            "runtime_eligible": standardize_ok,
            "priority": 20,
        })

        self.logger.info(
            "ENUM WORK UNITS | trial=%s | stage=standardize | std_complete=%s | deps=%s",
            trial_id,
            std_complete,
            std_deps,
        )

        # ------------------------------------------------------------------
        # Stage: refine
        # ------------------------------------------------------------------
        refine_ok, _ = self.stage_is_eligible("refine", runtime_report=runtime_report)

        refine_deps = []
        refine_dep_reasons = []
        if not std_complete:
            refine_deps.append("standardize")
            refine_dep_reasons.append("Standardize shared output missing.")

        refine_complete = False
        std_df = pd.DataFrame()
        try:
            if std_complete and not runs_df.empty:
                data_sig = self.standardize_stage_data_signature(runs_df)
                config_sig = self.stage_config_signature("standardize")
                cache_dir = self.standardize_stage_cache_dir(data_sig, config_sig)
                cache_csv = cache_dir / "objects_standardized.csv"

                if cache_csv.exists():
                    std_df = pd.read_csv(cache_csv)
                    ref_shared_sig = self.shared_signature_refine(std_df)

                    if register_shared_requirements and self.cfg.shared_artifacts.enable_shared_artifact_registry:
                        self.register_shared_requirement(
                            artifact_family="labeling.refine.outputs",
                            shared_signature=ref_shared_sig,
                            trial_id=trial_id,
                        )

                    refine_complete = self.shared_artifact_is_valid(
                        artifact_family="labeling.refine.outputs",
                        shared_signature=ref_shared_sig,
                    )

                    if not refine_complete:
                        ref_data_sig = self.refine_stage_data_signature(std_df)
                        ref_config_sig = self.stage_config_signature("refine")
                        ref_cache_dir = self.refine_stage_cache_dir(ref_data_sig, ref_config_sig)
                        ref_cache_csv = ref_cache_dir / "objects_refined.csv"
                        refine_complete = (
                            self.validate_stage_cache(
                                stage_name="refine",
                                stage_cache_dir=ref_cache_dir,
                                expected_data_signature=ref_data_sig,
                                expected_config_signature=ref_config_sig,
                            )
                            and ref_cache_csv.exists()
                        )
        except Exception:
            refine_complete = False

        units.append({
            "unit_id": f"{trial_id}:{self.pipeline_name}:refine",
            "trial_id": trial_id,
            "pipeline_name": self.pipeline_name,
            "config_signature": config_signature,
            "stage_name": "refine",
            "work_key": "refine",
            "scope": WorkUnitScope.STAGE.value,
            "status": WorkUnitStatus.COMPLETE.value if refine_complete else (
                WorkUnitStatus.PENDING.value if (refine_ok and not refine_deps) else (
                    WorkUnitStatus.BLOCKED.value if refine_ok else WorkUnitStatus.INELIGIBLE.value
                )
            ),
            "dependencies": refine_deps,
            "dependency_reasons": refine_dep_reasons,
            "runtime_required_capabilities": list(self.cfg.labeling_runtime.require_capability_refine),
            "runtime_eligible": refine_ok,
            "priority": 30,
        })

        self.logger.info(
            "ENUM WORK UNITS | trial=%s | stage=refine | refine_complete=%s | deps=%s",
            trial_id,
            refine_complete,
            refine_deps,
        )

        # ------------------------------------------------------------------
        # Plot-level transfer+rasterize units
        # ------------------------------------------------------------------
        transfer_ok, _ = self.stage_is_eligible("transfer", runtime_report=runtime_report)
        rasterize_ok, _ = self.stage_is_eligible("rasterize", runtime_report=runtime_report)

        if refine_complete:
            try:
                if not std_df.empty:
                    ref_data_sig = self.refine_stage_data_signature(std_df)
                    ref_config_sig = self.stage_config_signature("refine")
                    ref_cache_dir = self.refine_stage_cache_dir(ref_data_sig, ref_config_sig)
                    ref_cache_csv = ref_cache_dir / "objects_refined.csv"
                    refined_df = pd.read_csv(ref_cache_csv)

                    grouped = refined_df.groupby(["site_id", "plot_id", "source_version"], dropna=False)
                    for (site_id, plot_id, source_version), group_df in grouped:
                        transfer_shared_sig = self.shared_signature_transfer(
                            site=site_id,
                            plot_id=plot_id,
                            source_version=source_version,
                            objects_group=group_df,
                        )

                        if register_shared_requirements and self.cfg.shared_artifacts.enable_shared_artifact_registry:
                            self.register_shared_requirement(
                                artifact_family="labeling.transfer.outputs",
                                shared_signature=transfer_shared_sig,
                                trial_id=trial_id,
                                metadata={
                                    "site_id": site_id,
                                    "plot_id": plot_id,
                                    "source_version": source_version,
                                },
                            )

                        valid_shared = self.shared_artifact_is_valid(
                            artifact_family="labeling.transfer.outputs",
                            shared_signature=transfer_shared_sig,
                        )

                        valid_cache = valid_shared or self.has_valid_transfer_stage_cache(
                            site=site_id,
                            plot_id=plot_id,
                            source_version=source_version,
                            objects_group=group_df,
                        )

                        deps = []
                        dep_reasons = []
                        
                        if not valid_cache:
                            site_assets_shared_sig = self.shared_signature_site_assets(site_id)
                            if not self.shared_artifact_is_valid(
                                artifact_family="labeling.site_assets",
                                shared_signature=site_assets_shared_sig,
                            ):
                                deps.append(f"site_assets:{site_id}")
                                dep_reasons.append(f"Shared site assets missing for site={site_id}")

                        # site_assets_shared_sig = self.shared_signature_site_assets(site_id)
                        # if not self.shared_artifact_is_valid(
                        #     artifact_family="labeling.site_assets",
                        #     shared_signature=site_assets_shared_sig,
                        # ):
                        #     deps.append(f"site_assets:{site_id}")
                        #     dep_reasons.append(f"Shared site assets missing for site={site_id}")

                        transfer_status = (
                            WorkUnitStatus.COMPLETE.value if valid_cache else (
                                WorkUnitStatus.PENDING.value if (transfer_ok and rasterize_ok and not deps) else (
                                    WorkUnitStatus.BLOCKED.value if (transfer_ok and rasterize_ok) else WorkUnitStatus.INELIGIBLE.value
                                )
                            )
                        )
                        
                        transfer_statuses_by_site.setdefault(site_id, []).append(transfer_status)

                        units.append({
                            "unit_id": f"{trial_id}:{self.pipeline_name}:transfer:{site_id}:{plot_id}:{source_version}",
                            "trial_id": trial_id,
                            "pipeline_name": self.pipeline_name,
                            "config_signature": config_signature,
                            "stage_name": "transfer",
                            "work_key": f"{site_id}|{plot_id}|{source_version}",
                            "scope": WorkUnitScope.PLOT.value,
                            "status": transfer_status,
                            "dependencies": deps,
                            "dependency_reasons": dep_reasons,
                            "runtime_required_capabilities": sorted(
                                set(self.cfg.labeling_runtime.require_capability_transfer)
                                | set(self.cfg.labeling_runtime.require_capability_rasterize)
                            ),
                            "runtime_eligible": transfer_ok and rasterize_ok,
                            "priority": 100,
                            "site_id": site_id,
                            "plot_id": plot_id,
                            "source_version": source_version,
                            "shared_artifact_family": "labeling.transfer.outputs",
                            "shared_signature": transfer_shared_sig,
                        })

                        n_transfer_units = sum(1 for u in units if u.get("stage_name") == "transfer")
                        self.logger.info(
                            "ENUM WORK UNITS | trial=%s | stage=transfer | n_units=%d | transfer_ok=%s | rasterize_ok=%s",
                            trial_id,
                            n_transfer_units,
                            transfer_ok,
                            rasterize_ok,
                        )
            except Exception:
                pass

        for site, idx in site_asset_unit_idx_by_site.items():
            site_unit = units[idx]
            if site_unit["status"] == WorkUnitStatus.COMPLETE.value:
                continue
        
            downstream_statuses = transfer_statuses_by_site.get(site, [])
            all_site_transfers_complete = (
                len(downstream_statuses) > 0
                and all(s == WorkUnitStatus.COMPLETE.value for s in downstream_statuses)
            )
        
            if all_site_transfers_complete:
                site_unit["status"] = WorkUnitStatus.COMPLETE.value
                site_unit["notes"] = list(site_unit.get("notes", [])) + [
                    "Marked complete because all downstream transfer units for this site are already complete."
                ]
                self.logger.info(
                    "ENUM WORK UNITS | trial=%s | site_assets promoted to complete from downstream transfer completeness | site=%s",
                    trial_id,
                    site,
                )

        enum_t1 = perf_counter()
        self.logger.info(
            "ENUM WORK UNITS DONE  | trial=%s | pipeline=%s | total_units=%d | dt=%.2fs",
            trial_id,
            self.pipeline_name,
            len(units),
            enum_t1 - enum_t0,
        )

        self._enumeration_cache[enum_cache_key] = [dict(u) for u in units]
        return units

    # -------------------------------------------------------------------------
    # Stage 1: Sprint 3 manifest -> canonical objects
    # -------------------------------------------------------------------------

    def stage_run_sprint3(self) -> pd.DataFrame:
        """
        Discover PTX files, select the latest K per site, run configured Sprint 3
        variants with caching/cleanup, and update the Sprint 3 manifest CSV.
        """
        if not self.pipeline_config.run_sprint3:
            self.logger.info("Skipping Sprint 3 execution because run_sprint3=False")
            if self.sprint3_manifest_csv.exists():
                return pd.read_csv(self.sprint3_manifest_csv)
            return pd.DataFrame()

        self.sprint3_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        self.ptx_cache_root.mkdir(parents=True, exist_ok=True)

        if self.pipeline_config.cleanup_stale_ptx_before_run:
            stale_removed_df = cleanup_stale_ptx_cache(
                cache_root=self.ptx_cache_root,
                stale_days=self.pipeline_config.stale_ptx_days,
            )
            if not stale_removed_df.empty:
                self.logger.info("Removed %d stale PTX cache file(s)", len(stale_removed_df))

        ptx_summary_df = summarize_ptx_entries_by_site(
            cfg=self.cfg,
            site_ids=self.cfg.sites,
        )

        selected_ptx_df = select_ptx_entries(
            ptx_summary_df,
            max_ptx_per_site=self.pipeline_config.max_ptx_per_site,
        )

        self.logger.info(
            "Sprint 3 PTX discovery selected %d PTX row(s) across %d site(s)",
            len(selected_ptx_df),
            selected_ptx_df['site_id'].nunique() if not selected_ptx_df.empty else 0,
        )

        all_results = []

        for _, row in selected_ptx_df.iterrows():
            site = row["site_id"]
            ptx_entry = {
                "name": row["ptx_name"],
                "url": row["ptx_url"],
            }

            local_ptx = None
            all_variants_ok = True

            try:
                local_ptx = download_ptx_with_cache(
                    site_id=site,
                    ptx_entry=ptx_entry,
                    cache_root=self.ptx_cache_root,
                )
            except Exception:
                self.logger.exception("Failed to download PTX for site=%s entry=%s", site, ptx_entry)
                continue

            for variant in self.pipeline_config.sprint3_variants:
                self.logger.info(
                    "Starting Sprint 3 | site=%s | variant=%s | ptx=%s",
                    site, variant, local_ptx.name
                )
                try:
                    result = run_sprint3_for_ptx(
                        site_id=site,
                        ptx_path=local_ptx,
                        sprint3_base_dir=self.cfg.data.sprint3_base_dir,
                        output_root=self.sprint3_output_root,
                        variant=variant,
                        raise_on_error=True,
                        force_rerun=self.pipeline_config.force_rerun_sprint3,
                        require_success_artifacts=self.pipeline_config.require_success_artifacts_sprint3,
                    )
                    all_results.append(result)
                    append_results_manifest(self.sprint3_manifest_csv, [result])

                    if self.cfg.shared_artifacts.enable_shared_artifact_registry:
                        shared_sig = self.shared_signature_sprint3(
                            site=site,
                            variant=variant,
                            ptx_name=ptx_entry["name"],
                            ptx_url=ptx_entry["url"],
                        )
                        self.shared_registry.mark_available(
                            artifact_family="labeling.sprint3.outputs",
                            shared_signature=shared_sig,
                            producer_pipeline=self.pipeline_name,
                            source_trial=getattr(self, "_active_trial_id", None),
                            metadata={
                                "site": site,
                                "variant": variant,
                                "ptx_name": ptx_entry["name"],
                                "used_cache": getattr(result, "used_cache", None),
                            },
                            status=SharedArtifactStatus.VALID,
                        )

                    self.logger.info(
                        "Completed Sprint 3 | site=%s | variant=%s | ptx=%s | cache=%s",
                        site,
                        variant,
                        local_ptx.name,
                        result.used_cache,
                    )
                except Exception:
                    all_variants_ok = False
                    self.logger.exception(
                        "Sprint 3 failed | site=%s | variant=%s | ptx=%s",
                        site, variant, local_ptx.name
                    )

            if (
                local_ptx is not None
                and self.pipeline_config.cleanup_ptx_after_all_variants
                and all_variants_ok
            ):
                try:
                    from Final.labeling.sprint3_runner import cleanup_ptx_file
                    cleanup_ptx_file(local_ptx)
                except Exception:
                    self.logger.exception("Failed PTX cleanup after Sprint 3 for %s", local_ptx)

        if self.sprint3_manifest_csv.exists():
            manifest_df = pd.read_csv(self.sprint3_manifest_csv)
        else:
            manifest_df = pd.DataFrame()

        self.logger.info(
            "Sprint 3 stage complete | manifest_rows=%d | manifest_csv=%s",
            len(manifest_df),
            self.sprint3_manifest_csv,
        )
        return manifest_df

    def stage_load_and_standardize_objects(
        self,
        manifest_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        manifest_csv = Path(manifest_csv) if manifest_csv is not None else self.sprint3_manifest_csv
        if not manifest_csv.exists():
            self.logger.warning("Sprint 3 manifest missing: %s", manifest_csv)
            return pd.DataFrame()

        runs_df = pd.read_csv(manifest_csv)
        if "returncode" in runs_df.columns:
            runs_df = runs_df[runs_df["returncode"] == 0].copy()

        if "variant" in runs_df.columns and self.pipeline_config.sprint3_variants:
            runs_df = runs_df[runs_df["variant"].isin(self.pipeline_config.sprint3_variants)].copy()

        data_sig = self.standardize_stage_data_signature(runs_df)
        config_sig = self.stage_config_signature("standardize")
        cache_dir = self.standardize_stage_cache_dir(data_sig, config_sig)
        cache_csv = cache_dir / "objects_standardized.csv"

        if self.validate_stage_cache(
            stage_name="standardize",
            stage_cache_dir=cache_dir,
            expected_data_signature=data_sig,
            expected_config_signature=config_sig,
        ) and cache_csv.exists():
            self.logger.info("Using module-aware standardize cache | data=%s | config=%s", data_sig, config_sig)

            self.enforce_storage_policy_for_stage_cache(
                stage_name="standardize",
                stage_cache_dir=cache_dir,
            )

            return pd.read_csv(cache_csv)

        objects = standardize_sprint3_manifest(
            runs_df,
            source_version_prefix="sprint3",
            label_variant="base",
            keep_only_valid_runs=True,
            require_success_returncode=True,
        )
        objects.to_csv(cache_csv, index=False)

        self.write_stage_cache(
            stage_name="standardize",
            stage_cache_dir=cache_dir,
            data_signature=data_sig,
            config_signature=config_sig,
            artifact_paths={"objects_csv": str(cache_csv)},
            success=True,
            notes=["Standardized Sprint 3 manifest outputs."],
        )

        if self.cfg.shared_artifacts.enable_shared_artifact_registry and not runs_df.empty:
            shared_sig = self.shared_signature_standardize(runs_df)
            self.shared_registry.mark_available(
                artifact_family="labeling.standardize.outputs",
                shared_signature=shared_sig,
                producer_pipeline=self.pipeline_name,
                local_path=str(cache_csv),
                source_trial=getattr(self, "_active_trial_id", None),
                metadata={"n_rows": int(len(objects))},
                status=SharedArtifactStatus.VALID,
            )

        self.logger.info("Standardized %d object rows from Sprint 3 manifest after variant filtering", len(objects))
        return objects

    # -------------------------------------------------------------------------
    # Stage 2: refinement
    # -------------------------------------------------------------------------

    def stage_refine_objects(self, objects_df: pd.DataFrame) -> pd.DataFrame:
        if objects_df.empty:
            return objects_df.copy()

        data_sig = self.refine_stage_data_signature(objects_df)
        config_sig = self.stage_config_signature("refine")
        cache_dir = self.refine_stage_cache_dir(data_sig, config_sig)
        cache_csv = cache_dir / "objects_refined.csv"

        if self.validate_stage_cache(
            stage_name="refine",
            stage_cache_dir=cache_dir,
            expected_data_signature=data_sig,
            expected_config_signature=config_sig,
        ) and cache_csv.exists():
            self.logger.info("Using module-aware refine cache | data=%s | config=%s", data_sig, config_sig)

            self.enforce_storage_policy_for_stage_cache(
                stage_name="refine",
                stage_cache_dir=cache_dir,
            )

            return pd.read_csv(cache_csv)

        subspace_cfg = SubspaceReductionConfig(
            min_component_pixels=self.pipeline_config.subspace_min_component_pixels,
            min_object_confidence=self.pipeline_config.subspace_min_object_confidence,
            min_transform_confidence=self.pipeline_config.subspace_min_transform_confidence,
            min_temporal_confidence=self.pipeline_config.subspace_min_temporal_confidence,
            max_height_m=self.pipeline_config.subspace_max_height_m,
        )

        site_reference_dates = self.pipeline_config.site_reference_dates if self.pipeline_config.use_temporal_confidence else None

        refined = refine_shrub_objects(
            objects_df,
            self.cfg,
            site_reference_dates=site_reference_dates,
            apply_subspace_filter=self.pipeline_config.use_object_subspace_filter,
            subspace_config=subspace_cfg,
        )
        refined.to_csv(cache_csv, index=False)

        self.write_stage_cache(
            stage_name="refine",
            stage_cache_dir=cache_dir,
            data_signature=data_sig,
            config_signature=config_sig,
            artifact_paths={"objects_csv": str(cache_csv)},
            success=True,
            notes=["Refined object table with module-aware config."],
        )

        if self.cfg.shared_artifacts.enable_shared_artifact_registry and not objects_df.empty:
            shared_sig = self.shared_signature_refine(objects_df)
            self.shared_registry.mark_available(
                artifact_family="labeling.refine.outputs",
                shared_signature=shared_sig,
                producer_pipeline=self.pipeline_name,
                local_path=str(cache_csv),
                source_trial=getattr(self, "_active_trial_id", None),
                metadata={"n_rows": int(len(refined))},
                status=SharedArtifactStatus.VALID,
            )

        self.logger.info("Refined %d object rows", len(refined))
        return refined

    # -------------------------------------------------------------------------
    # Stage 3: site asset prep
    # -------------------------------------------------------------------------

    def validate_cached_naip(self, naip_path: Path) -> bool:
        try:
            with rasterio.open(naip_path) as src:
                samples = [
                    (0, 0),
                    (max(0, src.width // 2 - 8), max(0, src.height // 2 - 8)),
                    (max(0, src.width - 16), max(0, src.height - 16)),
                ]
                for x, y in samples:
                    w = min(16, src.width - x)
                    h = min(16, src.height - y)
                    if w <= 0 or h <= 0:
                        continue
                    src.read([1], window=Window(x, y, w, h))
            return True
        except Exception as e:
            self.logger.warning("Cached NAIP validation failed for %s: %s", naip_path, e)
            return False

    def prepare_site_assets(self, site: str, force_refresh: bool = False):
        config_signature = self._current_config_signature()
    
        # ------------------------------------------------------------
        # 1) Manifest-first reuse
        # ------------------------------------------------------------
        if not force_refresh:
            site_manifest = self._load_site_metadata_manifest(site)
            if site_manifest is not None:
                naip_local = Path(site_manifest["naip_local"])
                als_meta = site_manifest.get("als_metadata", [])
    
                if naip_local.exists() and self.validate_cached_naip(naip_local):
                    self.logger.info(
                        "Using site metadata manifest for site=%s | naip=%s | als_meta_rows=%d",
                        site, naip_local, len(als_meta)
                    )
                    return naip_local, als_meta
    
                self.logger.warning(
                    "Site metadata manifest found for site=%s but local NAIP path is missing/invalid: %s",
                    site, naip_local
                )
    
        # ------------------------------------------------------------
        # 2) Source inventory artifact
        # ------------------------------------------------------------
        inventory = None if force_refresh else self._load_source_inventory(site)
        if inventory is None:
            inventory = self._build_source_inventory(site)
            self._persist_source_inventory(site, inventory)
            self.logger.info("Built and persisted source inventory for site=%s", site)
        else:
            self.logger.info("Using cached source inventory for site=%s", site)
    
        site_base = inventory["site_base"]
        naip_name = inventory["naip"]["expected_name"]
        naip_url = inventory["naip"]["remote_url"]
    
        # ------------------------------------------------------------
        # 3) NAIP local cache with validation / repair / hydration
        # ------------------------------------------------------------
        naip_local = self._site_naip_cache_root(site) / naip_name
        naip_rel_path = self._render_rel_path(
            "site_naip_raster",
            site=site,
            filename=naip_name,
            #config_signature=self._current_config_signature(),
        )

        use_cached_naip = False
        if not force_refresh:
            repaired = self.repair_or_rehydrate_artifact(
                artifact_key="site_naip_raster",
                rel_path=naip_rel_path,
                local_path=naip_local,
                validator_key="site_naip_raster",
                recompute_fn=None,
            )
            if repaired is not None and self.validate_cached_naip(repaired):
                self.logger.info("Using cached/repaired/hydrated NAIP for site=%s: %s", site, repaired)
                naip_local = repaired
                use_cached_naip = True

        if not use_cached_naip:
            self.logger.info("Downloading NAIP for site=%s to %s", site, naip_local)
            download_file(naip_url, naip_local)
            if not self.validate_cached_naip(naip_local):
                raise RuntimeError(f"Downloaded NAIP for site={site} is unreadable: {naip_local}")

            self._persist_file_artifact(
                naip_local,
                "site_naip_raster",
                site=site,
                filename=naip_name,
            )

        if self.cfg.shared_artifacts.enable_shared_artifact_registry:
            self.shared_registry.mark_available(
                artifact_family="labeling.site_assets",
                shared_signature=self.shared_signature_site_assets(site),
                producer_pipeline=self.pipeline_name,
                rel_path=naip_rel_path,
                local_path=str(naip_local),
                metadata={"site": site, "asset_type": "naip"},
                status=SharedArtifactStatus.VALID,
            )

        # ------------------------------------------------------------
        # 4) ALS metadata: local cache -> remote artifact -> recompute
        # ------------------------------------------------------------
        als_cache_dir = self._site_als_cache_root(site)
        als_meta_json = als_cache_dir / "als_metadata.json"
        als_meta_rel_path = self._render_rel_path(
            "site_als_metadata_json",
            site=site,
        )

        als_meta = None

        if not force_refresh:
            repaired = self.repair_or_rehydrate_artifact(
                artifact_key="site_als_metadata_json",
                rel_path=als_meta_rel_path,
                local_path=als_meta_json,
                validator_key="site_als_metadata_json",
                recompute_fn=None,
            )
            if repaired is not None:
                self.logger.info("Using cached/repaired/hydrated ALS metadata for site=%s: %s", site, repaired)
                als_meta = json.loads(repaired.read_text(encoding="utf-8"))

        if als_meta is None:
            als_files = inventory.get("als", {}).get("files", [])
            if not als_files:
                als_url = f"{site_base}/{self.cfg.data.als_dir}"
                als_files = list_files_with_suffix(als_url, (".laz", ".las", ".copc.laz"))

            if not als_files:
                raise RuntimeError(f"No ALS files found for site '{site}'")

            self.logger.info("Downloading %d ALS file(s) for site=%s to extract metadata", len(als_files), site)

            records = []
            scratch_dir = Path(tempfile.mkdtemp(prefix=f"labeling_als_{site}_"))
            try:
                for entry in als_files:
                    local_path = scratch_dir / entry["name"]
                    download_file(entry["url"], local_path)
                    meta = extract_als_metadata(local_path)
                    meta["source_file"] = entry["name"]
                    records.append(meta)
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.warning("Failed to delete temporary ALS file %s: %s", local_path, e)
            finally:
                try:
                    scratch_dir.rmdir()
                except Exception:
                    pass

            als_meta_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
            als_meta = records

            self._persist_file_artifact(
                als_meta_json,
                "site_als_metadata_json",
                site=site,
            )
            self.logger.info("Cached ALS metadata for site=%s at %s", site, als_meta_json)

        if self.cfg.shared_artifacts.enable_shared_artifact_registry:
            self.shared_registry.mark_available(
                artifact_family="labeling.site_assets",
                shared_signature=self.shared_signature_site_assets(site),
                producer_pipeline=self.pipeline_name,
                rel_path=als_meta_rel_path,
                local_path=str(als_meta_json),
                metadata={"site": site, "asset_type": "als_metadata"},
                status=SharedArtifactStatus.VALID,
            )
    
        # ------------------------------------------------------------
        # 5) Persist site metadata manifest
        # ------------------------------------------------------------
        self._persist_site_metadata_manifest(site, naip_local=naip_local, als_meta=als_meta)
    
        self._mark_site_assets_shared_valid(
            site=site,
            naip_local=naip_local,
            als_meta=als_meta,
        )
        return naip_local, als_meta

    # -------------------------------------------------------------------------
    # Stage 4: transform lookup / cache
    # -------------------------------------------------------------------------

    def get_site_transform_index(self, site: str, force_refresh: bool = False) -> dict:
        if not force_refresh:
            payload = self._try_load_json_artifact(
                "transform_index",
                site=site,
                config_signature=self._current_config_signature(),
            )
            if payload is not None:
                self.logger.info("Using artifact-backed transform index for site=%s", site)
    
                # keep local mirror in the old cache location too
                index_path = self._site_transform_index_path(site)
                index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return payload
    
        remote_dir = f"{site_to_remote_base(self.cfg, site)}/{self.cfg.data.transformations_dir}"
        entries = list_files_with_suffix(remote_dir, (".txt",))
        index = {entry["name"]: entry["url"] for entry in entries}
    
        index_path = self._site_transform_index_path(site)
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        self._persist_json_artifact(
            index,
            "transform_index",
            site=site,
            config_signature=self._current_config_signature(),
        )
    
        self.logger.info("Cached transform index for site=%s with %d entries", site, len(index))
        return index

    def get_transform_local_cached(self, site: str, plot_id: str, force_refresh: bool = False) -> Path:
        transform_name = shrub_csv_to_transform_name(f"{plot_id}.csv")
        transform_local = self._site_transform_cache_root(site) / transform_name
    
        if (not force_refresh) and transform_local.exists():
            self.logger.info("Using cached transform for site=%s plot_id=%s", site, plot_id)
            return transform_local
    
        rel_path = self._render_rel_path(
            "transform_txt",
            site=site,
            config_signature=self._current_config_signature(),
            plot_id=plot_id,
        )
    
        if (not force_refresh) and self._remote_exists(rel_path):
            self.logger.info("Pulling transform from artifact store for site=%s plot_id=%s", site, plot_id)
            pulled = self.hydrate_and_validate_artifact(
                rel_path=rel_path,
                local_path=transform_local,
                artifact_key="transform_txt",
                reason=f"get_transform_local_cached:{site}:{plot_id}",
            )
            if pulled is not None:
                return Path(pulled)
    
        transform_index = self.get_site_transform_index(site, force_refresh=force_refresh)
    
        chosen_name = None
        transform_url = None
    
        if transform_name in transform_index:
            chosen_name = transform_name
            transform_url = transform_index[transform_name]
            self.logger.info("Downloading exact-match transform for site=%s plot_id=%s", site, plot_id)
        else:
            fallback_names = sorted(
                name for name in transform_index
                if name.startswith(plot_id) and name.endswith("toALS.txt")
            )
            if fallback_names:
                chosen_name = fallback_names[0]
                transform_url = transform_index[chosen_name]
                self.logger.warning(
                    "Exact transform missing for site=%s plot_id=%s; using fallback transform %s",
                    site, plot_id, chosen_name
                )
    
        if transform_url is None:
            raise FileNotFoundError(
                f"No transform file found for site={site}, plot_id={plot_id}. "
                f"Expected exact name {transform_name} or fallback starting with {plot_id}."
            )
    
        download_file(transform_url, transform_local)
    
        # Persist through artifact store under the plot_id-based rel path
        self._persist_file_artifact(
            transform_local,
            "transform_txt",
            site=site,
            config_signature=self._current_config_signature(),
            plot_id=plot_id,
        )
    
        return transform_local

    # -------------------------------------------------------------------------
    # Stage 5: output paths / output cache
    # -------------------------------------------------------------------------

    def artifact_paths_for_plot(self, site: str, plot_id: str, source_version: str) -> dict:
        sv = str(source_version)

        masks_dir = self.cfg.output.labeling_root / "masks" / site / sv
        conf_dir = self.cfg.output.labeling_root / "confidence" / site / sv
        objid_dir = self.cfg.output.labeling_root / "object_id" / site / sv
        objects_dir = self.cfg.output.labeling_root / "objects" / site / sv
        qa_dir = self.cfg.output.labeling_root / "qa" / site / sv

        for d in [masks_dir, conf_dir, objid_dir, objects_dir, qa_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return {
            "binary_path": masks_dir / f"{plot_id}_mask.tif",
            "confidence_path": conf_dir / f"{plot_id}_confidence.tif",
            "object_id_path": objid_dir / f"{plot_id}_object_id.tif",
            "object_table_path": objects_dir / f"{plot_id}_objects.csv",
            "qa_path": qa_dir / f"{plot_id}_overlay.png",
            "masks_dir": masks_dir,
            "conf_dir": conf_dir,
        }

    def has_successful_sprint4_outputs(self, site: str, plot_id: str, source_version: str) -> bool:
        paths = self.artifact_paths_for_plot(site, plot_id, source_version)

        required = [
            paths["binary_path"],
            paths["confidence_path"],
            paths["object_id_path"],
            paths["object_table_path"],
            paths["qa_path"],
        ]
        if not all(p.exists() for p in required):
            return False

        for res in self.pipeline_config.multires:
            if float(res) == 1.0:
                continue
            multires_binary = paths["masks_dir"] / f"{plot_id}_mask_{res:g}m.tif"
            multires_conf = paths["conf_dir"] / f"{plot_id}_confidence_{res:g}m.tif"
            if not multires_binary.exists() or not multires_conf.exists():
                return False

        return True

    def build_artifact_rows_from_disk(
        self,
        *,
        site: str,
        plot_id: str,
        source_version: str,
        objects_df: pd.DataFrame,
    ) -> pd.DataFrame:
        paths = self.artifact_paths_for_plot(site, plot_id, source_version)

        date_token = objects_df["ptx_date_token"].iloc[0] if "ptx_date_token" in objects_df.columns and len(objects_df) else None
        variant = objects_df["variant"].iloc[0] if "variant" in objects_df.columns and len(objects_df) else None
        plot_key = extract_plot_key(plot_id)

        rows = []
        for res in self.pipeline_config.multires:
            if float(res) == 1.0:
                multires_binary_path = paths["binary_path"]
                multires_conf_path = paths["confidence_path"]
            else:
                multires_binary_path = paths["masks_dir"] / f"{plot_id}_mask_{res:g}m.tif"
                multires_conf_path = paths["conf_dir"] / f"{plot_id}_confidence_{res:g}m.tif"

            rows.append(
                {
                    "site_id": site,
                    "plot_id": plot_id,
                    "plot_key": plot_key,
                    "variant": variant,
                    "label_variant": "base",
                    "resolution_m": float(res),
                    "binary_mask_path": str(multires_binary_path),
                    "confidence_mask_path": str(multires_conf_path),
                    "object_id_raster_path": str(paths["object_id_path"]),
                    "object_table_path": str(paths["object_table_path"]),
                    "qa_overlay_path": str(paths["qa_path"]),
                    "n_objects": int(len(objects_df)),
                    "n_valid_objects": int(objects_df["valid_object"].sum()) if "valid_object" in objects_df.columns else int(len(objects_df)),
                    "date_token": date_token,
                    "source_version": source_version,
                }
            )

        return pd.DataFrame(rows)

    def append_sprint4_artifacts_manifest(self, new_artifacts_df: pd.DataFrame) -> pd.DataFrame:
        manifest_csv = self.sprint4_manifest_csv
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)

        if manifest_csv.exists():
            old_df = pd.read_csv(manifest_csv)
            combined = pd.concat([old_df, new_artifacts_df], ignore_index=True)
        else:
            combined = new_artifacts_df.copy()

        dedup_subset = ["site_id", "plot_id", "source_version", "resolution_m", "label_variant"]
        combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
        combined.to_csv(manifest_csv, index=False)
        return combined

    # -------------------------------------------------------------------------
    # Stage 6: one plot/source-version transfer
    # -------------------------------------------------------------------------

    def process_one_object_group_to_labels(
        self,
        *,
        site: str,
        plot_id: str,
        source_version: str,
        objects_group: pd.DataFrame,
        naip_path: Path | None,
        als_meta: list | None,
        force_rerun: bool = False,
    ):
        self.logger.info(
            "Processing Sprint 4 transfer | site=%s | plot_id=%s | source_version=%s | n_objects=%d",
            site, plot_id, source_version, len(objects_group)
        )

        data_sig = self.transfer_stage_data_signature(
            site=site,
            plot_id=plot_id,
            source_version=source_version,
            objects_group=objects_group,
        )
        config_sig = self.stage_config_signature("transfer") + "__" + self.stage_config_signature("rasterize")
        cache_dir = self.transfer_stage_cache_dir(site, plot_id, source_version, data_sig, config_sig)

        paths = self.artifact_paths_for_plot(site, plot_id, source_version)
        cache_object_csv = cache_dir / "objects_transferred.csv"
        cache_artifacts_csv = cache_dir / "artifacts.csv"

        if (
            (not force_rerun)
            and self.validate_stage_cache(
                stage_name="transfer",
                stage_cache_dir=cache_dir,
                expected_data_signature=data_sig,
                expected_config_signature=config_sig,
            )
            and cache_object_csv.exists()
            and cache_artifacts_csv.exists()
        ):
            self.logger.info(
                "Using module-aware Sprint 4 cache | site=%s | plot_id=%s | source_version=%s | data=%s | config=%s",
                site, plot_id, source_version, data_sig, config_sig
            )
            cached_objects = pd.read_csv(cache_object_csv)
            cached_artifacts = pd.read_csv(cache_artifacts_csv)

            self.reconcile_stage_artifacts_with_storage_policy(
                stage_name="transfer",
                artifact_paths=self.read_stage_cache_artifact_paths(cache_dir),
            )
            return cached_objects, cached_artifacts

        if naip_path is None or als_meta is None:
            raise ValueError("naip_path and als_meta are required when no Sprint 4 output cache is available.")

        transform_local = self.get_transform_local_cached(site, plot_id, force_refresh=force_rerun)

        objects = objects_group.copy()
        objects, tile = transform_objects_to_als(objects, transform_local, als_meta)

        tile_wkt = tile.get("srs_wkt")
        if not tile_wkt:
            raise ValueError(f"ALS tile {tile.get('source_file')} is missing CRS/WKT metadata.")

        objects, grid = align_objects_to_naip(objects, naip_path, tile_wkt, self.cfg)
        binary, confidence, object_id = rasterize_objects(
            objects,
            grid,
            self.cfg,
            boundary_mode=self.pipeline_config.boundary_confidence_mode,
            apply_mask_subspace_reduction=self.pipeline_config.use_object_subspace_filter,
            min_component_pixels=self.pipeline_config.subspace_min_component_pixels,
        )

        paths = self.artifact_paths_for_plot(site, plot_id, source_version)

        write_single_band_geotiff(paths["binary_path"], binary, grid, dtype="uint8", nodata=self.cfg.raster.background_value)
        write_single_band_geotiff(paths["confidence_path"], confidence, grid, dtype="float32", nodata=self.cfg.raster.confidence_background)
        write_single_band_geotiff(paths["object_id_path"], object_id, grid, dtype="int32", nodata=0)
        export_table(objects, paths["object_table_path"])

        try:
            create_overlay_figure(naip_path, paths["binary_path"], paths["qa_path"])
        except Exception as e:
            if self.pipeline_config.nonfatal_qa_overlay:
                self.logger.warning(
                    "QA overlay failed for site=%s plot_id=%s source_version=%s: %s",
                    site, plot_id, source_version, e
                )
            else:
                raise

        multires_artifact_paths = {}
        for res in self.pipeline_config.multires:
            if float(res) == 1.0:
                continue

            b_res, b_grid = resample_single_band(binary, grid, res)
            c_res, c_grid = resample_single_band(confidence, grid, res)

            multires_binary_path = paths["masks_dir"] / f"{plot_id}_mask_{res:g}m.tif"
            multires_conf_path = paths["conf_dir"] / f"{plot_id}_confidence_{res:g}m.tif"

            write_single_band_geotiff(multires_binary_path, b_res, b_grid, dtype="uint8", nodata=self.cfg.raster.background_value)
            write_single_band_geotiff(multires_conf_path, c_res, c_grid, dtype="float32", nodata=self.cfg.raster.confidence_background)

        binary_rel_path, binary_remote_ref = self._persist_file_artifact(
            paths["binary_path"],
            "binary_mask",
            site=site,
            config_signature=self._current_config_signature(),
            source_version=str(source_version),
            plot_id=plot_id,
        )
        conf_rel_path, conf_remote_ref = self._persist_file_artifact(
            paths["confidence_path"],
            "confidence_mask",
            site=site,
            config_signature=self._current_config_signature(),
            source_version=str(source_version),
            plot_id=plot_id,
        )
        object_id_rel_path, object_id_remote_ref = self._persist_file_artifact(
            paths["object_id_path"],
            "object_id_raster",
            site=site,
            config_signature=self._current_config_signature(),
            source_version=str(source_version),
            plot_id=plot_id,
        )
        object_table_rel_path, object_table_remote_ref = self._persist_file_artifact(
            paths["object_table_path"],
            "object_table",
            site=site,
            config_signature=self._current_config_signature(),
            source_version=str(source_version),
            plot_id=plot_id,
        )

        qa_rel_path = None
        qa_remote_ref = None
        if paths["qa_path"].exists():
            qa_rel_path, qa_remote_ref = self._persist_file_artifact(
                paths["qa_path"],
                "qa_overlay",
                site=site,
                config_signature=self._current_config_signature(),
                source_version=str(source_version),
                plot_id=plot_id,
            )

        artifacts_df = self.build_artifact_rows_from_disk(
            site=site,
            plot_id=plot_id,
            source_version=source_version,
            objects_df=objects,
        )

        # attach rel paths for downstream hydration/reuse
        artifacts_df["binary_mask_rel_path"] = binary_rel_path
        artifacts_df["confidence_mask_rel_path"] = conf_rel_path
        artifacts_df["object_id_raster_rel_path"] = object_id_rel_path
        artifacts_df["object_table_rel_path"] = object_table_rel_path
        artifacts_df["qa_overlay_rel_path"] = qa_rel_path

        objects.to_csv(cache_object_csv, index=False)
        artifacts_df.to_csv(cache_artifacts_csv, index=False)

        self.write_stage_cache(
            stage_name="transfer",
            stage_cache_dir=cache_dir,
            data_signature=data_sig,
            config_signature=config_sig,
            artifact_paths={
                "objects_csv": str(cache_object_csv),
                "artifacts_csv": str(cache_artifacts_csv),
                "binary_rel_path": binary_rel_path,
                "confidence_rel_path": conf_rel_path,
                "object_id_rel_path": object_id_rel_path,
                "object_table_rel_path": object_table_rel_path,
                "qa_rel_path": qa_rel_path,
            },
            success=True,
            notes=["Transfer+rasterize outputs cached with remote-backed artifact rel paths."],
        )

        if self.cfg.shared_artifacts.enable_shared_artifact_registry:
            shared_sig = self.shared_signature_transfer(
                site=site,
                plot_id=plot_id,
                source_version=source_version,
                objects_group=objects_group,
            )
            self.shared_registry.mark_available(
                artifact_family="labeling.transfer.outputs",
                shared_signature=shared_sig,
                producer_pipeline=self.pipeline_name,
                local_path=str(cache_artifacts_csv),
                source_trial=getattr(self, "_active_trial_id", None),
                metadata={
                    "site_id": site,
                    "plot_id": plot_id,
                    "source_version": source_version,
                    "n_rows": int(len(artifacts_df)),
                },
                status=SharedArtifactStatus.VALID,
            )

        self.logger.info(
            "Finished Sprint 4 transfer | site=%s | plot_id=%s | source_version=%s",
            site, plot_id, source_version
        )
        return objects, artifacts_df

    # -------------------------------------------------------------------------
    # Stage 7: full site loop
    # -------------------------------------------------------------------------

    def site_has_pending_sprint4_work(self, site: str, site_objects: pd.DataFrame) -> bool:
        grouped = site_objects.groupby(["plot_id", "source_version"], dropna=False)
        for (plot_id, source_version), group_df in grouped:
            self.logger.info(
                "Transfer cache check | site=%s | plot_id=%s | source_version=%s | valid_cache=%s",
                site, plot_id, source_version,
                self.has_valid_transfer_stage_cache(
                    site=site,
                    plot_id=plot_id,
                    source_version=source_version,
                    objects_group=group_df,
                ),
            )
            if not self.has_valid_transfer_stage_cache(
                site=site,
                plot_id=plot_id,
                source_version=source_version,
                objects_group=group_df,
            ):
                return True
        return False

    def stage_transfer_all_sites(self, objects_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_site_objects = []
        all_site_artifacts = []

        for site in self.cfg.sites:
            self.logger.info("=" * 100)
            self.logger.info("Processing full Sprint 4 site loop for site=%s", site)

            site_objects = objects_df[objects_df["site_id"] == site].copy()
            if site_objects.empty:
                self.logger.info("No objects available for site=%s; skipping.", site)
                continue

            if (not self.pipeline_config.force_rerun_sprint4) and (not self.site_has_pending_sprint4_work(site, site_objects)):
                self.logger.info("All Sprint 4 outputs already cached for site=%s; skipping site asset prep.", site)

                grouped = site_objects.groupby(["site_id", "plot_id", "source_version"], dropna=False)
                for (_, plot_id, source_version), group_df in grouped:
                    cached_objects, cached_artifacts = self.process_one_object_group_to_labels(
                        site=site,
                        plot_id=plot_id,
                        source_version=source_version,
                        objects_group=group_df,
                        naip_path=None,
                        als_meta=None,
                        force_rerun=False,
                    )
                    all_site_objects.append(cached_objects)
                    all_site_artifacts.append(cached_artifacts)
                    self.append_sprint4_artifacts_manifest(cached_artifacts)
                continue

            try:
                naip_local, als_meta = self.prepare_site_assets(
                    site,
                    force_refresh=self.pipeline_config.force_refresh_site_assets,
                )

                grouped = site_objects.groupby(["site_id", "plot_id", "source_version"], dropna=False)
                for (_, plot_id, source_version), group_df in grouped:
                    try:
                        objects_out, artifacts_out = self.process_one_object_group_to_labels(
                            site=site,
                            plot_id=plot_id,
                            source_version=source_version,
                            objects_group=group_df,
                            naip_path=naip_local,
                            als_meta=als_meta,
                            force_rerun=self.pipeline_config.force_rerun_sprint4,
                        )
                        all_site_objects.append(objects_out)
                        all_site_artifacts.append(artifacts_out)
                        self.append_sprint4_artifacts_manifest(artifacts_out)
                    except Exception:
                        self.logger.exception(
                            "Sprint 4 failed | site=%s | plot_id=%s | source_version=%s",
                            site, plot_id, source_version
                        )
                        continue
            except Exception:
                self.logger.exception("Site-level Sprint 4 prep failed for site=%s", site)
                continue

        objects_all = pd.concat(all_site_objects, ignore_index=True) if all_site_objects else pd.DataFrame()
        artifacts_all = pd.concat(all_site_artifacts, ignore_index=True) if all_site_artifacts else pd.DataFrame()
        artifacts_all = deduplicate_artifact_table(artifacts_all)
        return objects_all, artifacts_all

    # -------------------------------------------------------------------------
    # Finalize / save
    # -------------------------------------------------------------------------

    def finalize_outputs(
        self,
        objects_df: pd.DataFrame,
        artifacts_df: pd.DataFrame,
        *,
        signature: str | None = None,
        write_global: bool = True,
    ) -> tuple[Path, Path]:
        signature = signature or self.config_signature()
    
        run_summary_dir = self.run_summary_dir(signature)
        objects_csv = run_summary_dir / "objects_all.csv"
        artifacts_csv = run_summary_dir / "artifacts_all.csv"
    
        objects_df.to_csv(objects_csv, index=False)
        artifacts_df.to_csv(artifacts_csv, index=False)
    
        self._persist_file_artifact(
            objects_csv,
            "objects_summary",
            config_signature=signature,
        )
        self._persist_file_artifact(
            artifacts_csv,
            "artifacts_summary",
            config_signature=signature,
        )
    
        if write_global:
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            global_objects_csv = self.summary_dir / "objects_all.csv"
            global_artifacts_csv = self.summary_dir / "artifacts_all.csv"
            objects_df.to_csv(global_objects_csv, index=False)
            artifacts_df.to_csv(global_artifacts_csv, index=False)
    
        self.logger.info(
            "Saved labeling summaries for signature=%s to %s and %s",
            signature, objects_csv, artifacts_csv
        )
        return objects_csv, artifacts_csv

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run_work_unit(self, unit: dict, *, trial_id: str, state=None):
        stage_name = unit["stage_name"]

        self._active_trial_id = trial_id

        if stage_name == "sprint3":
            return self.stage_run_sprint3()

        if stage_name == "standardize":
            manifest_source = self.sprint3_manifest_csv
            return self.stage_load_and_standardize_objects(manifest_csv=manifest_source)

        if stage_name == "refine":
            manifest_source = self.sprint3_manifest_csv
            objects_std = self.stage_load_and_standardize_objects(manifest_csv=manifest_source)
            return self.stage_refine_objects(objects_std)

        if stage_name == "transfer":
            manifest_source = self.sprint3_manifest_csv
            objects_std = self.stage_load_and_standardize_objects(manifest_csv=manifest_source)
            objects_refined = self.stage_refine_objects(objects_std)

            site_id = unit["site_id"]
            plot_id = unit["plot_id"]
            source_version = unit["source_version"]

            group_df = objects_refined[
                (objects_refined["site_id"] == site_id)
                & (objects_refined["plot_id"] == plot_id)
                & (objects_refined["source_version"] == source_version)
            ].copy()

            if group_df.empty:
                raise ValueError(f"No refined objects found for transfer unit {unit['unit_id']}")

            naip_local, als_meta = self.prepare_site_assets(
                site_id,
                force_refresh=self.pipeline_config.force_refresh_site_assets,
            )

            return self.process_one_object_group_to_labels(
                site=site_id,
                plot_id=plot_id,
                source_version=source_version,
                objects_group=group_df,
                naip_path=naip_local,
                als_meta=als_meta,
                force_rerun=self.pipeline_config.force_rerun_sprint4,
            )

        if stage_name == "site_assets":
            site_id = unit["site_id"]
            naip_local, als_meta = self.prepare_site_assets(
                site_id,
                force_refresh=self.pipeline_config.force_refresh_site_assets,
            )
        
            self._mark_site_assets_shared_valid(
                site=site_id,
                naip_local=naip_local,
                als_meta=als_meta,
                trial_id=trial_id,
            )
            return {
                "stage_name": "site_assets",
                "site_id": site_id,
                "naip_local": str(naip_local),
                "als_metadata_rows": len(als_meta or []),
            }

        raise NotImplementedError(f"Unknown labeling work unit stage={stage_name}")

    def run(
        self,
        *,
        manifest_csv: str | Path | None = None,
        notes: list[str] | None = None,
    ) -> PipelineRunResult:
        notes = notes or []
    
        self.sync_artifact_registry_if_available()
        if self.cfg.coordination.enabled and self.cfg.coordination.sync_registry_before_claim:
            self.coordination.sync()
    
        signature = self.config_signature()
        runtime_report = self.runtime_report()
    
        existing = self.try_load_existing_run(signature=signature)

        if existing is not None:
            reuse_ok, reuse_reason = self.should_reuse_existing_run(
                existing,
                runtime_report=runtime_report,
            )
            if reuse_ok:
                self.logger.info(
                    "Reusing existing labeling run for signature=%s | reason=%s",
                    signature, reuse_reason
                )
                self.enforce_storage_policy_for_existing_run(
                    signature=signature,
                    existing=existing,
                )
                return existing
        
            self.logger.info(
                "Existing labeling run will be resumed instead of reused for signature=%s | reason=%s",
                signature, reuse_reason
            )

            self.enforce_storage_policy_for_existing_run(
                signature=signature,
                existing=existing,
            )
    
        runtime_notes = [
            f"runtime_image={runtime_report.detected_image_key}",
            f"runtime_capabilities={','.join(runtime_report.capabilities)}",
        ]
        notes = list(notes) + runtime_notes
    
        executed_stages = []
        skipped_stages = {}
    
        # Stage: sprint3
        sprint3_ok, sprint3_elig = self.stage_is_eligible("sprint3", runtime_report=runtime_report)
        if sprint3_ok:
            sprint3_manifest_df = self.stage_run_sprint3()
            executed_stages.append("sprint3")
        else:
            sprint3_manifest_df = pd.DataFrame()
            skipped_stages["sprint3"] = self.stage_block_reason("sprint3", runtime_report=runtime_report)
            self.logger.info("Skipping stage=sprint3 | reason=%s", skipped_stages["sprint3"])
    
        manifest_source = manifest_csv if manifest_csv is not None else self.sprint3_manifest_csv
    
        # Stage: standardize
        standardize_ok, _ = self.stage_is_eligible("standardize", runtime_report=runtime_report)
        if standardize_ok:
            objects_std = self.stage_load_and_standardize_objects(manifest_csv=manifest_source)
            executed_stages.append("standardize")
        else:
            objects_std = pd.DataFrame()
            skipped_stages["standardize"] = self.stage_block_reason("standardize", runtime_report=runtime_report)
            self.logger.info("Skipping stage=standardize | reason=%s", skipped_stages["standardize"])
    
        # Stage: refine
        refine_ok, _ = self.stage_is_eligible("refine", runtime_report=runtime_report)
        if refine_ok and not objects_std.empty:
            objects_refined = self.stage_refine_objects(objects_std)
            executed_stages.append("refine")
        else:
            objects_refined = objects_std.copy()
            if not refine_ok:
                skipped_stages["refine"] = self.stage_block_reason("refine", runtime_report=runtime_report)
                self.logger.info("Skipping stage=refine | reason=%s", skipped_stages["refine"])
    
        # Stage: transfer/rasterize
        transfer_ok, _ = self.stage_is_eligible("transfer", runtime_report=runtime_report)
        rasterize_ok, _ = self.stage_is_eligible("rasterize", runtime_report=runtime_report)
    
        if transfer_ok and rasterize_ok and not objects_refined.empty:
            objects_all, artifacts_all = self.stage_transfer_all_sites(objects_refined)
            executed_stages.extend(["transfer", "rasterize"])
        else:
            objects_all, artifacts_all = pd.DataFrame(), pd.DataFrame()
            if not transfer_ok:
                skipped_stages["transfer"] = self.stage_block_reason("transfer", runtime_report=runtime_report)
                self.logger.info("Skipping stage=transfer | reason=%s", skipped_stages["transfer"])
            if not rasterize_ok:
                skipped_stages["rasterize"] = self.stage_block_reason("rasterize", runtime_report=runtime_report)
                self.logger.info("Skipping stage=rasterize | reason=%s", skipped_stages["rasterize"])
    
        objects_csv, artifacts_csv = self.finalize_outputs(
            objects_all,
            artifacts_all,
            signature=signature,
            write_global=True,
        )
    
        success = (not objects_all.empty) and (not artifacts_all.empty)
        if success:
            status = "success"
        elif executed_stages and skipped_stages:
            status = "partial_runtime_gated"
        elif executed_stages:
            status = "partial_or_empty_outputs"
        else:
            status = "no_eligible_stages"
    
        stage_cache_root = self.cfg.output.labeling_root / "stage_cache"
        stage_cache_manifest_count = len(list(stage_cache_root.rglob("stage_cache_manifest.json"))) if stage_cache_root.exists() else 0
    
        result = PipelineRunResult(
            pipeline_name=self.pipeline_name,
            success=success,
            status=status,
            raster_outputs=CanonicalRasterOutputs(
                labels=artifacts_all,
                qa_overlays=artifacts_all[["site_id", "plot_id", "qa_overlay_path"]].copy()
                if not artifacts_all.empty and "qa_overlay_path" in artifacts_all.columns
                else None,
            ),
            object_outputs=CanonicalObjectOutputs(
                objects=objects_all,
                source_provenance=objects_all[
                    [c for c in ["site_id", "plot_id", "source_version", "source_file"] if c in objects_all.columns]
                ].copy()
                if not objects_all.empty
                else None,
            ),
            qa_outputs={
                "objects_csv": str(objects_csv),
                "artifacts_csv": str(artifacts_csv),
                "artifacts_manifest_csv": str(self.sprint4_manifest_csv),
                "runtime_report": asdict(runtime_report),
                "executed_stages": executed_stages,
                "skipped_stages": skipped_stages,
            },
            metrics={
                "n_object_rows": int(len(objects_all)),
                "n_artifact_rows": int(len(artifacts_all)),
                "n_sites": int(objects_all["site_id"].nunique()) if "site_id" in objects_all.columns and not objects_all.empty else 0,
                "stage_cache_manifest_count": int(stage_cache_manifest_count),
                "n_executed_stages": int(len(executed_stages)),
                "n_skipped_stages": int(len(skipped_stages)),
            },
            notes=notes,
        )
    
        self.save_run_result(result, subdir=signature)
        self.save_pipeline_run_manifest(result, signature=signature)
    
        if self.cfg.coordination.enabled and self.cfg.coordination.sync_registry_after_stage:
            self.coordination.sync()
    
        return result