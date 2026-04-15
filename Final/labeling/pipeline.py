from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import tempfile
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
)
from Final.artifact_store import (
    LocalArtifactStore,
    DriveRegistryArtifactStore,
    HybridArtifactStore,
)
from Final.pipeline_caching import hash_payload
from Final.gating import (
    QACheckSpec,
    QACheckResult,
    ModuleQAProfile,
    ModuleQAEvaluation,
)
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
class LabelingStorageConfig:
    enable_local_store: bool = True
    enable_drive_store: bool = False
    use_hybrid_store: bool = False

    push_large_artifacts_to_remote: bool = False
    prune_local_after_remote_push: bool = False
    verify_remote_before_prune: bool = True

    local_storage_root: Path | None = None
    drive_registry_path: Path | None = None
    drive_config_path: Path | None = None
    client_secrets_path: Path | None = None
    credentials_path: Path | None = None

    fail_if_drive_missing: bool = False

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
    storage: LabelingStorageConfig = field(default_factory=LabelingStorageConfig)
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

        storage = self.pipeline_config.storage

        project_root = cfg.data.project_root
        if storage.local_storage_root is None:
            storage.local_storage_root = cfg.output.labeling_root / "artifact_store_local"
        if storage.drive_registry_path is None:
            storage.drive_registry_path = project_root / "Final" / "artifact_registry.yaml"
        if storage.drive_config_path is None:
            storage.drive_config_path = project_root / "drive_config.yaml"
        if storage.client_secrets_path is None:
            storage.client_secrets_path = project_root / "client_secrets.json"
        if storage.credentials_path is None:
            storage.credentials_path = project_root / "pydrive_credentials.json"

        self.artifact_store = self._build_artifact_store()

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
            ),
            "labeling.standardize.base": ModuleSpec(
                key="labeling.standardize.base",
                stage_name="standardize",
            ),
            "labeling.refine.shape_descriptors": ModuleSpec(
                key="labeling.refine.shape_descriptors",
                stage_name="refine",
                enabled_key="use_shape_descriptors",
            ),
            "labeling.refine.temporal_confidence": ModuleSpec(
                key="labeling.refine.temporal_confidence",
                stage_name="refine",
                enabled_key="use_temporal_confidence",
                param_keys=("site_reference_dates",),
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
            ),
            "labeling.transfer.base": ModuleSpec(
                key="labeling.transfer.base",
                stage_name="transfer",
            ),
            "labeling.rasterize.mode": ModuleSpec(
                key="labeling.rasterize.mode",
                stage_name="rasterize",
                variant_key="rasterization_mode",
            ),
            "labeling.boundary_confidence": ModuleSpec(
                key="labeling.boundary_confidence",
                stage_name="rasterize",
                enabled_key="use_boundary_confidence",
                variant_key="boundary_confidence_mode",
            ),
            "labeling.mask_subspace_reduction": ModuleSpec(
                key="labeling.mask_subspace_reduction",
                stage_name="rasterize",
                enabled_key="use_object_subspace_filter",
                param_keys=("subspace_min_component_pixels",),
            ),
            "labeling.multires_export": ModuleSpec(
                key="labeling.multires_export",
                stage_name="rasterize",
                param_keys=("multires",),
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
        storage = self.pipeline_config.storage

        local_store = LocalArtifactStore(
            repo_root=self.cfg.data.project_root,
            storage_root=Path(storage.local_storage_root),
        )

        drive_store = None
        if storage.enable_drive_store:
            missing = [
                str(p) for p in [
                    storage.client_secrets_path,
                    storage.drive_config_path,
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
                    registry_path=Path(storage.drive_registry_path),
                    drive_config_path=Path(storage.drive_config_path),
                    client_secrets_path=Path(storage.client_secrets_path),
                    credentials_path=Path(storage.credentials_path),
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

    def _remote_exists(self, rel_path: str) -> bool:
        try:
            return self.artifact_store.exists(rel_path)
        except Exception:
            return False

    def _push_if_needed(self, local_path: Path, artifact_key: str, rel_path: str) -> str | None:
        spec = self._artifact_spec(artifact_key)
        storage = self.pipeline_config.storage

        should_push = (
            storage.push_large_artifacts_to_remote
            and spec.storage_tier in {StorageTier.LOCAL_THEN_REMOTE, StorageTier.REMOTE_ONLY}
            and isinstance(self.artifact_store, (HybridArtifactStore, DriveRegistryArtifactStore))
        )
        if not should_push:
            return None

        self.logger.info("PUSH ARTIFACT | key=%s | rel_path=%s", artifact_key, rel_path)
        return self.artifact_store.push(local_path, rel_path=rel_path)

    def _prune_if_allowed(self, local_path: Path, artifact_key: str, rel_path: str) -> None:
        spec = self._artifact_spec(artifact_key)
        storage = self.pipeline_config.storage

        if not storage.prune_local_after_remote_push:
            return
        if not spec.prune_local_after_push:
            return
        if storage.verify_remote_before_prune and not self._remote_exists(rel_path):
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

    def _try_load_json_artifact(self, artifact_key: str, **fmt) -> dict | None:
        rel_path = self._render_rel_path(artifact_key, **fmt)
        local_path = self._local_artifact_path(rel_path)

        if local_path.exists():
            return json.loads(local_path.read_text(encoding="utf-8"))

        if self._remote_exists(rel_path):
            pulled = self.artifact_store.pull(rel_path, local_path=local_path)
            return json.loads(Path(pulled).read_text(encoding="utf-8"))

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

        # bridge mode: if global summaries exist, adopt them into this signature
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
        self.logger.info("Saved labeling run manifest to %s", path)
        return path

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

        self.logger.info("Refined %d object rows", len(refined))
        return refined

    # -------------------------------------------------------------------------
    # Stage 3: site asset prep
    # -------------------------------------------------------------------------

    def validate_cached_naip(self, naip_path: Path) -> bool:
        try:
            with rasterio.open(naip_path) as src:
                h = max(1, min(16, src.height))
                w = max(1, min(16, src.width))
                src.read([1], window=Window(0, 0, w, h))
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
        # 3) NAIP local cache with validation
        # ------------------------------------------------------------
        naip_local = self._site_naip_cache_root(site) / naip_name
    
        use_cached_naip = False
        if (not force_refresh) and naip_local.exists():
            if self.validate_cached_naip(naip_local):
                self.logger.info("Using cached NAIP for site=%s: %s", site, naip_local)
                use_cached_naip = True
            else:
                self.logger.warning("Deleting corrupt cached NAIP for site=%s: %s", site, naip_local)
                naip_local.unlink(missing_ok=True)
    
        if not use_cached_naip:
            self.logger.info("Downloading NAIP for site=%s to %s", site, naip_local)
            download_file(naip_url, naip_local)
            if not self.validate_cached_naip(naip_local):
                raise RuntimeError(f"Downloaded NAIP for site={site} is unreadable: {naip_local}")
    
        # ------------------------------------------------------------
        # 4) ALS metadata: local cache -> remote artifact -> recompute
        # ------------------------------------------------------------
        als_cache_dir = self._site_als_cache_root(site)
        als_meta_json = als_cache_dir / "als_metadata.json"
    
        als_meta = None
    
        if (not force_refresh) and als_meta_json.exists():
            self.logger.info("Using cached ALS metadata for site=%s: %s", site, als_meta_json)
            als_meta = json.loads(als_meta_json.read_text(encoding="utf-8"))
    
        if als_meta is None and not force_refresh:
            artifact_rows = self._load_cached_als_metadata_artifact(site)
            if artifact_rows is not None:
                self.logger.info("Using remote/local artifact-backed ALS metadata for site=%s", site)
                als_meta = artifact_rows
                als_meta_json.write_text(json.dumps(als_meta, indent=2), encoding="utf-8")
    
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
            self._persist_als_metadata_artifact(site, als_meta)
            self.logger.info("Cached ALS metadata for site=%s at %s", site, als_meta_json)
    
        # ------------------------------------------------------------
        # 5) Persist site metadata manifest
        # ------------------------------------------------------------
        self._persist_site_metadata_manifest(site, naip_local=naip_local, als_meta=als_meta)
    
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
            pulled = self.artifact_store.pull(rel_path, local_path=transform_local)
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

        for res in self.pipeline_config.multires:
            if float(res) == 1.0:
                continue

            b_res, b_grid = resample_single_band(binary, grid, res)
            c_res, c_grid = resample_single_band(confidence, grid, res)

            multires_binary_path = paths["masks_dir"] / f"{plot_id}_mask_{res:g}m.tif"
            multires_conf_path = paths["conf_dir"] / f"{plot_id}_confidence_{res:g}m.tif"

            write_single_band_geotiff(multires_binary_path, b_res, b_grid, dtype="uint8", nodata=self.cfg.raster.background_value)
            write_single_band_geotiff(multires_conf_path, c_res, c_grid, dtype="float32", nodata=self.cfg.raster.confidence_background)

        artifacts_df = self.build_artifact_rows_from_disk(
            site=site,
            plot_id=plot_id,
            source_version=source_version,
            objects_df=objects,
        )

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
                "binary_path": str(paths["binary_path"]),
                "confidence_path": str(paths["confidence_path"]),
                "object_id_path": str(paths["object_id_path"]),
                "qa_path": str(paths["qa_path"]),
            },
            success=True,
            notes=["Transfer+rasterize outputs cached with module-aware signatures."],
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

        if write_global:
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            global_objects_csv = self.summary_dir / "objects_all.csv"
            global_artifacts_csv = self.summary_dir / "artifacts_all.csv"
            objects_df.to_csv(global_objects_csv, index=False)
            artifacts_df.to_csv(global_artifacts_csv, index=False)

        self.logger.info(
            "Saved labeling summaries for signature=%s to %s and %s",
            signature,
            objects_csv,
            artifacts_csv,
        )
        return objects_csv, artifacts_csv

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run(
        self,
        *,
        manifest_csv: str | Path | None = None,
        notes: list[str] | None = None,
    ) -> PipelineRunResult:
        notes = notes or []

        signature = self.config_signature()

        existing = self.try_load_existing_run(signature=signature)
        if existing is not None:
            self.logger.info("Resuming/reusing existing labeling run for signature=%s", signature)
            return existing

        sprint3_manifest_df = self.stage_run_sprint3()
        manifest_source = manifest_csv if manifest_csv is not None else self.sprint3_manifest_csv

        objects_std = self.stage_load_and_standardize_objects(manifest_csv=manifest_source)
        objects_refined = self.stage_refine_objects(objects_std)
        objects_all, artifacts_all = self.stage_transfer_all_sites(objects_refined)
        objects_csv, artifacts_csv = self.finalize_outputs(
            objects_all,
            artifacts_all,
            signature=signature,
            write_global=True,
        )

        #module_specs = self.build_module_specs()
        #qa_evals = self.evaluate_labeling_qc(objects_df=objects_all, artifacts_df=artifacts_all)

        success = (not objects_all.empty) and (not artifacts_all.empty)
        status = "success" if success else "empty_outputs"

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
                # "module_specs": {k: vars(v) for k, v in module_specs.items()},
                # "module_qc": {
                #     k: {
                #         "pass_rate": v.pass_rate,
                #         "mean_score": v.mean_score,
                #         "results": [vars(r) for r in v.results],
                #     }
                #     for k, v in qa_evals.items()
                # },
            },
            metrics={
                "n_object_rows": int(len(objects_all)),
                "n_artifact_rows": int(len(artifacts_all)),
                "n_sites": int(objects_all["site_id"].nunique()) if "site_id" in objects_all.columns and not objects_all.empty else 0,
                "stage_cache_manifest_count": int(stage_cache_manifest_count),
            },
            notes=notes,
        )

        self.save_run_result(result, subdir=signature)
        self.save_pipeline_run_manifest(result, signature=signature)
        return result