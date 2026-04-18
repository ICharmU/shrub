from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable
import json
from time import perf_counter

import numpy as np
import hashlib
import rasterio
from rasterio.windows import Window
import pandas as pd

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
    ExecutionEligibilityStatus,
    WorkUnitScope,
    WorkUnitStatus,
    SharedArtifactStatus,
)
from Final.artifact_store import (
    LocalArtifactStore,
    DriveRegistryArtifactStore,
    HybridArtifactStore,
)
from Final.shared_artifact_registry import SharedArtifactRegistry
from Final.coordination import CoordinationManager
from Final.shared_utils import get_logger
from Final.pipeline_caching import hash_payload

from Final.features.config import fe_cfg
from Final.features.stack_registry import (
    load_or_init_stack_registry,
    save_stack_registry,
    deduplicate_stack_registry,
    stack_registry_frame,
)
from Final.features.object_aggregation import (
    build_and_persist_object_feature_table,
    modeling_stack_layer_selection,
)


@dataclass
class FeaturePipelineConfig:
    canonical_grid_source: str = "naip"

    enable_naip: bool = True
    enable_als: bool = True
    enable_3dep: bool = True
    enable_rap: bool = True

    naip_families: tuple[str, ...] = ("raw", "veg_idx", "texture", "multiscale")
    als_families: tuple[str, ...] = ("height_structure",)
    dep3_families: tuple[str, ...] = ("terrain",)
    rap_families: tuple[str, ...] = ("prior",)

    representation_mode: str = "raster"   # "raster", "object", "both"
    enable_object_aggregation: bool = True
    enable_representation_export: bool = False

    chunk_size_px: int = 1024
    default_halo_px: int = 32

    force_refresh_site_assets: bool = False
    force_refresh_canonical_grid: bool = False
    force_refresh_chunk_manifest: bool = False
    force_refresh_source_ready: bool = False
    force_refresh_family_chunks: bool = False
    force_refresh_stack_finalize: bool = False
    force_refresh_object_aggregation: bool = False

    def enabled_sources(self) -> list[str]:
        out: list[str] = []
        if self.enable_naip:
            out.append("naip")
        if self.enable_als:
            out.append("als")
        if self.enable_3dep:
            out.append("3dep")
        if self.enable_rap:
            out.append("rap")
        return out

    def enabled_families_for_source(self, source_name: str) -> tuple[str, ...]:
        mapping = {
            "naip": self.naip_families,
            "als": self.als_families,
            "3dep": self.dep3_families,
            "rap": self.rap_families,
        }
        return tuple(mapping.get(source_name, ()))

@dataclass
class FeaturePipelineOps:
    prepare_site_assets: Callable[..., Any]
    build_canonical_grid_for_site: Callable[..., Any]
    build_chunk_manifest: Callable[..., Any]
    ensure_source_ready: Callable[..., dict[str, Any]]

    run_naip_chunked_pipeline: Callable[..., Any]
    run_als_chunked_pipeline: Callable[..., Any]
    run_3dep_chunked_pipeline: Callable[..., Any]
    run_rap_chunked_pipeline: Callable[..., Any]

    label_objects_for_site: Callable[[str], pd.DataFrame]
    build_and_persist_object_feature_table: Callable[..., Any]

    clear_artifact_staging_dir: Callable[[], None] | None = None


class FeaturePipeline(BasePipeline):
    def __init__(
        self,
        cfg,
        *,
        ops: FeaturePipelineOps,
        pipeline_config: FeaturePipelineConfig | None = None,
    ):
        super().__init__(
            cfg,
            pipeline_name="features",
            output_root=cfg.output.features_root / "pipeline_runs",
        )
        self.logger = get_logger("features.pipeline")
        self.ops = ops
        self.pipeline_config = pipeline_config or FeaturePipelineConfig(
            canonical_grid_source=cfg.features.canonical_grid_source
        )
        self._enumeration_cache: dict[tuple, list[dict[str, Any]]] = {}

        self.artifact_store = self._build_artifact_store()
        self.coordination = CoordinationManager(
            self.artifact_store,
            root_prefix=self.cfg.coordination.root_prefix,
        )
        self.shared_registry = SharedArtifactRegistry(
            self.artifact_store,
            root_prefix=self.cfg.shared_artifacts.registry_prefix,
        )

    # -------------------------------------------------------------------------
    # Artifact specs
    # -------------------------------------------------------------------------

    def artifact_specs(self) -> dict[str, ArtifactSpec]:
        return {
            "site_metadata_manifest": ArtifactSpec(
                key="site_metadata_manifest",
                rel_path_template="features/{site}/{config_signature}/site_assets/site_metadata_manifest.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "source_inventory": ArtifactSpec(
                key="source_inventory",
                rel_path_template="features/{site}/{config_signature}/site_assets/source_inventory.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "site_naip_raster": ArtifactSpec(
                key="site_naip_raster",
                rel_path_template="features/{site}/shared/site_assets/naip/{filename}",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "site_3dep_raster": ArtifactSpec(
                key="site_3dep_raster",
                rel_path_template="features/{site}/shared/site_assets/3dep/{filename}",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "site_rap_raster": ArtifactSpec(
                key="site_rap_raster",
                rel_path_template="features/{site}/shared/site_assets/rap/{filename}",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "site_als_metadata_json": ArtifactSpec(
                key="site_als_metadata_json",
                rel_path_template="features/{site}/shared/site_assets/als_metadata/als_metadata.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "canonical_grid_json": ArtifactSpec(
                key="canonical_grid_json",
                rel_path_template="features/{site}/shared/canonical_grid/{shared_signature}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "chunk_manifest_json": ArtifactSpec(
                key="chunk_manifest_json",
                rel_path_template="features/{site}/shared/chunk_manifest/{shared_signature}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "source_ready_manifest": ArtifactSpec(
                key="source_ready_manifest",
                rel_path_template="features/{site}/shared/source_ready/{source_name}/{shared_signature}.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            # Future stages
            "family_chunk_npz": ArtifactSpec(
                key="family_chunk_npz",
                rel_path_template="features/{site}/{config_signature}/{source_name}/{family_name}/{chunk_id}.npz",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=True,
            ),
            "stack_registry": ArtifactSpec(
                key="stack_registry",
                rel_path_template="features/{site}/{config_signature}/stack/stack_registry.json",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
            "object_feature_table": ArtifactSpec(
                key="object_feature_table",
                rel_path_template="features/{site}/{config_signature}/objects/object_feature_table.csv",
                storage_tier=StorageTier.LOCAL_THEN_REMOTE,
                required_for_resume=True,
                prune_local_after_push=False,
            ),
        }

    # -------------------------------------------------------------------------
    # Pipeline spec
    # -------------------------------------------------------------------------

    def build_pipeline_spec(self) -> PipelineSpec:
        modules = {
            "features.site_assets.base": ModuleSpec(
                key="features.site_assets.base",
                stage_name="site_assets",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_site_assets,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.canonical_grid.base": ModuleSpec(
                key="features.canonical_grid.base",
                stage_name="canonical_grid",
                param_keys=("canonical_grid_source",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_canonical_grid,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.chunk_manifest.base": ModuleSpec(
                key="features.chunk_manifest.base",
                stage_name="chunk_manifest",
                param_keys=("chunk_size_px", "default_halo_px"),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_chunk_manifest,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.source_ready.naip": ModuleSpec(
                key="features.source_ready.naip",
                stage_name="source_ready",
                enabled_key="enable_naip",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_source_ready,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.source_ready.als": ModuleSpec(
                key="features.source_ready.als",
                stage_name="source_ready",
                enabled_key="enable_als",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_source_ready,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.source_ready.3dep": ModuleSpec(
                key="features.source_ready.3dep",
                stage_name="source_ready",
                enabled_key="enable_3dep",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_source_ready,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.source_ready.rap": ModuleSpec(
                key="features.source_ready.rap",
                stage_name="source_ready",
                enabled_key="enable_rap",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_source_ready,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.family_chunk.naip": ModuleSpec(
                key="features.family_chunk.naip",
                stage_name="family_chunk",
                enabled_key="enable_naip",
                param_keys=("naip_families",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_family_chunk,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.family_chunk.als": ModuleSpec(
                key="features.family_chunk.als",
                stage_name="family_chunk",
                enabled_key="enable_als",
                param_keys=("als_families",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_family_chunk,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.family_chunk.3dep": ModuleSpec(
                key="features.family_chunk.3dep",
                stage_name="family_chunk",
                enabled_key="enable_3dep",
                param_keys=("dep3_families",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_family_chunk,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.family_chunk.rap": ModuleSpec(
                key="features.family_chunk.rap",
                stage_name="family_chunk",
                enabled_key="enable_rap",
                param_keys=("rap_families",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_family_chunk,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.stack_finalize.base": ModuleSpec(
                key="features.stack_finalize.base",
                stage_name="stack_finalize",
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_stack_finalize,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.object_aggregation.base": ModuleSpec(
                key="features.object_aggregation.base",
                stage_name="object_aggregation",
                enabled_key="enable_object_aggregation",
                param_keys=("representation_mode",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_object_aggregation,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
            "features.representation_export.base": ModuleSpec(
                key="features.representation_export.base",
                stage_name="representation_export",
                enabled_key="enable_representation_export",
                param_keys=("representation_mode",),
                runtime_requirement=RuntimeRequirement(
                    required_capabilities=self.cfg.features_runtime.require_capability_representation_export,
                    mode=RuntimeRequirementMode.ALL,
                ),
            ),
        }

        stages = [
            StageSpec(
                name="site_assets",
                module_keys=["features.site_assets.base"],
                cache_policy=CachePolicy(require_manifest=False, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
            StageSpec(
                name="canonical_grid",
                module_keys=["features.canonical_grid.base"],
                cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
            StageSpec(
                name="chunk_manifest",
                module_keys=["features.chunk_manifest.base"],
                cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
            StageSpec(
                name="source_ready",
                module_keys=[
                    "features.source_ready.naip",
                    "features.source_ready.als",
                    "features.source_ready.3dep",
                    "features.source_ready.rap",
                ],
                cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
            StageSpec(
                name="family_chunk",
                module_keys=[
                    "features.family_chunk.naip",
                    "features.family_chunk.als",
                    "features.family_chunk.3dep",
                    "features.family_chunk.rap",
                ],
                cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
            StageSpec(
                name="stack_finalize",
                module_keys=["features.stack_finalize.base"],
                cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
            StageSpec(
                name="object_aggregation",
                module_keys=["features.object_aggregation.base"],
                cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
            StageSpec(
                name="representation_export",
                module_keys=["features.representation_export.base"],
                cache_policy=CachePolicy(require_manifest=True, allow_legacy_reuse=False, retention_mode=CacheRetentionMode.LEAN),
            ),
        ]

        search_axes = [
            SearchAxis(key="enable_als", values=[True, False], stage_name="source_ready", module_key="features.source_ready.als"),
            SearchAxis(key="enable_3dep", values=[True, False], stage_name="source_ready", module_key="features.source_ready.3dep"),
            SearchAxis(key="enable_rap", values=[True, False], stage_name="source_ready", module_key="features.source_ready.rap"),
            SearchAxis(key="representation_mode", values=["raster", "both"], stage_name="object_aggregation", module_key="features.object_aggregation.base"),
            SearchAxis(key="enable_object_aggregation", values=[False, True], stage_name="object_aggregation", module_key="features.object_aggregation.base"),
        ]

        return PipelineSpec(
            pipeline_name="features",
            domain=PipelineDomain.FEATURES,
            stages=stages,
            modules=modules,
            search_axes=search_axes,
        )

    # -------------------------------------------------------------------------
    # Storage / artifacts
    # -------------------------------------------------------------------------

    def _build_artifact_store(self):
        storage = self.cfg.features_runtime.storage
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
            self.logger.info("Using HybridArtifactStore for features pipeline")
            return HybridArtifactStore(local_store=local_store, remote_store=drive_store)

        if drive_store is not None and not storage.enable_local_store:
            self.logger.info("Using DriveRegistryArtifactStore only for features pipeline")
            return drive_store

        self.logger.info("Using LocalArtifactStore only for features pipeline")
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
        return self.cfg.output.features_root / "_remote_stage" / rel_path

    def _remote_exists(self, rel_path: str) -> bool:
        try:
            return self.artifact_store.exists(rel_path)
        except Exception:
            return False

    def _push_if_needed(self, local_path: Path, artifact_key: str, rel_path: str) -> str | None:
        spec = self._artifact_spec(artifact_key)
        policy = self.cfg.features_runtime.storage_policy

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
        policy = self.cfg.features_runtime.storage_policy

        if not policy.prune_local_after_remote_push:
            return
        if not spec.prune_local_after_push:
            return
        if policy.verify_remote_before_prune and not self._remote_exists(rel_path):
            return
        if local_path.exists():
            local_path.unlink()
            self.logger.info("PRUNED LOCAL ARTIFACT | rel_path=%s", rel_path)

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
            if artifact_key in {
                "site_metadata_manifest",
                "source_inventory",
                "canonical_grid_json",
                "chunk_manifest_json",
                "source_ready_manifest",
                "site_als_metadata_json",
                "stack_registry",
            }:
                json.loads(local_path.read_text(encoding="utf-8"))
                return True

            if artifact_key == "object_feature_table":
                pd.read_csv(local_path, nrows=5)
                return True

            if artifact_key in {"site_naip_raster", "site_3dep_raster", "site_rap_raster"}:
                with rasterio.open(local_path) as src:
                    src.read(1, window=Window(0, 0, min(16, src.width), min(16, src.height)))
                return True

            if artifact_key == "family_chunk_npz":
                with np.load(local_path) as _:
                    return True

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
        
    def enabled_sources(self) -> list[str]:
        return self.pipeline_config.enabled_sources()

    def enabled_families_for_source(self, source_name: str) -> tuple[str, ...]:
        return self.pipeline_config.enabled_families_for_source(source_name)

    def source_stage_priority(self, source_name: str) -> int:
        order = {"naip": 10, "als": 20, "3dep": 30, "rap": 40}
        return order.get(source_name, 100)

    def family_chunk_priority(self, source_name: str, family_name: str, chunk_idx: int) -> int:
        return self.source_stage_priority(source_name) * 1000 + chunk_idx

    def stage_is_eligible(self, stage_name: str, runtime_report=None) -> tuple[bool, dict]:
        runtime_report = runtime_report or self.runtime_report()
        elig = self.stage_runtime_eligibility(stage_name, runtime_report=runtime_report)
        all_ok = all(
            info.status == ExecutionEligibilityStatus.ELIGIBLE
            for info in elig.values()
        )
        return all_ok, elig
    
    
    def stage_block_reason(self, stage_name: str, runtime_report=None) -> str:
        ok, elig = self.stage_is_eligible(stage_name, runtime_report=runtime_report)
        if ok:
            return ""
    
        msgs = []
        for module_key, info in elig.items():
            if info.status != ExecutionEligibilityStatus.ELIGIBLE:
                missing = getattr(info, "missing_capabilities", None) or []
                reason = getattr(info, "reason", "") or ""
                if missing:
                    msgs.append(f"{module_key}: missing={missing}")
                elif reason:
                    msgs.append(f"{module_key}: {reason}")
                else:
                    msgs.append(f"{module_key}: ineligible")
        return "; ".join(msgs)

    # -------------------------------------------------------------------------
    # Shared artifact helpers
    # -------------------------------------------------------------------------

    def shared_artifact_family_for_stage(self, stage_name: str) -> str | None:
        mapping = {
            "site_assets": "features.site_assets",
            "canonical_grid": "features.canonical_grid",
            "chunk_manifest": "features.chunk_manifest",
            "source_ready": "features.source_ready",
            "family_chunk": "features.family_chunk",
            "stack_finalize": "features.stack_finalize",
            "object_aggregation": "features.object_aggregation",
        }
        return mapping.get(stage_name)

    def shared_signature_site_assets(self, site: str) -> str:
        payload = {
            "site": site,
            "als_dir": self.cfg.data.als_dir,
            "naip_3dep_dir": self.cfg.data.naip_3dep_dir,
            "artifact_repair": asdict(self.cfg.artifact_repair),
        }
        return hash_payload(payload)

    def shared_signature_canonical_grid(self, site: str) -> str:
        payload = {
            "site": site,
            "site_assets_sig": self.shared_signature_site_assets(site),
            "canonical_grid_source": self.pipeline_config.canonical_grid_source,
        }
        return hash_payload(payload)

    def shared_signature_chunk_manifest(self, site: str) -> str:
        payload = {
            "site": site,
            "canonical_grid_sig": self.shared_signature_canonical_grid(site),
            "chunk_size_px": self.pipeline_config.chunk_size_px,
            "default_halo_px": self.pipeline_config.default_halo_px,
        }
        return hash_payload(payload)

    def shared_signature_source_ready(self, site: str, source_name: str) -> str:
        payload = {
            "site": site,
            "source_name": source_name,
            "site_assets_sig": self.shared_signature_site_assets(site),
            "canonical_grid_sig": self.shared_signature_canonical_grid(site),
            "force_refresh_source_ready": bool(self.pipeline_config.force_refresh_source_ready),
        }
        return hash_payload(payload)

    def shared_signature_for_stage(self, stage_name: str, **kwargs) -> str | None:
        if stage_name == "site_assets":
            return self.shared_signature_site_assets(kwargs["site"])
        if stage_name == "canonical_grid":
            return self.shared_signature_canonical_grid(kwargs["site"])
        if stage_name == "chunk_manifest":
            return self.shared_signature_chunk_manifest(kwargs["site"])
        if stage_name == "source_ready":
            return self.shared_signature_source_ready(kwargs["site"], kwargs["source_name"])
        return None
    
    def shared_signature_family_chunk(
        self,
        *,
        site: str,
        source_name: str,
        family_name: str,
        chunk_id: str,
    ) -> str:
        payload = {
            "site": site,
            "source_name": source_name,
            "family_name": family_name,
            "chunk_id": chunk_id,
            "source_ready_sig": self.shared_signature_source_ready(site, source_name),
            "chunk_manifest_sig": self.shared_signature_chunk_manifest(site),
            "pipeline_config": asdict(self.pipeline_config),
        }
        return hash_payload(payload)

    def shared_signature_stack_finalize(self, site: str) -> str:
        payload = {
            "site": site,
            "enabled_sources": self.pipeline_config.enabled_sources(),
            "naip_families": self.pipeline_config.naip_families,
            "als_families": self.pipeline_config.als_families,
            "dep3_families": self.pipeline_config.dep3_families,
            "rap_families": self.pipeline_config.rap_families,
            "representation_mode": self.pipeline_config.representation_mode,
            "chunk_manifest_sig": self.shared_signature_chunk_manifest(site),
        }
        return hash_payload(payload)

    def shared_signature_object_aggregation(self, site: str) -> str:
        payload = {
            "site": site,
            "stack_sig": self.shared_signature_stack_finalize(site),
            "representation_mode": self.pipeline_config.representation_mode,
            "object_agg_cfg": asdict(fe_cfg.object_agg),
        }
        return hash_payload(payload)

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
    ) -> None:
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
        if not self.cfg.shared_artifacts.enable_shared_artifact_registry:
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

    # -------------------------------------------------------------------------
    # Config signatures / convenience
    # -------------------------------------------------------------------------

    def config_dict(self) -> dict:
        return asdict(self.pipeline_config)

    def config_signature(self, config_dict: dict | None = None) -> str:
        payload = config_dict if config_dict is not None else self.config_dict()
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def _enabled_sources(self) -> list[str]:
        return self.pipeline_config.enabled_sources()

    # -------------------------------------------------------------------------
    # Work-unit enumeration
    # -------------------------------------------------------------------------

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
            return [dict(u) for u in self._enumeration_cache[enum_cache_key]]

        units: list[dict[str, Any]] = []

        site_assets_ok, _ = self.stage_is_eligible("site_assets", runtime_report=runtime_report)
        canonical_ok, _ = self.stage_is_eligible("canonical_grid", runtime_report=runtime_report)
        chunk_ok, _ = self.stage_is_eligible("chunk_manifest", runtime_report=runtime_report)
        source_ready_ok, _ = self.stage_is_eligible("source_ready", runtime_report=runtime_report)
        family_chunk_ok, _ = self.stage_is_eligible("family_chunk", runtime_report=runtime_report)
        stack_ok, _ = self.stage_is_eligible("stack_finalize", runtime_report=runtime_report)
        object_ok, _ = self.stage_is_eligible("object_aggregation", runtime_report=runtime_report)
        export_ok, _ = self.stage_is_eligible("representation_export", runtime_report=runtime_report)

        site_assets_complete_by_site: dict[str, bool] = {}
        canonical_complete_by_site: dict[str, bool] = {}
        chunk_complete_by_site: dict[str, bool] = {}
        source_ready_complete: dict[tuple[str, str], bool] = {}
        family_chunk_complete: dict[tuple[str, str, str, str], bool] = {}

        # site_assets
        for site in self.cfg.sites:
            shared_sig = self.shared_signature_site_assets(site)
            if register_shared_requirements:
                self.register_shared_requirement(
                    artifact_family="features.site_assets",
                    shared_signature=shared_sig,
                    trial_id=trial_id,
                    metadata={"site": site},
                )
            complete = self.shared_artifact_is_valid(
                artifact_family="features.site_assets",
                shared_signature=shared_sig,
            )
            site_assets_complete_by_site[site] = complete

            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:site_assets:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "site_assets",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if complete else (
                    WorkUnitStatus.PENDING.value if site_assets_ok else WorkUnitStatus.INELIGIBLE.value
                ),
                "dependencies": [],
                "dependency_reasons": [],
                "runtime_required_capabilities": list(self.cfg.features_runtime.require_capability_site_assets),
                "runtime_eligible": site_assets_ok,
                "priority": 5,
                "site_id": site,
                "shared_artifact_family": "features.site_assets",
                "shared_signature": shared_sig,
            })

        # canonical_grid
        for site in self.cfg.sites:
            shared_sig = self.shared_signature_canonical_grid(site)
            if register_shared_requirements:
                self.register_shared_requirement(
                    artifact_family="features.canonical_grid",
                    shared_signature=shared_sig,
                    trial_id=trial_id,
                    metadata={"site": site},
                )
            complete = self.shared_artifact_is_valid(
                artifact_family="features.canonical_grid",
                shared_signature=shared_sig,
            )
            canonical_complete_by_site[site] = complete

            deps, dep_reasons = [], []
            if not site_assets_complete_by_site[site]:
                deps.append("site_assets")
                dep_reasons.append("Site assets not shared-valid yet.")

            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:canonical_grid:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "canonical_grid",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if complete else (
                    WorkUnitStatus.PENDING.value if (canonical_ok and not deps) else (
                        WorkUnitStatus.BLOCKED.value if canonical_ok else WorkUnitStatus.INELIGIBLE.value
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self.cfg.features_runtime.require_capability_canonical_grid),
                "runtime_eligible": canonical_ok,
                "priority": 10,
                "site_id": site,
                "shared_artifact_family": "features.canonical_grid",
                "shared_signature": shared_sig,
            })

        # chunk_manifest
        for site in self.cfg.sites:
            shared_sig = self.shared_signature_chunk_manifest(site)
            if register_shared_requirements:
                self.register_shared_requirement(
                    artifact_family="features.chunk_manifest",
                    shared_signature=shared_sig,
                    trial_id=trial_id,
                    metadata={"site": site},
                )
            complete = self.shared_artifact_is_valid(
                artifact_family="features.chunk_manifest",
                shared_signature=shared_sig,
            )
            chunk_complete_by_site[site] = complete

            deps, dep_reasons = [], []
            if not canonical_complete_by_site[site]:
                deps.append("canonical_grid")
                dep_reasons.append("Canonical grid not shared-valid yet.")

            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:chunk_manifest:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "chunk_manifest",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if complete else (
                    WorkUnitStatus.PENDING.value if (chunk_ok and not deps) else (
                        WorkUnitStatus.BLOCKED.value if chunk_ok else WorkUnitStatus.INELIGIBLE.value
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self.cfg.features_runtime.require_capability_chunk_manifest),
                "runtime_eligible": chunk_ok,
                "priority": 15,
                "site_id": site,
                "shared_artifact_family": "features.chunk_manifest",
                "shared_signature": shared_sig,
            })

        # source_ready
        for site in self.cfg.sites:
            for source_name in self.enabled_sources():
                shared_sig = self.shared_signature_source_ready(site, source_name)
                if register_shared_requirements:
                    self.register_shared_requirement(
                        artifact_family=f"features.source_ready.{source_name}",
                        shared_signature=shared_sig,
                        trial_id=trial_id,
                        metadata={"site": site, "source_name": source_name},
                    )
                complete = self.shared_artifact_is_valid(
                    artifact_family=f"features.source_ready.{source_name}",
                    shared_signature=shared_sig,
                )
                source_ready_complete[(site, source_name)] = complete

                deps, dep_reasons = [], []
                if not site_assets_complete_by_site[site]:
                    deps.append("site_assets")
                    dep_reasons.append("Site assets not shared-valid yet.")
                if not canonical_complete_by_site[site]:
                    deps.append("canonical_grid")
                    dep_reasons.append("Canonical grid not shared-valid yet.")

                units.append({
                    "unit_id": f"{trial_id}:{self.pipeline_name}:source_ready:{site}:{source_name}",
                    "trial_id": trial_id,
                    "pipeline_name": self.pipeline_name,
                    "config_signature": config_signature,
                    "stage_name": "source_ready",
                    "work_key": f"{site}:{source_name}",
                    "scope": WorkUnitScope.SITE.value,
                    "status": WorkUnitStatus.COMPLETE.value if complete else (
                        WorkUnitStatus.PENDING.value if (source_ready_ok and not deps) else (
                            WorkUnitStatus.BLOCKED.value if source_ready_ok else WorkUnitStatus.INELIGIBLE.value
                        )
                    ),
                    "dependencies": deps,
                    "dependency_reasons": dep_reasons,
                    "runtime_required_capabilities": list(self.cfg.features_runtime.require_capability_source_ready),
                    "runtime_eligible": source_ready_ok,
                    "priority": self.source_stage_priority(source_name),
                    "site_id": site,
                    "source_version": source_name,
                    "shared_artifact_family": f"features.source_ready.{source_name}",
                    "shared_signature": shared_sig,
                })

        # family_chunk
        for site in self.cfg.sites:
            try:
                site_assets = self.ops.prepare_site_assets(site, artifact_store=self.artifact_store, force_refresh=False)
                grid = self.ops.build_canonical_grid_for_site(site, artifact_store=self.artifact_store, assets=site_assets, force_refresh=False)
                manifest = self.ops.build_chunk_manifest(grid, artifact_store=self.artifact_store, force_refresh=False)
                chunk_records = list(getattr(manifest, "records", []) or [])
            except Exception:
                chunk_records = []

            for source_name in self.enabled_sources():
                for family_name in self.enabled_families_for_source(source_name):
                    for chunk_idx, record in enumerate(chunk_records):
                        shared_sig = self.shared_signature_family_chunk(
                            site=site,
                            source_name=source_name,
                            family_name=family_name,
                            chunk_id=record.chunk_id,
                        )
                        if register_shared_requirements:
                            self.register_shared_requirement(
                                artifact_family=f"features.family_chunk.{source_name}.{family_name}",
                                shared_signature=shared_sig,
                                trial_id=trial_id,
                                metadata={"site": site, "source_name": source_name, "family_name": family_name, "chunk_id": record.chunk_id},
                            )

                        complete = self.shared_artifact_is_valid(
                            artifact_family=f"features.family_chunk.{source_name}.{family_name}",
                            shared_signature=shared_sig,
                        )
                        family_chunk_complete[(site, source_name, family_name, record.chunk_id)] = complete

                        deps, dep_reasons = [], []
                        if not chunk_complete_by_site.get(site, False):
                            deps.append("chunk_manifest")
                            dep_reasons.append("Chunk manifest not shared-valid yet.")
                        if not source_ready_complete.get((site, source_name), False):
                            deps.append("source_ready")
                            dep_reasons.append(f"Source {source_name} is not source-ready yet.")

                        units.append({
                            "unit_id": f"{trial_id}:{self.pipeline_name}:family_chunk:{site}:{source_name}:{family_name}:{record.chunk_id}",
                            "trial_id": trial_id,
                            "pipeline_name": self.pipeline_name,
                            "config_signature": config_signature,
                            "stage_name": "family_chunk",
                            "work_key": f"{site}:{source_name}:{family_name}:{record.chunk_id}",
                            "scope": WorkUnitScope.CHUNK.value,
                            "status": WorkUnitStatus.COMPLETE.value if complete else (
                                WorkUnitStatus.PENDING.value if (family_chunk_ok and not deps) else (
                                    WorkUnitStatus.BLOCKED.value if family_chunk_ok else WorkUnitStatus.INELIGIBLE.value
                                )
                            ),
                            "dependencies": deps,
                            "dependency_reasons": dep_reasons,
                            "runtime_required_capabilities": list(self.cfg.features_runtime.require_capability_family_chunk),
                            "runtime_eligible": family_chunk_ok,
                            "priority": self.family_chunk_priority(source_name, family_name, chunk_idx),
                            "site_id": site,
                            "source_version": source_name,
                            "family_name": family_name,
                            "chunk_id": record.chunk_id,
                            "shared_artifact_family": f"features.family_chunk.{source_name}.{family_name}",
                            "shared_signature": shared_sig,
                        })

        # stack_finalize
        for site in self.cfg.sites:
            shared_sig = self.shared_signature_stack_finalize(site)
            if register_shared_requirements:
                self.register_shared_requirement(
                    artifact_family="features.stack_finalize",
                    shared_signature=shared_sig,
                    trial_id=trial_id,
                    metadata={"site": site},
                )

            complete = self.shared_artifact_is_valid(
                artifact_family="features.stack_finalize",
                shared_signature=shared_sig,
            )

            deps, dep_reasons = [], []
            for source_name in self.enabled_sources():
                for family_name in self.enabled_families_for_source(source_name):
                    expected_any = [
                        k for k in family_chunk_complete
                        if k[0] == site and k[1] == source_name and k[2] == family_name
                    ]
                    if expected_any and not all(family_chunk_complete[k] for k in expected_any):
                        deps.append("family_chunk")
                        dep_reasons.append(f"Not all family chunks are ready for {source_name}/{family_name}.")
                        break

            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:stack_finalize:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "stack_finalize",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": WorkUnitStatus.COMPLETE.value if complete else (
                    WorkUnitStatus.PENDING.value if (stack_ok and not deps) else (
                        WorkUnitStatus.BLOCKED.value if stack_ok else WorkUnitStatus.INELIGIBLE.value
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self.cfg.features_runtime.require_capability_stack_finalize),
                "runtime_eligible": stack_ok,
                "priority": 9000,
                "site_id": site,
                "shared_artifact_family": "features.stack_finalize",
                "shared_signature": shared_sig,
            })

        # object_aggregation
        for site in self.cfg.sites:
            shared_sig = self.shared_signature_object_aggregation(site)
            if register_shared_requirements:
                self.register_shared_requirement(
                    artifact_family="features.object_aggregation",
                    shared_signature=shared_sig,
                    trial_id=trial_id,
                    metadata={"site": site},
                )

            complete = self.shared_artifact_is_valid(
                artifact_family="features.object_aggregation",
                shared_signature=shared_sig,
            )

            deps, dep_reasons = [], []
            if self.pipeline_config.enable_object_aggregation and not self.shared_artifact_is_valid(
                artifact_family="features.stack_finalize",
                shared_signature=self.shared_signature_stack_finalize(site),
            ):
                deps.append("stack_finalize")
                dep_reasons.append("Stack registry is not ready yet.")

            units.append({
                "unit_id": f"{trial_id}:{self.pipeline_name}:object_aggregation:{site}",
                "trial_id": trial_id,
                "pipeline_name": self.pipeline_name,
                "config_signature": config_signature,
                "stage_name": "object_aggregation",
                "work_key": site,
                "scope": WorkUnitScope.SITE.value,
                "status": (
                    WorkUnitStatus.COMPLETE.value if complete else (
                        WorkUnitStatus.PENDING.value if (object_ok and self.pipeline_config.enable_object_aggregation and not deps) else (
                            WorkUnitStatus.BLOCKED.value if (object_ok and self.pipeline_config.enable_object_aggregation) else WorkUnitStatus.INELIGIBLE.value
                        )
                    )
                ),
                "dependencies": deps,
                "dependency_reasons": dep_reasons,
                "runtime_required_capabilities": list(self.cfg.features_runtime.require_capability_object_aggregation),
                "runtime_eligible": object_ok and self.pipeline_config.enable_object_aggregation,
                "priority": 9500,
                "site_id": site,
                "shared_artifact_family": "features.object_aggregation",
                "shared_signature": shared_sig,
            })

        self._enumeration_cache[enum_cache_key] = [dict(u) for u in units]
        return units

    # -------------------------------------------------------------------------
    # Work-unit execution
    # -------------------------------------------------------------------------

    def run_work_unit(self, unit: dict, *, trial_id: str, state=None):
        stage_name = unit["stage_name"]
        site_id = unit.get("site_id")

        if stage_name == "site_assets":
            bundle = self.ops.prepare_site_assets(
                site_id,
                artifact_store=self.artifact_store,
                force_refresh=self.pipeline_config.force_refresh_site_assets,
            )
            self._mark_shared_available(
                artifact_family="features.site_assets",
                shared_signature=self.shared_signature_site_assets(site_id),
                metadata={
                    "site": site_id,
                    "source_asset_keys": sorted(list((bundle.source_assets or {}).keys())),
                    "n_notes": len(bundle.notes or []),
                },
                trial_id=trial_id,
            )
            return {"stage_name": "site_assets", "site_id": site_id}

        if stage_name == "canonical_grid":
            site_assets = self.ops.prepare_site_assets(
                site_id,
                artifact_store=self.artifact_store,
                force_refresh=False,
            )
            grid = self.ops.build_canonical_grid_for_site(
                site_id,
                artifact_store=self.artifact_store,
                assets=site_assets,
                force_refresh=self.pipeline_config.force_refresh_canonical_grid,
            )
            self._mark_shared_available(
                artifact_family="features.canonical_grid",
                shared_signature=self.shared_signature_canonical_grid(site_id),
                metadata={
                    "site": site_id,
                    "width": int(grid.width),
                    "height": int(grid.height),
                    "crs": str(grid.crs),
                    "source_name": str(grid.source_name),
                },
                trial_id=trial_id,
            )
            return {"stage_name": "canonical_grid", "site_id": site_id}

        if stage_name == "chunk_manifest":
            site_assets = self.ops.prepare_site_assets(site_id, artifact_store=self.artifact_store, force_refresh=False)
            grid = self.ops.build_canonical_grid_for_site(site_id, artifact_store=self.artifact_store, assets=site_assets, force_refresh=False)
            manifest = self.ops.build_chunk_manifest(
                grid,
                artifact_store=self.artifact_store,
                force_refresh=self.pipeline_config.force_refresh_chunk_manifest,
            )
            self._mark_shared_available(
                artifact_family="features.chunk_manifest",
                shared_signature=self.shared_signature_chunk_manifest(site_id),
                metadata={
                    "site": site_id,
                    "n_chunks": int(len(getattr(manifest, "records", []) or [])),
                    "chunk_size_px": int(self.pipeline_config.chunk_size_px),
                    "default_halo_px": int(self.pipeline_config.default_halo_px),
                },
                trial_id=trial_id,
            )
            return {"stage_name": "chunk_manifest", "site_id": site_id}

        if stage_name == "source_ready":
            source_name = unit["source_version"]
            site_assets = self.ops.prepare_site_assets(site_id, artifact_store=self.artifact_store, force_refresh=False)
            grid = self.ops.build_canonical_grid_for_site(site_id, artifact_store=self.artifact_store, assets=site_assets, force_refresh=False)
            record = self.ops.ensure_source_ready(
                site_id=site_id,
                source_name=source_name,
                artifact_store=self.artifact_store,
                site_assets=site_assets,
                canonical_grid=grid,
                force_refresh=self.pipeline_config.force_refresh_source_ready,
            )
            self._mark_shared_available(
                artifact_family=f"features.source_ready.{source_name}",
                shared_signature=self.shared_signature_source_ready(site_id, source_name),
                metadata=dict(record or {}),
                trial_id=trial_id,
            )
            return {"stage_name": "source_ready", "site_id": site_id, "source_name": source_name}

        if stage_name == "family_chunk":
            source_name = unit["source_version"]
            family_name = unit["family_name"]

            site_assets = self.ops.prepare_site_assets(site_id, artifact_store=self.artifact_store, force_refresh=False)
            grid = self.ops.build_canonical_grid_for_site(site_id, artifact_store=self.artifact_store, assets=site_assets, force_refresh=False)
            chunk_manifest = self.ops.build_chunk_manifest(grid, artifact_store=self.artifact_store, force_refresh=False)

            registry = load_or_init_stack_registry(
                site_id,
                artifact_store=self.artifact_store,
                config_signature=self.config_signature(),
            )

            if source_name == "naip":
                registry = self.ops.run_naip_chunked_pipeline(
                    site_id,
                    artifact_store=self.artifact_store,
                    assets=site_assets,
                    grid=grid,
                    chunk_manifest=chunk_manifest,
                    rebuild_registry=False,
                    enabled_family_keys=(family_name,),
                )
            elif source_name == "als":
                registry = self.ops.run_als_chunked_pipeline(
                    site_id,
                    artifact_store=self.artifact_store,
                    site_assets=site_assets,
                    canonical_grid=grid,
                    chunk_manifest=chunk_manifest,
                    registry=registry,
                    enabled_family_keys=(family_name,),
                )
            elif source_name == "3dep":
                registry = self.ops.run_3dep_chunked_pipeline(
                    site_id,
                    artifact_store=self.artifact_store,
                    site_assets=site_assets,
                    canonical_grid=grid,
                    chunk_manifest=chunk_manifest,
                    registry=registry,
                    force_refresh_source=self.pipeline_config.force_refresh_family_chunks,
                    enabled_family_keys=(family_name,),
                )
            elif source_name == "rap":
                registry = self.ops.run_rap_chunked_pipeline(
                    site_id,
                    artifact_store=self.artifact_store,
                    site_assets=site_assets,
                    canonical_grid=grid,
                    chunk_manifest=chunk_manifest,
                    registry=registry,
                    force_refresh_source=self.pipeline_config.force_refresh_family_chunks,
                    enabled_family_keys=(family_name,),
                )
            else:
                raise KeyError(f"Unsupported source_name={source_name}")

            self._mark_shared_available(
                artifact_family=f"features.family_chunk.{source_name}.{family_name}",
                shared_signature=self.shared_signature_family_chunk(
                    site=site_id,
                    source_name=source_name,
                    family_name=family_name,
                    chunk_id=unit["chunk_id"],
                ),
                metadata={
                    "site": site_id,
                    "source_name": source_name,
                    "family_name": family_name,
                    "chunk_id": unit["chunk_id"],
                    "n_layers_registry": len(registry.layers),
                },
                trial_id=trial_id,
            )
            return {
                "stage_name": "family_chunk",
                "site_id": site_id,
                "source_name": source_name,
                "family_name": family_name,
                "chunk_id": unit["chunk_id"],
            }

        if stage_name == "stack_finalize":
            registry = load_or_init_stack_registry(
                site_id,
                artifact_store=self.artifact_store,
                config_signature=self.config_signature(),
            )
            registry = deduplicate_stack_registry(registry)
            stack_rec = save_stack_registry(
                registry,
                artifact_store=self.artifact_store,
            )
            self._mark_shared_available(
                artifact_family="features.stack_finalize",
                shared_signature=self.shared_signature_stack_finalize(site_id),
                rel_path=stack_rec.rel_path,
                local_path=stack_rec.local_path,
                remote_ref=stack_rec.remote_ref,
                metadata={
                    "site": site_id,
                    "n_layers": len(registry.layers),
                },
                trial_id=trial_id,
            )
            return {"stage_name": "stack_finalize", "site_id": site_id, "n_layers": len(registry.layers)}

        if stage_name == "object_aggregation":
            site_assets = self.ops.prepare_site_assets(site_id, artifact_store=self.artifact_store, force_refresh=False)
            grid = self.ops.build_canonical_grid_for_site(site_id, artifact_store=self.artifact_store, assets=site_assets, force_refresh=False)
            chunk_manifest = self.ops.build_chunk_manifest(grid, artifact_store=self.artifact_store, force_refresh=False)
            registry = load_or_init_stack_registry(
                site_id,
                artifact_store=self.artifact_store,
                config_signature=self.config_signature(),
            )
            objects_df = self.ops.label_objects_for_site(site_id)
            obj_df, obj_rec = self.ops.build_and_persist_object_feature_table(
                registry,
                artifact_store=self.artifact_store,
                site_id=site_id,
                canonical_grid=grid,
                chunk_manifest=chunk_manifest,
                objects_df=objects_df,
            )
            self._mark_shared_available(
                artifact_family="features.object_aggregation",
                shared_signature=self.shared_signature_object_aggregation(site_id),
                rel_path=(obj_rec.rel_path if obj_rec else None),
                local_path=(obj_rec.local_path if obj_rec else None),
                remote_ref=(obj_rec.remote_ref if obj_rec else None),
                metadata={
                    "site": site_id,
                    "n_rows": int(len(obj_df)),
                    "n_cols": int(obj_df.shape[1]) if not obj_df.empty else 0,
                },
                trial_id=trial_id,
            )
            return {"stage_name": "object_aggregation", "site_id": site_id, "n_rows": int(len(obj_df))}

        if stage_name == "representation_export":
            return {"stage_name": "representation_export", "site_id": site_id, "status": "noop_for_now"}

        raise NotImplementedError(f"Unknown features work unit stage={stage_name}")

    def run(self, *, trial_id: str = "adhoc_features", state=None, **kwargs) -> PipelineRunResult:
        runtime_report = self.runtime_report()
        executed_units: list[str] = []
        loop_guard = 0

        while True:
            loop_guard += 1
            if loop_guard > 50000:
                raise RuntimeError("FeaturePipeline debug run exceeded loop guard.")

            units = self.enumerate_work_units(
                trial_id=trial_id,
                config_signature=self.config_signature(),
                runtime_report=runtime_report,
                register_shared_requirements=False,
            )

            runnable = [
                u for u in units
                if u["status"] == WorkUnitStatus.PENDING.value and not u.get("dependencies")
            ]
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

        raster_outputs = CanonicalRasterOutputs()
        object_outputs = CanonicalObjectOutputs()

        return PipelineRunResult(
            pipeline_name=self.pipeline_name,
            success=success,
            status=status,
            raster_outputs=raster_outputs,
            object_outputs=object_outputs,
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
            },
            notes=[
                "Full FE pipeline execution through family chunks, stack finalization, and optional object aggregation.",
            ],
        )
    

def build_feature_pipeline_ops():
    from Final.features.assets import prepare_site_assets
    from Final.features.canonical_grid import build_canonical_grid_for_site
    from Final.features.chunking import build_chunk_manifest
    from Final.features.source_naip import run_naip_chunked_pipeline, prepare_naip_asset
    from Final.features.source_als import run_als_chunked_pipeline
    from Final.features.source_3dep import run_3dep_chunked_pipeline, prepare_3dep_asset
    from Final.features.source_rap import run_rap_chunked_pipeline, prepare_rap_asset
    from Final.features.object_aggregation import build_and_persist_object_feature_table
    from Final.features.labeling_bridge import label_objects_for_site
    from Final.features.assets import find_cached_rap_asset
    from Final.features.source_3dep import build_3dep_source_inventory_record
    from Final.features.source_rap import build_rap_source_inventory_record
    from Final.features.source_als import validate_als_tile_inventory

    def ensure_source_ready(
        *,
        site_id: str,
        source_name: str,
        artifact_store,
        site_assets,
        canonical_grid,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        if source_name == "naip":
            path = prepare_naip_asset(
                site_id,
                artifact_store=artifact_store,
                force_refresh=force_refresh,
                inventory=site_assets.source_assets.get("source_inventory"),
                canonical_grid=canonical_grid,
                site_assets=site_assets,
            )
            return {"source_name": "naip", "status": "ready", "asset_path": str(path)}

        if source_name == "3dep":
            path = prepare_3dep_asset(
                site_id,
                artifact_store=artifact_store,
                canonical_grid=canonical_grid,
                force_refresh=force_refresh,
                inventory=site_assets.source_assets.get("source_inventory"),
            )
            rec = build_3dep_source_inventory_record(
                site_id,
                artifact_store=artifact_store,
                site_assets=site_assets,
            )
            return {"source_name": "3dep", "status": "ready", "asset_path": str(path), "inventory_status": rec.status}

        if source_name == "rap":
            path = prepare_rap_asset(
                site_id,
                artifact_store=artifact_store,
                canonical_grid=canonical_grid,
                force_refresh=force_refresh,
                inventory=site_assets.source_assets.get("source_inventory"),
            )
            rec = build_rap_source_inventory_record(
                site_id,
                artifact_store=artifact_store,
                site_assets=site_assets,
            )
            return {"source_name": "rap", "status": "ready", "asset_path": str(path), "inventory_status": rec.status}

        if source_name == "als":
            tiles_df = validate_als_tile_inventory(
                site_assets.source_assets.get("als_metadata_df", None)
                if False else
                __import__("Final.features.source_als", fromlist=["select_intersecting_als_tiles"]).select_intersecting_als_tiles(site_assets, canonical_grid),
                site_id=site_id,
            )
            return {"source_name": "als", "status": "ready", "n_tiles": int(len(tiles_df))}

        raise KeyError(f"Unsupported source_name={source_name}")

    return FeaturePipelineOps(
        prepare_site_assets=prepare_site_assets,
        build_canonical_grid_for_site=build_canonical_grid_for_site,
        build_chunk_manifest=build_chunk_manifest,
        ensure_source_ready=ensure_source_ready,
        run_naip_chunked_pipeline=run_naip_chunked_pipeline,
        run_als_chunked_pipeline=run_als_chunked_pipeline,
        run_3dep_chunked_pipeline=run_3dep_chunked_pipeline,
        run_rap_chunked_pipeline=run_rap_chunked_pipeline,
        label_objects_for_site=label_objects_for_site,
        build_and_persist_object_feature_table=build_and_persist_object_feature_table,
        clear_artifact_staging_dir=getattr(
            __import__("Final.features.source_3dep", fromlist=["clear_artifact_staging_dir"]),
            "clear_artifact_staging_dir",
            None,
        ),
    )