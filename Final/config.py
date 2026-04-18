from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from Final.paths import (
    PROJECT_ROOT,
    FINAL_ROOT,
    SPRINT3_BASE_ROOT,
    LABELING_ROOT,
    FEATURES_ROOT,
    MODELING_ROOT,
    POSTPROCESSING_ROOT,
)

DEFAULT_SITES = [
    "calaveras-big-trees",
    "dl-bliss",
    "independence-lake",
    "pacific-union-college",
    "sedgwick",
    "shaver-lake",
]

@dataclass
class SiteConfig:
    sites: list[str] = field(default_factory=lambda: list(DEFAULT_SITES))

@dataclass
class ArtifactStoreConfig:
    enabled: bool = False
    mode: str = "local"   # "local", "drive", "hybrid"

    local_storage_root: Path = FINAL_ROOT / "artifact_store_local"

    drive_registry_path: Path = FINAL_ROOT / "artifact_registry.yaml"
    drive_config_path: Path = PROJECT_ROOT / "drive_config.yaml"
    drive_client_secrets_path: Path = PROJECT_ROOT / "client_secrets.json"
    drive_credentials_path: Path = PROJECT_ROOT / "pydrive_credentials.json"

@dataclass
class ArtifactStorePolicyConfig:
    push_large_artifacts_to_remote: bool = False
    prune_local_after_remote_push: bool = False
    verify_remote_before_prune: bool = True

@dataclass
class ArtifactRepairPolicyConfig:
    repair_invalid_local_assets: bool = True
    prefer_remote_hydration_for_invalid_local_assets: bool = True
    recompute_only_if_local_and_remote_invalid: bool = True
    revalidate_on_hydrate: bool = True


@dataclass
class SharedArtifactPolicyConfig:
    enable_shared_artifact_registry: bool = True
    registry_prefix: str = "shared_artifacts"
    prefer_shared_artifacts_over_trial_local: bool = True
    allow_cross_trial_reuse: bool = True
    attach_trial_requirements_to_registry: bool = True


@dataclass
class GridHealthConfig:
    enable_preflight_reports: bool = True
    include_runtime_report: bool = True
    include_dependency_report: bool = True
    include_shared_artifact_report: bool = True

@dataclass
class RuntimeImageSpecConfig:
    key: str
    aliases: tuple[str, ...] = ()
    marker_env_vars: dict[str, str] = field(default_factory=dict)
    marker_files: tuple[str, ...] = ()
    required_executables: tuple[str, ...] = ()
    required_python_modules: tuple[str, ...] = ()
    provided_capabilities: tuple[str, ...] = ()


@dataclass
class RuntimeDetectionConfig:
    image_env_var_names: tuple[str, ...] = (
        "SHRUBWISE_IMAGE",
        "CONTAINER_IMAGE",
        "DOCKER_IMAGE",
        "JUPYTER_IMAGE_SPEC",
    )
    conda_env_var_names: tuple[str, ...] = ("CONDA_DEFAULT_ENV",)
    detect_executables: tuple[str, ...] = ("python", "Rscript", "pdal")
    detect_python_modules: tuple[str, ...] = ("numpy", "pandas", "rasterio", "scipy")
    marker_file_candidates: tuple[str, ...] = (
        "/etc/shrubwise_image.json",
        "/etc/container_image_name",
    )
    strict_image_name_match: bool = False

DEFAULT_RUNTIME_IMAGES = [
    RuntimeImageSpecConfig(
        key="intelimon-shrubs",
        aliases=("pramonettivega/intelimon-shrubs", "intelimon-shrubs"),
        required_executables=("Rscript",),
        provided_capabilities=(
            "runtime:rscript",
            "runtime:intelimon_sprint3",
        ),
    ),
    RuntimeImageSpecConfig(
        key="shrubs-labels-v1",
        aliases=("pramonettivega/shrubs-labels:v1", "shrubs-labels:v1"),
        required_executables=("python", "pdal"),
        required_python_modules=("numpy", "pandas", "rasterio"),
        provided_capabilities=(
            "runtime:python",
            "runtime:pdal",
            "runtime:rasterio",
            "runtime:labeling_transfer",
            "runtime:features",
            "runtime:modeling",
        ),
    ),
]

@dataclass
class CoordinationConfig:
    enabled: bool = True
    root_prefix: str = "coordination"
    locks_prefix: str = "locks"
    status_prefix: str = "status"
    heartbeat_interval_sec: int = 60
    stale_lock_timeout_sec: int = 15 * 60
    sync_registry_before_claim: bool = True
    sync_registry_after_stage: bool = True
    hydrate_remote_before_compute: bool = True

    refresh_skip_window_sec: int = 30
    live_unit_refresh_window_sec: int = 10


@dataclass
class RuntimePolicyConfig:
    skip_ineligible_stages: bool = True
    wait_for_upstream_outputs: bool = True
    allow_partial_pipeline_execution: bool = True
    claim_work_before_run: bool = True
    attach_runtime_report_to_manifests: bool = True
    hydrate_remote_before_rerun: bool = True


@dataclass
class PipelineStorageConfig:
    enable_local_store: bool = True
    enable_drive_store: bool = False
    use_hybrid_store: bool = False
    fail_if_drive_missing: bool = False

@dataclass
class DataConfig:
    # Remote Sprint 4-style locations
    base_root: str = "https://wifire-data.sdsc.edu/nc/public.php/dav/files"
    shrubs_dir: str = "shrub_lists"
    revised_shrubs_dir: str = "shrub_lists_revised"
    transformations_dir: str = "transformations"
    als_dir: str = "ALS"
    naip_3dep_dir: str = "NAIP_3DEP_product"

    # Remote Sprint 3 TLS locations
    original_tls_dir: str = "original_TLS"

    # Local project roots
    project_root: Path = PROJECT_ROOT
    final_root: Path = FINAL_ROOT

    # Local Sprint 3 locations
    sprint3_base_dir: Path = SPRINT3_BASE_ROOT
    sprint3_original_script: Path = SPRINT3_BASE_ROOT / "IntELiMon_1_1_1.R"
    sprint3_revised_script: Path = SPRINT3_BASE_ROOT / "IntELiMon_1_1_1_revised.R"
    sprint3_default_input: Path = SPRINT3_BASE_ROOT

    # Runtime behavior
    keep_temp: bool = False


@dataclass
class RasterizationConfig:
    default_radius_m: float = 1.0
    pad_m: float = 20.0

    background_value: int = 0
    shrub_value: int = 1

    confidence_background: float = 0.0
    confidence_center: float = 1.0
    confidence_edge: float = 0.35

    # resolutions (meters) for exported label products
    create_multires: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


@dataclass
class RefinementConfig:
    infer_radius_from_area: bool = True
    clamp_inferred_radius: bool = True
    min_radius_m: float = 0.25
    max_radius_m: float = 4.0

    temporal_half_life_days: float = 365.0

    # room for later label-engineering improvements
    attach_shape_descriptors: bool = True
    attach_confidence_defaults: bool = True


@dataclass
class OutputConfig:
    root: Path = FINAL_ROOT / "artifacts"

    labeling_root: Path = FINAL_ROOT / "artifacts" / "labeling"
    features_root: Path = FINAL_ROOT / "artifacts" / "features"
    modeling_root: Path = FINAL_ROOT / "artifacts" / "modeling"
    postprocessing_root: Path = FINAL_ROOT / "artifacts" / "postprocessing"
    logs_root: Path = FINAL_ROOT / "artifacts" / "logs"


@dataclass
class LabelingConfig:
    use_revised_sprint3: bool = True
    sprint3_variant_name: str = "revised"

    build_confidence_masks: bool = True
    save_object_id_raster: bool = True

    # notebook/pipeline behavior
    run_sprint3_locally_when_possible: bool = True
    standardize_sprint3_outputs: bool = True
    perform_deduplication: bool = True

    # Sprint 3 execution behavior
    sprint3_variants: tuple[str, ...] = ("original", "revised")
    sprint3_force_rerun: bool = False
    sprint3_require_success_artifacts: bool = True

    # PTX selection / storage
    max_ptx_per_site: int | None = 1
    cleanup_ptx_after_all_variants: bool = True
    cleanup_stale_ptx_before_run: bool = True
    stale_ptx_days: int = 2

    # logging / manifests
    sprint3_manifest_name: str = "sprint3_runs.csv"


@dataclass
class FeatureConfig:
    canonical_grid_source: str = "naip"
    object_table_enabled: bool = True

@dataclass
class LabelingRuntimeConfig:
    storage: PipelineStorageConfig = field(default_factory=PipelineStorageConfig)
    storage_policy: ArtifactStorePolicyConfig = field(default_factory=ArtifactStorePolicyConfig)
    
    require_capability_sprint3: tuple[str, ...] = ("runtime:intelimon_sprint3",)
    require_capability_standardize: tuple[str, ...] = ("runtime:python",)
    require_capability_refine: tuple[str, ...] = ("runtime:python",)
    require_capability_transfer: tuple[str, ...] = ("runtime:labeling_transfer",)
    require_capability_rasterize: tuple[str, ...] = ("runtime:labeling_transfer",)

    resume_partial_runs: bool = True
    reuse_successful_runs: bool = True
    reuse_partial_runs_when_no_new_stages_are_eligible: bool = True


@dataclass
class FeatureRuntimeConfig:
    storage: PipelineStorageConfig = field(default_factory=PipelineStorageConfig)
    storage_policy: ArtifactStorePolicyConfig = field(default_factory=ArtifactStorePolicyConfig)

    require_capability_site_assets: tuple[str, ...] = ("runtime:python",)
    require_capability_canonical_grid: tuple[str, ...] = ("runtime:python", "runtime:rasterio")
    require_capability_chunk_manifest: tuple[str, ...] = ("runtime:python",)
    require_capability_source_ready: tuple[str, ...] = ("runtime:python",)
    require_capability_family_chunk: tuple[str, ...] = ("runtime:python",)
    require_capability_stack_finalize: tuple[str, ...] = ("runtime:python",)
    require_capability_object_aggregation: tuple[str, ...] = ("runtime:python",)
    require_capability_representation_export: tuple[str, ...] = ("runtime:python", "runtime:rasterio")

    resume_partial_runs: bool = True
    reuse_successful_runs: bool = True
    reuse_partial_runs_when_no_new_stages_are_eligible: bool = True

@dataclass
class ProjectConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    data: DataConfig = field(default_factory=DataConfig)
    raster: RasterizationConfig = field(default_factory=RasterizationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    artifact_store: ArtifactStoreConfig = field(default_factory=ArtifactStoreConfig)
    runtime_detection: RuntimeDetectionConfig = field(default_factory=RuntimeDetectionConfig)
    coordination: CoordinationConfig = field(default_factory=CoordinationConfig)
    runtime_policy: RuntimePolicyConfig = field(default_factory=RuntimePolicyConfig)
    runtime_images: list[RuntimeImageSpecConfig] = field(default_factory=lambda: list(DEFAULT_RUNTIME_IMAGES))
    labeling_runtime: LabelingRuntimeConfig = field(default_factory=LabelingRuntimeConfig)
    features_runtime: FeatureRuntimeConfig = field(default_factory=FeatureRuntimeConfig)

    artifact_repair: ArtifactRepairPolicyConfig = field(default_factory=ArtifactRepairPolicyConfig)
    shared_artifacts: SharedArtifactPolicyConfig = field(default_factory=SharedArtifactPolicyConfig)
    grid_health: GridHealthConfig = field(default_factory=GridHealthConfig) 

    debug: bool = False

    @property
    def sites(self) -> list[str]:
        return self.site.sites

    def ensure_dirs(self) -> None:
        roots = [
            self.output.root,
            self.output.labeling_root,
            self.output.features_root,
            self.output.modeling_root,
            self.output.postprocessing_root,
            self.output.logs_root,
        ]
        for p in roots:
            p.mkdir(parents=True, exist_ok=True)
    
        labeling_subdirs = [
            "manifests",
            "ptx_cache",
            "sprint3",
            "objects",
            "masks",
            "confidence",
            "object_id",
            "qa",
            "summaries",
        ]
        for sub in labeling_subdirs:
            (self.output.labeling_root / sub).mkdir(parents=True, exist_ok=True)

    def resolve(self) -> "ProjectConfig":
        """
        Normalize all path-like fields to absolute paths and ensure output dirs exist.
        """
        self.data.project_root = Path(self.data.project_root).resolve()
        self.data.final_root = Path(self.data.final_root).resolve()
        self.data.sprint3_base_dir = Path(self.data.sprint3_base_dir).resolve()
        self.data.sprint3_original_script = Path(self.data.sprint3_original_script).resolve()
        self.data.sprint3_revised_script = Path(self.data.sprint3_revised_script).resolve()
        self.data.sprint3_default_input = Path(self.data.sprint3_default_input).resolve()

        self.output.root = Path(self.output.root).resolve()
        self.output.labeling_root = Path(self.output.labeling_root).resolve()
        self.output.features_root = Path(self.output.features_root).resolve()
        self.output.modeling_root = Path(self.output.modeling_root).resolve()
        self.output.postprocessing_root = Path(self.output.postprocessing_root).resolve()
        self.output.logs_root = Path(self.output.logs_root).resolve()

        self.artifact_store.local_storage_root = Path(self.artifact_store.local_storage_root).resolve()
        self.artifact_store.drive_registry_path = Path(self.artifact_store.drive_registry_path).resolve()
        self.artifact_store.drive_config_path = Path(self.artifact_store.drive_config_path).resolve()
        self.artifact_store.drive_client_secrets_path = Path(self.artifact_store.drive_client_secrets_path).resolve()
        self.artifact_store.drive_credentials_path = Path(self.artifact_store.drive_credentials_path).resolve()

        self.ensure_dirs()
        return self


PipelineConfig = ProjectConfig


def default_config(root: str | Path | None = None) -> PipelineConfig:
    cfg = PipelineConfig()

    if root is not None:
        root = Path(root)
        if not root.is_absolute():
            root = FINAL_ROOT / root

        cfg.output.root = root
        cfg.output.labeling_root = root / "labeling"
        cfg.output.features_root = root / "features"
        cfg.output.modeling_root = root / "modeling"
        cfg.output.postprocessing_root = root / "postprocessing"
        cfg.output.logs_root = root / "logs"

    return cfg.resolve()