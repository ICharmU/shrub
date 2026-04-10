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
class ProjectConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    data: DataConfig = field(default_factory=DataConfig)
    raster: RasterizationConfig = field(default_factory=RasterizationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

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