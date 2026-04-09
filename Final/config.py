from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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

    # Local Sprint 3 locations
    sprint3_base_dir: Path = Path("../Sprint 3/Base")
    sprint3_original_script: Path = Path("../Sprint 3/Base/IntELiMon_1_1_1.R")
    sprint3_revised_script: Path = Path("../Sprint 3/Base/IntELiMon_1_1_1_revised.R")
    sprint3_default_input: Path = Path("../Sprint 3/Base")
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
    create_multires: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


@dataclass
class RefinementConfig:
    infer_radius_from_area: bool = True
    min_radius_m: float = 0.25
    max_radius_m: float = 4.0
    temporal_half_life_days: float = 365.0


@dataclass
class OutputConfig:
    root: Path = Path("artifacts")
    labeling_root: Path = Path("artifacts/labeling")
    features_root: Path = Path("artifacts/features")
    logs_root: Path = Path("artifacts/logs")


@dataclass
class LabelingConfig:
    use_revised_sprint3: bool = True
    sprint3_variant_name: str = "revised"
    build_confidence_masks: bool = True
    save_object_id_raster: bool = True


@dataclass
class ProjectConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    data: DataConfig = field(default_factory=DataConfig)
    raster: RasterizationConfig = field(default_factory=RasterizationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    debug: bool = False

    def ensure_dirs(self) -> None:
        for p in [self.output.root, self.output.labeling_root, self.output.features_root, self.output.logs_root]:
            p.mkdir(parents=True, exist_ok=True)
        for sub in ["manifests", "objects", "masks", "confidence", "object_id", "qa", "sprint3"]:
            (self.output.labeling_root / sub).mkdir(parents=True, exist_ok=True)


def default_config(root: str | Path = "artifacts") -> ProjectConfig:
    cfg = ProjectConfig()
    root = Path(root)
    cfg.output.root = root
    cfg.output.labeling_root = root / "labeling"
    cfg.output.features_root = root / "features"
    cfg.output.logs_root = root / "logs"
    cfg.ensure_dirs()
    return cfg
