from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from Final.config import default_config

cfg = default_config()

NOTEBOOK_VERSION = "fe_pipeline_v1"

DEFAULT_SOURCE_ORDER = ("naip", "als", "3dep", "rap")


@dataclass
class TimingConfig:
    enabled: bool = True
    log_start_end: bool = True
    warn_if_seconds_over: float = 30.0
    enable_rate_tracking: bool = True
    extrapolation_unit: str = "megapixels"


@dataclass
class ChunkingConfig:
    enabled: bool = True
    chunk_size_px: int = 1024
    halo_px_default: int = 32
    allow_full_raster_fallback: bool = True
    materialize_full_rasters_by_default: bool = False


@dataclass
class PersistenceConfig:
    persist_manifests: bool = True
    persist_chunk_outputs: bool = True
    persist_family_outputs: bool = True
    persist_stack_registry: bool = True
    persist_object_tables: bool = True
    write_runtime_csv: bool = True
    write_summary_csvs: bool = True

    push_large_artifacts_to_remote: bool = cfg.features_runtime.storage_policy.push_large_artifacts_to_remote
    prune_local_after_remote_push: bool = cfg.features_runtime.storage_policy.prune_local_after_remote_push
    verify_remote_before_prune: bool = cfg.features_runtime.storage_policy.verify_remote_before_prune


@dataclass
class ObjectAggregationConfig:
    enabled: bool = True
    include_centroid_sample: bool = True
    centroid_only: bool = False
    square_window_radius_px: int = 3
    use_radius_scaled_window: bool = True
    min_radius_px: int = 2
    max_radius_px: int = 12
    stats: tuple[str, ...] = ("mean", "std", "min", "max")


@dataclass
class NAIPFamilyConfig:
    enabled_families: tuple[str, ...] = ("raw", "veg_idx", "texture", "multiscale")
    heavy_families_enabled: tuple[str, ...] = ()

    texture_entropy_window: int = 9
    texture_lbp_radius: int = 2
    texture_blur_sizes: tuple[int, ...] = (5, 11)
    multiscale_sizes: tuple[int, ...] = (3, 7, 15)

    granulometry_scales: tuple[int, ...] = (3, 5, 7, 11)
    fractal_scales: tuple[int, ...] = (3, 5, 7, 9)
    wavelet_type: str = "db2"
    wavelet_level: int = 2
    gabor_frequencies: tuple[float, ...] = (0.1, 0.2)
    gabor_thetas: tuple[float, ...] = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
    ldp_k: int = 3
    morphology_window_sizes: tuple[int, ...] = (3, 7, 11)

    quantize_for_texture: bool = True
    quantize_levels: int = 256
    base_band_preference: tuple[str, ...] = ("nir", "green", "red", "blue")


@dataclass
class ALSFamilyConfig:
    enabled_families: tuple[str, ...] = ("height_structure",)
    heavy_families_enabled: tuple[str, ...] = ()

    structural_raster_resolution_m: float = 1.0
    tall_canopy_threshold_m: float = 5.0
    local_relief_window_px: int = 5
    knn_k: int = 30


@dataclass
class TerrainFamilyConfig:
    enabled_families: tuple[str, ...] = ("terrain",)
    heavy_families_enabled: tuple[str, ...] = ()

    ruggedness_kernel_radius: int = 3
    tpi_radius_meters: int = 150
    hillshade_azimuth: int = 270
    hillshade_zenith: int = 45


@dataclass
class RAPFamilyConfig:
    enabled_families: tuple[str, ...] = ("prior",)
    context_radius_meters: int = 500


@dataclass
class FeaturePipelineConfig:
    sites: list[str] = field(default_factory=lambda: list(cfg.sites))
    enabled_sources: dict[str, bool] = field(default_factory=lambda: {k: True for k in DEFAULT_SOURCE_ORDER})
    source_order: tuple[str, ...] = DEFAULT_SOURCE_ORDER

    canonical_grid_source: str = cfg.features.canonical_grid_source
    object_table_enabled: bool = cfg.features.object_table_enabled

    force_refresh_assets: bool = False
    force_refresh_features: bool = False
    force_refresh_objects: bool = False
    force_rebuild_chunk_manifest: bool = False

    timing: TimingConfig = field(default_factory=TimingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    object_agg: ObjectAggregationConfig = field(default_factory=ObjectAggregationConfig)

    naip: NAIPFamilyConfig = field(default_factory=NAIPFamilyConfig)
    als: ALSFamilyConfig = field(default_factory=ALSFamilyConfig)
    terrain: TerrainFamilyConfig = field(default_factory=TerrainFamilyConfig)
    rap: RAPFamilyConfig = field(default_factory=RAPFamilyConfig)

    cache_root: Path = cfg.output.features_root / "notebook_cache"
    summary_root: Path = cfg.output.features_root / "notebook_summaries"
    qa_root: Path = cfg.output.features_root / "notebook_qa"

    version: str = NOTEBOOK_VERSION

    def resolve(self) -> "FeaturePipelineConfig":
        self.cache_root = Path(self.cache_root).resolve()
        self.summary_root = Path(self.summary_root).resolve()
        self.qa_root = Path(self.qa_root).resolve()

        for root in (self.cache_root, self.summary_root, self.qa_root):
            root.mkdir(parents=True, exist_ok=True)

        return self


fe_cfg = FeaturePipelineConfig().resolve()