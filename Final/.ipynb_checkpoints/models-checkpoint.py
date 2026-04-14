from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# -----------------------------------------------------------------------------
# Column name registries
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ShrubObjectColumns:
    site_id: str = "site_id"
    plot_id: str = "plot_id"
    source_file: str = "source_file"
    source_version: str = "source_version"
    object_id: str = "object_id"

    x_tls: str = "x_tls"
    y_tls: str = "y_tls"
    z_tls: str = "z_tls"

    area_tls: str = "area_tls"
    perimeter_tls: str = "perimeter_tls"
    height_tls: str = "height_tls"
    n_points: str = "n_points"

    radius_m: str = "radius_m"
    radius_source: str = "radius_source"
    compactness: str = "compactness"
    elongation: str = "elongation"
    bbox_minx: str = "bbox_minx"
    bbox_miny: str = "bbox_miny"
    bbox_maxx: str = "bbox_maxx"
    bbox_maxy: str = "bbox_maxy"

    date_tls: str = "date_tls"
    temporal_confidence: str = "temporal_confidence"
    transform_confidence: str = "transform_confidence"
    object_confidence: str = "object_confidence"
    boundary_confidence_mode: str = "boundary_confidence_mode"

    x_als: str = "x_als"
    y_als: str = "y_als"
    x_naip: str = "x_naip"
    y_naip: str = "y_naip"
    row: str = "row"
    col: str = "col"

    valid_object: str = "valid_object"
    label_variant: str = "label_variant"
    dedup_keep: str = "dedup_keep"
    dedup_reason: str = "dedup_reason"


@dataclass(frozen=True)
class LabelArtifactColumns:
    site_id: str = "site_id"
    label_variant: str = "label_variant"
    plot_id: str = "plot_id"
    resolution_m: str = "resolution_m"
    binary_mask_path: str = "binary_mask_path"
    confidence_mask_path: str = "confidence_mask_path"
    object_id_raster_path: str = "object_id_raster_path"
    object_table_path: str = "object_table_path"
    qa_overlay_path: str = "qa_overlay_path"
    n_objects: str = "n_objects"
    n_valid_objects: str = "n_valid_objects"


# -----------------------------------------------------------------------------
# Generic pipeline / scalability framework models
# -----------------------------------------------------------------------------


class PipelineDomain(str, Enum):
    LABELING = "label_engineering"
    FEATURES = "feature_engineering"
    MODELING = "modeling"
    POSTPROCESSING = "postprocessing"
    SHARED = "shared"


class RepresentationTarget(str, Enum):
    RASTER = "raster"
    OBJECT = "object"
    BOTH = "both"


class SpatialScope(str, Enum):
    PIXEL = "pixel-level"
    OBJECT = "object-level"
    PATCH = "patch-level"
    SITE = "site-level"


class ResolutionScope(str, Enum):
    NATIVE = "native_only"
    MULTI = "multi_resolution"
    COARSE = "coarse_only_prior"


class AvailabilityTier(str, Enum):
    UNIVERSAL = "universal"
    MOST = "most_sites"
    PARTIAL = "partial_site_conditional"


class RuntimeTier(str, Enum):
    CHEAP = "cheap"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"


class ModuleStatus(str, Enum):
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass
class ModuleCard:
    name: str
    domain: PipelineDomain
    representation_target: RepresentationTarget
    spatial_scope: SpatialScope
    resolution_scope: ResolutionScope
    availability_tier: AvailabilityTier
    runtime_tier: RuntimeTier

    description: str
    inputs_required: list[str] = field(default_factory=list)
    outputs_produced: list[str] = field(default_factory=list)

    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    status: ModuleStatus = ModuleStatus.CANDIDATE

    gate_coverage: str | None = None
    gate_alignment: str | None = None
    gate_information: str | None = None
    gate_resource: str | None = None
    gate_robustness: str | None = None

    coverage_score: float | None = None
    alignment_score: float | None = None
    information_score: float | None = None
    resource_score: float | None = None
    robustness_score: float | None = None

    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRasterOutputs:
    labels: Any = None
    features: Any = None
    predictions: Any = None
    qa_overlays: Any = None


@dataclass
class CanonicalObjectOutputs:
    objects: Any = None
    predicted_objects: Any = None
    source_provenance: Any = None
    quality_flags: Any = None


@dataclass
class ExperimentState:
    experiment_name: str
    active_modules: list[str]
    notes: str = ""

    raster_outputs: CanonicalRasterOutputs = field(default_factory=CanonicalRasterOutputs)
    object_outputs: CanonicalObjectOutputs = field(default_factory=CanonicalObjectOutputs)
    qa_outputs: dict[str, Any] = field(default_factory=dict)
    section_status: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineRunResult:
    pipeline_name: str
    success: bool
    status: str
    raster_outputs: CanonicalRasterOutputs = field(default_factory=CanonicalRasterOutputs)
    object_outputs: CanonicalObjectOutputs = field(default_factory=CanonicalObjectOutputs)
    qa_outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

@dataclass
class ModuleVariant:
    module_name: str
    enabled: bool = True
    variant_name: str = "default"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageCacheRecord:
    stage_name: str
    data_signature: str
    config_signature: str
    module_variants: list[dict[str, Any]] = field(default_factory=list)
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    created_at: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class PipelineModuleState:
    pipeline_name: str
    stage_modules: dict[str, list[ModuleVariant]] = field(default_factory=dict)

class CacheRetentionMode(str, Enum):
    FULL = "full"
    LEAN = "lean"
    MANIFEST_ONLY = "manifest_only"


@dataclass
class SearchAxis:
    key: str
    values: list[Any]
    stage_name: str | None = None
    module_key: str | None = None


@dataclass
class CachePolicy:
    require_manifest: bool = True
    allow_legacy_reuse: bool = False
    retention_mode: CacheRetentionMode = CacheRetentionMode.LEAN
    artifact_keys_to_prune: tuple[str, ...] = ()
    prune_after_success: bool = False


@dataclass
class ModuleSpec:
    key: str
    stage_name: str
    enabled_key: str | None = None
    variant_key: str | None = None
    param_keys: tuple[str, ...] = ()
    include_in_state: bool = True


@dataclass
class StageSpec:
    name: str
    module_keys: list[str]
    cache_policy: CachePolicy = field(default_factory=CachePolicy)


@dataclass
class PipelineSpec:
    pipeline_name: str
    domain: PipelineDomain
    stages: list[StageSpec]
    modules: dict[str, ModuleSpec]
    search_axes: list[SearchAxis] = field(default_factory=list)


@dataclass
class PipelineStateUpdate:
    section_name: str
    status: str
    active_modules: list[str] = field(default_factory=list)
    qa_outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)