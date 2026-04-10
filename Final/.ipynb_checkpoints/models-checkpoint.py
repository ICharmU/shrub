from __future__ import annotations

from dataclasses import dataclass


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
