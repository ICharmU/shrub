from __future__ import annotations

import math

import numpy as np
import pandas as pd

from Final.config import ProjectConfig
from Final.models import ShrubObjectColumns


COLS = ShrubObjectColumns()


def radius_from_area(area_m2: float | np.ndarray, min_radius: float = 0.25, max_radius: float = 4.0) -> np.ndarray:
    area = np.asarray(area_m2, dtype=float)
    radius = np.sqrt(np.clip(area, 0.0, None) / math.pi)
    return np.clip(radius, min_radius, max_radius)


def compute_basic_geometry_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    area = pd.to_numeric(out[COLS.area_tls], errors="coerce").to_numpy(dtype=float)
    perimeter = np.where(np.isfinite(area) & (area > 0), 2.0 * np.sqrt(math.pi * area), np.nan)
    compactness = np.where(
        np.isfinite(area) & np.isfinite(perimeter) & (perimeter > 0),
        4.0 * math.pi * area / (perimeter ** 2),
        np.nan,
    )
    out[COLS.perimeter_tls] = perimeter
    out[COLS.compactness] = compactness
    out[COLS.elongation] = np.nan  # populated after better Sprint 3 geometry extraction lands
    out[COLS.bbox_minx] = out[COLS.x_tls]
    out[COLS.bbox_miny] = out[COLS.y_tls]
    out[COLS.bbox_maxx] = out[COLS.x_tls]
    out[COLS.bbox_maxy] = out[COLS.y_tls]
    return out


def attach_radius(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    out = df.copy()
    has_radius = COLS.radius_m in out.columns
    if has_radius:
        radius = pd.to_numeric(out[COLS.radius_m], errors="coerce").to_numpy(dtype=float)
        source = np.where(np.isfinite(radius) & (radius > 0), "observed", "missing")
    else:
        radius = np.full(len(out), np.nan, dtype=float)
        source = np.array(["missing"] * len(out), dtype=object)

    if cfg.refinement.infer_radius_from_area:
        inferred = radius_from_area(
            out[COLS.area_tls].to_numpy(dtype=float),
            min_radius=cfg.refinement.min_radius_m,
            max_radius=cfg.refinement.max_radius_m,
        )
        replace = ~np.isfinite(radius) | (radius <= 0)
        radius[replace] = inferred[replace]
        source[replace] = "inferred_from_area"

    replace_default = ~np.isfinite(radius) | (radius <= 0)
    radius[replace_default] = cfg.raster.default_radius_m
    source[replace_default] = "default"

    out[COLS.radius_m] = radius
    out[COLS.radius_source] = source
    return out


def compute_object_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    area = pd.to_numeric(out[COLS.area_tls], errors="coerce").to_numpy(dtype=float)
    npts = pd.to_numeric(out[COLS.n_points], errors="coerce").to_numpy(dtype=float)
    height = pd.to_numeric(out[COLS.height_tls], errors="coerce").to_numpy(dtype=float)
    score = np.ones(len(out), dtype=float)

    score *= np.where(np.isfinite(area) & (area > 0), 1.0, 0.75)
    score *= np.where(np.isfinite(npts) & (npts >= 100), 1.0, 0.8)
    score *= np.where(np.isfinite(height) & (height <= 3.0), 1.0, 0.85)

    out[COLS.object_confidence] = np.clip(score, 0.0, 1.0)
    out[COLS.boundary_confidence_mode] = "radial"
    return out


def refine_shrub_objects(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    out = df.copy()
    out = compute_basic_geometry_descriptors(out)
    out = attach_radius(out, cfg)
    out = compute_object_confidence(out)

    out[COLS.transform_confidence] = np.nan
    if COLS.temporal_confidence not in out.columns:
        out[COLS.temporal_confidence] = np.nan
    if COLS.dedup_keep not in out.columns:
        out[COLS.dedup_keep] = True
    if COLS.dedup_reason not in out.columns:
        out[COLS.dedup_reason] = ""
    return out
