from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import center_of_mass, label, find_objects
from scipy.signal import convolve2d


def radial_confidence(
    distance_px: np.ndarray,
    radius_px: float,
    center: float = 1.0,
    edge: float = 0.35,
) -> np.ndarray:
    if radius_px <= 0:
        return np.zeros_like(distance_px, dtype=float)
    norm = np.clip(distance_px / radius_px, 0.0, 1.0)
    return center - (center - edge) * norm


def temporal_confidence(date_a, date_b, half_life_days: float = 365.0) -> float:
    if date_a is None or date_b is None:
        return float("nan")
    try:
        delta = abs((date_a - date_b).days)
    except Exception:
        return float("nan")
    return float(math.exp(-math.log(2) * (delta / max(half_life_days, 1e-6))))


def safe_confidence_scalar(value, default: float = 1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(np.clip(value, 0.0, 1.0))


def _connected_components(binary_mask: np.ndarray):
    structure = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )
    labeled, _ = label(binary_mask.astype(bool), structure=structure)
    slices = find_objects(labeled)
    return labeled, slices


def universal_component_confidence(
    shrub_mask: np.ndarray,
    *,
    mean: float | None = None,
    var: float | None = None,
    eps: float = math.e,
) -> np.ndarray:
    """
    Convolution-based confidence for one connected shrub patch.
    Higher confidence in dense interior, lower toward boundaries.
    """
    shrub_mask = shrub_mask.astype(float)

    transform_diag = -math.sqrt(1.0 / (2.0 + eps**2))
    transform_adj = 1.0 / (1.0 + eps)
    kernel = np.array(
        [
            [transform_diag, transform_adj, transform_diag],
            [transform_adj, 1.0, transform_adj],
            [transform_diag, transform_adj, transform_diag],
        ],
        dtype=float,
    )
    kernel = kernel / np.sum(np.abs(kernel))

    weighted = convolve2d(np.pad(shrub_mask, 1), kernel, mode="same")
    weighted = weighted[1:-1, 1:-1]

    if mean is None:
        mean = float(np.nanmean(weighted))
    if var is None:
        var = float(np.nanvar(weighted))

    var = max(var, 1e-6)
    conf = 1.0 - np.exp(-((weighted - mean) ** 2) / (2.0 * var))
    conf = np.where(shrub_mask > 0, conf, 0.0)
    return np.clip(conf, 0.0, 1.0)


def boundary_confidence_from_mask(
    binary_mask: np.ndarray,
    *,
    mode: str = "radial",
    center: float = 1.0,
    edge: float = 0.35,
    eps: float = math.e,
) -> np.ndarray:
    """
    Build a pixel-level boundary confidence map from a binary object mask.

    Modes
    -----
    radial:
        Per connected component, confidence falls from centroid to boundary.
    universal:
        Convolution-style local-density weighting per connected component.
    """
    binary_mask = binary_mask.astype(bool)
    out = np.zeros(binary_mask.shape, dtype=float)

    if not binary_mask.any():
        return out

    labeled, slices = _connected_components(binary_mask)

    for group_id, bbox in enumerate(slices, start=1):
        if bbox is None:
            continue

        comp = (labeled[bbox] == group_id)
        if not comp.any():
            continue

        if mode == "universal":
            comp_conf = universal_component_confidence(comp, eps=eps)
        elif mode == "radial":
            rr, cc = np.indices(comp.shape, dtype=float)
            cy, cx = center_of_mass(comp.astype(np.uint8))
            dist = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)

            boundary_rows, boundary_cols = np.where(comp)
            radius_px = float(np.max(np.sqrt((boundary_rows - cy) ** 2 + (boundary_cols - cx) ** 2)))
            radius_px = max(radius_px, 1e-6)

            comp_conf = radial_confidence(dist, radius_px, center=center, edge=edge)
            comp_conf = np.where(comp, comp_conf, 0.0)
        else:
            raise ValueError(f"Unknown boundary confidence mode: {mode}")

        out[bbox] = np.maximum(out[bbox], comp_conf)

    return np.clip(out, 0.0, 1.0)


def compose_pixel_confidence(
    boundary_conf: np.ndarray,
    *,
    object_confidence: float | None = None,
    temporal_confidence_value: float | None = None,
    transform_confidence_value: float | None = None,
) -> np.ndarray:
    scale = (
        safe_confidence_scalar(object_confidence, 1.0)
        * safe_confidence_scalar(temporal_confidence_value, 1.0)
        * safe_confidence_scalar(transform_confidence_value, 1.0)
    )
    return np.clip(boundary_conf * scale, 0.0, 1.0)