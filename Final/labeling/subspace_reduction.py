from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import label

from Final.models import ShrubObjectColumns


COLS = ShrubObjectColumns()


@dataclass
class SubspaceReductionConfig:
    min_component_pixels: int = 4
    min_object_confidence: float = 0.55
    min_transform_confidence: float = 0.50
    min_temporal_confidence: float = 0.40
    max_height_m: float = 3.5


def binarize_labels(mask: np.ndarray) -> np.ndarray:
    return (mask != 0).astype(np.uint8)


def contains_shrub_attributes(*attrs: bool) -> bool:
    return any(bool(x) for x in attrs)


def apply_object_subspace_filter(
    df: pd.DataFrame,
    config: SubspaceReductionConfig,
) -> pd.DataFrame:
    """
    Object-level pruning:
    keep objects that still look shrub-like according to the currently
    available metadata/confidence fields.
    """
    out = df.copy()

    keep = np.ones(len(out), dtype=bool)
    reasons = np.array([""] * len(out), dtype=object)

    if COLS.object_confidence in out.columns:
        oc = pd.to_numeric(out[COLS.object_confidence], errors="coerce").to_numpy(dtype=float)
        fail = np.isfinite(oc) & (oc < config.min_object_confidence)
        keep &= ~fail
        reasons[fail] = np.where(reasons[fail] == "", "low_object_confidence", reasons[fail] + "|low_object_confidence")

    if COLS.transform_confidence in out.columns:
        tc = pd.to_numeric(out[COLS.transform_confidence], errors="coerce").to_numpy(dtype=float)
        fail = np.isfinite(tc) & (tc < config.min_transform_confidence)
        keep &= ~fail
        reasons[fail] = np.where(reasons[fail] == "", "low_transform_confidence", reasons[fail] + "|low_transform_confidence")

    if COLS.temporal_confidence in out.columns:
        tmp = pd.to_numeric(out[COLS.temporal_confidence], errors="coerce").to_numpy(dtype=float)
        fail = np.isfinite(tmp) & (tmp < config.min_temporal_confidence)
        keep &= ~fail
        reasons[fail] = np.where(reasons[fail] == "", "low_temporal_confidence", reasons[fail] + "|low_temporal_confidence")

    if COLS.height_tls in out.columns:
        height = pd.to_numeric(out[COLS.height_tls], errors="coerce").to_numpy(dtype=float)
        fail = np.isfinite(height) & (height > config.max_height_m)
        keep &= ~fail
        reasons[fail] = np.where(reasons[fail] == "", "too_tall_for_shrub", reasons[fail] + "|too_tall_for_shrub")

    if COLS.valid_object in out.columns:
        out[COLS.valid_object] = out[COLS.valid_object].fillna(True).astype(bool) & keep
    else:
        out[COLS.valid_object] = keep

    out["subspace_filter_reason"] = reasons
    return out


def filter_small_mask_components(
    binary_mask: np.ndarray,
    confidence_mask: np.ndarray | None = None,
    object_id_mask: np.ndarray | None = None,
    *,
    min_component_pixels: int = 4,
):
    """
    Pixel-level mask cleanup:
    remove tiny connected components after rasterization.
    """
    structure = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )
    labeled, n = label(binary_mask.astype(bool), structure=structure)

    keep = np.zeros(binary_mask.shape, dtype=bool)
    for component_id in range(1, n + 1):
        comp = labeled == component_id
        if int(comp.sum()) >= int(min_component_pixels):
            keep |= comp

    filtered_binary = np.where(keep, binary_mask, 0).astype(binary_mask.dtype)

    filtered_conf = None
    if confidence_mask is not None:
        filtered_conf = np.where(keep, confidence_mask, 0).astype(confidence_mask.dtype)

    filtered_object_id = None
    if object_id_mask is not None:
        filtered_object_id = np.where(keep, object_id_mask, 0).astype(object_id_mask.dtype)

    return filtered_binary, filtered_conf, filtered_object_id