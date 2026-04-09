from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_COLUMNS = {
    "treeID": "object_id",
    "Z": "height_tls",
    "npoints": "n_points",
    "convhull_area": "area_tls",
    "X": "x_tls",
    "Y": "y_tls",
}


def _safe_radius_from_area(area: pd.Series) -> pd.Series:
    area = pd.to_numeric(area, errors="coerce")
    return np.sqrt(area / np.pi)


def load_sprint3_shrub_csv(
    shrub_csv: str | Path,
    *,
    site_id: str,
    source_version: str,
    label_variant: str = "base",
    metrics_csv: str | Path | None = None,
    tree_inventory_csv: str | Path | None = None,
    fuels_raster: str | Path | None = None,
    dtm_raster: str | Path | None = None,
    chm_raster: str | Path | None = None,
    input_ptx: str | Path | None = None,
) -> pd.DataFrame:
    shrub_csv = Path(shrub_csv)
    df = pd.read_csv(shrub_csv)

    rename_map = {k: v for k, v in STANDARD_COLUMNS.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    required = ["object_id", "x_tls", "y_tls"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{shrub_csv} is missing required standardized column '{col}'")

    if "height_tls" not in df.columns:
        df["height_tls"] = np.nan
    if "n_points" not in df.columns:
        df["n_points"] = np.nan
    if "area_tls" not in df.columns:
        df["area_tls"] = np.nan

    df["radius_m"] = _safe_radius_from_area(df["area_tls"])
    df["radius_source"] = np.where(df["radius_m"].notna(), "area_tls", "default")

    df["site_id"] = site_id
    df["plot_id"] = shrub_csv.stem
    df["source_file"] = str(shrub_csv)
    df["source_version"] = source_version
    df["label_variant"] = label_variant

    df["metrics_csv"] = str(metrics_csv) if metrics_csv else None
    df["tree_inventory_csv"] = str(tree_inventory_csv) if tree_inventory_csv else None
    df["fuels_raster"] = str(fuels_raster) if fuels_raster else None
    df["dtm_raster"] = str(dtm_raster) if dtm_raster else None
    df["chm_raster"] = str(chm_raster) if chm_raster else None
    df["input_ptx"] = str(input_ptx) if input_ptx else None

    df["valid_object"] = True
    df["object_confidence"] = 1.0
    df["temporal_confidence"] = 1.0
    df["transform_confidence"] = np.nan
    df["boundary_confidence_mode"] = "uniform"

    canonical_order = [
        "site_id",
        "plot_id",
        "object_id",
        "x_tls",
        "y_tls",
        "height_tls",
        "area_tls",
        "radius_m",
        "radius_source",
        "n_points",
        "valid_object",
        "object_confidence",
        "temporal_confidence",
        "transform_confidence",
        "boundary_confidence_mode",
        "source_file",
        "source_version",
        "label_variant",
        "metrics_csv",
        "tree_inventory_csv",
        "fuels_raster",
        "dtm_raster",
        "chm_raster",
        "input_ptx",
    ]

    extra_cols = [c for c in df.columns if c not in canonical_order]
    return df[canonical_order + extra_cols]


def combine_standardized_objects(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)