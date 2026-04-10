from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Final.shared_utils import extract_first_date_token, get_logger, safe_stem, yyyymmdd_to_timestamp

logger = get_logger("labeling.sprint3_standardize")


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


def _find_single(pattern_dir: Path, pattern: str) -> Path | None:
    if not pattern_dir.exists():
        return None
    matches = sorted(pattern_dir.glob(pattern))
    return matches[0] if matches else None


def detect_sprint3_run_dirs(root: str | Path) -> list[Path]:
    """
    Detect Sprint 3 run directories.

    A run directory is expected to directly contain some subset of:
    - Shrubs/
    - metrics/
    - Inventory/
    - Fuels/
    - dtm/
    - chm/

    This supports:
    - passing one specific PTX run dir
    - passing a site/variant dir containing many PTX run dirs
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sprint 3 path does not exist: {root}")

    marker_dirs = {"Shrubs", "metrics", "Inventory", "Fuels", "dtm", "chm"}

    # if root itself is already a run dir
    if any((root / m).exists() for m in marker_dirs):
        return [root]

    run_dirs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any((child / m).exists() for m in marker_dirs):
            run_dirs.append(child)

    return run_dirs


def build_local_sprint3_manifest(
    root: str | Path,
    *,
    site_id: str | None = None,
    variant: str | None = None,
) -> pd.DataFrame:
    """
    Build a manifest of locally generated Sprint 3 run outputs.

    Parameters
    ----------
    root
        Can be:
        - one PTX run directory
        - a site/variant directory containing PTX run dirs
    """
    run_dirs = detect_sprint3_run_dirs(root)

    rows = []
    for run_dir in run_dirs:
        shrub_csv = _find_single(run_dir / "Shrubs", "*.csv")
        metrics_csv = _find_single(run_dir / "metrics", "*.csv")
        tree_inventory_csv = _find_single(run_dir / "Inventory", "*_inv.csv")
        fuels_raster = _find_single(run_dir / "Fuels", "*.tif")
        dtm_raster = _find_single(run_dir / "dtm", "*.tif")
        chm_raster = _find_single(run_dir / "chm", "*.tif")
        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"

        ptx_stem = run_dir.name
        date_token = extract_first_date_token(ptx_stem)
        ptx_date = yyyymmdd_to_timestamp(date_token)

        # infer site/variant if not passed
        inferred_site = site_id
        inferred_variant = variant
        parts = run_dir.parts

        if inferred_variant is None:
            for v in ("original", "revised"):
                if v in parts:
                    inferred_variant = v
                    break

        if inferred_site is None:
            # best guess: directory just before "sprint3"
            if "sprint3" in parts:
                idx = parts.index("sprint3")
                if idx - 1 >= 0:
                    inferred_site = parts[idx - 1]

        rows.append(
            {
                "site_id": inferred_site,
                "variant": inferred_variant,
                "ptx_stem": ptx_stem,
                "date_token": date_token,
                "ptx_date": ptx_date,
                "run_dir": str(run_dir),
                "shrub_csv": str(shrub_csv) if shrub_csv else None,
                "metrics_csv": str(metrics_csv) if metrics_csv else None,
                "tree_inventory_csv": str(tree_inventory_csv) if tree_inventory_csv else None,
                "fuels_raster": str(fuels_raster) if fuels_raster else None,
                "dtm_raster": str(dtm_raster) if dtm_raster else None,
                "chm_raster": str(chm_raster) if chm_raster else None,
                "stdout_log": str(stdout_log) if stdout_log.exists() else None,
                "stderr_log": str(stderr_log) if stderr_log.exists() else None,
                "has_shrub_csv": shrub_csv is not None,
                "has_metrics_csv": metrics_csv is not None,
                "has_tree_inventory_csv": tree_inventory_csv is not None,
                "has_fuels_raster": fuels_raster is not None,
                "has_dtm_raster": dtm_raster is not None,
                "has_chm_raster": chm_raster is not None,
            }
        )

    manifest_df = pd.DataFrame(rows)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(
            by=["site_id", "variant", "ptx_date", "ptx_stem"],
            ascending=[True, True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    logger.info("Built Sprint 3 manifest with %d run(s) from %s", len(manifest_df), root)
    return manifest_df


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
    run_dir: str | Path | None = None,
    variant: str | None = None,
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

    plot_id = shrub_csv.stem
    date_token = extract_first_date_token(plot_id)
    ptx_date = yyyymmdd_to_timestamp(date_token)

    df["site_id"] = site_id
    df["plot_id"] = plot_id
    df["ptx_stem"] = plot_id
    df["ptx_date_token"] = date_token
    df["ptx_date"] = ptx_date

    df["source_file"] = str(shrub_csv)
    df["source_version"] = source_version
    df["variant"] = variant
    df["label_variant"] = label_variant

    df["run_dir"] = str(run_dir) if run_dir else None
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
        "ptx_stem",
        "ptx_date_token",
        "ptx_date",
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
        "variant",
        "label_variant",
        "run_dir",
        "metrics_csv",
        "tree_inventory_csv",
        "fuels_raster",
        "dtm_raster",
        "chm_raster",
        "input_ptx",
    ]

    extra_cols = [c for c in df.columns if c not in canonical_order]
    out = df[canonical_order + extra_cols]
    logger.info("Loaded %d shrub object(s) from %s", len(out), shrub_csv)
    return out


def standardize_sprint3_manifest(
    manifest_df: pd.DataFrame,
    *,
    source_version_prefix: str = "sprint3",
    label_variant: str = "base",
    keep_only_valid_runs: bool = True,
    require_success_returncode: bool = True,
) -> pd.DataFrame:
    """
    Standardize Sprint 3 outputs from either:
    1. build_local_sprint3_manifest(...)
    2. the runner/results manifest CSV

    Expected minimum columns:
    - site_id
    - variant (optional but preferred)
    - shrub_csv
    """
    if manifest_df.empty:
        return pd.DataFrame()

    work_df = manifest_df.copy()

    # If this came from sprint3_results manifest, keep only successful runs.
    if require_success_returncode and "returncode" in work_df.columns:
        work_df = work_df[work_df["returncode"] == 0].copy()

    if keep_only_valid_runs:
        if "has_shrub_csv" in work_df.columns:
            work_df = work_df[work_df["has_shrub_csv"]].copy()
        elif "shrub_csv" in work_df.columns:
            work_df = work_df[work_df["shrub_csv"].notna()].copy()
        else:
            raise ValueError(
                "Manifest does not contain either 'has_shrub_csv' or 'shrub_csv'."
            )

    frames = []
    for _, row in work_df.iterrows():
        shrub_csv = row.get("shrub_csv")
        if not shrub_csv or pd.isna(shrub_csv):
            continue

        variant = row.get("variant")
        source_version = (
            f"{source_version_prefix}_{variant}"
            if pd.notna(variant)
            else source_version_prefix
        )

        df = load_sprint3_shrub_csv(
            shrub_csv=shrub_csv,
            site_id=row.get("site_id"),
            source_version=source_version,
            label_variant=label_variant,
            metrics_csv=row.get("metrics_csv"),
            tree_inventory_csv=row.get("tree_inventory_csv"),
            fuels_raster=row.get("fuels_raster"),
            dtm_raster=row.get("dtm_raster"),
            chm_raster=row.get("chm_raster"),
            input_ptx=row.get("input_ptx"),
            run_dir=row.get("output_dir", row.get("run_dir")),
            variant=variant,
        )
        frames.append(df)

    out = combine_standardized_objects(frames)
    logger.info("Standardized %d Sprint 3 object rows from %d manifest rows", len(out), len(work_df))
    return out


def standardize_sprint3_directory(
    root: str | Path,
    *,
    site_id: str | None = None,
    variant: str | None = None,
    source_version_prefix: str = "sprint3",
    label_variant: str = "base",
    keep_only_valid_runs: bool = True,
) -> pd.DataFrame:
    manifest_df = build_local_sprint3_manifest(root, site_id=site_id, variant=variant)
    return standardize_sprint3_manifest(
        manifest_df,
        source_version_prefix=source_version_prefix,
        label_variant=label_variant,
        keep_only_valid_runs=keep_only_valid_runs,
    )


def combine_standardized_objects(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_objects(objects_df: pd.DataFrame) -> pd.DataFrame:
    if objects_df.empty:
        return pd.DataFrame()

    summary = (
        objects_df.groupby(["site_id", "variant", "plot_id"], dropna=False)
        .agg(
            n_objects=("object_id", "count"),
            n_valid=("valid_object", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_height_tls=("height_tls", "mean"),
            mean_area_tls=("area_tls", "mean"),
            mean_radius_m=("radius_m", "mean"),
            mean_n_points=("n_points", "mean"),
        )
        .reset_index()
        .sort_values(["site_id", "variant", "plot_id"])
    )
    return summary