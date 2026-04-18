from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Final.shared_utils import get_logger
from Final.config import default_config

cfg = default_config()
LOGGER = get_logger("features.labeling_bridge")

def labeling_summary_dir() -> Path:
    return cfg.output.labeling_root / "summaries"


def labeling_manifest_dir() -> Path:
    return cfg.output.labeling_root / "manifests"


def labeling_pipeline_runs_root() -> Path:
    return cfg.output.labeling_root / "pipeline_runs"


def latest_labeling_run_summary_paths() -> tuple[Path | None, Path | None]:
    runs_root = labeling_pipeline_runs_root()
    if not runs_root.exists():
        return None, None

    candidate_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    candidate_dirs = sorted(candidate_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in candidate_dirs:
        objects_csv = run_dir / "summaries" / "objects_all.csv"
        artifacts_csv = run_dir / "summaries" / "artifacts_all.csv"
        if objects_csv.exists() and artifacts_csv.exists():
            return objects_csv, artifacts_csv

    return None, None


def normalize_site_id_value(x: Any) -> str:
    return str(x).strip().lower().replace("_", "-").replace(" ", "-")


def load_labeling_objects_summary() -> pd.DataFrame:
    # First prefer the newest config-specific run summaries
    run_objects_csv, _ = latest_labeling_run_summary_paths()
    if run_objects_csv is not None and run_objects_csv.exists():
        df = pd.read_csv(run_objects_csv)
        if "site_id" in df.columns:
            df["_site_id_norm"] = df["site_id"].map(normalize_site_id_value)
        LOGGER.info("Loaded labeling objects summary from latest pipeline run | rows=%d | path=%s", len(df), run_objects_csv)
        return df

    # Fall back to global summaries
    obj_csv = labeling_summary_dir() / "objects_all.csv"
    if obj_csv.exists():
        df = pd.read_csv(obj_csv)
        if "site_id" in df.columns:
            df["_site_id_norm"] = df["site_id"].map(normalize_site_id_value)
        LOGGER.info("Loaded labeling objects summary from global summaries | rows=%d | path=%s", len(df), obj_csv)
        return df

    LOGGER.warning("No labeling object summary found in pipeline_runs or global summaries.")
    return pd.DataFrame()


def load_labeling_artifacts_summary() -> pd.DataFrame:
    # First prefer the newest config-specific run summaries
    _, run_artifacts_csv = latest_labeling_run_summary_paths()
    if run_artifacts_csv is not None and run_artifacts_csv.exists():
        df = pd.read_csv(run_artifacts_csv)
        if "site_id" in df.columns:
            df["_site_id_norm"] = df["site_id"].map(normalize_site_id_value)
        LOGGER.info("Loaded labeling artifacts summary from latest pipeline run | rows=%d | path=%s", len(df), run_artifacts_csv)
        return df

    art_csv = labeling_summary_dir() / "artifacts_all.csv"
    if art_csv.exists():
        df = pd.read_csv(art_csv)
        if "site_id" in df.columns:
            df["_site_id_norm"] = df["site_id"].map(normalize_site_id_value)
        LOGGER.info("Loaded labeling artifacts summary from global summaries | rows=%d | path=%s", len(df), art_csv)
        return df

    manifest_csv = labeling_manifest_dir() / "sprint4_artifacts.csv"
    if manifest_csv.exists():
        df = pd.read_csv(manifest_csv)
        if "site_id" in df.columns:
            df["_site_id_norm"] = df["site_id"].map(normalize_site_id_value)
        LOGGER.info("Loaded labeling artifact manifest fallback | rows=%d | path=%s", len(df), manifest_csv)
        return df

    LOGGER.warning("No labeling artifact summary/manifest found.")
    return pd.DataFrame()

def label_objects_for_site(site_id: str) -> pd.DataFrame:
    if LABEL_OBJECTS_DF.empty:
        LOGGER.warning("LABEL_OBJECTS_DF is empty.")
        return pd.DataFrame()

    wanted = normalize_site_id_value(site_id)

    if "_site_id_norm" in LABEL_OBJECTS_DF.columns:
        out = LABEL_OBJECTS_DF[LABEL_OBJECTS_DF["_site_id_norm"] == wanted].copy()
    else:
        out = LABEL_OBJECTS_DF[LABEL_OBJECTS_DF["site_id"].map(normalize_site_id_value) == wanted].copy()

    LOGGER.info(
        "label_objects_for_site | requested=%s | matched_rows=%d | unique_sites=%s",
        site_id,
        len(out),
        sorted(LABEL_OBJECTS_DF["site_id"].astype(str).unique().tolist())[:20] if "site_id" in LABEL_OBJECTS_DF.columns else [],
    )
    return out


def label_artifacts_for_site(site_id: str) -> pd.DataFrame:
    if LABEL_ARTIFACTS_DF.empty:
        return pd.DataFrame()

    wanted = normalize_site_id_value(site_id)

    if "_site_id_norm" in LABEL_ARTIFACTS_DF.columns:
        return LABEL_ARTIFACTS_DF[LABEL_ARTIFACTS_DF["_site_id_norm"] == wanted].copy()

    return LABEL_ARTIFACTS_DF[LABEL_ARTIFACTS_DF["site_id"].map(normalize_site_id_value) == wanted].copy()


def best_label_artifact_for_site(site_id: str, resolution_m: float = 1.0) -> pd.Series | None:
    site_df = label_artifacts_for_site(site_id)
    if site_df.empty:
        return None

    if "resolution_m" in site_df.columns:
        site_df = site_df[np.isclose(site_df["resolution_m"].astype(float), float(resolution_m))].copy()

    if site_df.empty:
        return None

    sort_cols = [c for c in ["plot_id", "source_version"] if c in site_df.columns]
    if sort_cols:
        site_df = site_df.sort_values(sort_cols)

    return site_df.iloc[0]

LABEL_OBJECTS_DF = load_labeling_objects_summary()
LABEL_ARTIFACTS_DF = load_labeling_artifacts_summary()