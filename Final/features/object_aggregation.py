from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Iterable

from Final.pipeline_caching import hash_payload
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from Final.artifact_store import ArtifactStore
from Final.shared_utils import ensure_dir, get_logger

from Final.features.config import fe_cfg
from Final.features.models import CanonicalGrid, ChunkManifest, RasterStackRegistry
from Final.features.artifact_io import (
    PersistedArtifactRecord,
    current_fe_config_signature,
    load_npz_dict,
    local_artifact_abs_path,
    persist_json_artifact,
    push_artifact_if_needed,
    remote_artifact_exists,
    render_artifact_rel_path,
)

LOGGER = get_logger("features.object_aggregation")

RUNTIME_LOGS: list[dict[str, Any]] = []
RUNTIME_ACCUM = defaultdict(list)


def log_info(msg: str, *args):
    LOGGER.info(msg, *args)


def stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def config_signature(payload: dict[str, Any]) -> str:
    return hash_payload(payload)


def fe_config_signature() -> str:
    return config_signature(asdict(fe_cfg))


def runtime_frame() -> pd.DataFrame:
    return pd.DataFrame(RUNTIME_LOGS) if RUNTIME_LOGS else pd.DataFrame()


def array_megapixels(arr: np.ndarray) -> float:
    return float(arr.shape[0] * arr.shape[1]) / 1_000_000.0


def estimate_remaining_time(stage_name: str, remaining_units: float) -> float | None:
    rows = RUNTIME_ACCUM.get(stage_name, [])
    rates = [
        r["sec_per_unit"]
        for r in rows
        if np.isfinite(r.get("sec_per_unit", np.nan))
    ]
    if not rates:
        return None
    return float(np.mean(rates) * remaining_units)

@contextmanager
def timed_stage(
    stage_name: str,
    *,
    site_id: str | None = None,
    source_name: str | None = None,
    family_name: str | None = None,
    unit_amount: float | None = None,
    extra: dict[str, Any] | None = None,
):
    start = time.perf_counter()

    if fe_cfg.timing.log_start_end:
        log_info(
            "START | stage=%s | site=%s | source=%s | family=%s",
            stage_name, site_id, source_name, family_name
        )

    try:
        yield
    finally:
        duration = time.perf_counter() - start
        sec_per_unit = duration / unit_amount if (unit_amount is not None and unit_amount > 0) else np.nan

        row = {
            "stage_name": stage_name,
            "site_id": site_id,
            "source_name": source_name,
            "family_name": family_name,
            "duration_sec": duration,
            "unit_amount": unit_amount,
            "sec_per_unit": sec_per_unit,
            **(extra or {}),
        }
        RUNTIME_LOGS.append(row)
        RUNTIME_ACCUM[stage_name].append(row)

        suffix = ""
        if np.isfinite(sec_per_unit):
            suffix = f" | sec_per_unit={sec_per_unit:.4f}"

        log_info(
            "END   | stage=%s | site=%s | source=%s | family=%s | duration=%.2fs%s",
            stage_name, site_id, source_name, family_name, duration, suffix
        )

        if duration >= fe_cfg.timing.warn_if_seconds_over:
            log_info(
                "SLOW  | stage=%s | duration=%.2fs exceeded threshold=%.2fs",
                stage_name, duration, fe_cfg.timing.warn_if_seconds_over
            )

def pixel_from_xy(transform: Affine, x: float, y: float) -> tuple[int, int]:
    col, row = ~transform * (x, y)
    return int(round(row)), int(round(col))


def clip_window(row: int, col: int, radius: int, height: int, width: int) -> tuple[slice, slice]:
    r0 = max(0, row - radius)
    r1 = min(height, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(width, col + radius + 1)
    return slice(r0, r1), slice(c0, c1)


def summarize_patch(arr: np.ndarray, rs: slice, cs: slice, stats: tuple[str, ...]) -> dict[str, float]:
    patch = arr[rs, cs]
    vals = patch[np.isfinite(patch)]
    if vals.size == 0:
        return {stat: np.nan for stat in stats}

    out = {}
    if "mean" in stats:
        out["mean"] = float(np.mean(vals))
    if "std" in stats:
        out["std"] = float(np.std(vals))
    if "min" in stats:
        out["min"] = float(np.min(vals))
    if "max" in stats:
        out["max"] = float(np.max(vals))
    return out

def stack_registry_frame(registry: RasterStackRegistry) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in registry.layers]) if registry.layers else pd.DataFrame()


def parse_chunked_layer_name(layer_name: str) -> tuple[str, str]:
    if "::" not in layer_name:
        raise ValueError(f"Expected chunked layer name with '::', got: {layer_name}")
    base_name, chunk_id = layer_name.split("::", 1)
    return base_name, chunk_id


def selected_registry_rows(
    registry: RasterStackRegistry,
    *,
    source_name: str | None = None,
    family_names: Iterable[str] | None = None,
    base_layer_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = stack_registry_frame(registry)
    if df.empty:
        return df

    if source_name is not None:
        df = df[df["source_name"] == source_name].copy()

    if family_names is not None:
        family_names = set(family_names)
        df = df[df["family_name"].isin(family_names)].copy()

    if base_layer_names is not None:
        wanted = set(base_layer_names)
        df["base_layer_name"] = df["layer_name"].str.split("::").str[0]
        df = df[df["base_layer_name"].isin(wanted)].copy()

    return df.reset_index(drop=True)

def load_chunk_layer_array(
    row: pd.Series,
    *,
    artifact_store: ArtifactStore,
) -> np.ndarray:
    layer_name = row["layer_name"]
    base_name, _ = parse_chunked_layer_name(layer_name)

    local_path = row.get("local_path")
    rel_path = row.get("rel_path")

    path = Path(local_path) if local_path is not None else None

    if path is not None and path.exists():
        payload = load_npz_dict(path)
        if base_name not in payload:
            raise KeyError(f"Base layer {base_name} not present in chunk npz at {path}. Keys={list(payload.keys())}")
        return payload[base_name]

    if rel_path is not None and remote_artifact_exists(rel_path, artifact_store=artifact_store):
        LOGGER.info("PULL CHUNK ARTIFACT | rel_path=%s", rel_path)
        target_local = local_artifact_abs_path(rel_path, artifact_store=artifact_store)
        target_local.parent.mkdir(parents=True, exist_ok=True)

        pulled = artifact_store.pull(rel_path, local_path=target_local)
        try:
            payload = load_npz_dict(pulled)
            if base_name not in payload:
                raise KeyError(f"Base layer {base_name} not present in pulled chunk npz at {pulled}. Keys={list(payload.keys())}")
            return payload[base_name]
        finally:
            try:
                pulled = Path(pulled)
                if pulled.exists():
                    pulled.unlink()
                    LOGGER.info("PRUNED HYDRATED CHUNK NPZ | rel_path=%s", rel_path)
            except Exception as e:
                LOGGER.warning("Failed to prune hydrated chunk NPZ | rel_path=%s | err=%s", rel_path, e)

    raise FileNotFoundError(
        f"Chunk artifact unavailable for layer={layer_name}. "
        f"local_path={local_path}, rel_path={rel_path}, remote_ref={row.get('remote_ref')}"
    )

def modeling_stack_layer_selection(registry: RasterStackRegistry) -> dict[str, list[str]]:
    df = stack_registry_frame(registry)
    if df.empty:
        return {"pixel_layers": [], "object_layers": []}

    df["base_layer_name"] = df["layer_name"].str.split("::").str[0]

    pixel_layers = sorted(df["base_layer_name"].unique().tolist())

    # For now object aggregation can use the same layer universe;
    # later you may restrict to compact/high-value families only.
    object_layers = pixel_layers.copy()

    return {
        "pixel_layers": pixel_layers,
        "object_layers": object_layers,
    }

def assemble_full_layer_from_registry(
    registry: RasterStackRegistry,
    chunk_manifest: ChunkManifest,
    canonical_grid: CanonicalGrid,
    *,
    artifact_store: ArtifactStore,
    target_layer_name: str,
) -> np.ndarray:
    rows = selected_registry_rows(registry, base_layer_names=[target_layer_name])
    if rows.empty:
        raise ValueError(f"No registered chunks found for target layer {target_layer_name}")

    out = np.full((canonical_grid.height, canonical_grid.width), np.nan, dtype=np.float32)
    chunk_lookup = {r.chunk_id: r for r in chunk_manifest.records}

    with timed_stage(
        "stack_assemble",
        site_id=canonical_grid.site_id,
        source_name="assembled_stack",
        family_name=target_layer_name,
        unit_amount=(canonical_grid.height * canonical_grid.width) / 1_000_000.0,
    ):
        for _, row in rows.iterrows():
            _, chunk_id = parse_chunked_layer_name(row["layer_name"])
            record = chunk_lookup[chunk_id]
            arr = load_chunk_layer_array(row, artifact_store=artifact_store)

            if arr.shape != (record.height, record.width):
                raise ValueError(
                    f"Chunk array shape mismatch for {chunk_id}: expected {(record.height, record.width)}, got {arr.shape}"
                )

            out[record.row_start:record.row_end, record.col_start:record.col_end] = arr

    return out

def assemble_selected_layers(
    registry: RasterStackRegistry,
    chunk_manifest: ChunkManifest,
    canonical_grid: CanonicalGrid,
    *,
    artifact_store: ArtifactStore,
    source_name: str | None = None,
    family_names: Iterable[str] | None = None,
    base_layer_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    rows = selected_registry_rows(
        registry,
        source_name=source_name,
        family_names=family_names,
        base_layer_names=base_layer_names,
    )
    if rows.empty:
        return {}

    rows = rows.copy()
    rows["base_layer_name"] = rows["layer_name"].str.split("::").str[0]

    arrays: dict[str, np.ndarray] = {}
    for base_layer_name in sorted(rows["base_layer_name"].unique()):
        arrays[base_layer_name] = assemble_full_layer_from_registry(
            registry,
            chunk_manifest,
            canonical_grid,
            artifact_store=artifact_store,
            target_layer_name=base_layer_name,
        )
    return arrays

def aggregate_selected_registered_layers_to_objects(
    registry: RasterStackRegistry,
    *,
    artifact_store: ArtifactStore,
    site_id: str,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    objects_df: pd.DataFrame,
    source_name: str | None = None,
    family_names: Iterable[str] | None = None,
    base_layer_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    if objects_df.empty:
        LOGGER.info("OBJECT AGG | site=%s | no objects provided", site_id)
        return pd.DataFrame()

    layers = assemble_selected_layers(
        registry,
        chunk_manifest,
        canonical_grid,
        artifact_store=artifact_store,
        source_name=source_name,
        family_names=family_names,
        base_layer_names=base_layer_names,
    )
    if not layers:
        LOGGER.warning("OBJECT AGG | site=%s | no assembled layers selected", site_id)
        return pd.DataFrame()

    work = objects_df.copy()

    x_col = "x_naip" if "x_naip" in work.columns else ("x_als" if "x_als" in work.columns else None)
    y_col = "y_naip" if "y_naip" in work.columns else ("y_als" if "y_als" in work.columns else None)
    if x_col is None or y_col is None:
        raise ValueError("Could not find object coordinates for aggregation.")

    rows = []
    h, w = canonical_grid.height, canonical_grid.width
    px_size = canonical_grid.pixel_size[0]

    with timed_stage(
        "object_aggregation",
        site_id=site_id,
        source_name=source_name or "selected_layers",
        unit_amount=len(work),
        extra={"n_layers": len(layers)},
    ):
        for _, obj in work.iterrows():
            x = float(obj[x_col])
            y = float(obj[y_col])
            row, col = pixel_from_xy(canonical_grid.transform, x, y)

            base = {
                "site_id": site_id,
                "object_id": obj.get("object_id"),
                "plot_id": obj.get("plot_id"),
                "row": row,
                "col": col,
            }

            radius_px = fe_cfg.object_agg.square_window_radius_px
            if fe_cfg.object_agg.use_radius_scaled_window and "radius_m" in obj and pd.notna(obj["radius_m"]):
                radius_px = int(round(float(obj["radius_m"]) / px_size))
                radius_px = max(fe_cfg.object_agg.min_radius_px, min(fe_cfg.object_agg.max_radius_px, radius_px))

            if row < 0 or row >= h or col < 0 or col >= w:
                base["valid_sample"] = False
                rows.append(base)
                continue

            base["valid_sample"] = True
            rs, cs = clip_window(row, col, radius_px, h, w)

            for feat_name, arr in layers.items():
                if fe_cfg.object_agg.include_centroid_sample:
                    base[f"{feat_name}__centroid"] = float(arr[row, col]) if np.isfinite(arr[row, col]) else np.nan
                if not fe_cfg.object_agg.centroid_only:
                    stats = summarize_patch(arr, rs, cs, fe_cfg.object_agg.stats)
                    for stat_name, value in stats.items():
                        base[f"{feat_name}__{stat_name}"] = value

            rows.append(base)

    out_df = pd.DataFrame(rows)
    LOGGER.info(
        "OBJECT AGG DONE | site=%s | n_objects=%d | n_rows=%d | n_cols=%d",
        site_id, len(work), len(out_df), out_df.shape[1]
    )
    return out_df

def persist_object_feature_table(
    obj_df: pd.DataFrame,
    *,
    artifact_store: ArtifactStore,
    site_id: str,
    config_sig: str | None = None,
) -> PersistedArtifactRecord:
    config_sig = config_sig or current_fe_config_signature()
    rel_path = render_artifact_rel_path(
        "object_feature_table",
        site_id=site_id,
        config_signature=config_sig,
    )
    local_path = local_artifact_abs_path(rel_path, artifact_store=artifact_store)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    obj_df.to_csv(local_path, index=False)

    return push_artifact_if_needed(
        local_path,
        artifact_key="object_feature_table",
        rel_path=rel_path,
        artifact_store=artifact_store,
    )


def build_and_persist_object_feature_table(
    registry: RasterStackRegistry,
    *,
    artifact_store: ArtifactStore,
    site_id: str,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    objects_df: pd.DataFrame,
    source_name: str | None = None,
    family_names: Iterable[str] | None = None,
    base_layer_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, PersistedArtifactRecord | None]:
    obj_df = aggregate_selected_registered_layers_to_objects(
        registry,
        artifact_store=artifact_store,
        site_id=site_id,
        canonical_grid=canonical_grid,
        chunk_manifest=chunk_manifest,
        objects_df=objects_df,
        source_name=source_name,
        family_names=family_names,
        base_layer_names=base_layer_names,
    )

    if obj_df.empty:
        LOGGER.warning("No object features produced for site=%s; skipping persistence.", site_id)
        return obj_df, None

    rec = persist_object_feature_table(
        obj_df,
        artifact_store=artifact_store,
        site_id=site_id,
    )
    LOGGER.info("Persisted object feature table | site=%s | path=%s", site_id, rec.local_path)
    return obj_df, rec

def export_single_band_geotiff(
    arr: np.ndarray,
    *,
    canonical_grid: CanonicalGrid,
    out_path: str | Path,
    nodata: float = np.nan,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = canonical_grid.profile(dtype="float32", count=1, nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def export_selected_layers_as_geotiffs(
    assembled_layers: dict[str, np.ndarray],
    *,
    canonical_grid: CanonicalGrid,
    site_id: str,
    export_root: str | Path | None = None,
) -> dict[str, Path]:
    export_root = Path(export_root) if export_root is not None else ensure_dir(fe_cfg.summary_root / site_id / "selected_stack_exports")
    export_root.mkdir(parents=True, exist_ok=True)

    out_paths = {}
    for layer_name, arr in assembled_layers.items():
        out_path = export_root / f"{layer_name}.tif"
        export_single_band_geotiff(arr, canonical_grid=canonical_grid, out_path=out_path)
        out_paths[layer_name] = out_path

    LOGGER.info("Exported %d selected layers as GeoTIFFs | site=%s | root=%s", len(out_paths), site_id, export_root)
    return out_paths


def export_selected_layers_as_npz(
    assembled_layers: dict[str, np.ndarray],
    *,
    site_id: str,
    export_root: str | Path | None = None,
) -> Path:
    export_root = Path(export_root) if export_root is not None else ensure_dir(fe_cfg.summary_root / site_id / "selected_stack_exports")
    export_root.mkdir(parents=True, exist_ok=True)

    out_path = export_root / "selected_layers.npz"
    np.savez_compressed(out_path, **assembled_layers)
    LOGGER.info("Exported selected layers as NPZ | site=%s | path=%s", site_id, out_path)
    return out_path

def build_modeling_ready_feature_views(
    registry: RasterStackRegistry,
    *,
    site_id: str,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    objects_df: pd.DataFrame,
    pixel_layer_names: Iterable[str] | None = None,
    object_layer_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    selection = modeling_stack_layer_selection(registry)

    pixel_layer_names = list(pixel_layer_names) if pixel_layer_names is not None else selection["pixel_layers"]
    object_layer_names = list(object_layer_names) if object_layer_names is not None else selection["object_layers"]

    assembled_pixel_layers = assemble_selected_layers(
        registry,
        chunk_manifest,
        canonical_grid,
        base_layer_names=pixel_layer_names,
    )

    object_feature_df = aggregate_selected_registered_layers_to_objects(
        registry,
        site_id=site_id,
        canonical_grid=canonical_grid,
        chunk_manifest=chunk_manifest,
        objects_df=objects_df,
        base_layer_names=object_layer_names,
    )

    return {
        "site_id": site_id,
        "pixel_layer_names": pixel_layer_names,
        "object_layer_names": object_layer_names,
        "assembled_pixel_layers": assembled_pixel_layers,
        "object_feature_df": object_feature_df,
    }

def persist_modeling_view_manifest(
    *,
    site_id: str,
    pixel_layer_names: list[str],
    object_layer_names: list[str],
    object_feature_shape: tuple[int, int],
) -> PersistedArtifactRecord:
    payload = {
        "site_id": site_id,
        "config_signature": current_fe_config_signature(),
        "pixel_layer_names": pixel_layer_names,
        "object_layer_names": object_layer_names,
        "object_feature_shape": list(object_feature_shape),
    }
    return persist_json_artifact(
        payload,
        artifact_key="runtime_log",  # replace with dedicated spec later if you add one
        site_id=site_id,
    )

