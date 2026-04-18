from __future__ import annotations

from pathlib import Path
from typing import Any

from Final.features.source_specs import SOURCE_SPECS
from Final.labeling.io import download_file
import laspy
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from scipy.stats import binned_statistic_2d

from Final.artifact_store import ArtifactStore
from Final.shared_utils import ensure_dir, get_logger
from Final.models import RuntimeTier, RepresentationTarget

from Final.features.config import fe_cfg
from Final.features.models import (
    ChunkManifest,
    CanonicalGrid,
    FeatureFamilySpec,
    RasterStackRegistry,
    SiteAssetBundle,
    SourceRasterBundle,
)
from Final.features.artifact_io import read_json, write_json
from Final.features.source_registry import run_source_chunked_pipeline
from Final.features.raster_io import align_bundle_to_grid
from Final.features.fe_als import (
    calculate_distance_to_tall_canopy,
    calculate_local_relief,
    rasterize_als_feature_dict_from_xyz,
)
from Final.features.fe_2d import get_uniform_blur
from Final.features.source_naip import nan_to_num_copy, gradient_magnitude, local_std

LOGGER = get_logger("features.source_als")

def als_metadata_frame(site_assets: SiteAssetBundle) -> pd.DataFrame:
    rows = site_assets.source_assets.get("als_metadata", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def normalize_als_bounds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {}
    for src, dst in [
        ("minx", "min_x"),
        ("maxx", "max_x"),
        ("miny", "min_y"),
        ("maxy", "max_y"),
        ("xmin", "min_x"),
        ("xmax", "max_x"),
        ("ymin", "min_y"),
        ("ymax", "max_y"),
    ]:
        if src in out.columns and dst not in out.columns:
            rename_map[src] = dst
    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def canonical_grid_bounds(grid: CanonicalGrid) -> dict[str, float]:
    x_min, y_max = grid.transform * (0, 0)
    x_max, y_min = grid.transform * (grid.width, grid.height)
    return {
        "min_x": min(x_min, x_max),
        "max_x": max(x_min, x_max),
        "min_y": min(y_min, y_max),
        "max_y": max(y_min, y_max),
    }


def bbox_intersects(a: dict[str, float], b: dict[str, float]) -> bool:
    return not (
        a["max_x"] < b["min_x"] or
        a["min_x"] > b["max_x"] or
        a["max_y"] < b["min_y"] or
        a["min_y"] > b["max_y"]
    )

def enrich_als_metadata_with_inventory_urls(
    df: pd.DataFrame,
    *,
    site_assets: SiteAssetBundle,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    inventory = site_assets.source_assets.get("source_inventory", {}) or {}
    inv_files = inventory.get("als", {}).get("files", []) or []

    if not inv_files:
        return out

    inv_by_name = {}
    for entry in inv_files:
        name = entry.get("name")
        if name is not None:
            inv_by_name[str(name)] = entry

    def _lookup_entry(row: pd.Series) -> dict[str, Any] | None:
        candidates = [
            row.get("source_file"),
            row.get("name"),
            row.get("filename"),
            row.get("tile_name"),
        ]
        for c in candidates:
            if c is None:
                continue
            c = str(c)
            if c in inv_by_name:
                return inv_by_name[c]
        return None

    hrefs = []
    urls = []
    download_urls = []
    names = []

    for _, row in out.iterrows():
        entry = _lookup_entry(row)
        if entry is None:
            hrefs.append(None)
            urls.append(None)
            download_urls.append(None)
            names.append(row.get("source_file") or row.get("name"))
        else:
            hrefs.append(entry.get("href"))
            urls.append(entry.get("url"))
            download_urls.append(entry.get("download_url") or entry.get("url") or entry.get("href"))
            names.append(entry.get("name"))

    if "href" not in out.columns:
        out["href"] = hrefs
    else:
        out["href"] = out["href"].where(out["href"].notna(), pd.Series(hrefs, index=out.index))

    if "url" not in out.columns:
        out["url"] = urls
    else:
        out["url"] = out["url"].where(out["url"].notna(), pd.Series(urls, index=out.index))

    if "download_url" not in out.columns:
        out["download_url"] = download_urls
    else:
        out["download_url"] = out["download_url"].where(
            out["download_url"].notna(),
            pd.Series(download_urls, index=out.index),
        )

    if "name" not in out.columns:
        out["name"] = names
    else:
        out["name"] = out["name"].where(out["name"].notna(), pd.Series(names, index=out.index))

    return out

def build_als_tile_inventory(site_assets: SiteAssetBundle, canonical_grid: CanonicalGrid) -> pd.DataFrame:
    df = als_metadata_frame(site_assets)
    if df.empty:
        LOGGER.warning("ALS tile inventory empty.")
        return df

    df = normalize_als_bounds(df)
    df = enrich_als_metadata_with_inventory_urls(df, site_assets=site_assets)

    grid_bbox = canonical_grid_bounds(canonical_grid)

    if {"min_x", "max_x", "min_y", "max_y"}.issubset(df.columns):
        overlaps = []
        for _, row in df.iterrows():
            tile_bbox = {
                "min_x": float(row["min_x"]),
                "max_x": float(row["max_x"]),
                "min_y": float(row["min_y"]),
                "max_y": float(row["max_y"]),
            }
            overlaps.append(bbox_intersects(tile_bbox, grid_bbox))
        df["intersects_canonical_grid"] = overlaps
    else:
        df["intersects_canonical_grid"] = True

    LOGGER.info("Built ALS tile inventory | rows=%d", len(df))
    return df

def validate_als_tile_inventory(
    tiles_df: pd.DataFrame,
    *,
    site_id: str,
) -> pd.DataFrame:
    if tiles_df.empty:
        raise ValueError(f"ALS tile inventory is empty for site={site_id}")

    out = tiles_df.copy()

    if "intersects_canonical_grid" in out.columns:
        out = out[out["intersects_canonical_grid"]].copy()

    if out.empty:
        raise ValueError(f"No ALS tiles intersect the canonical grid for site={site_id}")

    url_cols = [c for c in ["href", "url", "download_url"] if c in out.columns]
    if not url_cols:
        raise ValueError(
            f"ALS tile inventory has no URL-bearing columns for site={site_id}. "
            f"Columns={list(out.columns)}"
        )

    has_url = pd.Series(False, index=out.index)
    for c in url_cols:
        has_url = has_url | out[c].notna()

    n_bad = int((~has_url).sum())
    if n_bad > 0:
        LOGGER.warning(
            "ALS inventory has rows with no download URL | site=%s | n_bad=%d",
            site_id,
            n_bad,
        )
        out = out[has_url].copy()

    if out.empty:
        raise ValueError(f"ALS tile inventory became empty after URL validation for site={site_id}")

    return out.reset_index(drop=True)

def structural_raster_cache_root(site_id: str) -> Path:
    return ensure_dir(fe_cfg.cache_root / "als_structural_rasters" / site_id)


def structural_raster_manifest_path(site_id: str) -> Path:
    return structural_raster_cache_root(site_id) / "structural_raster_manifest.json"


def load_structural_raster_manifest(site_id: str) -> dict[str, Any]:
    path = structural_raster_manifest_path(site_id)
    if path.exists():
        return read_json(path)
    return {"site_id": site_id, "products": []}


def save_structural_raster_manifest(site_id: str, payload: dict[str, Any]) -> Path:
    path = structural_raster_manifest_path(site_id)
    write_json(path, payload)
    LOGGER.info("Saved structural raster manifest | site=%s | path=%s", site_id, path)
    return path

def maybe_prune_local_raw_als_tile(local_tile: str | Path) -> None:
    local_tile = Path(local_tile)

    if not fe_cfg.persistence.prune_local_after_remote_push:
        return

    try:
        if local_tile.exists():
            local_tile.unlink()
            LOGGER.info("Pruned local raw ALS tile | path=%s", local_tile)
    except Exception as e:
        LOGGER.warning("Failed to prune local raw ALS tile | path=%s | err=%s", local_tile, e)

ALS_FAMILY_SPECS = {
    "height_structure": FeatureFamilySpec(
        key="height_structure",
        source_name="als",
        runtime_tier=RuntimeTier.EXPENSIVE,
        required_halo_px=8,
        representation_target=RepresentationTarget.RASTER,
        notes="ALS-derived structural rasters: canopy height, mean height, density, canopy distance, local relief, and roughness proxies.",
    ),
}

def select_intersecting_als_tiles(
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
) -> pd.DataFrame:
    df = build_als_tile_inventory(site_assets, canonical_grid)
    if df.empty:
        return df

    df = validate_als_tile_inventory(df, site_id=canonical_grid.site_id)
    return df.reset_index(drop=True)


def _safe_height_values_from_las(las) -> np.ndarray:
    if hasattr(las, "HeightAboveGround"):
        return np.asarray(las.HeightAboveGround, dtype=np.float32)
    return np.asarray(las.z, dtype=np.float32)


def rasterize_als_structural_metrics_from_las(
    las_path: str | Path,
    *,
    site_id: str,
    tile_id: str,
    resolution: float = 1.0,
) -> SourceRasterBundle:
    las_path = Path(las_path)
    las = laspy.read(las_path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = _safe_height_values_from_las(las)

    arrays, bounds = rasterize_als_feature_dict_from_xyz(
        x,
        y,
        z,
        resolution=resolution,
        tall_canopy_threshold_m=fe_cfg.als.tall_canopy_threshold_m,
        local_relief_window_px=fe_cfg.als.local_relief_window_px,
        knn_k=fe_cfg.als.knn_k,
    )

    x_min, _, _, y_max = bounds
    transform = from_origin(x_min, y_max, resolution, resolution)

    return SourceRasterBundle(
        site_id=site_id,
        source_name="als",
        arrays=arrays,
        transform=transform,
        crs=las.header.parse_crs(),
        nodata=np.nan,
        metadata={
            "tile_id": tile_id,
            "path": str(las_path),
            "resolution": resolution,
            "has_hag": hasattr(las, "HeightAboveGround"),
        },
    )

def validate_als_merged_bundle(bundle: SourceRasterBundle, *, site_id: str) -> None:
    if not bundle.arrays:
        raise ValueError(f"Merged ALS bundle has no arrays for site={site_id}")

    expected_core = {"als_chm_max", "als_height_mean", "als_point_count"}
    missing = sorted(expected_core - set(bundle.arrays.keys()))
    if missing:
        raise ValueError(f"Merged ALS bundle missing core layers for site={site_id}: {missing}")

    sample = next(iter(bundle.arrays.values()))
    if sample.ndim != 2:
        raise ValueError(f"Merged ALS bundle arrays are not 2D for site={site_id}")

    LOGGER.info(
        "Validated ALS merged bundle | site=%s | layers=%d | shape=%s",
        site_id,
        len(bundle.arrays),
        sample.shape,
    )

def validate_cached_als_tile(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        backends = laspy.LazBackend.detect_available()
        las = laspy.read(
            path,
            laz_backend=backends[0] if backends else None,
        )
        # touch a little real data so a truncated file fails here
        _ = las.header.point_count
        _ = np.asarray(las.x[: min(32, len(las.x))])
        return True
    except Exception as e:
        LOGGER.warning("ALS tile validation failed | path=%s | err=%s", path, e)
        return False

def prepare_als_tile_local_copy(
    site_id: str,
    *,
    tile_row: pd.Series,
    force_refresh: bool = False,
    max_download_attempts: int = 3,
) -> Path:
    local_dir = structural_raster_cache_root(site_id) / "raw_tiles"
    local_dir.mkdir(parents=True, exist_ok=True)

    name = tile_row.get("name") or tile_row.get("filename") or f"{tile_row.name}.laz"
    local_path = local_dir / str(name)

    href = tile_row.get("href") or tile_row.get("url") or tile_row.get("download_url")
    if href is None:
        raise ValueError(f"Could not find ALS download URL for tile row: {tile_row.to_dict()}")

    if not force_refresh and local_path.exists():
        if validate_cached_als_tile(local_path):
            return local_path
        LOGGER.warning("Removing invalid cached ALS tile | site=%s | tile=%s | path=%s", site_id, name, local_path)
        local_path.unlink(missing_ok=True)

    last_err = None
    for attempt in range(1, max_download_attempts + 1):
        try:
            local_path.unlink(missing_ok=True)
            download_file(href, local_path)

            if not validate_cached_als_tile(local_path):
                raise RuntimeError("Downloaded ALS tile failed validation.")

            LOGGER.info(
                "Downloaded ALS tile | site=%s | tile=%s | attempt=%d | path=%s",
                site_id, name, attempt, local_path,
            )
            return local_path

        except Exception as e:
            last_err = e
            LOGGER.warning(
                "ALS tile download/validation failed | site=%s | tile=%s | attempt=%d | err=%s",
                site_id, name, attempt, e,
            )
            local_path.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to prepare ALS tile after {max_download_attempts} attempts: {name} | err={last_err}")


def align_and_merge_als_tile_bundles(
    bundles: list[SourceRasterBundle],
    *,
    canonical_grid: CanonicalGrid,
) -> SourceRasterBundle:
    if not bundles:
        raise ValueError("No ALS bundles provided for merge.")

    aligned_bundles = [
        align_bundle_to_grid(
            bundle,
            dst_grid=canonical_grid,
            resampling=SOURCE_SPECS["als"].default_resampling,
        )
        for bundle in bundles
    ]

    layer_names = sorted(set().union(*[set(b.arrays.keys()) for b in aligned_bundles]))
    merged: dict[str, np.ndarray] = {}

    for layer_name in layer_names:
        stack = []
        for bundle in aligned_bundles:
            if layer_name in bundle.arrays:
                stack.append(bundle.arrays[layer_name])

        if not stack:
            continue

        stacked = np.stack(stack, axis=0)

        if layer_name == "als_point_count":
            merged[layer_name] = np.nansum(stacked, axis=0).astype(np.float32)
        else:
            with np.errstate(all="ignore"):
                merged_arr = np.nanmax(stacked, axis=0)
            merged[layer_name] = merged_arr.astype(np.float32)

    out = SourceRasterBundle(
        site_id=canonical_grid.site_id,
        source_name="als",
        arrays=merged,
        transform=canonical_grid.transform,
        crs=canonical_grid.crs,
        nodata=np.nan,
        metadata={
            "n_tiles_merged": len(bundles),
            "aligned_to": canonical_grid.source_name,
        },
    )
    validate_als_merged_bundle(out, site_id=canonical_grid.site_id)
    return out

def make_als_height_structure_compute_fn():
    def _compute(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}

        for name, arr in arrays.items():
            out[name] = nan_to_num_copy(arr).astype(np.float32)

        if "als_chm_max" in arrays:
            chm = nan_to_num_copy(arrays["als_chm_max"])
            out["als_chm_gradmag"] = gradient_magnitude(chm)
            out["als_chm_localstd_5"] = local_std(chm, size=5)

        if "als_height_mean" in arrays and "als_chm_max" in arrays:
            out["als_gap_max_minus_mean"] = (
                nan_to_num_copy(arrays["als_chm_max"]) - nan_to_num_copy(arrays["als_height_mean"])
            ).astype(np.float32)

        if "als_point_count" in arrays:
            count = nan_to_num_copy(arrays["als_point_count"])
            out["als_point_density_blur_5"] = get_uniform_blur(count, neighborhood_size=5).astype(np.float32)

        return out

    return _compute


def run_als_chunked_pipeline(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    registry: RasterStackRegistry | None = None,
) -> RasterStackRegistry:
    tiles_df = select_intersecting_als_tiles(site_assets, canonical_grid)
    tiles_df = validate_als_tile_inventory(tiles_df, site_id=site_id)

    if tiles_df.empty:
        raise ValueError(f"No ALS tiles intersect the canonical grid for site={site_id}")

    tile_bundles: list[SourceRasterBundle] = []
    failed_tiles: list[str] = []

    for _, row in tiles_df.iterrows():
        tile_name = str(row.get("name") or row.get("filename") or row.name)
        local_tile = None

        try:
            local_tile = prepare_als_tile_local_copy(site_id, tile_row=row)
            tile_id = Path(local_tile).stem

            bundle = rasterize_als_structural_metrics_from_las(
                local_tile,
                site_id=site_id,
                tile_id=tile_id,
                resolution=fe_cfg.als.structural_raster_resolution_m,
            )
            tile_bundles.append(bundle)

        except Exception as e:
            LOGGER.warning(
                "ALS tile processing failed | site=%s | tile=%s | err=%s",
                site_id, tile_name, e
            )
            failed_tiles.append(tile_name)

        finally:
            if local_tile is not None:
                try:
                    maybe_prune_local_raw_als_tile(local_tile)
                except Exception as e:
                    LOGGER.warning(
                        "ALS raw tile prune failed | site=%s | tile=%s | err=%s",
                        site_id, tile_name, e
                    )

    if not tile_bundles:
        raise ValueError(f"All ALS tiles failed for site={site_id}; failed_tiles={failed_tiles}")

    LOGGER.info(
        "ALS rasterization complete | site=%s | successful_tiles=%d | failed_tiles=%d",
        site_id, len(tile_bundles), len(failed_tiles)
    )

    merged_bundle = align_and_merge_als_tile_bundles(
        tile_bundles,
        canonical_grid=canonical_grid,
    )
    validate_als_merged_bundle(merged_bundle, site_id=site_id)

    return run_source_chunked_pipeline(
        artifact_store=artifact_store,
        site_id=site_id,
        source_name="als",
        raw_bundle=merged_bundle,
        canonical_grid=canonical_grid,
        chunk_manifest=chunk_manifest,
        family_specs=ALS_FAMILY_SPECS,
        family_compute_fns={
            "height_structure": make_als_height_structure_compute_fn(),
        },
        family_cfg_payloads={
            "height_structure": {
                "structural_raster_resolution_m": fe_cfg.als.structural_raster_resolution_m,
                "tall_canopy_threshold_m": fe_cfg.als.tall_canopy_threshold_m,
                "local_relief_window_px": fe_cfg.als.local_relief_window_px,
                "knn_k": fe_cfg.als.knn_k,
            }
        },
        registry=registry,
    )

