from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
import numpy as np

from Final.artifact_store import ArtifactStore
from Final.shared_utils import get_logger
from Final.models import RuntimeTier, RepresentationTarget

from Final.features.config import fe_cfg
from Final.features.models import (
    ChunkManifest,
    CanonicalGrid,
    FeatureFamilySpec,
    RasterStackRegistry,
    SiteAssetBundle,
    SourceRasterBundle,
    SourceIngestRecord,
)
from Final.features.artifact_io import (
    persist_existing_file_artifact,
    remote_artifact_exists,
)
from Final.features.raster_io import (
    align_bundle_to_grid,
    read_raster_bundle,
    validate_cached_raster,
    validate_optional_raster_asset,
)
from Final.features.source_registry import run_source_chunked_pipeline
from Final.features.assets import site_rap_cache_root, site_rap_rel_path
from Final.features.ee_utils import (
    ensure_ee_initialized,
    canonical_grid_bounds_wgs84,
    canonical_grid_bbox_region_coords,
    ee_rap_image_for_grid,
    export_ee_image_to_geotiff,
)
from Final.features.source_specs import SOURCE_SPECS
from Final.features.fe_2d import get_uniform_blur
from Final.features.fe_rap import build_rap_prior_feature_dict

LOGGER = get_logger("features.source_rap")

RAP_FAMILY_SPECS = {
    "prior": FeatureFamilySpec(
        key="prior",
        source_name="rap",
        runtime_tier=RuntimeTier.CHEAP,
        required_halo_px=8,
        representation_target=RepresentationTarget.RASTER,
        notes="Local shrub presence and neighborhood ecological priors.",
    ),
}

def prepare_rap_asset(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    canonical_grid: CanonicalGrid | None = None,
    force_refresh: bool = False,
    inventory: dict | None = None,
) -> Path:
    local_path = site_rap_cache_root(site_id) / f"{site_id}_rap_ee_10m.tif"
    rel_path = site_rap_rel_path(site_id, filename=local_path.name)

    if (not force_refresh) and local_path.exists() and validate_cached_raster(local_path):
        LOGGER.info("Using cached RAP | site=%s | path=%s", site_id, local_path)
        return local_path

    if not force_refresh and remote_artifact_exists(rel_path, artifact_store=artifact_store):
        LOGGER.info("Hydrating RAP from artifact store | site=%s | rel_path=%s", site_id, rel_path)
        pulled = artifact_store.pull(rel_path, local_path=local_path)
        if validate_cached_raster(pulled):
            return pulled
        LOGGER.warning("Hydrated RAP failed validation | site=%s | path=%s", site_id, pulled)
        Path(pulled).unlink(missing_ok=True)

    if canonical_grid is None:
        raise ValueError("prepare_rap_asset requires canonical_grid when no cached RAP raster is available.")

    ensure_ee_initialized(project="shrubwise-dc-488219")

    west, south, east, north = canonical_grid_bounds_wgs84(canonical_grid)
    region_coords = canonical_grid_bbox_region_coords(canonical_grid)
    image = ee_rap_image_for_grid(canonical_grid)

    LOGGER.info(
        "Exporting RAP from Earth Engine | site=%s | out=%s | bounds_wgs84=(%.6f, %.6f, %.6f, %.6f)",
        site_id, local_path, west, south, east, north
    )

    export_ee_image_to_geotiff(
        image,
        out_path=local_path,
        region_coords=region_coords,
        scale=10.0,
        crs="EPSG:4326",
    )

    if not validate_cached_raster(local_path):
        raise RuntimeError(f"Exported RAP is unreadable for site={site_id}: {local_path}")

    persist_existing_file_artifact(
        local_path,
        artifact_key="site_rap_raster",
        site_id=site_id,
        artifact_store=artifact_store,
        filename=local_path.name,
    )

    if not local_path.exists():
        LOGGER.info("RAP local file was pruned after push; rehydrating for current use | site=%s", site_id)
        artifact_store.pull(rel_path, local_path=local_path)

    LOGGER.info("Prepared RAP asset | site=%s | path=%s", site_id, local_path)
    return local_path

def infer_rap_band_names(path: str | Path) -> list[str]:
    path = Path(path)
    with rasterio.open(path) as src:
        count = src.count

    if count >= 4:
        default = ["SHR", "TRE", "PFG", "AFG"]
        if count > 4:
            default += [f"band_{i}" for i in range(5, count + 1)]
        return default[:count]

    return [f"band_{i}" for i in range(1, count + 1)]

def build_rap_source_inventory_record(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
) -> SourceIngestRecord:
    local_cached = site_assets.source_assets.get("rap")
    if validate_optional_raster_asset(local_cached):
        return SourceIngestRecord(
            site_id=site_id,
            source_name="rap",
            asset_path=str(local_cached),
            status="cached_local",
            notes=["RAP cached locally."],
        )

    rel_path = site_rap_rel_path(site_id)
    if remote_artifact_exists(rel_path, artifact_store=artifact_store):
        return SourceIngestRecord(
            site_id=site_id,
            source_name="rap",
            asset_path=rel_path,
            status="cached_remote_lazy",
            notes=["RAP cached remotely and will be hydrated lazily."],
        )

    return SourceIngestRecord(
        site_id=site_id,
        source_name="rap",
        asset_path=None,
        status="not_materialized_yet",
        notes=["RAP has not been materialized yet."],
    )

def maybe_prune_local_site_rap_after_load(
    site_id: str,
    tif_path: str | Path,
    *,
    artifact_store: ArtifactStore,
) -> None:
    tif_path = Path(tif_path)
    rel_path = site_rap_rel_path(site_id, filename=tif_path.name)

    if not fe_cfg.persistence.prune_local_after_remote_push:
        return
    if not remote_artifact_exists(rel_path, artifact_store=artifact_store):
        return

    try:
        if tif_path.exists():
            tif_path.unlink()
            LOGGER.info("Pruned local RAP site asset after in-memory load | site=%s | path=%s", site_id, tif_path)
    except Exception as e:
        LOGGER.warning("Failed to prune local RAP asset | site=%s | path=%s | err=%s", site_id, tif_path, e)

def load_rap_source_bundle(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
    force_refresh: bool = False,
) -> SourceRasterBundle:
    tif_path = prepare_rap_asset(
        site_id,
        artifact_store=artifact_store,
        canonical_grid=canonical_grid,
        force_refresh=force_refresh,
        inventory=site_assets.source_assets.get("source_inventory"),
    )

    bundle = read_raster_bundle(
        tif_path,
        site_id=site_id,
        source_name="rap",
        band_names=infer_rap_band_names(tif_path),
    )

    maybe_prune_local_site_rap_after_load(
        site_id,
        tif_path,
        artifact_store=artifact_store,
    )
    return bundle

def align_rap_to_canonical_grid(
    bundle: SourceRasterBundle,
    *,
    canonical_grid: CanonicalGrid,
) -> SourceRasterBundle:
    return align_bundle_to_grid(
        bundle,
        dst_grid=canonical_grid,
        resampling=SOURCE_SPECS["rap"].default_resampling,
    )

def make_rap_prior_compute_fn(
    *,
    context_radius_meters: float,
    approx_native_resolution_m: float = 10.0,
):
    def _compute(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return build_rap_prior_feature_dict(
            arrays,
            shrub_band_name="SHR",
            context_radius_meters=context_radius_meters,
            approx_native_resolution_m=approx_native_resolution_m,
            blur_fn=get_uniform_blur,
        )
    return _compute


def run_rap_chunked_pipeline(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    registry: RasterStackRegistry | None = None,
    force_refresh_source: bool = False,
) -> RasterStackRegistry:
    raw_bundle = load_rap_source_bundle(
        site_id,
        artifact_store=artifact_store,
        site_assets=site_assets,
        canonical_grid=canonical_grid,
        force_refresh=force_refresh_source,
    )

    prior_compute_fn = make_rap_prior_compute_fn(
        context_radius_meters=fe_cfg.rap.context_radius_meters,
        approx_native_resolution_m=10.0,
    )

    return run_source_chunked_pipeline(
        artifact_store=artifact_store,
        site_id=site_id,
        source_name="rap",
        raw_bundle=raw_bundle,
        canonical_grid=canonical_grid,
        chunk_manifest=chunk_manifest,
        family_specs=RAP_FAMILY_SPECS,
        family_compute_fns={"prior": prior_compute_fn},
        family_cfg_payloads={
            "prior": {
                "context_radius_meters": fe_cfg.rap.context_radius_meters,
                "approx_native_resolution_m": 10.0,
            }
        },
        registry=registry,
    )
