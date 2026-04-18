from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from Final.artifact_store import ArtifactStore
from Final.shared_utils import ensure_dir, get_logger
from Final.models import RuntimeTier, RepresentationTarget

from Final.features.config import fe_cfg
from Final.features.models import (
    CanonicalGrid,
    ChunkManifest,
    FeatureFamilySpec,
    RasterStackRegistry,
    SiteAssetBundle,
    SourceIngestRecord,
    SourceRasterBundle,
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
from Final.features.ee_utils import (
    ensure_ee_initialized,
    canonical_grid_bounds_wgs84,
    canonical_grid_bbox_region_coords,
    ee_3dep_elevation_for_grid,
    export_ee_image_to_geotiff,
)
from Final.features.assets import site_3dep_cache_root, site_3dep_rel_path
from Final.features.source_specs import SOURCE_SPECS
from Final.features.fe_2d import get_uniform_blur
from Final.features.fe_3dep import build_terrain_feature_dict_from_array

LOGGER = get_logger("features.source_3dep")


TERRAIN_FAMILY_SPECS = {
    "terrain": FeatureFamilySpec(
        key="terrain",
        source_name="3dep",
        runtime_tier=RuntimeTier.MODERATE,
        required_halo_px=16,
        representation_target=RepresentationTarget.RASTER,
        notes="Elevation, slope, aspect, northness/eastness, curvature, ruggedness, TPI, exposure.",
    ),
}

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def expected_3dep_filename(site_id: str) -> str:
    slug = site_id.replace("-", "_")
    return f"{slug}_3dep.tif"


def render_3dep_asset_rel_path(site_id: str, filename: str | None = None) -> str:
    filename = filename or expected_3dep_filename(site_id)
    return f"features/{site_id}/shared/site_assets/3dep/{filename}"


def local_3dep_asset_path(site_id: str, filename: str | None = None) -> Path:
    filename = filename or expected_3dep_filename(site_id)
    return site_3dep_cache_root(site_id) / filename


def validate_cached_3dep(tif_path: str | Path) -> bool:
    tif_path = Path(tif_path)
    if not tif_path.exists():
        return False

    try:
        with rasterio.open(tif_path) as src:
            if src.count < 1 or src.width <= 0 or src.height <= 0:
                return False

            # tiny probe read
            w = min(16, src.width)
            h = min(16, src.height)
            _ = src.read(1, window=rasterio.windows.Window(0, 0, w, h))
        return True
    except Exception as e:
        LOGGER.warning("Cached 3DEP validation failed for %s: %s", tif_path, e)
        return False

def build_3dep_source_inventory_record(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
) -> SourceIngestRecord:
    local_cached = site_assets.source_assets.get("3dep")
    if validate_optional_raster_asset(local_cached):
        return SourceIngestRecord(
            site_id=site_id,
            source_name="3dep",
            asset_path=str(local_cached),
            status="cached_local",
            notes=["3DEP cached locally."],
        )

    rel_path = site_3dep_rel_path(site_id)
    if remote_artifact_exists(rel_path, artifact_store=artifact_store):
        return SourceIngestRecord(
            site_id=site_id,
            source_name="3dep",
            asset_path=rel_path,
            status="cached_remote_lazy",
            notes=["3DEP cached remotely and will be hydrated lazily."],
        )

    return SourceIngestRecord(
        site_id=site_id,
        source_name="3dep",
        asset_path=None,
        status="not_materialized_yet",
        notes=["3DEP has not been materialized yet."],
    )

def make_terrain_family_compute_fn(
    *,
    pixel_size_m: float,
    ruggedness_kernel_radius: int,
    tpi_radius_meters: float,
    hillshade_azimuth: float,
    hillshade_zenith: float,
):
    def _compute(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        elevation = np.asarray(next(iter(arrays.values())), dtype=np.float32)
        return build_terrain_feature_dict_from_array(
            elevation,
            pixel_size_m=pixel_size_m,
            ruggedness_kernel_radius=ruggedness_kernel_radius,
            tpi_radius_meters=tpi_radius_meters,
            hillshade_azimuth=hillshade_azimuth,
            hillshade_zenith=hillshade_zenith,
            blur_fn=get_uniform_blur,
        )
    return _compute


def prepare_3dep_asset(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    canonical_grid: CanonicalGrid | None = None,
    force_refresh: bool = False,
    inventory: dict | None = None,
) -> Path:
    local_path = site_3dep_cache_root(site_id) / f"{site_id}_3dep_ee_10m.tif"
    rel_path = site_3dep_rel_path(site_id, filename=local_path.name)

    if (not force_refresh) and local_path.exists() and validate_cached_raster(local_path):
        LOGGER.info("Using cached 3DEP | site=%s | path=%s", site_id, local_path)
        return local_path

    if not force_refresh and remote_artifact_exists(rel_path, artifact_store=artifact_store):
        LOGGER.info("Hydrating 3DEP from artifact store | site=%s | rel_path=%s", site_id, rel_path)
        pulled = artifact_store.pull(rel_path, local_path=local_path)
        if validate_cached_raster(pulled):
            return pulled
        LOGGER.warning("Hydrated 3DEP failed validation | site=%s | path=%s", site_id, pulled)
        Path(pulled).unlink(missing_ok=True)

    if canonical_grid is None:
        raise ValueError("prepare_3dep_asset requires canonical_grid when no cached 3DEP raster is available.")

    ensure_ee_initialized(project="shrubwise-dc-488219")

    west, south, east, north = canonical_grid_bounds_wgs84(canonical_grid)
    region_coords = canonical_grid_bbox_region_coords(canonical_grid)
    image = ee_3dep_elevation_for_grid(canonical_grid)

    LOGGER.info(
        "Exporting 3DEP from Earth Engine | site=%s | out=%s | bounds_wgs84=(%.6f, %.6f, %.6f, %.6f)",
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
        raise RuntimeError(f"Exported 3DEP is unreadable for site={site_id}: {local_path}")

    persist_existing_file_artifact(
        local_path,
        artifact_key="site_3dep_raster",
        site_id=site_id,
        artifact_store=artifact_store,
        filename=local_path.name,
    )

    if not local_path.exists():
        LOGGER.info("3DEP local file was pruned after push; rehydrating for current use | site=%s", site_id)
        artifact_store.pull(rel_path, local_path=local_path)

    LOGGER.info("Prepared 3DEP asset | site=%s | path=%s", site_id, local_path)
    return local_path

def maybe_prune_local_site_3dep_after_load(
    site_id: str,
    tif_path: str | Path,
    *,
    artifact_store: ArtifactStore,
) -> None:
    tif_path = Path(tif_path)
    rel_path = site_3dep_rel_path(site_id, filename=tif_path.name)

    if not fe_cfg.persistence.prune_local_after_remote_push:
        return
    if not remote_artifact_exists(rel_path, artifact_store=artifact_store):
        return

    try:
        if tif_path.exists():
            tif_path.unlink()
            LOGGER.info("Pruned local 3DEP site asset after in-memory load | site=%s | path=%s", site_id, tif_path)
    except Exception as e:
        LOGGER.warning("Failed to prune local 3DEP asset | site=%s | path=%s | err=%s", site_id, tif_path, e)


def load_3dep_source_bundle(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
    force_refresh: bool = False,
) -> SourceRasterBundle:
    tif_path = prepare_3dep_asset(
        site_id,
        artifact_store=artifact_store,
        canonical_grid=canonical_grid,
        force_refresh=force_refresh,
        inventory=site_assets.source_assets.get("source_inventory"),
    )

    bundle = read_raster_bundle(
        tif_path,
        site_id=site_id,
        source_name="3dep",
        band_names=["elevation"],
    )

    maybe_prune_local_site_3dep_after_load(
        site_id,
        tif_path,
        artifact_store=artifact_store,
    )
    return bundle


def align_3dep_to_canonical_grid(
    bundle: SourceRasterBundle,
    *,
    canonical_grid: CanonicalGrid,
) -> SourceRasterBundle:
    return align_bundle_to_grid(
        bundle,
        dst_grid=canonical_grid,
        resampling=SOURCE_SPECS["3dep"].default_resampling,
    )

def run_3dep_chunked_pipeline(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    registry: RasterStackRegistry | None = None,
    force_refresh_source: bool = False,
) -> RasterStackRegistry:
    raw_bundle = load_3dep_source_bundle(
        site_id,
        artifact_store=artifact_store,
        site_assets=site_assets,
        canonical_grid=canonical_grid,
        force_refresh=force_refresh_source,
    )

    terrain_compute_fn = make_terrain_family_compute_fn(
        pixel_size_m=canonical_grid.pixel_size[0],
        ruggedness_kernel_radius=fe_cfg.terrain.ruggedness_kernel_radius,
        tpi_radius_meters=fe_cfg.terrain.tpi_radius_meters,
        hillshade_azimuth=fe_cfg.terrain.hillshade_azimuth,
        hillshade_zenith=fe_cfg.terrain.hillshade_zenith,
    )

    return run_source_chunked_pipeline(
        artifact_store=artifact_store,
        site_id=site_id,
        source_name="3dep",
        raw_bundle=raw_bundle,
        canonical_grid=canonical_grid,
        chunk_manifest=chunk_manifest,
        family_specs=TERRAIN_FAMILY_SPECS,
        family_compute_fns={"terrain": terrain_compute_fn},
        family_cfg_payloads={
            "terrain": {
                "ruggedness_kernel_radius": fe_cfg.terrain.ruggedness_kernel_radius,
                "tpi_radius_meters": fe_cfg.terrain.tpi_radius_meters,
                "hillshade_azimuth": fe_cfg.terrain.hillshade_azimuth,
                "hillshade_zenith": fe_cfg.terrain.hillshade_zenith,
                "pixel_size_m": canonical_grid.pixel_size[0],
            }
        },
        registry=registry,
    )

def clear_artifact_staging_dir() -> None:
    staging_dir = ensure_dir(fe_cfg.cache_root / "_artifact_staging")
    for p in staging_dir.rglob("*"):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass
    LOGGER.info("Cleared artifact staging directory | path=%s", staging_dir)