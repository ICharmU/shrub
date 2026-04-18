from __future__ import annotations

import numpy as np

from Final.shared_utils import get_logger
from Final.artifact_store import ArtifactStore
from Final.features.config import fe_cfg
from Final.features.models import (
    CanonicalGrid,
    ChunkManifest,
    FeatureFamilySpec,
    RasterStackRegistry,
    SiteAssetBundle,
)
from Final.features.raster_io import read_raster_bundle
from Final.features.source_registry import run_source_chunked_pipeline

LOGGER = get_logger("features.source_3dep")


TERRAIN_FAMILY_SPECS = {
    "terrain": FeatureFamilySpec(
        key="terrain",
        source_name="3dep",
        runtime_tier="moderate",
        required_halo_px=8,
        representation_target="raster",
        notes="Terrain and exposure features from 3DEP.",
    ),
}


def load_3dep_source_bundle(
    site_id: str,
    *,
    site_assets: SiteAssetBundle,
):
    tif_path = site_assets.source_assets.get("3dep")
    if tif_path is None:
        raise ValueError(f"No 3DEP asset available for site={site_id}")

    return read_raster_bundle(
        tif_path,
        site_id=site_id,
        source_name="3dep",
        band_names=["elevation"],
    )


def _grad_x(arr: np.ndarray) -> np.ndarray:
    return np.gradient(arr, axis=1).astype(np.float32)


def _grad_y(arr: np.ndarray) -> np.ndarray:
    return np.gradient(arr, axis=0).astype(np.float32)


def make_terrain_family_compute_fn(
    *,
    pixel_size_m: float,
    ruggedness_kernel_radius: int,
    tpi_radius_meters: float,
    hillshade_azimuth: float,
    hillshade_zenith: float,
):
    def _compute(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        dem = arrays["elevation"].astype(np.float32)

        gx = _grad_x(dem) / max(pixel_size_m, 1e-6)
        gy = _grad_y(dem) / max(pixel_size_m, 1e-6)

        slope = np.sqrt(gx**2 + gy**2).astype(np.float32)
        aspect = np.arctan2(gy, gx).astype(np.float32)

        northness = np.cos(aspect).astype(np.float32)
        eastness = np.sin(aspect).astype(np.float32)

        relief_local = (
            np.nanmax(dem) - np.nanmin(dem)
            if np.isfinite(dem).any() else np.nan
        )
        relief_arr = np.full_like(dem, relief_local, dtype=np.float32)

        return {
            "terrain_elevation": dem,
            "terrain_slope": slope,
            "terrain_aspect": aspect,
            "terrain_northness": northness,
            "terrain_eastness": eastness,
            "terrain_relief_proxy": relief_arr,
        }

    return _compute


def run_3dep_chunked_pipeline(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    registry: RasterStackRegistry,
) -> RasterStackRegistry:
    raw_bundle = load_3dep_source_bundle(site_id, site_assets=site_assets)

    compute_fn = make_terrain_family_compute_fn(
        pixel_size_m=canonical_grid.pixel_size[0],
        ruggedness_kernel_radius=fe_cfg.terrain.ruggedness_kernel_radius,
        tpi_radius_meters=fe_cfg.terrain.tpi_radius_meters,
        hillshade_azimuth=fe_cfg.terrain.hillshade_azimuth,
        hillshade_zenith=fe_cfg.terrain.hillshade_zenith,
    )

    enabled = {
        k: v
        for k, v in TERRAIN_FAMILY_SPECS.items()
        if k in fe_cfg.terrain.enabled_families
    }

    return run_source_chunked_pipeline(
        artifact_store=artifact_store,
        site_id=site_id,
        source_name="3dep",
        raw_bundle=raw_bundle,
        canonical_grid=canonical_grid,
        chunk_manifest=chunk_manifest,
        family_specs=enabled,
        family_compute_fns={"terrain": compute_fn},
        family_cfg_payloads={},
        registry=registry,
    )