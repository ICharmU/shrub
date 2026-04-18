from __future__ import annotations

from pathlib import Path
from typing import Any

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
from Final.features.source_specs import SOURCE_SPECS
from Final.features.raster_io import (
    infer_naip_band_names,
    read_raster_bundle,
)
from Final.features.source_registry import run_source_chunked_pipeline

LOGGER = get_logger("features.source_naip")


NAIP_FAMILY_SPECS = {
    "raw": FeatureFamilySpec(
        key="raw",
        source_name="naip",
        runtime_tier="cheap",
        required_halo_px=0,
        representation_target="raster",
        notes="Raw NAIP channels.",
    ),
    "veg_idx": FeatureFamilySpec(
        key="veg_idx",
        source_name="naip",
        runtime_tier="cheap",
        required_halo_px=0,
        representation_target="raster",
        notes="Vegetation indices.",
    ),
}


def load_naip_source_bundle(
    site_id: str,
    *,
    site_assets: SiteAssetBundle,
):
    tif_path = site_assets.source_assets.get("naip")
    if tif_path is None:
        raise ValueError(f"No NAIP asset available for site={site_id}")
    return read_raster_bundle(
        tif_path,
        site_id=site_id,
        source_name="naip",
        band_names=infer_naip_band_names(tif_path),
    )


def make_naip_raw_compute_fn():
    def _compute(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {f"naip_{k}": v.astype(np.float32) for k, v in arrays.items()}
    return _compute


def make_naip_veg_idx_compute_fn():
    def _compute(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        red = arrays.get("red")
        green = arrays.get("green")
        blue = arrays.get("blue")
        nir = arrays.get("nir")

        eps = 1e-6

        if nir is not None and red is not None:
            out["naip_ndvi"] = ((nir - red) / (nir + red + eps)).astype(np.float32)

        if green is not None and red is not None and blue is not None:
            out["naip_vari"] = ((green - red) / (green + red - blue + eps)).astype(np.float32)

        return out
    return _compute


def run_naip_chunked_pipeline(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    registry: RasterStackRegistry,
) -> RasterStackRegistry:
    raw_bundle = load_naip_source_bundle(site_id, site_assets=site_assets)

    enabled = {
        k: v
        for k, v in NAIP_FAMILY_SPECS.items()
        if k in fe_cfg.naip.enabled_families
    }

    compute_map = {}
    if "raw" in enabled:
        compute_map["raw"] = make_naip_raw_compute_fn()
    if "veg_idx" in enabled:
        compute_map["veg_idx"] = make_naip_veg_idx_compute_fn()

    return run_source_chunked_pipeline(
        artifact_store=artifact_store,
        site_id=site_id,
        source_name="naip",
        raw_bundle=raw_bundle,
        canonical_grid=canonical_grid,
        chunk_manifest=chunk_manifest,
        family_specs=enabled,
        family_compute_fns=compute_map,
        family_cfg_payloads={},
        registry=registry,
    )