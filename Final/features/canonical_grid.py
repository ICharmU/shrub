from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pyproj import CRS
import numpy as np
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds

from Final.artifact_store import ArtifactStore
from Final.shared_utils import get_logger
from Final.pipeline_caching import hash_payload

from Final.features.config import fe_cfg
from Final.features.models import CanonicalGrid, SiteAssetBundle
from Final.features.artifact_io import (
    current_fe_config_signature,
    try_load_json_artifact,
    persist_json_artifact,
)
from Final.features.raster_io import (
    infer_naip_band_names,
    read_raster_bundle,
    validate_optional_raster_asset,
)


LOGGER = get_logger("features.canonical_grid")

PrepareNAIPFn = Callable[..., Path | None]
PrepareSiteAssetsFn = Callable[..., SiteAssetBundle]


def canonical_grid_data_signature(site_id: str, naip_ref: str | None) -> str:
    return hash_payload(
        {
            "site_id": site_id,
            "naip_ref": naip_ref,
            "canonical_grid_source": fe_cfg.canonical_grid_source,
        }
    )


def canonical_grid_config_signature() -> str:
    return hash_payload(
        {
            "canonical_grid_source": fe_cfg.canonical_grid_source,
            "version": fe_cfg.version,
        }
    )


def resolve_canonical_grid_naip_path(
    site_id: str,
    *,
    assets: SiteAssetBundle,
    prepare_naip_asset_fn: PrepareNAIPFn | None = None,
    force_refresh: bool = False,
) -> Path:
    naip_path = assets.source_assets.get("naip")

    if validate_optional_raster_asset(naip_path):
        return Path(naip_path)

    if prepare_naip_asset_fn is None:
        raise ValueError(
            f"NAIP asset for site={site_id} is missing or invalid, and no prepare_naip_asset_fn was supplied."
        )

    repaired = prepare_naip_asset_fn(
        site_id,
        force_refresh=force_refresh,
        inventory=assets.source_assets.get("source_inventory"),
        site_assets=assets,
        canonical_grid=None,
    )

    if not validate_optional_raster_asset(repaired):
        raise ValueError(f"NAIP asset repair failed for site={site_id}: {repaired}")

    return Path(repaired)


def build_canonical_grid_for_site(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    assets: SiteAssetBundle | None = None,
    force_refresh: bool = False,
    prepare_site_assets_fn: PrepareSiteAssetsFn | None = None,
    prepare_naip_asset_fn: PrepareNAIPFn | None = None,
) -> CanonicalGrid:
    if fe_cfg.canonical_grid_source != "naip":
        raise NotImplementedError(
            f"Only canonical_grid_source='naip' is currently supported, got {fe_cfg.canonical_grid_source!r}"
        )

    if assets is None:
        if prepare_site_assets_fn is None:
            raise ValueError("assets is None and no prepare_site_assets_fn was supplied.")
        assets = prepare_site_assets_fn(site_id, force_refresh=force_refresh)

    naip_ref = str(assets.source_assets.get("naip"))
    data_sig = canonical_grid_data_signature(site_id, naip_ref)
    config_sig = canonical_grid_config_signature()

    payload = None if force_refresh else try_load_json_artifact(
        artifact_key="canonical_grid",
        site_id=site_id,
        artifact_store=artifact_store,
        config_sig=current_fe_config_signature(),
    )

    if payload is not None:
        if payload.get("data_signature") == data_sig and payload.get("config_signature") == config_sig:
            LOGGER.info("Using cached canonical grid | site=%s", site_id)
            return CanonicalGrid(
                site_id=payload["site_id"],
                width=payload["width"],
                height=payload["height"],
                transform=Affine(*payload["transform"]),
                crs=CRS.from_user_input(payload["crs"]),
                source_name=payload["source_name"],
                nodata=payload["nodata"],
            )

    naip_path = resolve_canonical_grid_naip_path(
        site_id,
        assets=assets,
        prepare_naip_asset_fn=prepare_naip_asset_fn,
        force_refresh=force_refresh,
    )

    bundle = read_raster_bundle(
        naip_path,
        site_id=site_id,
        source_name="naip",
        band_names=infer_naip_band_names(naip_path),
    )

    h, w = next(iter(bundle.arrays.values())).shape
    grid = CanonicalGrid(
        site_id=site_id,
        width=w,
        height=h,
        transform=bundle.transform,
        crs=bundle.crs,
        source_name="naip",
        nodata=np.nan,
    )

    persist_json_artifact(
        {
            "site_id": grid.site_id,
            "width": grid.width,
            "height": grid.height,
            "transform": list(grid.transform)[:6],
            "crs": str(grid.crs),
            "source_name": grid.source_name,
            "nodata": grid.nodata,
            "data_signature": data_sig,
            "config_signature": config_sig,
        },
        artifact_key="canonical_grid",
        site_id=site_id,
        artifact_store=artifact_store,
        config_sig=current_fe_config_signature(),
    )

    LOGGER.info(
        "Built canonical grid | site=%s | shape=(%d, %d) | pixel_size=%s",
        site_id,
        grid.height,
        grid.width,
        grid.pixel_size,
    )
    return grid


def canonical_grid_bounds_native(grid: CanonicalGrid) -> tuple[float, float, float, float]:
    left, bottom, right, top = array_bounds(
        grid.height,
        grid.width,
        grid.transform,
    )
    return float(left), float(bottom), float(right), float(top)


def canonical_grid_bounds_wgs84(grid: CanonicalGrid) -> tuple[float, float, float, float]:
    src_crs = CRS.from_user_input(grid.crs)
    left, bottom, right, top = canonical_grid_bounds_native(grid)

    if str(src_crs).upper() == "EPSG:4326":
        out = (left, bottom, right, top)
    else:
        out = transform_bounds(
            src_crs,
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )

    out = tuple(float(x) for x in out)

    west, south, east, north = out
    west = max(-180.0, min(180.0, west))
    east = max(-180.0, min(180.0, east))
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))

    if not (west < east and south < north):
        raise ValueError(f"Invalid WGS84 bounds after transform: {(west, south, east, north)}")

    return west, south, east, north


def canonical_grid_bbox_region_coords(grid: CanonicalGrid) -> list[list[float]]:
    west, south, east, north = canonical_grid_bounds_wgs84(grid)
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]