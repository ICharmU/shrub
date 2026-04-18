from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from Final.labeling.manifests import list_files_with_suffix
import ee
import geemap
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_bounds

from Final.shared_utils import get_logger
from Final.features.models import CanonicalGrid, SiteAssetBundle
from Final.features.raster_io import validate_cached_raster
from Final.features.labeling_bridge import best_label_artifact_for_site

LOGGER = get_logger("features.ee_utils")

def list_remote_tif_candidates(remote_dir: str) -> list[dict[str, Any]]:
    try:
        return list_files_with_suffix(remote_dir, (".tif", ".tiff"))
    except Exception as e:
        LOGGER.warning("Failed tif listing for remote_dir=%s | err=%s", remote_dir, e)
        return []

def export_ee_image_to_geotiff(
    image: ee.Image,
    *,
    out_path: str | Path,
    region_coords: list[list[float]],
    scale: float = 10.0,
    crs: str = "EPSG:4326",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    geemap.ee_export_image(
        image,
        filename=str(out_path),
        scale=scale,
        region=region_coords,
        file_per_band=False,
        crs=crs,
    )
    return out_path

def score_3dep_candidate(entry: dict[str, Any], *, naip_name: str) -> tuple[int, int, str]:
    """
    Higher is better.
    We want a tif in the NAIP_3DEP product dir that is not the NAIP orthophoto.
    """
    name = str(entry.get("name", "")).lower()

    score = 0

    if name == str(naip_name).lower():
        score -= 1000

    keywords_hi = ["3dep", "dem", "elev", "elevation", "terrain", "dtm"]
    keywords_mid = ["dsm", "usgs", "slope"]

    for kw in keywords_hi:
        if kw in name:
            score += 50
    for kw in keywords_mid:
        if kw in name:
            score += 20

    if "naip" in name:
        score -= 50
    if "rgb" in name or "nir" in name or "ortho" in name or "ortho" in name:
        score -= 40

    return (score, -len(name), name)


def infer_3dep_entry_from_candidates(
    tif_candidates: list[dict[str, Any]],
    *,
    naip_name: str,
) -> dict[str, Any] | None:
    if not tif_candidates:
        return None

    ranked = sorted(
        tif_candidates,
        key=lambda x: score_3dep_candidate(x, naip_name=naip_name),
        reverse=True,
    )

    best = ranked[0]
    best_score = score_3dep_candidate(best, naip_name=naip_name)[0]

    if best_score < 0:
        return None
    return best

def ensure_ee_initialized(project: str | None = None):
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()

def label_artifact_bounds_wgs84(site_id: str, resolution_m: float = 1.0) -> tuple[float, float, float, float]:
    art = best_label_artifact_for_site(site_id, resolution_m=resolution_m)
    if art is None:
        raise ValueError(f"No label artifact found for site={site_id}")

    mask_path = (
        art.get("binary_mask_path")
        or art.get("confidence_mask_path")
        or art.get("object_id_raster_path")
    )
    if mask_path is None:
        raise ValueError(f"No raster path available in label artifact row for site={site_id}")

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Label artifact raster missing for site={site_id}: {mask_path}")

    with rasterio.open(mask_path) as src:
        left, bottom, right, top = rasterio.transform.array_bounds(
            src.height,
            src.width,
            src.transform,
        )
        src_crs = CRS.from_user_input(src.crs)
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
    if not np.all(np.isfinite(out)):
        raise ValueError(f"Non-finite label-derived bounds for site={site_id}: {out}")

    west, south, east, north = out
    west = max(-180.0, min(180.0, west))
    east = max(-180.0, min(180.0, east))
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))

    if not (west < east and south < north):
        raise ValueError(f"Invalid label-derived bounds for site={site_id}: {(west, south, east, north)}")

    return west, south, east, north

def site_bounds_wgs84_for_naip_fallback(
    site_id: str,
    *,
    canonical_grid: CanonicalGrid | None = None,
    resolution_m: float = 1.0,
) -> tuple[float, float, float, float]:
    if canonical_grid is not None:
        return canonical_grid_bounds_wgs84(canonical_grid)
    return label_artifact_bounds_wgs84(site_id, resolution_m=resolution_m)


def site_region_coords_for_naip_fallback(
    site_id: str,
    *,
    canonical_grid: CanonicalGrid | None = None,
    resolution_m: float = 1.0,
) -> list[list[float]]:
    west, south, east, north = site_bounds_wgs84_for_naip_fallback(
        site_id,
        canonical_grid=canonical_grid,
        resolution_m=resolution_m,
    )
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


def site_geometry_for_naip_fallback(
    site_id: str,
    *,
    canonical_grid: CanonicalGrid | None = None,
    resolution_m: float = 1.0,
) -> ee.Geometry:
    west, south, east, north = site_bounds_wgs84_for_naip_fallback(
        site_id,
        canonical_grid=canonical_grid,
        resolution_m=resolution_m,
    )
    return ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)

def ee_naip_image_for_site(
    site_id: str,
    *,
    canonical_grid: CanonicalGrid | None = None,
    resolution_m: float = 1.0,
    year: int | None = None,
) -> ee.Image:
    geom = site_geometry_for_naip_fallback(
        site_id,
        canonical_grid=canonical_grid,
        resolution_m=resolution_m,
    )

    collection = ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(geom)

    if year is not None:
        collection = collection.filterDate(f"{year}-01-01", f"{year}-12-31")

    # Prefer newest available NAIP covering the site
    img = collection.sort("system:time_start", False).first()
    if img is None:
        raise ValueError(f"No NAIP image found in EE for site={site_id}")

    return ee.Image(img).clip(geom).toFloat()

def infer_naip_band_names_from_ee(image: ee.Image) -> list[str]:
    names = [str(x) for x in image.bandNames().getInfo()]
    # Normalize common NAIP naming to your expected convention
    lowered = [n.lower() for n in names]

    mapping = []
    for n in lowered:
        if n in {"r", "red"}:
            mapping.append("red")
        elif n in {"g", "green"}:
            mapping.append("green")
        elif n in {"b", "blue"}:
            mapping.append("blue")
        elif n in {"n", "nir"}:
            mapping.append("nir")
        else:
            mapping.append(n)

    return mapping

def canonical_grid_bounds_native(grid: CanonicalGrid) -> tuple[float, float, float, float]:
    left, bottom, right, top = rasterio.transform.array_bounds(
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

    if not np.all(np.isfinite(out)):
        raise ValueError(f"Non-finite WGS84 bounds derived from canonical grid: {out}")

    west, south, east, north = out

    # Sanity checks / clamping
    west = max(-180.0, min(180.0, west))
    east = max(-180.0, min(180.0, east))
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))

    if not (west < east and south < north):
        raise ValueError(
            f"Invalid WGS84 bounds after transformation: {(west, south, east, north)}"
        )

    return west, south, east, north


def canonical_grid_bbox_geometry(grid: CanonicalGrid) -> ee.Geometry:
    west, south, east, north = canonical_grid_bounds_wgs84(grid)
    return ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)


def canonical_grid_bbox_region_coords(grid: CanonicalGrid) -> list[list[float]]:
    west, south, east, north = canonical_grid_bounds_wgs84(grid)
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


def ee_3dep_elevation_for_grid(grid: CanonicalGrid) -> ee.Image:
    geom = canonical_grid_bbox_geometry(grid)

    dem = (
        ee.ImageCollection("USGS/3DEP/10m_collection")
        .select("elevation")
        .mosaic()
        .toFloat()
        .rename("elevation")
        .clip(geom)
    )
    return dem

def ee_rap_image_for_grid(
    grid: CanonicalGrid,
    *,
    image_id: str = "projects/rap-data-365417/assets/vegetation-cover-10m",
) -> ee.Image:
    geom = canonical_grid_bbox_geometry(grid)

    img = (
        ee.ImageCollection(image_id)
        .mosaic()
        .toFloat()
        .clip(geom)
    )
    return img


def infer_rap_band_names_from_ee(image: ee.Image) -> list[str]:
    names = image.bandNames().getInfo()
    return [str(x) for x in names]

def als_site_bounds_native(als_meta: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    if not als_meta:
        raise ValueError("ALS metadata is empty; cannot derive site-wide bounds.")

    mins_x, mins_y, maxs_x, maxs_y = [], [], [], []

    for meta in als_meta:
        nb = meta.get("native_bounds")
        if not isinstance(nb, dict):
            continue
        mins_x.append(float(nb["minx"]))
        mins_y.append(float(nb["miny"]))
        maxs_x.append(float(nb["maxx"]))
        maxs_y.append(float(nb["maxy"]))

    if not mins_x:
        raise ValueError("ALS metadata had no usable native_bounds entries.")

    return min(mins_x), min(mins_y), max(maxs_x), max(maxs_y)

def als_site_bounds_wgs84(als_meta: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    first = next((m for m in als_meta if m.get("srs_wkt")), None)
    if first is None:
        raise ValueError("No ALS metadata entry contains srs_wkt; cannot derive site-wide CRS.")

    src_crs = CRS.from_wkt(first["srs_wkt"])
    left, bottom, right, top = als_site_bounds_native(als_meta)

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
    if not np.all(np.isfinite(out)):
        raise ValueError(f"Non-finite ALS-derived site bounds: {out}")

    west, south, east, north = out
    west = max(-180.0, min(180.0, west))
    east = max(-180.0, min(180.0, east))
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))

    if not (west < east and south < north):
        raise ValueError(f"Invalid ALS-derived site bounds: {(west, south, east, north)}")

    return west, south, east, north

def site_bounds_wgs84_for_naip_sitewide_fallback(
    site_id: str,
    *,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid | None = None,
    resolution_m: float = 1.0,
) -> tuple[float, float, float, float]:
    if canonical_grid is not None:
        return canonical_grid_bounds_wgs84(canonical_grid)

    als_meta = site_assets.source_assets.get("als_metadata", [])
    if als_meta:
        return als_site_bounds_wgs84(als_meta)

    return label_artifact_bounds_wgs84(site_id, resolution_m=resolution_m)

def site_region_coords_for_naip_sitewide_fallback(
    site_id: str,
    *,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid | None = None,
    resolution_m: float = 1.0,
) -> list[list[float]]:
    west, south, east, north = site_bounds_wgs84_for_naip_sitewide_fallback(
        site_id,
        site_assets=site_assets,
        canonical_grid=canonical_grid,
        resolution_m=resolution_m,
    )
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]

def split_wgs84_bbox_into_tiles(
    bounds: tuple[float, float, float, float],
    *,
    nx: int,
    ny: int,
) -> list[tuple[float, float, float, float]]:
    west, south, east, north = bounds
    xs = np.linspace(west, east, nx + 1)
    ys = np.linspace(south, north, ny + 1)

    tiles = []
    for j in range(ny):
        for i in range(nx):
            tiles.append((xs[i], ys[j], xs[i + 1], ys[j + 1]))
    return tiles

def region_coords_from_bounds(bounds: tuple[float, float, float, float]) -> list[list[float]]:
    west, south, east, north = bounds
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]

def ee_geometry_from_bounds(bounds: tuple[float, float, float, float]) -> ee.Geometry:
    west, south, east, north = bounds
    return ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)

def ee_naip_image_for_bounds(
    bounds: tuple[float, float, float, float],
    *,
    year: int | None = None,
) -> ee.Image:
    geom = ee_geometry_from_bounds(bounds)

    collection = ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(geom)
    if year is not None:
        collection = collection.filterDate(f"{year}-01-01", f"{year}-12-31")

    img = collection.sort("system:time_start", False).first()
    if img is None:
        raise ValueError(f"No NAIP image found in EE for bounds={bounds}")

    return ee.Image(img).clip(geom).toFloat()

def export_ee_naip_tiled_to_geotiff(
    *,
    out_path: str | Path,
    bounds_wgs84: tuple[float, float, float, float],
    scale: float = 1.0,
    crs: str = "EPSG:4326",
    nx: int = 2,
    ny: int = 2,
    year: int | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tile_bounds_list = split_wgs84_bbox_into_tiles(bounds_wgs84, nx=nx, ny=ny)
    tmp_dir = out_path.parent / f"{out_path.stem}_tiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = []

    try:
        for idx, tile_bounds in enumerate(tile_bounds_list):
            tile_path = tmp_dir / f"tile_{idx:02d}.tif"
            tile_img = ee_naip_image_for_bounds(tile_bounds, year=year)

            export_ee_image_to_geotiff(
                tile_img,
                out_path=tile_path,
                region_coords=region_coords_from_bounds(tile_bounds),
                scale=scale,
                crs=crs,
            )

            if not validate_cached_raster(tile_path):
                raise RuntimeError(f"Exported NAIP tile unreadable: {tile_path}")

            tile_paths.append(tile_path)

        srcs = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic_arr, mosaic_transform = rio_merge(srcs)
            meta = srcs[0].meta.copy()
            meta.update(
                {
                    "height": mosaic_arr.shape[1],
                    "width": mosaic_arr.shape[2],
                    "transform": mosaic_transform,
                    "count": mosaic_arr.shape[0],
                    "dtype": str(mosaic_arr.dtype),
                }
            )

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(mosaic_arr)
        finally:
            for src in srcs:
                src.close()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path

