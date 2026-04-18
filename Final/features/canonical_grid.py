from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from Final.artifact_store import ArtifactStore
from Final.shared_utils import ensure_dir, get_logger

from Final.features.artifact_io import (
    read_json,
    remote_artifact_exists,
    render_artifact_rel_path,
    write_json,
)
from Final.features.assets import site_3dep_cache_root, site_naip_cache_root, site_rap_cache_root
from Final.features.config import fe_cfg
from Final.features.models import CanonicalGrid, SiteAssetBundle
from Final.features.raster_io import (
    infer_naip_band_names,
    read_raster_bundle,
    validate_cached_raster,
)

LOGGER = get_logger("features.canonical_grid")


def _stable_sig(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def canonical_grid_cache_dir(site_id: str, data_signature: str, config_signature: str) -> Path:
    return ensure_dir(fe_cfg.cache_root / "canonical_grid" / site_id / data_signature / config_signature)


def resolve_site_asset_for_read(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    asset_path: str | Path | None,
    artifact_key: str,
    filename: str | None = None,
) -> Path:
    if asset_path is None:
        raise ValueError(f"No asset path registered for site={site_id} and artifact_key={artifact_key}")

    p = Path(asset_path)
    if p.exists():
        return p

    if artifact_key == "site_naip_raster":
        rel_path = render_artifact_rel_path("site_naip_raster", site_id=site_id, filename=filename or p.name)
        local_path = site_naip_cache_root(site_id) / (filename or p.name)
    elif artifact_key == "site_3dep_raster":
        rel_path = render_artifact_rel_path("site_3dep_raster", site_id=site_id, filename=filename or p.name)
        local_path = site_3dep_cache_root(site_id) / (filename or p.name)
    elif artifact_key == "site_rap_raster":
        rel_path = render_artifact_rel_path("site_rap_raster", site_id=site_id, filename=filename or p.name)
        local_path = site_rap_cache_root(site_id) / (filename or p.name)
    else:
        raise ValueError(f"Unsupported artifact_key for site-asset rehydration: {artifact_key}")

    if remote_artifact_exists(rel_path, artifact_store=artifact_store):
        LOGGER.info(
            "Rehydrating pruned site asset | site=%s | key=%s | rel_path=%s",
            site_id, artifact_key, rel_path,
        )
        return _pull_with_validation(
            artifact_store=artifact_store,
            rel_path=rel_path,
            local_path=local_path,
            validator=validate_cached_raster,
            max_attempts=2,
        )

    raise FileNotFoundError(
        f"Site asset missing locally and not found remotely | site={site_id} | key={artifact_key} | path={p}"
    )

def _pull_with_validation(
    *,
    artifact_store: ArtifactStore,
    rel_path: str,
    local_path: Path,
    validator,
    max_attempts: int = 2,
) -> Path:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass

            pulled = artifact_store.pull(rel_path, local_path=local_path)

            if validator(pulled):
                return Path(pulled)

            try:
                Path(pulled).unlink(missing_ok=True)
            except Exception:
                pass

            last_err = RuntimeError(f"Hydrated artifact failed validation: {rel_path}")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to hydrate valid artifact after {max_attempts} attempts | rel_path={rel_path} | err={last_err}")

def build_canonical_grid_for_site(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    assets: SiteAssetBundle,
    force_refresh: bool = False,
) -> CanonicalGrid:
    naip_asset = assets.source_assets.get("naip")
    if naip_asset is None:
        raise ValueError(f"No NAIP asset available for site={site_id}; cannot build canonical grid.")

    naip_path = resolve_site_asset_for_read(
        site_id,
        artifact_store=artifact_store,
        asset_path=naip_asset,
        artifact_key="site_naip_raster",
        filename=Path(naip_asset).name if naip_asset is not None else None,
    )

    data_sig = _stable_sig(
        {
            "site_id": site_id,
            "naip_filename": naip_path.name,
            "canonical_grid_source": fe_cfg.canonical_grid_source,
        }
    )
    config_sig = _stable_sig(
        {
            "canonical_grid_source": fe_cfg.canonical_grid_source,
            "version": fe_cfg.version,
        }
    )

    cache_dir = canonical_grid_cache_dir(site_id, data_sig, config_sig)
    grid_json = cache_dir / "canonical_grid.json"

    if (not force_refresh) and grid_json.exists():
        payload = read_json(grid_json)
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

    write_json(
        grid_json,
        {
            "site_id": grid.site_id,
            "width": grid.width,
            "height": grid.height,
            "transform": list(grid.transform)[:6],
            "crs": str(grid.crs),
            "source_name": grid.source_name,
            "nodata": grid.nodata,
        },
    )

    LOGGER.info(
        "Built canonical grid | site=%s | shape=(%d, %d) | pixel_size=%s",
        site_id,
        grid.height,
        grid.width,
        grid.pixel_size,
    )
    return grid