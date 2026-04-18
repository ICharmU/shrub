from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import json

from Final.config import default_config
from Final.shared_utils import ensure_dir, get_logger
from Final.artifact_store import ArtifactStore
from Final.labeling.manifests import (
    list_files_with_suffix,
    site_to_remote_base,
    site_to_tif_name,
)

from Final.features.config import fe_cfg
from Final.features.models import SiteAssetBundle
from Final.features.artifact_io import (
    current_fe_config_signature,
    persist_json_artifact,
    try_load_json_artifact,
)
from Final.features.artifact_io import render_artifact_rel_path
from Final.features.raster_io import validate_optional_raster_asset, validate_cached_raster


cfg = default_config()
LOGGER = get_logger("features.assets")


PrepareNAIPFn = Callable[..., Path | None]
PrepareALSMetadataFn = Callable[..., list[dict[str, Any]]]
Prepare3DEPFn = Callable[..., Path | None]
PrepareRAPFn = Callable[..., Path | None]
FindCachedRasterFn = Callable[..., Path | None]


def site_asset_cache_root(site_id: str) -> Path:
    return ensure_dir(fe_cfg.cache_root / "site_assets" / site_id)


def site_naip_cache_root(site_id: str) -> Path:
    return ensure_dir(site_asset_cache_root(site_id) / "naip")


def site_als_cache_root(site_id: str) -> Path:
    return ensure_dir(site_asset_cache_root(site_id) / "als_metadata")


def site_3dep_cache_root(site_id: str) -> Path:
    return ensure_dir(site_asset_cache_root(site_id) / "3dep")


def site_rap_cache_root(site_id: str) -> Path:
    return ensure_dir(site_asset_cache_root(site_id) / "rap")

def site_3dep_rel_path(site_id: str, filename: str | None = None) -> str:
    filename = filename or f"{site_id}_3dep_ee_10m.tif"
    return render_artifact_rel_path(
        "site_3dep_raster",
        site_id=site_id,
        filename=filename,
    )

def site_rap_rel_path(site_id: str, filename: str | None = None) -> str:
    filename = filename or f"{site_id}_rap_ee_10m.tif"
    return render_artifact_rel_path(
        "site_rap_raster",
        site_id=site_id,
        filename=filename,
    )

def serialize_site_asset_bundle(bundle: SiteAssetBundle) -> dict[str, Any]:
    payload = {
        "site_id": bundle.site_id,
        "source_assets": {},
        "notes": list(bundle.notes),
    }

    for key, value in bundle.source_assets.items():
        if isinstance(value, Path):
            payload["source_assets"][key] = {"kind": "path", "value": str(value)}
        else:
            payload["source_assets"][key] = {"kind": "json", "value": value}

    return payload


def deserialize_site_asset_bundle(payload: dict[str, Any]) -> SiteAssetBundle:
    bundle = SiteAssetBundle(site_id=payload["site_id"])
    bundle.notes = list(payload.get("notes", []))

    for key, wrapped in payload.get("source_assets", {}).items():
        if wrapped["kind"] == "path":
            bundle.source_assets[key] = Path(wrapped["value"])
        else:
            bundle.source_assets[key] = wrapped["value"]

    return bundle


def site_asset_bundle_payload(bundle: SiteAssetBundle | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return serialize_site_asset_bundle(bundle)

def find_cached_3dep_asset(site_id: str) -> Path | None:
    dep_cache_dir = site_3dep_cache_root(site_id)

    # Prefer the new EE-exported naming convention first
    preferred = sorted(dep_cache_dir.glob("*_3dep_ee_10m.tif"))
    fallback = sorted(dep_cache_dir.glob("*.tif"))

    candidates = preferred + [p for p in fallback if p not in preferred]

    for p in candidates:
        if validate_cached_raster(p):
            return p
        else:
            try:
                LOGGER.warning("Removing invalid cached 3DEP candidate | site=%s | path=%s", site_id, p)
                p.unlink(missing_ok=True)
            except Exception as e:
                LOGGER.warning("Failed to remove invalid cached 3DEP candidate | path=%s | err=%s", p, e)

    return None

def find_cached_rap_asset(site_id: str) -> Path | None:
    rap_cache_dir = site_rap_cache_root(site_id)
    tif_candidates = sorted(rap_cache_dir.glob("*.tif"))
    valid_candidates = [p for p in tif_candidates if validate_cached_raster(p)]
    return valid_candidates[0] if valid_candidates else None

def _dedupe_notes(notes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for note in notes:
        note = str(note).strip()
        if not note:
            continue
        if note in seen:
            continue
        seen.add(note)
        out.append(note)
    return out

def _drop_notes_with_prefix(notes: list[str], prefix: str) -> list[str]:
    prefix = str(prefix).strip()
    return [n for n in notes if not str(n).strip().startswith(prefix)]


def _replace_source_notes(
    notes: list[str],
    *,
    clear_prefixes: tuple[str, ...],
    new_note: str | None = None,
) -> list[str]:
    out = list(notes)
    for prefix in clear_prefixes:
        out = _drop_notes_with_prefix(out, prefix)
    if new_note:
        out.append(new_note)
    return _dedupe_notes(out)

def list_remote_tif_candidates(remote_dir: str) -> list[dict[str, Any]]:
    try:
        return list_files_with_suffix(remote_dir, (".tif", ".tiff"))
    except Exception as e:
        LOGGER.warning("Failed tif listing for remote_dir=%s | err=%s", remote_dir, e)
        return []


def score_3dep_candidate(entry: dict[str, Any], *, naip_name: str) -> tuple[int, int, str]:
    name = str(entry.get("name", "")).lower()
    score = 0

    if name == str(naip_name).lower():
        score -= 1000

    for kw in ["3dep", "dem", "elev", "elevation", "terrain", "dtm"]:
        if kw in name:
            score += 50

    for kw in ["dsm", "usgs", "slope"]:
        if kw in name:
            score += 20

    if "naip" in name:
        score -= 50
    if any(kw in name for kw in ("rgb", "nir", "ortho")):
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
    return best if best_score >= 0 else None

def build_source_inventory(site_id: str) -> dict[str, Any]:
    site_base = site_to_remote_base(cfg, site_id)
    product_dir = f"{site_base}/{cfg.data.naip_3dep_dir}"
    naip_name = site_to_tif_name(site_id)

    inventory = {
        "site_id": site_id,
        "site_base": site_base,
        "naip": {
            "expected_name": naip_name,
            "remote_url": f"{product_dir}/{naip_name}",
        },
        "als": {
            "remote_url": f"{site_base}/{cfg.data.als_dir}",
            "files": [],
        },
        "3dep": {
            "remote_dir": product_dir,
            "remote_url": None,
            "expected_name": None,
            "candidates": [],
        },
        "rap": {
            "remote_url": None,
        },
    }

    try:
        als_files = list_files_with_suffix(
            inventory["als"]["remote_url"],
            (".las", ".laz", ".copc.laz"),
        )
        inventory["als"]["files"] = als_files
    except Exception as e:
        inventory["als"]["error"] = str(e)

    try:
        tif_candidates = list_remote_tif_candidates(product_dir)
        inventory["3dep"]["candidates"] = tif_candidates

        chosen = infer_3dep_entry_from_candidates(
            tif_candidates,
            naip_name=naip_name,
        )
        if chosen is not None:
            inventory["3dep"]["expected_name"] = chosen["name"]
            inventory["3dep"]["remote_url"] = chosen.get("url") or chosen.get("href")
        else:
            inventory["3dep"]["warning"] = (
                "No convincing 3DEP tif candidate found in NAIP_3DEP product directory."
            )
    except Exception as e:
        inventory["3dep"]["error"] = str(e)

    return inventory


def persist_source_inventory(
    site_id: str,
    inventory: dict[str, Any],
    *,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
):
    return persist_json_artifact(
        inventory,
        artifact_key="source_inventory",
        site_id=site_id,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )


def try_load_source_inventory(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
) -> dict[str, Any] | None:
    return try_load_json_artifact(
        artifact_key="source_inventory",
        site_id=site_id,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )


def try_load_site_metadata_manifest(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
) -> SiteAssetBundle | None:
    payload = try_load_json_artifact(
        artifact_key="site_metadata_manifest",
        site_id=site_id,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )
    if payload is None:
        return None

    LOGGER.info("Using site metadata manifest | site=%s", site_id)
    return deserialize_site_asset_bundle(payload)


def persist_site_metadata_manifest(
    bundle: SiteAssetBundle,
    *,
    artifact_store: ArtifactStore,
    config_sig: str | None = None,
):
    payload = serialize_site_asset_bundle(bundle)
    return persist_json_artifact(
        payload,
        artifact_key="site_metadata_manifest",
        site_id=bundle.site_id,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )


def enrich_existing_site_asset_bundle(
    bundle: SiteAssetBundle,
    *,
    site_id: str,
    inventory: dict[str, Any],
    artifact_store: ArtifactStore,
    force_refresh: bool = False,
    prepare_naip_asset_fn: PrepareNAIPFn | None = None,
    prepare_als_metadata_fn: PrepareALSMetadataFn | None = None,
    prepare_3dep_asset_fn: Prepare3DEPFn | None = None,
    prepare_rap_asset_fn: PrepareRAPFn | None = None,
    find_cached_rap_asset_fn: FindCachedRasterFn | None = None,
) -> SiteAssetBundle:
    source_assets = dict(bundle.source_assets or {})
    notes = list(bundle.notes or [])

    if force_refresh or (not validate_optional_raster_asset(source_assets.get("naip"))):
        if prepare_naip_asset_fn is not None:
            try:
                source_assets["naip"] = prepare_naip_asset_fn(
                    site_id,
                    artifact_store=artifact_store,
                    force_refresh=force_refresh,
                    inventory=inventory,
                    site_assets=bundle,
                )
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "NAIP unavailable during manifest enrichment:",
                        "NAIP unavailable:",
                    ),
                )
            except Exception as e:
                source_assets["naip"] = None
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "NAIP unavailable during manifest enrichment:",
                        "NAIP unavailable:",
                    ),
                    new_note=f"NAIP unavailable during manifest enrichment: {e}",
                )
    
    als_meta = source_assets.get("als_metadata")
    if force_refresh or als_meta is None or (isinstance(als_meta, list) and len(als_meta) == 0):
        if prepare_als_metadata_fn is not None:
            try:
                source_assets["als_metadata"] = prepare_als_metadata_fn(
                    site_id,
                    artifact_store=artifact_store,
                    force_refresh=force_refresh,
                    inventory=inventory,
                    config_sig=current_fe_config_signature(),
                )
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "ALS metadata unavailable during manifest enrichment:",
                        "ALS metadata unavailable:",
                    ),
                )
            except Exception as e:
                source_assets["als_metadata"] = []
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "ALS metadata unavailable during manifest enrichment:",
                        "ALS metadata unavailable:",
                    ),
                    new_note=f"ALS metadata unavailable during manifest enrichment: {e}",
                )
    
    if force_refresh or (not validate_optional_raster_asset(source_assets.get("3dep"))):
        if prepare_3dep_asset_fn is not None:
            try:
                source_assets["3dep"] = prepare_3dep_asset_fn(
                    site_id,
                    artifact_store=artifact_store,
                    force_refresh=force_refresh,
                    inventory=inventory,
                )
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "3DEP unavailable during manifest enrichment:",
                        "3DEP unavailable:",
                    ),
                )
            except Exception as e:
                source_assets["3dep"] = None
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "3DEP unavailable during manifest enrichment:",
                        "3DEP unavailable:",
                    ),
                    new_note=f"3DEP unavailable during manifest enrichment: {e}",
                )
    
    if force_refresh or (not validate_optional_raster_asset(source_assets.get("rap"))):
        if prepare_rap_asset_fn is not None:
            try:
                source_assets["rap"] = prepare_rap_asset_fn(
                    site_id,
                    artifact_store=artifact_store,
                    force_refresh=force_refresh,
                    inventory=inventory,
                )
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "RAP unavailable during manifest enrichment:",
                        "RAP unavailable:",
                        "RAP cache lookup failed:",
                    ),
                )
            except Exception as e:
                source_assets["rap"] = None
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "RAP unavailable during manifest enrichment:",
                        "RAP unavailable:",
                        "RAP cache lookup failed:",
                    ),
                    new_note=f"RAP unavailable during manifest enrichment: {e}",
                )
        elif find_cached_rap_asset_fn is not None:
            try:
                source_assets["rap"] = find_cached_rap_asset_fn(site_id)
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "RAP unavailable during manifest enrichment:",
                        "RAP unavailable:",
                        "RAP cache lookup failed:",
                    ),
                )
            except Exception as e:
                source_assets["rap"] = None
                notes = _replace_source_notes(
                    notes,
                    clear_prefixes=(
                        "RAP unavailable during manifest enrichment:",
                        "RAP unavailable:",
                        "RAP cache lookup failed:",
                    ),
                    new_note=f"RAP cache lookup failed: {e}",
                )
    else:
        source_assets.setdefault("rap", None)

    source_assets["source_inventory"] = inventory

    out = SiteAssetBundle(site_id=site_id)
    out.source_assets.update(source_assets)
    out.notes.extend(_dedupe_notes(notes))
    return out


def prepare_site_assets(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    force_refresh: bool = False,
    prepare_naip_asset_fn: PrepareNAIPFn | None = None,
    prepare_als_metadata_fn: PrepareALSMetadataFn | None = None,
    prepare_3dep_asset_fn: Prepare3DEPFn | None = None,
    prepare_rap_asset_fn: PrepareRAPFn | None = None,
    find_cached_rap_asset_fn: FindCachedRasterFn | None = None,
) -> SiteAssetBundle:
    config_sig = current_fe_config_signature()

    existing = None if force_refresh else try_load_site_metadata_manifest(
        site_id,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )

    inventory = None if force_refresh else try_load_source_inventory(
        site_id,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )
    if inventory is None:
        inventory = build_source_inventory(site_id)
        persist_source_inventory(
            site_id,
            inventory,
            artifact_store=artifact_store,
            config_sig=config_sig,
        )
        LOGGER.info("Built and persisted source inventory | site=%s", site_id)
    else:
        LOGGER.info("Using cached source inventory | site=%s", site_id)

    if existing is not None:
        enriched = enrich_existing_site_asset_bundle(
            existing,
            site_id=site_id,
            inventory=inventory,
            force_refresh=force_refresh,
            prepare_naip_asset_fn=prepare_naip_asset_fn,
            prepare_als_metadata_fn=prepare_als_metadata_fn,
            prepare_3dep_asset_fn=prepare_3dep_asset_fn,
            find_cached_rap_asset_fn=find_cached_rap_asset_fn,
            artifact_store=artifact_store,
            prepare_rap_asset_fn=prepare_rap_asset_fn,
        )

        old_payload = site_asset_bundle_payload(existing)
        new_payload = site_asset_bundle_payload(enriched)

        if old_payload != new_payload:
            persist_site_metadata_manifest(
                enriched,
                artifact_store=artifact_store,
                config_sig=config_sig,
            )
            LOGGER.info(
                "Prepared site assets from enriched manifest reuse | site=%s | keys=%s | notes=%d",
                site_id,
                list(enriched.source_assets.keys()),
                len(enriched.notes),
            )
        else:
            LOGGER.info(
                "Prepared site assets from enriched manifest reuse (no manifest changes) | site=%s | keys=%s | notes=%d",
                site_id,
                list(enriched.source_assets.keys()),
                len(enriched.notes),
            )

        return enriched

    bundle = SiteAssetBundle(site_id=site_id)

    if prepare_naip_asset_fn is not None:
        try:
            bundle.source_assets["naip"] = prepare_naip_asset_fn(
                site_id,
                artifact_store=artifact_store,
                force_refresh=force_refresh,
                inventory=inventory,
                site_assets=bundle,
            )
        except Exception as e:
            bundle.source_assets["naip"] = None
            bundle.notes.append(f"NAIP unavailable: {e}")
    
    if prepare_als_metadata_fn is not None:
        try:
            bundle.source_assets["als_metadata"] = prepare_als_metadata_fn(
                site_id,
                artifact_store=artifact_store,
                force_refresh=force_refresh,
                inventory=inventory,
                config_sig=config_sig,
            )
        except Exception as e:
            bundle.source_assets["als_metadata"] = []
            bundle.notes.append(f"ALS metadata unavailable: {e}")
    
    if prepare_3dep_asset_fn is not None:
        try:
            bundle.source_assets["3dep"] = prepare_3dep_asset_fn(
                site_id,
                artifact_store=artifact_store,
                force_refresh=force_refresh,
                inventory=inventory,
            )
        except Exception as e:
            bundle.source_assets["3dep"] = None
            bundle.notes.append(f"3DEP unavailable: {e}")
    
    if prepare_rap_asset_fn is not None:
        try:
            bundle.source_assets["rap"] = prepare_rap_asset_fn(
                site_id,
                artifact_store=artifact_store,
                force_refresh=force_refresh,
                inventory=inventory,
            )
        except Exception as e:
            bundle.source_assets["rap"] = None
            bundle.notes.append(f"RAP unavailable: {e}")
    elif find_cached_rap_asset_fn is not None:
        try:
            bundle.source_assets["rap"] = find_cached_rap_asset_fn(site_id)
        except Exception as e:
            bundle.notes.append(f"RAP cache lookup failed: {e}")
            bundle.source_assets["rap"] = None
    else:
        bundle.source_assets["rap"] = None

    bundle.source_assets["source_inventory"] = inventory
    bundle.notes = _dedupe_notes(bundle.notes)

    persist_site_metadata_manifest(
        bundle,
        artifact_store=artifact_store,
        config_sig=config_sig,
    )

    LOGGER.info(
        "Prepared site assets | site=%s | keys=%s | notes=%d",
        site_id,
        list(bundle.source_assets.keys()),
        len(bundle.notes),
    )
    return bundle