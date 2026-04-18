from __future__ import annotations
from pathlib import Path
from typing import Any

from Final.features.artifact_io import (
    current_fe_config_signature,
    persist_existing_file_artifact,
    remote_artifact_exists,
    render_artifact_rel_path
)
from Final.features.ee_utils import ee_naip_image_for_site, ensure_ee_initialized, export_ee_naip_tiled_to_geotiff, site_bounds_wgs84_for_naip_sitewide_fallback, site_region_coords_for_naip_sitewide_fallback
from Final.labeling.io import download_file
import numpy as np

from Final.artifact_store import ArtifactStore
from Final.shared_utils import get_logger

from Final.features.config import fe_cfg
from Final.features.models import (
    ChunkManifest,
    CanonicalGrid,
    FeatureFamilySpec,
    RasterStackRegistry,
    SiteAssetBundle,
    SourceRasterBundle,
    RuntimeTier,
    RepresentationTarget
)
from Final.features.raster_io import infer_naip_band_names, read_raster_bundle, validate_cached_raster
from Final.features.source_registry import run_source_chunked_pipeline
from Final.features.stack_registry import load_or_init_stack_registry
from Final.features.assets import build_source_inventory, site_naip_cache_root
from Final.features.fe_2d import (
    grey_opening,
    grey_closing,
    get_uniform_blur,
    get_entropy_feature,
    get_fast_lbp_texture,
    get_color_diff,
    get_granulometry_features,
    get_fractal_dimension_map,
    get_wavelet_features,
    fast_cpu_gabor,
    get_ldp_feature,
)

LOGGER = get_logger("features.source_naip")


NAIP_FAMILY_SPECS = {
    "raw": FeatureFamilySpec(
        key="raw",
        source_name="naip",
        runtime_tier=RuntimeTier.CHEAP,
        required_halo_px=0,
        representation_target=RepresentationTarget.RASTER,
        notes="Raw NAIP channels.",
    ),
    "veg_idx": FeatureFamilySpec(
        key="veg_idx",
        source_name="naip",
        runtime_tier=RuntimeTier.CHEAP,
        required_halo_px=0,
        representation_target=RepresentationTarget.RASTER,
        notes="Vegetation indices like NDVI / VARI.",
    ),
    "texture": FeatureFamilySpec(
        key="texture",
        source_name="naip",
        runtime_tier=RuntimeTier.MODERATE,
        required_halo_px=max(fe_cfg.naip.texture_entropy_window, max(fe_cfg.naip.texture_blur_sizes, default=0)),
        representation_target=RepresentationTarget.RASTER,
        notes="Entropy / LBP / blur / local structure features.",
    ),
    "multiscale": FeatureFamilySpec(
        key="multiscale",
        source_name="naip",
        runtime_tier=RuntimeTier.MODERATE,
        required_halo_px=max(fe_cfg.naip.multiscale_sizes, default=0),
        representation_target=RepresentationTarget.RASTER,
        notes="Multi-neighborhood statistics.",
    ),
    "color_diff": FeatureFamilySpec(
        key="color_diff",
        source_name="naip",
        runtime_tier=RuntimeTier.MODERATE,
        required_halo_px=1,
        representation_target=RepresentationTarget.RASTER,
    ),
    "granulometry": FeatureFamilySpec(
        key="granulometry",
        source_name="naip",
        runtime_tier=RuntimeTier.EXPENSIVE,
        required_halo_px=max(fe_cfg.naip.granulometry_scales, default=0),
        representation_target=RepresentationTarget.RASTER,
    ),
    "fractal": FeatureFamilySpec(
        key="fractal",
        source_name="naip",
        runtime_tier=RuntimeTier.EXPENSIVE,
        required_halo_px=max(fe_cfg.naip.fractal_scales, default=0),
        representation_target=RepresentationTarget.RASTER,
    ),
    "wavelet": FeatureFamilySpec(
        key="wavelet",
        source_name="naip",
        runtime_tier=RuntimeTier.EXPENSIVE,
        required_halo_px=16,
        representation_target=RepresentationTarget.RASTER,
    ),
    "gabor": FeatureFamilySpec(
        key="gabor",
        source_name="naip",
        runtime_tier=RuntimeTier.EXPENSIVE,
        required_halo_px=32,
        representation_target=RepresentationTarget.RASTER,
    ),
    "ldp": FeatureFamilySpec(
        key="ldp",
        source_name="naip",
        runtime_tier=RuntimeTier.MODERATE,
        required_halo_px=1,
        representation_target=RepresentationTarget.RASTER,
    ),
    "morphology": FeatureFamilySpec(
        key="morphology",
        source_name="naip",
        runtime_tier=RuntimeTier.EXPENSIVE,
        required_halo_px=max(fe_cfg.naip.morphology_window_sizes, default=0),
        representation_target=RepresentationTarget.RASTER,
    ),
}

def nan_to_num_copy(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = fill_value
    return out


def safe_divide(num: np.ndarray, denom: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.full_like(num, fill_value, dtype=np.float32)
    mask = np.isfinite(num) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    out[mask] = (num[mask] / denom[mask]).astype(np.float32)
    return out


def quantize_image(arr: np.ndarray, levels: int = 256) -> np.ndarray:
    arr = nan_to_num_copy(arr)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * (levels - 1)).astype(np.uint8)


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    arr = nan_to_num_copy(arr)
    gy, gx = np.gradient(arr)
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def local_std(arr: np.ndarray, size: int = 5) -> np.ndarray:
    arr = nan_to_num_copy(arr)
    mean = get_uniform_blur(arr, neighborhood_size=size)
    mean_sq = get_uniform_blur(arr**2, neighborhood_size=size)
    var = np.maximum(mean_sq - mean**2, 0.0)
    return np.sqrt(var).astype(np.float32)


def choose_base_band(arrays: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    for name in fe_cfg.naip.base_band_preference:
        if name in arrays:
            return name, arrays[name]
    first = next(iter(arrays.keys()))
    return first, arrays[first]


def get_simple_color_diff_kernels() -> dict[str, np.ndarray]:
    return {
        "sobel_x_like": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        "sobel_y_like": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        "laplacian_like": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    }

def active_naip_family_names() -> tuple[str, ...]:
    enabled = set(fe_cfg.naip.enabled_families) | set(fe_cfg.naip.heavy_families_enabled)
    ordered = []
    for key in NAIP_FAMILY_SPECS:
        if key in enabled:
            ordered.append(key)
    return tuple(ordered)


def compute_naip_family_on_chunk(
    family_name: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out = {}

    if family_name == "raw":
        for name, arr in arrays.items():
            out[f"naip_raw_{name}"] = arr.astype(np.float32)
        return out

    if family_name == "veg_idx":
        red = arrays.get("red")
        green = arrays.get("green")
        blue = arrays.get("blue")
        nir = arrays.get("nir")
        if red is not None and nir is not None:
            out["naip_idx_ndvi"] = safe_divide(nir - red, nir + red)
        if red is not None and green is not None and blue is not None:
            out["naip_idx_vari"] = safe_divide(green - red, green + red - blue)
        if red is not None and green is not None:
            out["naip_idx_g_minus_r"] = (green - red).astype(np.float32)
        if green is not None and blue is not None:
            out["naip_idx_g_minus_b"] = (green - blue).astype(np.float32)
        return out

    base_name, base = choose_base_band(arrays)
    base_clean = nan_to_num_copy(base)
    base_tex = quantize_image(base_clean, levels=fe_cfg.naip.quantize_levels) if fe_cfg.naip.quantize_for_texture else base_clean

    if family_name == "texture":
        out["naip_tex_gradmag"] = gradient_magnitude(base_clean)
        out["naip_tex_localstd_5"] = local_std(base_clean, size=5)
        out["naip_tex_entropy"] = get_entropy_feature(
            base_tex,
            neighborhood_size=fe_cfg.naip.texture_entropy_window
        ).astype(np.float32)
        out[f"naip_tex_lbp_r{fe_cfg.naip.texture_lbp_radius}"] = get_fast_lbp_texture(
            base_tex, radius=fe_cfg.naip.texture_lbp_radius
        ).astype(np.float32)
        for size in fe_cfg.naip.texture_blur_sizes:
            out[f"naip_tex_blur_{size}"] = get_uniform_blur(base_clean, neighborhood_size=size).astype(np.float32)
        return out

    if family_name == "multiscale":
        selected = [name for name in ["red", "green", "nir"] if name in arrays]
        if not selected:
            selected = [base_name]
        for band_name in selected:
            band = nan_to_num_copy(arrays[band_name])
            for size in fe_cfg.naip.multiscale_sizes:
                out[f"naip_ms_mean_{band_name}_{size}"] = get_uniform_blur(band, neighborhood_size=size).astype(np.float32)
                out[f"naip_ms_std_{band_name}_{size}"] = local_std(band, size=size)
        return out

    if family_name == "color_diff":
        for kernel_name, kernel in get_simple_color_diff_kernels().items():
            out[f"naip_cd_{kernel_name}"] = get_color_diff(base_clean, kernel).astype(np.float32)
        return out

    if family_name == "granulometry":
        cube = get_granulometry_features(base_clean, scales=list(fe_cfg.naip.granulometry_scales))
        for i in range(cube.shape[2]):
            out[f"naip_gran_{i:02d}"] = cube[:, :, i].astype(np.float32)
        return out

    if family_name == "fractal":
        out["naip_frac_fd"] = get_fractal_dimension_map(base_clean, scales=list(fe_cfg.naip.fractal_scales)).astype(np.float32)
        return out

    if family_name == "wavelet":
        cube = get_wavelet_features(
            base_clean,
            wavelet_type=fe_cfg.naip.wavelet_type,
            level=fe_cfg.naip.wavelet_level,
        )
        for i in range(cube.shape[2]):
            out[f"naip_wav_{i:02d}"] = cube[:, :, i].astype(np.float32)
        return out

    if family_name == "gabor":
        idx = 0
        for freq in fe_cfg.naip.gabor_frequencies:
            for theta in fe_cfg.naip.gabor_thetas:
                out[f"naip_gabor_{idx:02d}"] = fast_cpu_gabor(base_clean, frequency=freq, theta=theta).astype(np.float32)
                idx += 1
        return out

    if family_name == "ldp":
        out["naip_ldp"] = get_ldp_feature(base_tex, k=fe_cfg.naip.ldp_k).astype(np.float32)
        return out

    if family_name == "morphology":
        for size in fe_cfg.naip.morphology_window_sizes:
            opened = grey_opening(base_clean, size=(size, size)).astype(np.float32)
            closed = grey_closing(base_clean, size=(size, size)).astype(np.float32)
            out[f"naip_morph_open_{size}"] = opened
            out[f"naip_morph_close_{size}"] = closed
            out[f"naip_morph_open_resid_{size}"] = (base_clean - opened).astype(np.float32)
            out[f"naip_morph_close_resid_{size}"] = (closed - base_clean).astype(np.float32)
        return out

    raise KeyError(f"Unsupported NAIP family: {family_name}")



def active_naip_family_specs() -> dict[str, FeatureFamilySpec]:
    enabled = set(fe_cfg.naip.enabled_families) | set(fe_cfg.naip.heavy_families_enabled)
    return {k: v for k, v in NAIP_FAMILY_SPECS.items() if k in enabled}

def prepare_naip_asset(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    force_refresh: bool = False,
    inventory: dict[str, Any] | None = None,
    canonical_grid: CanonicalGrid | None = None,
    site_assets: SiteAssetBundle | None = None,
) -> Path:
    inventory = inventory or build_source_inventory(site_id)
    naip_name = inventory["naip"]["expected_name"]
    naip_url = inventory["naip"]["remote_url"]

    local_path = site_naip_cache_root(site_id) / naip_name
    rel_path = render_artifact_rel_path(
        "site_naip_raster",
        site_id=site_id,
        filename=naip_name,
    )

    # 1) valid local cache
    if (not force_refresh) and local_path.exists() and validate_cached_raster(local_path):
        LOGGER.info("Using cached NAIP | site=%s | path=%s", site_id, local_path)
        return local_path

    # 2) remove invalid local copy if present
    if local_path.exists():
        LOGGER.warning("Removing invalid cached NAIP | site=%s | path=%s", site_id, local_path)
        local_path.unlink(missing_ok=True)

    # 3) try remote artifact-store copy
    if not force_refresh and remote_artifact_exists(rel_path, artifact_store=artifact_store):
        LOGGER.info("Hydrating NAIP from artifact store | site=%s | rel_path=%s", site_id, rel_path)
        pulled = artifact_store.pull(rel_path, local_path=local_path)
        if validate_cached_raster(pulled):
            return pulled

        LOGGER.warning(
            "Hydrated NAIP failed validation; removing broken hydrated copy | site=%s | path=%s",
            site_id, pulled
        )
        Path(pulled).unlink(missing_ok=True)

    # 4) try DAV download first
    dav_ok = False
    try:
        LOGGER.info("Downloading NAIP | site=%s | url=%s", site_id, naip_url)
        download_file(naip_url, local_path)
        dav_ok = validate_cached_raster(local_path)
    except Exception as e:
        LOGGER.warning("DAV NAIP download failed | site=%s | err=%s", site_id, e)
        dav_ok = False

    if dav_ok:
        try:
            persist_existing_file_artifact(
                local_path,
                artifact_key="site_naip_raster",
                site_id=site_id,
                filename=naip_name,
                artifact_store=artifact_store,
            )
        except Exception as e:
            LOGGER.warning("Persisting DAV NAIP artifact failed | site=%s | err=%s", site_id, e)
        return local_path

    if local_path.exists():
        local_path.unlink(missing_ok=True)

    # 5) EE fallback
    ensure_ee_initialized(project="shrubwise-dc-488219")

    west, south, east, north = site_bounds_wgs84_for_naip_sitewide_fallback(
        site_id,
        site_assets=site_assets,
        canonical_grid=canonical_grid,
        resolution_m=1.0,
    )
    region_coords = site_region_coords_for_naip_sitewide_fallback(
        site_id,
        site_assets=site_assets,
        canonical_grid=canonical_grid,
        resolution_m=1.0,
    )
    image = ee_naip_image_for_site(
        site_id,
        canonical_grid=canonical_grid,
        resolution_m=1.0,
    )

    LOGGER.info(
        "Exporting NAIP from Earth Engine fallback | site=%s | out=%s | bounds_wgs84=(%.6f, %.6f, %.6f, %.6f)",
        site_id, local_path, west, south, east, north
    )

    bounds_wgs84 = site_bounds_wgs84_for_naip_sitewide_fallback(
        site_id,
        site_assets=site_assets,
        canonical_grid=canonical_grid,
        resolution_m=1.0,
    )

    LOGGER.info(
        "Exporting NAIP from Earth Engine fallback (tiled) | site=%s | out=%s | bounds_wgs84=(%.6f, %.6f, %.6f, %.6f)",
        site_id, local_path, *bounds_wgs84
    )

    export_ee_naip_tiled_to_geotiff(
        out_path=local_path,
        bounds_wgs84=bounds_wgs84,
        scale=1.0,
        crs="EPSG:4326",
        nx=2,
        ny=2,
        year=None,
    )

    if not validate_cached_raster(local_path):
        local_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"NAIP unavailable for site={site_id}: DAV download failed validation and EE fallback export was unreadable."
        )

    try:
        persist_existing_file_artifact(
            local_path,
            artifact_key="site_naip_raster",
            site_id=site_id,
            filename=naip_name,
        )
    except Exception as e:
        LOGGER.warning("Persisting EE NAIP artifact failed | site=%s | err=%s", site_id, e)

    return local_path

def load_naip_source_bundle(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    site_assets: SiteAssetBundle,
    canonical_grid: CanonicalGrid | None = None,
    force_refresh: bool = False,
) -> SourceRasterBundle:
    tif_path = prepare_naip_asset(
        site_id,
        artifact_store=artifact_store,
        force_refresh=force_refresh,
        inventory=site_assets.source_assets.get("source_inventory"),
        canonical_grid=canonical_grid,
        site_assets=site_assets,
    )
    return read_raster_bundle(
        tif_path,
        site_id=site_id,
        source_name="naip",
        band_names=infer_naip_band_names(tif_path),
    )


def _make_naip_compute_fn(family_name: str):
    def _compute(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return compute_naip_family_on_chunk(family_name, arrays)
    return _compute


def run_naip_chunked_pipeline(
    site_id: str,
    *,
    artifact_store: ArtifactStore,
    assets: SiteAssetBundle,
    grid: CanonicalGrid,
    chunk_manifest: ChunkManifest,
    registry: RasterStackRegistry | None = None,
    rebuild_registry: bool = True,
    force_refresh_source: bool = False,
) -> RasterStackRegistry:
    raw_bundle = load_naip_source_bundle(
        site_id,
        artifact_store=artifact_store,
        site_assets=assets,
        canonical_grid=grid,
        force_refresh=force_refresh_source,
    )

    family_specs = active_naip_family_specs()
    family_compute_fns = {name: _make_naip_compute_fn(name) for name in family_specs}

    if registry is None or rebuild_registry:
        registry = load_or_init_stack_registry(
            site_id,
            artifact_store=artifact_store,
            config_signature=current_fe_config_signature(),
        )

    return run_source_chunked_pipeline(
        artifact_store=artifact_store,
        site_id=site_id,
        source_name="naip",
        raw_bundle=raw_bundle,
        canonical_grid=grid,
        chunk_manifest=chunk_manifest,
        family_specs=family_specs,
        family_compute_fns=family_compute_fns,
        family_cfg_payloads={name: fe_cfg.naip.__dict__ for name in family_specs},
        registry=registry,
    )