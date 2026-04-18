from __future__ import annotations

from rasterio.enums import Resampling
import pandas as pd

from Final.features.models import SourceSpec


SOURCE_SPECS: dict[str, SourceSpec] = {
    "naip": SourceSpec(
        name="naip",
        native_kind="multiband_raster",
        default_resampling=Resampling.nearest,
        notes="Canonical appearance source and default canonical grid anchor.",
    ),
    "als": SourceSpec(
        name="als",
        native_kind="point_cloud_or_rasterized_structure",
        default_resampling=Resampling.bilinear,
        notes="Structural source for canopy height, density, and roughness context.",
    ),
    "3dep": SourceSpec(
        name="3dep",
        native_kind="terrain_raster",
        default_resampling=Resampling.bilinear,
        notes="Terrain and site-context raster source.",
    ),
    "rap": SourceSpec(
        name="rap",
        native_kind="coarse_vegetation_raster",
        default_resampling=Resampling.bilinear,
        notes="Coarse shrub/ecological prior source.",
    ),
}


def source_spec_frame() -> pd.DataFrame:
    rows = []
    for key, spec in SOURCE_SPECS.items():
        rows.append(
            {
                "source": key,
                "native_kind": spec.native_kind,
                "default_resampling": spec.default_resampling.name,
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)