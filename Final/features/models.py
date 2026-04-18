from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rasterio.transform import Affine
from rasterio.enums import Resampling


@dataclass
class SiteAssetBundle:
    site_id: str
    source_assets: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalGrid:
    site_id: str
    width: int
    height: int
    transform: Affine
    crs: Any
    source_name: str
    nodata: float | int | None = None

    @property
    def pixel_size(self) -> tuple[float, float]:
        return (abs(float(self.transform.a)), abs(float(self.transform.e)))

    def profile(
        self,
        *,
        dtype: str = "float32",
        count: int = 1,
        nodata: float | int | None = None,
        driver: str = "GTiff",
    ) -> dict[str, Any]:
        return {
            "driver": driver,
            "height": self.height,
            "width": self.width,
            "count": count,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata if nodata is None else nodata,
        }


@dataclass(frozen=True)
class ChunkRecord:
    site_id: str
    chunk_id: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    halo_px: int

    @property
    def height(self) -> int:
        return self.row_end - self.row_start

    @property
    def width(self) -> int:
        return self.col_end - self.col_start


@dataclass
class ChunkManifest:
    site_id: str
    chunk_size_px: int
    halo_px_default: int
    records: list[ChunkRecord] = field(default_factory=list)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    native_kind: str
    default_resampling: Resampling
    notes: str = ""


@dataclass
class SourceRasterBundle:
    site_id: str
    source_name: str
    arrays: dict[str, Any]
    transform: Affine
    crs: Any
    nodata: float | int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceIngestRecord:
    site_id: str
    source_name: str
    asset_path: str | None = None
    status: str = "unknown"
    notes: list[str] = field(default_factory=list)


@dataclass
class SourceReadyRecord:
    site_id: str
    source_name: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class RasterStackLayer:
    site_id: str
    config_signature: str
    source_name: str
    family_name: str
    layer_name: str
    rel_path: str | None = None
    local_path: str | None = None
    remote_ref: str | None = None
    storage_tier: str | None = None
    dtype: str = "float32"
    chunk_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RasterStackRegistry:
    site_id: str
    config_signature: str
    layers: list[RasterStackLayer] = field(default_factory=list)

    def add_layer(self, layer: RasterStackLayer) -> None:
        self.layers.append(layer)