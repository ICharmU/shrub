from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from skimage.transform import resize

from Final.models import ShrubObjectColumns


COLS = ShrubObjectColumns()


def summarize_objects(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(COLS.site_id)
        .agg(
            n_objects=(COLS.object_id, "count"),
            mean_area=(COLS.area_tls, "mean"),
            mean_radius=(COLS.radius_m, "mean"),
            mean_confidence=(COLS.object_confidence, "mean"),
        )
        .reset_index()
    )
    return summary


def create_overlay_figure(naip_path: str | Path, mask_path: str | Path, out_path: str | Path | None = None):
    with rasterio.open(mask_path) as msrc:
        mask_bounds = msrc.bounds
        mask = msrc.read(1).astype(float)
        nodata = msrc.nodata
    if nodata is not None:
        mask = np.where(mask == nodata, np.nan, mask)

    with rasterio.open(naip_path) as nsrc:
        win = from_bounds(*mask_bounds, transform=nsrc.transform)
        rgb = nsrc.read([1, 2, 3], window=win).astype(float)

    for i in range(3):
        b = rgb[i]
        rgb[i] = (b - b.min()) / (b.max() - b.min() + 1e-9)
    rgb_display = np.moveaxis(rgb, 0, -1)

    mask_resized = resize(mask, (rgb_display.shape[0], rgb_display.shape[1]), order=0, preserve_range=True, anti_aliasing=False)
    overlay = np.zeros((*mask_resized.shape, 4), dtype=float)
    overlay[mask_resized == 1] = [1.0, 0.1, 0.1, 0.6]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(rgb_display)
    axes[0].set_title("NAIP crop at mask extent")
    axes[0].axis("off")
    axes[1].imshow(rgb_display)
    axes[1].imshow(overlay)
    axes[1].set_title("+ shrub mask overlay")
    axes[1].axis("off")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
