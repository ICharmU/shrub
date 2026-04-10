from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


PLOT_KEY_RE = re.compile(r"^(?P<sitecode>[A-Za-z]{5})_(?P<plotnum>\d{4})")
DATE_RE = re.compile(r"(20\d{6})")
MASK_RE = re.compile(
    r"^(?P<site>[A-Za-z]{5})_(?P<plot>\d{4})_(?P<date>\d{8})_\d+_mask(?:_\d+(?:\.\d+)?m)?\.tif$",
    re.IGNORECASE,
)


def extract_plot_key(plot_id: str) -> str:
    m = PLOT_KEY_RE.match(str(plot_id))
    if m:
        return f"{m.group('sitecode').upper()}_{m.group('plotnum')}"
    return str(plot_id)


def extract_date_token(text: str) -> str | None:
    m = DATE_RE.search(str(text))
    return m.group(1) if m else None


def deduplicate_artifact_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "plot_id" not in df.columns:
        return df.copy()

    out = df.copy()

    if "date_token" not in out.columns:
        out["date_token"] = out["plot_id"].astype(str).str.extract(r"(20\d{6})", expand=False)

    out["plot_key"] = out["plot_id"].astype(str).map(extract_plot_key)
    out["date_token"] = out["date_token"].fillna("99999999")

    # Keep oldest by plot_key, but separately for each resolution / label variant / source version
    group_cols = ["site_id", "plot_key"]
    if "resolution_m" in out.columns:
        group_cols.append("resolution_m")
    if "label_variant" in out.columns:
        group_cols.append("label_variant")
    if "source_version" in out.columns:
        group_cols.append("source_version")

    out = out.sort_values(group_cols + ["date_token", "plot_id"])
    out["_rank"] = out.groupby(group_cols).cumcount()
    out["dedup_keep"] = out["_rank"] == 0
    out["dedup_reason"] = out["_rank"].map(
        lambda x: "kept_oldest_for_plot_key" if x == 0 else "removed_newer_duplicate_for_plot_key"
    )

    return out.drop(columns=["_rank"])


def deduplicate_masks_on_disk(output_root: Path) -> list[dict]:
    total_removed = []
    site_dirs = sorted(p for p in output_root.iterdir() if p.is_dir())

    for site_dir in site_dirs:
        groups = defaultdict(list)
        for tif in site_dir.glob("*.tif"):
            m = MASK_RE.match(tif.name)
            if m:
                plot_key = f"{m.group('site').upper()}_{m.group('plot')}"
                date = m.group("date")
                resolution_tag = tif.stem.split("_mask")[-1]  # "", "_1m", "_2m", etc.
                groups[(plot_key, resolution_tag)].append((date, tif))

        for (plot_key, resolution_tag), items in groups.items():
            if len(items) <= 1:
                continue
            items_sorted = sorted(items, key=lambda x: x[0])  # oldest first
            keeper = items_sorted[0][1]
            for _, dup in items_sorted[1:]:
                dup.unlink(missing_ok=True)
                total_removed.append(
                    {
                        "site_dir": site_dir.name,
                        "plot_key": plot_key,
                        "resolution_tag": resolution_tag,
                        "kept": keeper.name,
                        "removed": dup.name,
                    }
                )
    return total_removed