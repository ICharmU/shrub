from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from Final.models import ShrubObjectColumns


COLS = ShrubObjectColumns()

MASK_RE = re.compile(r"^(?P<site>[A-Za-z]{5})_(?P<plot>\d{4})_(?P<date>\d{8})_\d+_mask\.tif$", re.IGNORECASE)


def deduplicate_artifact_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "plot_id" not in df.columns:
        return df.copy()
    out = df.copy()
    if "date_token" not in out.columns:
        out["date_token"] = out["plot_id"].astype(str).str.extract(r"(20\d{6})", expand=False)
    out["date_token"] = out["date_token"].fillna("99999999")
    out = out.sort_values(["site_id", "plot_id", "date_token"])
    out["_rank"] = out.groupby(["site_id", "plot_id"]).cumcount()
    out["dedup_keep"] = out["_rank"] == 0
    out["dedup_reason"] = out["_rank"].map(lambda x: "kept_oldest" if x == 0 else "removed_newer_duplicate")
    return out.drop(columns=["_rank"])


def deduplicate_masks_on_disk(output_root: Path) -> list[dict]:
    total_removed = []
    site_dirs = sorted(p for p in output_root.iterdir() if p.is_dir())
    for site_dir in site_dirs:
        groups = defaultdict(list)
        for tif in site_dir.glob("*.tif"):
            m = MASK_RE.match(tif.name)
            if m:
                groups[(m.group("site").upper(), m.group("plot"))].append(tif)

        for (site_code, plot_id), files in groups.items():
            if len(files) <= 1:
                continue
            files_sorted = sorted(files, key=lambda p: MASK_RE.match(p.name).group("date"))
            keeper = files_sorted[0]
            for dup in files_sorted[1:]:
                dup.unlink(missing_ok=True)
                total_removed.append(
                    {"site_dir": site_dir.name, "site_code": site_code, "plot_id": plot_id, "kept": keeper.name, "removed": dup.name}
                )
    return total_removed
