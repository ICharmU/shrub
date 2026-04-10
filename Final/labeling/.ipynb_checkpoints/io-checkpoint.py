from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rasterio

from Final.labeling.manifests import SESSION


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with SESSION.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dest


def parse_transform_txt(txt_path: Path) -> np.ndarray:
    text = txt_path.read_text(errors="ignore")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    vals = np.array([float(x) for x in nums], dtype=float)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        row_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ln)
        if row_nums:
            rows.append([float(x) for x in row_nums])

    if len(rows) >= 4 and all(len(r) >= 4 for r in rows[:4]):
        return np.array([r[:4] for r in rows[:4]], dtype=float)
    if len(rows) >= 3 and all(len(r) >= 4 for r in rows[:3]):
        M = np.eye(4, dtype=float)
        M[:3, :] = np.array([r[:4] for r in rows[:3]], dtype=float)
        return M
    if vals.size == 16:
        return vals.reshape(4, 4)
    if vals.size == 12:
        M = np.eye(4, dtype=float)
        M[:3, :] = vals.reshape(3, 4)
        return M
    raise ValueError(f"Could not parse transform file: {txt_path}")


def open_naip(naip_path: Path):
    return rasterio.open(naip_path)


def trim_pdal_metadata(info: dict) -> dict:
    summary = {}
    md = info.get("metadata", {})
    stats = info.get("stats", {})

    for key in ["count", "compressed", "major_version", "minor_version", "dataformat_id", "srs"]:
        if key in md:
            summary[key] = md[key]

    native_bounds = {}
    for src_key, dst_key in [
        ("minx", "minx"), ("maxx", "maxx"),
        ("miny", "miny"), ("maxy", "maxy"),
        ("minz", "minz"), ("maxz", "maxz"),
    ]:
        if src_key in md:
            native_bounds[dst_key] = md[src_key]
    if native_bounds:
        summary["native_bounds"] = native_bounds

    if "srs" in md and isinstance(md["srs"], dict):
        srs = md["srs"]
        summary["srs_wkt"] = srs.get("compoundwkt") or srs.get("wkt")
        summary["srs_json"] = srs.get("json")

    summary["stats_keys"] = sorted(stats.keys())
    summary["metadata_keys"] = sorted(md.keys())
    return summary


def extract_als_metadata(laz_path: Path) -> dict:
    cmd = ["pdal", "info", "--metadata", "--stats", str(laz_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pdal info failed for {laz_path}:\n{proc.stderr}")
    raw = json.loads(proc.stdout)
    return trim_pdal_metadata(raw)
