from __future__ import annotations

import shutil
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import requests

from Final.config import ProjectConfig


SESSION = requests.Session()


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def list_webdav(url: str, depth: int = 1) -> list[dict]:
    headers = {"Depth": str(depth), "Content-Type": "application/xml"}
    body = """<?xml version="1.0" encoding="utf-8" ?>
    <d:propfind xmlns:d="DAV:">
      <d:prop>
        <d:resourcetype/>
        <d:getcontentlength/>
        <d:getlastmodified/>
      </d:prop>
    </d:propfind>"""
    r = SESSION.request("PROPFIND", url, headers=headers, data=body, timeout=120)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    entries = []
    for resp in root.iter():
        if _strip_ns(resp.tag) != "response":
            continue

        href = None
        is_collection = False
        content_length = None
        last_modified = None

        for child in resp.iter():
            tag = _strip_ns(child.tag)
            if tag == "href":
                href = child.text
            elif tag == "collection":
                is_collection = True
            elif tag == "getcontentlength" and child.text:
                try:
                    content_length = int(child.text)
                except ValueError:
                    content_length = None
            elif tag == "getlastmodified":
                last_modified = child.text

        if not href:
            continue

        decoded_href = urllib.parse.unquote(href)
        name = decoded_href.rstrip("/").split("/")[-1]
        if not name:
            continue

        entry_url = urllib.parse.urljoin(url.rstrip("/") + "/", urllib.parse.quote(name))
        entries.append(
            {
                "name": name,
                "url": entry_url,
                "is_dir": is_collection,
                "size": content_length,
                "last_modified": last_modified,
            }
        )

    dedup = {(e["name"], e["is_dir"]): e for e in entries}
    return list(dedup.values())


def list_files_with_suffix(url: str, suffixes: tuple[str, ...]) -> list[dict]:
    suffixes = tuple(s.lower() for s in suffixes)
    return [
        e for e in list_webdav(url, depth=1)
        if (not e["is_dir"]) and e["name"].lower().endswith(suffixes)
    ]


def site_to_remote_base(cfg: ProjectConfig, site: str) -> str:
    return f"{cfg.data.base_root}/ucca-{site}"


def site_to_tif_name(site: str) -> str:
    return site.replace("-", "_") + ".tif"


def build_remote_manifest(cfg: ProjectConfig, use_revised_shrubs: bool = True) -> pd.DataFrame:
    rows = []
    for site in cfg.sites:
        site_base = site_to_remote_base(cfg, site)
        shrubs_dir = cfg.data.revised_shrubs_dir if use_revised_shrubs else cfg.data.shrubs_dir
        shrub_url = f"{site_base}/{shrubs_dir}"
        transform_url = f"{site_base}/{cfg.data.transformations_dir}"
        als_url = f"{site_base}/{cfg.data.als_dir}"
        naip_url = f"{site_base}/{cfg.data.naip_3dep_dir}/{site_to_tif_name(site)}"

        shrub_files = list_files_with_suffix(shrub_url, (".csv",))
        transform_files = list_files_with_suffix(transform_url, (".txt",))
        als_files = list_files_with_suffix(als_url, (".laz", ".las", ".copc.laz"))

        rows.append(
            {
                "site_id": site,
                "site_base": site_base,
                "shrub_url": shrub_url,
                "transform_url": transform_url,
                "als_url": als_url,
                "naip_url": naip_url,
                "n_shrub_csv": len(shrub_files),
                "n_transform_txt": len(transform_files),
                "n_als_files": len(als_files),
                "shrub_files": shrub_files,
                "transform_files": transform_files,
                "als_files": als_files,
            }
        )
    return pd.DataFrame(rows)


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)