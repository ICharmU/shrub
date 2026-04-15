from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path

from Final.config import ProjectConfig
from Final.models import RuntimeCapabilityReport


def _try_read_marker_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def detect_runtime_capabilities(cfg: ProjectConfig) -> RuntimeCapabilityReport:
    rep = RuntimeCapabilityReport()

    env = os.environ

    # 1) detect image alias from env vars
    raw_image_name = None
    for key in cfg.runtime_detection.image_env_var_names:
        val = env.get(key)
        if val:
            raw_image_name = val.strip()
            rep.marker_env_matches[key] = raw_image_name
            break

    # 2) marker files
    for marker in cfg.runtime_detection.marker_file_candidates:
        val = _try_read_marker_file(marker)
        if val:
            rep.marker_files_found.append(marker)
            if raw_image_name is None:
                raw_image_name = val

    # 3) conda env
    for key in cfg.runtime_detection.conda_env_var_names:
        val = env.get(key)
        if val:
            rep.detected_conda_env = val.strip()
            break

    # 4) executables
    for exe in cfg.runtime_detection.detect_executables:
        if shutil.which(exe):
            rep.available_executables.append(exe)

    # 5) python modules
    for mod in cfg.runtime_detection.detect_python_modules:
        if _module_available(mod):
            rep.available_python_modules.append(mod)

    # 6) match configured images
    detected_key = None
    detected_alias = None
    for image_spec in cfg.runtime_images:
        aliases = set(image_spec.aliases) | {image_spec.key}
        if raw_image_name and any(alias in raw_image_name for alias in aliases):
            detected_key = image_spec.key
            detected_alias = raw_image_name
            break

    rep.detected_image_key = detected_key
    rep.detected_image_alias = detected_alias or raw_image_name

    # 7) infer capabilities from detected image
    capabilities = set()
    if detected_key is not None:
        for image_spec in cfg.runtime_images:
            if image_spec.key == detected_key:
                capabilities.update(image_spec.provided_capabilities)

    # 8) infer capabilities from executables/modules
    if "python" in rep.available_executables:
        capabilities.add("runtime:python")
    if "Rscript" in rep.available_executables:
        capabilities.add("runtime:rscript")
    if "pdal" in rep.available_executables:
        capabilities.add("runtime:pdal")
    if "rasterio" in rep.available_python_modules:
        capabilities.add("runtime:rasterio")

    rep.capabilities = sorted(capabilities)
    return rep