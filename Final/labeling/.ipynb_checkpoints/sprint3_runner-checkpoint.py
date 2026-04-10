from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from Final.shared_utils import (
    extract_first_date_token,
    get_logger,
    safe_stem,
    tail_text_file,
    yyyymmdd_to_timestamp,
)
from Final.labeling.manifests import list_files_with_suffix, site_to_remote_base

logger = get_logger("labeling.sprint3_runner")

Sprint3Variant = Literal["original", "revised"]


@dataclass
class Sprint3RunResult:
    site_id: str
    input_ptx: Path
    output_dir: Path
    variant: Sprint3Variant
    shrub_csv: Path | None
    metrics_csv: Path | None
    tree_inventory_csv: Path | None
    fuels_raster: Path | None
    dtm_raster: Path | None
    chm_raster: Path | None
    stdout_log: Path
    stderr_log: Path
    returncode: int
    used_cache: bool = False
    ptx_name: str | None = None
    ptx_date_token: str | None = None
    ptx_date: pd.Timestamp | pd.NaT = pd.NaT
    ptx_quality_score: float | None = None


def _variant_script_name(variant: Sprint3Variant) -> str:
    if variant == "original":
        return "IntELiMon_1_1_1.R"
    if variant == "revised":
        return "IntELiMon_1_1_1_revised.R"
    raise ValueError(f"Unknown Sprint 3 variant: {variant}")


def _find_single(pattern_dir: Path, pattern: str) -> Path | None:
    if not pattern_dir.exists():
        return None
    matches = sorted(pattern_dir.glob(pattern))
    return matches[0] if matches else None


def _validate_rscript_executable(rscript_executable: str) -> str:
    resolved = shutil.which(rscript_executable)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find Rscript executable '{rscript_executable}' on PATH."
        )
    return resolved


def _collect_outputs(run_dir: Path) -> dict[str, Path | None]:
    shrubs_dir = run_dir / "Shrubs"
    metrics_dir = run_dir / "metrics"
    inv_dir = run_dir / "Inventory"
    fuels_dir = run_dir / "Fuels"
    dtm_dir = run_dir / "dtm"
    chm_dir = run_dir / "chm"

    return {
        "shrub_csv": _find_single(shrubs_dir, "*.csv"),
        "metrics_csv": _find_single(metrics_dir, "*.csv"),
        "tree_inventory_csv": _find_single(inv_dir, "*_inv.csv"),
        "fuels_raster": _find_single(fuels_dir, "*.tif"),
        "dtm_raster": _find_single(dtm_dir, "*.tif"),
        "chm_raster": _find_single(chm_dir, "*.tif"),
    }


def _result_from_existing_run(
    *,
    site_id: str,
    ptx_path: Path,
    run_dir: Path,
    variant: Sprint3Variant,
) -> Sprint3RunResult:
    outputs = _collect_outputs(run_dir)
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    ptx_name = ptx_path.name
    ptx_date_token = extract_first_date_token(ptx_name)
    ptx_date = yyyymmdd_to_timestamp(ptx_date_token)

    return Sprint3RunResult(
        site_id=site_id,
        input_ptx=ptx_path,
        output_dir=run_dir,
        variant=variant,
        shrub_csv=outputs["shrub_csv"],
        metrics_csv=outputs["metrics_csv"],
        tree_inventory_csv=outputs["tree_inventory_csv"],
        fuels_raster=outputs["fuels_raster"],
        dtm_raster=outputs["dtm_raster"],
        chm_raster=outputs["chm_raster"],
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        returncode=0,
        used_cache=True,
        ptx_name=ptx_name,
        ptx_date_token=ptx_date_token,
        ptx_date=ptx_date,
        ptx_quality_score=None,
    )


def has_successful_sprint3_outputs(run_dir: Path, require_artifacts: bool = True) -> bool:
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    if not run_dir.exists():
        return False
    if not stdout_log.exists() or not stderr_log.exists():
        return False

    outputs = _collect_outputs(run_dir)
    if not require_artifacts:
        return True

    required = [
        outputs["shrub_csv"],
        outputs["metrics_csv"],
        outputs["tree_inventory_csv"],
        outputs["fuels_raster"],
        outputs["dtm_raster"],
        outputs["chm_raster"],
    ]
    return all(x is not None for x in required)


def discover_remote_ptx_files_for_site(*, cfg, site_id: str) -> list[dict]:
    remote_dir = f"{site_to_remote_base(cfg, site_id)}/{cfg.data.original_tls_dir}"
    logger.info("Discovering remote PTX files for site=%s at %s", site_id, remote_dir)
    return list_files_with_suffix(remote_dir, (".ptx",))


def summarize_ptx_entries_by_site(*, cfg, site_ids: list[str]) -> pd.DataFrame:
    rows = []

    for site_id in site_ids:
        try:
            entries = discover_remote_ptx_files_for_site(cfg=cfg, site_id=site_id)
        except Exception as e:
            logger.exception("Failed to discover PTX files for site=%s", site_id)
            rows.append(
                {
                    "site_id": site_id,
                    "ptx_name": None,
                    "ptx_url": None,
                    "date_token": None,
                    "date": pd.NaT,
                    "ptx_quality_score": None,
                    "discovery_error": str(e),
                }
            )
            continue

        if not entries:
            rows.append(
                {
                    "site_id": site_id,
                    "ptx_name": None,
                    "ptx_url": None,
                    "date_token": None,
                    "date": pd.NaT,
                    "ptx_quality_score": None,
                    "discovery_error": None,
                }
            )
            continue

        for entry in entries:
            name = entry["name"]
            date_token = extract_first_date_token(name)
            date = yyyymmdd_to_timestamp(date_token)
            rows.append(
                {
                    "site_id": site_id,
                    "ptx_name": name,
                    "ptx_url": entry["url"],
                    "date_token": date_token,
                    "date": date,
                    "ptx_quality_score": compute_ptx_quality_score(
                        site_id=site_id,
                        ptx_name=name,
                        ptx_date=date,
                    ),
                    "discovery_error": None,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df = df.sort_values(
            by=["site_id", "date", "ptx_name"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)
    return df


def compute_ptx_quality_score(
    *,
    site_id: str,
    ptx_name: str,
    ptx_date: pd.Timestamp | pd.NaT,
) -> float:
    """
    Lightweight pre-run PTX quality score for ranking/selection.

    Current heuristic:
    - newer dated files score higher
    - valid date token beats missing date token
    - leaves room to add future quality signals later
    """
    score = 0.0

    if pd.notna(ptx_date):
        score += 1_000_000_000 + ptx_date.timestamp()
    else:
        score -= 1_000_000

    # small deterministic tiebreaker based on filename
    score += sum(ord(c) for c in ptx_name) * 1e-6
    return float(score)


def select_ptx_entries(
    ptx_summary_df: pd.DataFrame,
    *,
    max_ptx_per_site: int | None = None,
) -> pd.DataFrame:
    df = ptx_summary_df.copy()
    if df.empty:
        return df

    df = df[df["ptx_name"].notna()].copy()

    df = df.sort_values(
        by=["site_id", "ptx_quality_score", "date", "ptx_name"],
        ascending=[True, False, False, True],
        na_position="last",
    )

    if max_ptx_per_site is None:
        return df.reset_index(drop=True)

    return (
        df.groupby("site_id", group_keys=False)
          .head(max_ptx_per_site)
          .reset_index(drop=True)
    )


def cleanup_stale_ptx_cache(
    *,
    cache_root: str | Path,
    stale_days: int = 2,
) -> pd.DataFrame:
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
    removed = []

    for ptx_path in cache_root.rglob("*.ptx"):
        try:
            mtime = datetime.fromtimestamp(ptx_path.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                size_bytes = ptx_path.stat().st_size
                ptx_path.unlink(missing_ok=True)
                removed.append(
                    {
                        "path": str(ptx_path),
                        "size_bytes": size_bytes,
                        "mtime_utc": mtime.isoformat(),
                    }
                )
        except Exception as e:
            logger.warning("Failed stale PTX cleanup for %s: %s", ptx_path, e)

    removed_df = pd.DataFrame(removed)
    if not removed_df.empty:
        logger.info("Removed %d stale PTX file(s) from cache", len(removed_df))
    return removed_df


def download_ptx_with_cache(
    *,
    site_id: str,
    ptx_entry: dict,
    cache_root: str | Path,
) -> Path:
    cache_root = Path(cache_root).resolve()
    site_cache = cache_root / site_id
    site_cache.mkdir(parents=True, exist_ok=True)

    local_path = site_cache / ptx_entry["name"]
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.info("Using cached PTX for site=%s: %s", site_id, local_path)
        return local_path

    logger.info("Downloading PTX for site=%s from %s", site_id, ptx_entry["url"])
    try:
        with requests.get(ptx_entry["url"], stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception:
        # remove partial file on failed download
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    logger.info("Downloaded PTX to %s", local_path)
    return local_path


def cleanup_ptx_file(ptx_path: str | Path) -> None:
    ptx_path = Path(ptx_path)
    if ptx_path.exists():
        try:
            size_bytes = ptx_path.stat().st_size
            ptx_path.unlink()
            logger.info("Deleted PTX after processing: %s (%d bytes)", ptx_path, size_bytes)
        except Exception as e:
            logger.warning("Failed to delete PTX %s: %s", ptx_path, e)


def run_sprint3_for_ptx(
    *,
    site_id: str,
    ptx_path: str | Path,
    sprint3_base_dir: str | Path,
    output_root: str | Path,
    variant: Sprint3Variant = "revised",
    rscript_executable: str = "Rscript",
    raise_on_error: bool = True,
    force_rerun: bool = False,
    require_success_artifacts: bool = True,
) -> Sprint3RunResult:
    ptx_path = Path(ptx_path).resolve()
    sprint3_base_dir = Path(sprint3_base_dir).resolve()
    output_root = Path(output_root).resolve()

    script_path = sprint3_base_dir / _variant_script_name(variant)
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find Sprint 3 script: {script_path}")

    run_dir = output_root / site_id / "sprint3" / variant / safe_stem(ptx_path)

    # IMPORTANT: allow cached outputs even if the PTX has been deleted
    if not force_rerun and has_successful_sprint3_outputs(run_dir, require_artifacts=require_success_artifacts):
        logger.info(
            "Using cached Sprint 3 outputs for site=%s variant=%s ptx=%s",
            site_id,
            variant,
            ptx_path.name,
        )
        return _result_from_existing_run(
            site_id=site_id,
            ptx_path=ptx_path,
            run_dir=run_dir,
            variant=variant,
        )

    if not ptx_path.exists():
        raise FileNotFoundError(
            f"PTX file does not exist and no cached outputs were found: {ptx_path}"
        )

    resolved_rscript = _validate_rscript_executable(rscript_executable)
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    cmd = [resolved_rscript, str(script_path), str(ptx_path), str(run_dir)]

    logger.info("Running Sprint 3 for site=%s variant=%s", site_id, variant)
    logger.info("PTX: %s", ptx_path)
    logger.info("Script: %s", script_path)
    logger.info("Output dir: %s", run_dir)
    logger.info("Command: %s", " ".join(cmd))

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")

    outputs = _collect_outputs(run_dir)
    ptx_name = ptx_path.name
    ptx_date_token = extract_first_date_token(ptx_name)
    ptx_date = yyyymmdd_to_timestamp(ptx_date_token)

    result = Sprint3RunResult(
        site_id=site_id,
        input_ptx=ptx_path,
        output_dir=run_dir,
        variant=variant,
        shrub_csv=outputs["shrub_csv"],
        metrics_csv=outputs["metrics_csv"],
        tree_inventory_csv=outputs["tree_inventory_csv"],
        fuels_raster=outputs["fuels_raster"],
        dtm_raster=outputs["dtm_raster"],
        chm_raster=outputs["chm_raster"],
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        returncode=completed.returncode,
        used_cache=False,
        ptx_name=ptx_name,
        ptx_date_token=ptx_date_token,
        ptx_date=ptx_date,
        ptx_quality_score=compute_ptx_quality_score(
            site_id=site_id,
            ptx_name=ptx_name,
            ptx_date=ptx_date,
        ),
    )

    logger.info(
        "Sprint 3 finished with returncode=%s | shrub_csv=%s | metrics_csv=%s | tree_inventory_csv=%s | cache=%s",
        result.returncode,
        result.shrub_csv,
        result.metrics_csv,
        result.tree_inventory_csv,
        result.used_cache,
    )

    if completed.returncode != 0:
        stderr_tail = tail_text_file(stderr_log, n_lines=60)
        logger.error("Sprint 3 failed for site=%s variant=%s", site_id, variant)
        logger.error("stderr tail:\n%s", stderr_tail)

        if raise_on_error:
            raise RuntimeError(
                f"Sprint 3 failed for site={site_id}, variant={variant}, returncode={completed.returncode}.\n"
                f"stderr tail:\n{stderr_tail}"
            )

    return result


def sprint3_results_to_frame(results: list[Sprint3RunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "site_id": r.site_id,
                "input_ptx": str(r.input_ptx),
                "ptx_name": r.ptx_name,
                "ptx_date_token": r.ptx_date_token,
                "ptx_date": r.ptx_date,
                "ptx_quality_score": r.ptx_quality_score,
                "output_dir": str(r.output_dir),
                "variant": r.variant,
                "shrub_csv": str(r.shrub_csv) if r.shrub_csv else None,
                "metrics_csv": str(r.metrics_csv) if r.metrics_csv else None,
                "tree_inventory_csv": str(r.tree_inventory_csv) if r.tree_inventory_csv else None,
                "fuels_raster": str(r.fuels_raster) if r.fuels_raster else None,
                "dtm_raster": str(r.dtm_raster) if r.dtm_raster else None,
                "chm_raster": str(r.chm_raster) if r.chm_raster else None,
                "stdout_log": str(r.stdout_log),
                "stderr_log": str(r.stderr_log),
                "returncode": r.returncode,
                "used_cache": r.used_cache,
            }
        )
    return pd.DataFrame(rows)


def append_results_manifest(
    manifest_csv: str | Path,
    new_results: list[Sprint3RunResult],
) -> pd.DataFrame:
    manifest_csv = Path(manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    new_df = sprint3_results_to_frame(new_results)

    if manifest_csv.exists():
        old_df = pd.read_csv(manifest_csv)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["site_id", "input_ptx", "variant"],
            keep="last",
        )
    else:
        combined = new_df.copy()

    combined.to_csv(manifest_csv, index=False)
    return combined