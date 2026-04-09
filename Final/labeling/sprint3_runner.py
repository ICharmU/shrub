from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


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


def _variant_script_name(variant: Sprint3Variant) -> str:
    if variant == "original":
        return "IntELiMon_1_1_1.R"
    if variant == "revised":
        return "IntELiMon_1_1_1_revised.R"
    raise ValueError(f"Unknown Sprint 3 variant: {variant}")


def _find_single(pattern_dir: Path, pattern: str) -> Path | None:
    matches = sorted(pattern_dir.glob(pattern))
    return matches[0] if matches else None


def run_sprint3_for_ptx(
    *,
    site_id: str,
    ptx_path: str | Path,
    sprint3_base_dir: str | Path,
    output_root: str | Path,
    variant: Sprint3Variant = "revised",
    rscript_executable: str = "Rscript",
) -> Sprint3RunResult:
    ptx_path = Path(ptx_path).resolve()
    sprint3_base_dir = Path(sprint3_base_dir).resolve()
    output_root = Path(output_root).resolve()

    script_path = sprint3_base_dir / _variant_script_name(variant)
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find Sprint 3 script: {script_path}")
    if not ptx_path.exists():
        raise FileNotFoundError(f"Could not find PTX file: {ptx_path}")

    run_dir = output_root / site_id / "sprint3" / variant / ptx_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    cmd = [rscript_executable, str(script_path), str(ptx_path), str(run_dir)]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")

    shrubs_dir = run_dir / "Shrubs"
    metrics_dir = run_dir / "metrics"
    inv_dir = run_dir / "Inventory"
    fuels_dir = run_dir / "Fuels"
    dtm_dir = run_dir / "dtm"
    chm_dir = run_dir / "chm"

    shrub_csv = _find_single(shrubs_dir, "*.csv")
    metrics_csv = _find_single(metrics_dir, "*.csv")
    tree_inventory_csv = _find_single(inv_dir, "*_inv.csv")
    fuels_raster = _find_single(fuels_dir, "*.tif")
    dtm_raster = _find_single(dtm_dir, "*.tif")
    chm_raster = _find_single(chm_dir, "*.tif")

    return Sprint3RunResult(
        site_id=site_id,
        input_ptx=ptx_path,
        output_dir=run_dir,
        variant=variant,
        shrub_csv=shrub_csv,
        metrics_csv=metrics_csv,
        tree_inventory_csv=tree_inventory_csv,
        fuels_raster=fuels_raster,
        dtm_raster=dtm_raster,
        chm_raster=chm_raster,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        returncode=completed.returncode,
    )


def sprint3_results_to_frame(results: list[Sprint3RunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "site_id": r.site_id,
                "input_ptx": str(r.input_ptx),
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
            }
        )
    return pd.DataFrame(rows)