from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """
    Find the project repository root by walking upward until both `Final`
    and `Sprint 3` exist.

    This is meant to work whether code is run from:
    - Final/
    - Final/labeling/
    - a notebook directory
    - repo root itself
    """
    base = Path(start).resolve() if start is not None else Path.cwd().resolve()

    for candidate in [base, *base.parents]:
        if (candidate / "Final").exists() and (candidate / "Sprint 3").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate repository root containing both 'Final' and 'Sprint 3'. "
        f"Start path was: {base}"
    )


PROJECT_ROOT: Path = repo_root()
FINAL_ROOT: Path = PROJECT_ROOT / "Final"

SPRINT1_ROOT: Path = PROJECT_ROOT / "Sprint 1"
SPRINT2_ROOT: Path = PROJECT_ROOT / "Sprint 2"
SPRINT3_ROOT: Path = PROJECT_ROOT / "Sprint 3"
SPRINT4_ROOT: Path = PROJECT_ROOT / "Sprint 4"

SPRINT3_BASE_ROOT: Path = SPRINT3_ROOT / "Base"
SPRINT3_EXTRA_ROOT: Path = SPRINT3_ROOT / "Extra"

LABELING_ROOT: Path = FINAL_ROOT / "labeling"
FEATURES_ROOT: Path = FINAL_ROOT / "features"
MODELING_ROOT: Path = FINAL_ROOT / "modeling"
POSTPROCESSING_ROOT: Path = FINAL_ROOT / "postprocessing"


def final_root(start: str | Path | None = None) -> Path:
    return repo_root(start) / "Final"


def sprint3_base_root(start: str | Path | None = None) -> Path:
    return repo_root(start) / "Sprint 3" / "Base"


def artifacts_root(start: str | Path | None = None) -> Path:
    return final_root(start) / "artifacts"