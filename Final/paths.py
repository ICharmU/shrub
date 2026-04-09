from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    base = Path(start) if start is not None else Path.cwd()
    for candidate in [base, *base.parents]:
        if (candidate / 'Final').exists() and (candidate / 'Sprint 3').exists():
            return candidate
    return base


def final_root(start: str | Path | None = None) -> Path:
    return repo_root(start) / 'Final'
