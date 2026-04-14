from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import time

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RuntimeStats:
    started_at: str
    finished_at: str | None = None
    wall_seconds: float | None = None
    max_rss_mb: float | None = None
    bytes_written_local: int = 0
    bytes_pruned_local: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    notes: list[str] = None


class RuntimeMonitor:
    def __init__(self):
        self.stats = RuntimeStats(started_at=utc_now_iso(), notes=[])
        self._t0 = time.perf_counter()
        self._max_rss_bytes = 0

    def sample_memory(self) -> None:
        if psutil is None:
            return
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        self._max_rss_bytes = max(self._max_rss_bytes, rss)

    def add_cache_hit(self, n: int = 1) -> None:
        self.stats.cache_hits += int(n)

    def add_cache_miss(self, n: int = 1) -> None:
        self.stats.cache_misses += int(n)

    def add_bytes_written(self, n: int) -> None:
        self.stats.bytes_written_local += int(n)

    def add_bytes_pruned(self, n: int) -> None:
        self.stats.bytes_pruned_local += int(n)

    def add_note(self, text: str) -> None:
        self.stats.notes.append(text)

    def finalize(self) -> RuntimeStats:
        self.sample_memory()
        self.stats.finished_at = utc_now_iso()
        self.stats.wall_seconds = float(time.perf_counter() - self._t0)
        self.stats.max_rss_mb = round(self._max_rss_bytes / (1024 * 1024), 3) if self._max_rss_bytes else None
        return self.stats

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.finalize())
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


def file_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.exists() and path.is_file():
        return path.stat().st_size
    return 0


def dir_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total