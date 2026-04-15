from __future__ import annotations

from pathlib import Path

from Final.artifact_store import ArtifactStore
from Final.resource_monitor import RuntimeMonitor, file_size_bytes


def push_outputs_from_qa_outputs(
    store: ArtifactStore,
    qa_outputs: dict,
    *,
    repo_root: Path,
) -> dict[str, str]:
    pushed = {}
    for key, value in (qa_outputs or {}).items():
        if not isinstance(value, str):
            continue
        p = Path(value)
        if not p.exists() or not p.is_file():
            continue
        try:
            rel = str(p.relative_to(repo_root))
        except Exception:
            continue
        store.push(p, rel_path=rel)
        pushed[key] = rel
    return pushed


def execute_pipeline_section(
    pipeline,
    *,
    state,
    artifact_store: ArtifactStore | None = None,
    push_remote: bool = False,
):
    monitor = RuntimeMonitor()
    monitor.sample_memory()

    result = pipeline.run()
    monitor.sample_memory()

    if artifact_store is not None and push_remote:
        repo_root = pipeline.cfg.data.project_root
        pushed = push_outputs_from_qa_outputs(
            artifact_store,
            result.qa_outputs,
            repo_root=repo_root,
        )
        if pushed:
            result.qa_outputs["remote_artifacts"] = pushed

    stats = monitor.finalize()
    result.metrics["runtime_wall_seconds"] = stats.wall_seconds
    result.metrics["runtime_max_rss_mb"] = stats.max_rss_mb
    result.metrics["runtime_cache_hits"] = stats.cache_hits
    result.metrics["runtime_cache_misses"] = stats.cache_misses

    state = pipeline.apply_to_experiment_state(state, result)
    return result, state, stats