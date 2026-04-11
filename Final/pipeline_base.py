from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
import json

from Final.shared_utils import get_logger
from Final.models import PipelineRunResult


class BasePipeline(ABC):
    """
    Generic pipeline abstraction for section pipelines:
    labeling, features, modeling, postprocessing.

    Concrete subclasses should implement:
    - build_module_registry()
    - run()
    """

    def __init__(self, cfg, *, pipeline_name: str, output_root: str | Path):
        self.cfg = cfg
        self.pipeline_name = pipeline_name
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(f"pipeline.{pipeline_name}")

    #@abstractmethod
    def build_module_registry(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs) -> PipelineRunResult:
        raise NotImplementedError

    def save_run_result(self, result: PipelineRunResult, filename: str | None = None) -> Path:
        filename = filename or f"{self.pipeline_name}_run_result.json"
        path = self.output_root / filename
        path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        self.logger.info("Saved pipeline run result to %s", path)
        return path