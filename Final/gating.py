from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from Final.models import ModuleCard, ModuleStatus


# -----------------------------------------------------------------------------
# Generic QA representation
# -----------------------------------------------------------------------------


@dataclass
class QACheckSpec:
    name: str
    description: str
    level: str  # e.g. "module", "stage", "pipeline"
    required: bool = True
    threshold: float | None = None
    direction: str = "higher_is_better"  # or "lower_is_better", "boolean"


@dataclass
class QACheckResult:
    name: str
    passed: bool
    score: float | None = None
    value: Any = None
    message: str = ""


@dataclass
class ModuleQAProfile:
    module_name: str
    checks: list[QACheckSpec] = field(default_factory=list)


@dataclass
class ModuleQAEvaluation:
    module_name: str
    results: list[QACheckResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(int(r.passed) for r in self.results) / len(self.results)

    @property
    def mean_score(self) -> float | None:
        scores = [r.score for r in self.results if r.score is not None]
        if not scores:
            return None
        return float(sum(scores) / len(scores))


# -----------------------------------------------------------------------------
# Shared gatekeeping logic
# -----------------------------------------------------------------------------


def evaluate_coverage_gate(module: ModuleCard) -> tuple[str, float]:
    mapping = {
        "universal": ("pass", 1.0),
        "most_sites": ("conditional_pass", 0.75),
        "partial_site_conditional": ("warning", 0.4),
    }
    return mapping[module.availability_tier.value]


def evaluate_resource_gate(module: ModuleCard) -> tuple[str, float]:
    mapping = {
        "cheap": ("pass", 1.0),
        "moderate": ("conditional_pass", 0.7),
        "expensive": ("warning", 0.35),
    }
    return mapping[module.runtime_tier.value]


def evaluate_alignment_gate(module: ModuleCard) -> tuple[str, float]:
    if module.representation_target.value in {"raster", "both"}:
        return "pass", 0.8
    return "conditional_pass", 0.7


def evaluate_information_gate(module: ModuleCard) -> tuple[str, float]:
    if "structure" in module.name or "transfer" in module.name:
        return "conditional_pass", 0.75
    return "pass", 0.8


def evaluate_robustness_gate(module: ModuleCard) -> tuple[str, float]:
    if module.availability_tier.value == "partial_site_conditional":
        return "warning", 0.45
    return "conditional_pass", 0.7


def evaluate_module_card(module: ModuleCard) -> ModuleCard:
    module.gate_coverage, module.coverage_score = evaluate_coverage_gate(module)
    module.gate_alignment, module.alignment_score = evaluate_alignment_gate(module)
    module.gate_information, module.information_score = evaluate_information_gate(module)
    module.gate_resource, module.resource_score = evaluate_resource_gate(module)
    module.gate_robustness, module.robustness_score = evaluate_robustness_gate(module)
    return module


def decide_module_status(module: ModuleCard) -> ModuleStatus:
    scores = [
        module.coverage_score or 0.0,
        module.alignment_score or 0.0,
        module.information_score or 0.0,
        module.resource_score or 0.0,
        module.robustness_score or 0.0,
    ]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.85:
        return ModuleStatus.PROMOTED
    if avg_score >= 0.65:
        return ModuleStatus.EXPERIMENTAL
    if avg_score >= 0.45:
        return ModuleStatus.CANDIDATE
    return ModuleStatus.REJECTED


def module_cards_to_frame(modules: dict[str, ModuleCard]) -> pd.DataFrame:
    rows = []
    for module in modules.values():
        rows.append(
            {
                "name": module.name,
                "domain": module.domain.value,
                "representation_target": module.representation_target.value,
                "availability_tier": module.availability_tier.value,
                "runtime_tier": module.runtime_tier.value,
                "gate_coverage": module.gate_coverage,
                "gate_alignment": module.gate_alignment,
                "gate_information": module.gate_information,
                "gate_resource": module.gate_resource,
                "gate_robustness": module.gate_robustness,
                "coverage_score": module.coverage_score,
                "alignment_score": module.alignment_score,
                "information_score": module.information_score,
                "resource_score": module.resource_score,
                "robustness_score": module.robustness_score,
                "status": module.status.value,
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "name"]).reset_index(drop=True)