from dataclasses import dataclass
from typing import Callable
from Final.features.models import FeatureFamilySpec


@dataclass(frozen=True)
class FamilyAdapter:
    family_spec: FeatureFamilySpec
    source_name: str
    compute_fn_factory: Callable[..., Callable]
    cfg_payload_builder: Callable[..., dict]