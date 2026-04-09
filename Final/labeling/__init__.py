"""Modular label-engineering pipeline for shrub mapping.

This package pulls together the key ideas from Sprint 3 (IntELiMon shrub-object
generation from TLS) and Sprint 4 (TLS->ALS->NAIP label transfer and mask
rasterization) into a reusable Python scaffold.
"""

from .config import PipelineConfig, default_config
from .models import ShrubObjectColumns, LabelArtifactColumns

__all__ = [
    "PipelineConfig",
    "default_config",
    "ShrubObjectColumns",
    "LabelArtifactColumns",
]
