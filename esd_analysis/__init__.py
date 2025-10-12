"""
ESD Analysis Pipeline for Large Language Models

A high-performance batch processing pipeline for computing Empirical Spectral Density (ESD)
metrics on HuggingFace models, with full support for LoRA/PEFT adapters and multi-GPU acceleration.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .batch_analyzer import (
    BatchESDAnalyzer,
    ModelInfo,
    ESDResult,
    load_models_from_csv
)

from .model_utils import (
    load_model_smart,
    load_and_merge_adapter,
    get_base_model_from_adapter,
    detect_model_type,
    safe_model_cleanup,
    get_model_size_gb
)

# Make key classes available at package level
__all__ = [
    "BatchESDAnalyzer",
    "ModelInfo",
    "ESDResult",
    "load_models_from_csv",
    "load_model_smart",
    "load_and_merge_adapter",
    "get_base_model_from_adapter",
    "detect_model_type",
    "safe_model_cleanup",
    "get_model_size_gb"
]
