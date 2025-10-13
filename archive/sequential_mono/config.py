"""
Configuration settings for batch ESD analysis pipeline.
"""
import os
from pathlib import Path
from typing import Optional

# GPU Configuration
DEFAULT_GPU_IDS = None  # Auto-detect available GPUs
MAX_WORKERS = None  # Auto-set based on available GPUs
RAM_THRESHOLD_GB = 80  # Wait if RAM below this

# Model Loading Configuration
DEFAULT_DTYPE = "float16"
DEFAULT_DEVICE_MAP = "balanced"  # Load models on CPU first
LOW_CPU_MEM_USAGE = False
TRUST_REMOTE_CODE = True
MAX_MEMORY = {"cpu": "64GiB", "cuda:0": "40GiB", "cuda:1": "40GiB", "cuda:2": "40GiB", "cuda:3": "40GiB"}
OFFLOAD_FOLDER = "./.offload"

# ESD Analysis Configuration
EVALS_THRESH = 1e-5
BINS = 100
DEFAULT_FIX_FINGERS = "DKS"
XMIN_POS = 2
CONV_NORM = 0.5
FILTER_ZEROS = True
PARALLEL_COMPUTE = True  # Use parallel GPU computation

# Output Configuration
OUTPUT_FORMAT = "csv"  # csv, h5, or both
SAVE_EIGENVALUES = False  # Whether to save raw eigenvalues
CHECKPOINT_FREQUENCY = 10  # Save checkpoint every N models

# Retry Configuration
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds

# HuggingFace Configuration
def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variables."""
    return (os.environ.get("HF_TOKEN") or
            os.environ.get("HUGGINGFACE_HUB_TOKEN") or
            os.environ.get("HUGGINGFACE_TOKEN"))

# Cache Configuration
HF_CACHE_DIR = os.environ.get("HF_HOME", '/scratch/kinshuk/.cache/huggingface')
TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", HF_CACHE_DIR)

# Adapter Detection Patterns
ADAPTER_RELATIONS = {"adapter", "lora", "peft", "qlora"}
ADAPTER_FILE_PATTERNS = ["adapter_*.safetensors", "adapter_*.bin"]
MODEL_FILE_PATTERNS = ["model*.safetensors", "pytorch_model*.bin"]

# Layer Name Patterns for Parsing
LAYER_PREFIXES = {"layers", "layer", "h", "block", "blocks", "decoder.layers", "encoder.layers"}
