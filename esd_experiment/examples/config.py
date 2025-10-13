"""
Example configuration file for ESD experiments.

You can import this in your scripts to maintain consistent settings across runs.
"""

# GPU Configuration
GPU_CONFIG = {
    "gpus": [0, 1, 2, 3],  # Available GPU indices
    "num_gpus_per_job": 1,  # GPUs per model
    "gpu_memory_threshold": 500,  # MB threshold for "free" GPU
    "max_check": 10,  # Number of checks before allocating GPU
}

# ESD Analysis Parameters
ESD_CONFIG = {
    "fix_fingers": "xmin_mid",  # Options: xmin_mid, xmin_peak, DKS
    "evals_thresh": 1e-5,  # Eigenvalue filtering threshold
    "bins": 100,  # Histogram bins
    "filter_zeros": True,  # Filter near-zero eigenvalues
    "parallel_esd": False,  # Multi-GPU ESD computation (experimental)
}

# Model Loading Parameters
MODEL_CONFIG = {
    "device_map": "cpu",  # Device map for initial loading
    "torch_dtype": "float16",  # Model precision
    "max_retries": 2,  # Retry attempts on failure
}

# Experiment Control
EXPERIMENT_CONFIG = {
    "overwrite": False,  # Recompute existing results
    "skip_failed": True,  # Skip previously failed models
    "limit": None,  # Process only first N models (None = all)
}

# Paths
PATHS = {
    "cache_dir": "/path/to/cache",  # HuggingFace cache
    "output_dir": "./results",  # Results directory
    "log_dir": None,  # Log directory (None = output_dir/logs)
}

# Model Lists
MODEL_LISTS = {
    # Example: Different model collections
    "llama_family": [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
    ],
    
    "small_models": [
        "microsoft/phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "stabilityai/stablelm-2-zephyr-1_6b",
    ],
    
    "adapters": [
        # Format: (adapter_repo, base_model_relation, source_model)
        ("some/lora-adapter", "adapter", "meta-llama/Llama-2-7b-hf"),
    ],
}

# Analysis Configurations
ANALYSIS_CONFIGS = {
    "standard": {
        "fix_fingers": "xmin_mid",
        "filter_zeros": True,
    },
    
    "conservative": {
        "fix_fingers": "xmin_peak",
        "evals_thresh": 1e-6,
        "bins": 200,
    },
    
    "dks": {
        "fix_fingers": "DKS",
        "filter_zeros": False,
    },
}


def build_command(model_list: str, config_name: str = "standard", **overrides):
    """
    Helper function to build experiment command with configuration.
    
    Args:
        model_list: Path to model list CSV
        config_name: Name of analysis configuration to use
        **overrides: Override any configuration parameters
    
    Returns:
        Command string
    """
    # Start with base config
    config = {**GPU_CONFIG, **ESD_CONFIG, **EXPERIMENT_CONFIG}
    
    # Apply analysis config
    if config_name in ANALYSIS_CONFIGS:
        config.update(ANALYSIS_CONFIGS[config_name])
    
    # Apply overrides
    config.update(overrides)
    
    # Build command
    cmd_parts = [
        "python run_experiment.py",
        f"--model_list {model_list}",
        f"--output_dir {PATHS['output_dir']}",
        f"--gpus {' '.join(map(str, config['gpus']))}",
        f"--num_gpus_per_job {config['num_gpus_per_job']}",
        f"--gpu_memory_threshold {config['gpu_memory_threshold']}",
        f"--max_check {config['max_check']}",
        f"--fix_fingers {config['fix_fingers']}",
        f"--evals_thresh {config['evals_thresh']}",
        f"--bins {config['bins']}",
    ]
    
    if config.get("filter_zeros"):
        cmd_parts.append("--filter_zeros")
    
    if config.get("parallel_esd"):
        cmd_parts.append("--parallel_esd")
    
    if config.get("overwrite"):
        cmd_parts.append("--overwrite")
    
    if config.get("skip_failed"):
        cmd_parts.append("--skip_failed")
    
    if config.get("limit"):
        cmd_parts.append(f"--limit {config['limit']}")
    
    return " \\\n    ".join(cmd_parts)


if __name__ == "__main__":
    # Example usage
    print("Standard configuration:")
    print(build_command("models.csv", "standard"))
    print()
    
    print("Conservative configuration:")
    print(build_command("models.csv", "conservative"))
    print()
    
    print("Custom configuration:")
    print(build_command("models.csv", "standard", gpus=[0, 1], limit=10))
