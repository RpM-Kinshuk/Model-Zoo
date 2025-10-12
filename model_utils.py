"""
Model loading utilities with LoRA/PEFT adapter merging support.
"""
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
from huggingface_hub import HfApi, hf_hub_download
import config

logger = logging.getLogger(__name__)


def load_model_smart(
    model_name: str,
    model_type: Optional[str] = None,
    base_model: Optional[str] = None,
    dtype: str = config.DEFAULT_DTYPE,
    device_map: str = config.DEFAULT_DEVICE_MAP,
    **kwargs
) -> nn.Module:
    """
    Smart model loader that handles base models and adapters.

    Args:
        model_name: HuggingFace model ID or local path
        model_type: 'base', 'adapter', or None (auto-detect)
        base_model: Base model for adapters (if known)
        dtype: Model dtype (float16, float32, bfloat16)
        device_map: Device mapping strategy
        **kwargs: Additional arguments for from_pretrained

    Returns:
        Loaded model with adapters merged if applicable
    """
    # Auto-detect model type if not specified
    if model_type is None:
        model_type = detect_model_type(model_name)

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "auto": "auto"
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Set default kwargs
    kwargs.setdefault("trust_remote_code", config.TRUST_REMOTE_CODE)
    kwargs.setdefault("low_cpu_mem_usage", config.LOW_CPU_MEM_USAGE)
    kwargs.setdefault("torch_dtype", torch_dtype)
    kwargs.setdefault("device_map", device_map)

    # Add token if available
    token = config.get_hf_token()
    if token:
        kwargs.setdefault("token", token)

    if model_type == "adapter":
        logger.info(f"Loading adapter model: {model_name}")
        return load_and_merge_adapter(model_name, base_model, **kwargs)
    else:
        logger.info(f"Loading base model: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model.eval()
            return model
        except TypeError as e:
            # Fallback for older transformers versions
            if "token" in kwargs and "use_auth_token" not in kwargs:
                kwargs["use_auth_token"] = kwargs.pop("token")
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                model.eval()
                return model
            raise e


def load_and_merge_adapter(
    adapter_name: str,
    base_model_name: Optional[str] = None,
    merge: bool = True,
    **kwargs
) -> nn.Module:
    """
    Load a PEFT/LoRA adapter and optionally merge it with the base model.

    Args:
        adapter_name: HuggingFace adapter model ID
        base_model_name: Base model ID (auto-detected if None)
        merge: Whether to merge adapter weights into base model
        **kwargs: Additional arguments for from_pretrained

    Returns:
        Model with adapter (merged or separate)
    """
    # Auto-detect base model if not provided
    if base_model_name is None:
        base_model_name = get_base_model_from_adapter(adapter_name)
        if base_model_name is None:
            raise ValueError(f"Could not detect base model for adapter: {adapter_name}")

    logger.info(f"Loading base model: {base_model_name}")

    # Load base model
    base_kwargs = kwargs.copy()
    base_kwargs.pop("is_trainable", None)  # Remove PEFT-specific args
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **base_kwargs
    )
    base_model.eval()

    logger.info(f"Loading adapter: {adapter_name}")

    # Load adapter with PEFT
    token = config.get_hf_token()
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_name,
        is_trainable=False,
        token=token
    )

    if merge:
        logger.info("Merging adapter weights into base model")
        # Merge LoRA weights into base model
        merged_model = peft_model.merge_and_unload()
        merged_model.eval()

        # Clean up to save memory
        del base_model
        del peft_model
        torch.cuda.empty_cache()

        return merged_model
    else:
        return peft_model


def get_base_model_from_adapter(adapter_name: str) -> Optional[str]:
    """
    Extract base model name from adapter configuration.

    Args:
        adapter_name: HuggingFace adapter model ID

    Returns:
        Base model name or None if not found
    """
    token = config.get_hf_token()

    # Try PEFT config first
    try:
        peft_config = PeftConfig.from_pretrained(adapter_name, token=token)
        if hasattr(peft_config, "base_model_name_or_path"):
            return peft_config.base_model_name_or_path
    except Exception as e:
        logger.debug(f"Could not load PEFT config: {e}")

    # Try AutoConfig as fallback
    try:
        auto_config = AutoConfig.from_pretrained(
            adapter_name,
            token=token,
            trust_remote_code=True
        )

        # Check various possible fields
        for field in ["base_model_name_or_path", "base_model",
                      "model_name", "parent_model_name_or_path"]:
            if hasattr(auto_config, field):
                value = getattr(auto_config, field)
                if value:
                    return value
    except Exception as e:
        logger.debug(f"Could not load AutoConfig: {e}")

    # Try to parse from model card or README
    try:
        api = HfApi()
        model_info = api.model_info(adapter_name, token=token)

        # Check model card for base model mentions
        if model_info.card_data:
            card_dict = model_info.card_data.to_dict()
            if "base_model" in card_dict:
                return card_dict["base_model"]
    except Exception as e:
        logger.debug(f"Could not parse model card: {e}")

    return None


def detect_model_type(model_name: str) -> str:
    """
    Detect if a model is a base model or adapter.

    Args:
        model_name: HuggingFace model ID

    Returns:
        'base' or 'adapter'
    """
    api = HfApi()
    token = config.get_hf_token()

    try:
        # List files in the repository
        files = api.list_repo_files(
            repo_id=model_name,
            repo_type="model",
            token=token
        )

        # Check for adapter files
        has_adapter = any(
            any(pattern.replace("*", "") in f.lower()
                for pattern in config.ADAPTER_FILE_PATTERNS)
            for f in files
        )

        # Check for full model files
        has_model = any(
            any(pattern.replace("*", "") in f.lower()
                for pattern in config.MODEL_FILE_PATTERNS)
            for f in files
        )

        # If has adapter files but no model files, it's likely an adapter
        if has_adapter and not has_model:
            return "adapter"

        # Check for PEFT config
        if "adapter_config.json" in files:
            return "adapter"

    except Exception as e:
        logger.warning(f"Could not detect model type for {model_name}: {e}")

    # Default to base model
    return "base"


def safe_model_cleanup(model: Optional[nn.Module]) -> None:
    """
    Safely clean up model and free memory.

    Args:
        model: PyTorch model to clean up
    """
    if model is not None:
        try:
            # Move to CPU first to free GPU memory
            if hasattr(model, 'cpu'):
                model.cpu()

            # Delete model
            del model
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_size_gb(model: nn.Module) -> float:
    """
    Calculate model size in GB.

    Args:
        model: PyTorch model

    Returns:
        Model size in GB
    """
    total_params = sum(p.numel() for p in model.parameters())
    # Assume float16 (2 bytes per param) by default
    bytes_per_param = 2

    # Check actual dtype of first parameter
    for p in model.parameters():
        if p.dtype == torch.float32:
            bytes_per_param = 4
        elif p.dtype == torch.bfloat16:
            bytes_per_param = 2
        break

    return (total_params * bytes_per_param) / (1024**3)
