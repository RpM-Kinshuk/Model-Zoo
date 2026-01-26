"""
Robust model loader with PEFT adapter support.
Heavily inspired by calculate_adapters.py and run_metric.py patterns.
"""
import os
import re
import torch
import warnings
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
from huggingface_hub import HfApi, get_token


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or folder."""
    return (
        os.environ.get("HF_TOKEN") or 
        os.environ.get("HUGGINGFACE_HUB_TOKEN") or 
        os.environ.get("HUGGINGFACE_TOKEN") or
        get_token()
    )


def hf_from_pretrained(AutoModelCls, repo_id: str, **kwargs):
    """
    Load HuggingFace model with token support.
    Handles both token= and use_auth_token= for compatibility.
    """
    token = get_hf_token()
    kwargs.setdefault("trust_remote_code", True)
    kwargs.setdefault("low_cpu_mem_usage", True)
    
    if token:
        try:
            return AutoModelCls.from_pretrained(repo_id, token=token, **kwargs)
        except TypeError:
            # Fallback for older transformers versions
            return AutoModelCls.from_pretrained(repo_id, use_auth_token=token, **kwargs)
    else:
        return AutoModelCls.from_pretrained(repo_id, **kwargs)


def hf_repo_has_prefix(repo_id: str, prefix: str) -> bool:
    """Check if HuggingFace repo has files with given prefix."""
    api = HfApi()
    try:
        token = get_hf_token()
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        bnames = [Path(p).name.lower() for p in files]
        return any(name.startswith(prefix) and name.endswith(".safetensors") for name in bnames)
    except Exception as e:
        warnings.warn(f"Could not list repo files for {repo_id}: {e}")
        return False


def is_adapter_model(repo_id: str, base_model_relation: Optional[str] = None) -> bool:
    """
    Determine if a model is a PEFT adapter.
    
    Args:
        repo_id: HuggingFace repository ID
        base_model_relation: Optional relation string (e.g., "adapter", "lora", "peft")
    
    Returns:
        True if this is an adapter model
    """
    # Check explicit relation first
    if base_model_relation:
        relation = str(base_model_relation).strip().lower()
        if relation in {"adapter", "lora", "peft"}:
            return True
    
    # Check for adapter config
    try:
        token = get_hf_token()
        PeftConfig.from_pretrained(repo_id, token=token)
        return True
    except Exception:
        pass
    
    # Check file structure: has adapter*.safetensors but no model*.safetensors
    if hf_repo_has_prefix(repo_id, "adapter") and not hf_repo_has_prefix(repo_id, "model"):
        return True
    
    return False


def resolve_base_model(adapter_repo: str, source_model: Optional[str] = None) -> str:
    """
    Resolve base model for an adapter.
    
    Args:
        adapter_repo: The adapter repository ID
        source_model: Optional explicitly specified base model
    
    Returns:
        Base model repository ID
    """
    # Use explicit source_model if provided
    if source_model and isinstance(source_model, str) and source_model.strip():
        return source_model.strip()
    
    # Try to infer from PeftConfig
    token = get_hf_token()
    try:
        cfg = PeftConfig.from_pretrained(adapter_repo, token=token)
        if hasattr(cfg, "base_model_name_or_path") and cfg.base_model_name_or_path:
            return cfg.base_model_name_or_path
    except Exception as e:
        warnings.warn(f"Could not load PeftConfig for {adapter_repo}: {e}")
    
    # Try to infer from AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(adapter_repo, token=token, trust_remote_code=True)
        for key in ["base_model_name_or_path", "base_model", "model_name", "parent_model_name_or_path"]:
            if hasattr(cfg, key) and getattr(cfg, key):
                return getattr(cfg, key)
    except Exception as e:
        warnings.warn(f"Could not load AutoConfig for {adapter_repo}: {e}")
    
    raise RuntimeError(
        f"Could not resolve base model for adapter {adapter_repo}. "
        f"Please provide source_model explicitly in the model list."
    )


def load_and_merge_adapter(
    adapter_repo: str,
    base_repo: Optional[str] = None,
    device_map: str = "cpu",
    torch_dtype = torch.float16
) -> torch.nn.Module:
    """
    Load base model and merge PEFT adapter weights.
    
    Args:
        adapter_repo: Adapter repository ID
        base_repo: Base model repository ID (will be inferred if None)
        device_map: Device map for loading
        torch_dtype: Data type for model
    
    Returns:
        Merged model with adapter weights incorporated
    """
    # Resolve base model
    if base_repo is None:
        base_repo = resolve_base_model(adapter_repo)
    
    print(f"Loading base model: {base_repo}")
    base = hf_from_pretrained(
        AutoModelForCausalLM,
        base_repo,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    base.eval()
    
    print(f"Loading adapter: {adapter_repo}")
    token = get_hf_token()
    peft_model = PeftModel.from_pretrained(
        base,
        adapter_repo,
        is_trainable=False,
        token=token
    )
    
    print("Merging adapter weights into base model...")
    merged = peft_model.merge_and_unload() # type: ignore
    merged.eval()
    
    return merged


def load_model(
    repo_id: str,
    base_model_relation: Optional[str] = None,
    source_model: Optional[str] = None,
    device_map: str = "cpu",
    torch_dtype = torch.float16,
    revision: Optional[str] = None,
) -> Tuple[torch.nn.Module, bool]:
    """
    Load a model, handling both regular models and PEFT adapters.
    
    Args:
        repo_id: HuggingFace repository ID
        base_model_relation: Optional relation indicator ("adapter", "lora", etc.)
        source_model: Optional base model for adapters
        device_map: Device map for model loading
        torch_dtype: Data type for model
        revision: Optional git revision/commit to load
    
    Returns:
        Tuple of (model, is_adapter)
    """
    # Check if this is an adapter
    if is_adapter_model(repo_id, base_model_relation):
        print(f"[ADAPTER] Loading adapter model: {repo_id}")
        model = load_and_merge_adapter(
            adapter_repo=repo_id,
            base_repo=source_model,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        return model, True
    else:
        print(f"[STANDARD] Loading standard model: {repo_id}")
        model = hf_from_pretrained(
            AutoModelForCausalLM,
            repo_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            revision=revision,
        )
        model.eval()
        return model, False


def parse_model_string(model_str: str) -> Tuple[str, Optional[str]]:
    """
    Parse model string that may contain revision info.
    
    Format: "org/model@revision" or just "org/model"
    
    Returns:
        Tuple of (repo_id, revision)
    """
    model_str = model_str.strip()
    if "@" in model_str and not model_str.startswith("@"):
        repo_id, revision = model_str.split("@", 1)
        return repo_id.strip(), revision.strip() or None
    return model_str, None


def safe_filename(model_id: str) -> str:
    """Convert model ID to safe filename."""
    # Replace / with --
    safe = model_id.replace("/", "--")
    # Replace @ with __
    safe = safe.replace("@", "__")
    # Remove other problematic characters
    safe = re.sub(r"[^\w\-\.]", "_", safe)
    return safe
