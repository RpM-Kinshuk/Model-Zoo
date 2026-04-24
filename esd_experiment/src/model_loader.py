"""
Robust model loader with PEFT adapter support.
Heavily inspired by calculate_adapters.py and run_metric.py patterns.
"""
import importlib
import os
import re
import torch
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
try:
    from transformers import AutoModelForSeq2SeqLM
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForSeq2SeqLM = None
try:
    from transformers import AutoModelForSequenceClassification
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForSequenceClassification = None
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None
from peft import PeftModel, PeftConfig
from huggingface_hub import HfApi, get_token


@dataclass
class LoaderFailure(Exception):
    stage: str
    reason: str
    message: str

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


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

    def _call(load_kwargs):
        if token:
            try:
                return AutoModelCls.from_pretrained(repo_id, token=token, **load_kwargs)
            except TypeError:
                # Fallback for older transformers versions
                return AutoModelCls.from_pretrained(repo_id, use_auth_token=token, **load_kwargs)
        return AutoModelCls.from_pretrained(repo_id, **load_kwargs)

    try:
        return _call(kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if kwargs.get("low_cpu_mem_usage") and any(
            marker in message for marker in ("meta tensor", "meta tensors")
        ):
            retry_kwargs = dict(kwargs)
            retry_kwargs["low_cpu_mem_usage"] = False
            return _call(retry_kwargs)
        raise


def ensure_optimum_gptq_backend_compat() -> None:
    """Patch Optimum's legacy EXLLAMA_V1 reference for current GPTQModel builds."""
    try:
        import optimum.gptq.quantizer as optimum_gptq_quantizer
    except Exception:
        return
    backend_enum = getattr(optimum_gptq_quantizer, "BACKEND", None)
    if backend_enum is None:
        return
    if hasattr(backend_enum, "EXLLAMA_V1"):
        return
    if hasattr(backend_enum, "EXLLAMA_V2"):
        setattr(backend_enum, "EXLLAMA_V1", backend_enum.EXLLAMA_V2)


def ensure_compressed_tensors_backend_compat() -> None:
    """Import compressed_tensors, retrying after GPTQModel's torch setup side effects."""
    try:
        importlib.import_module("compressed_tensors")
        return
    except Exception as direct_exc:
        try:
            importlib.import_module("gptqmodel")
            importlib.import_module("compressed_tensors")
            return
        except Exception:
            raise direct_exc


def classify_loader_scenario_support(loader_scenario: Optional[str]) -> Optional[LoaderFailure]:
    scenario = (loader_scenario or "").strip().lower()
    if not scenario or scenario in {
        "standard_transformers",
        "standard_causal",
        "adapter_requires_base",
        "quantized_transformers_native",
        "compressed_tensors",
        "gptq",
        "awq",
        "multimodal_transformers",
        "multimodal",
        "seq2seq",
        "sequence_classification",
        "gguf",
    }:
        return None
    if scenario in {"quantized_alt_format"}:
        return LoaderFailure(
            "load",
            "unsupported_loader_scenario",
            "Quantized alternate-format repos are not supported by the current loader",
        )
    return LoaderFailure(
        "load",
        "unsupported_loader_scenario",
        f"Unsupported loader scenario: {loader_scenario}",
    )


def classify_quantized_dependency_failure(error: Exception) -> Optional[LoaderFailure]:
    message = str(error)
    lowered = message.lower()
    if (
        "meta tensors" in lowered
        or "meta tensor" in lowered
        or "incompatible torch version" in lowered
        or "duplicate template name" in lowered
    ):
        return LoaderFailure(
            "load",
            "quantized_backend_incompatible",
            f"Quantized-native backend is incompatible with the current runtime: {message}",
        )
    package_hints = {
        "gptqmodel": "gptqmodel",
        "autoawq": "autoawq",
        "bitsandbytes": "bitsandbytes",
        "compressed_tensors": "compressed-tensors",
        "compressed-tensors": "compressed-tensors",
    }
    for marker, package in package_hints.items():
        if marker in lowered:
            return LoaderFailure(
                "load",
                "quantized_dependency_missing",
                f"Quantized-native loading requires the optional dependency `{package}`: {message}",
            )
    return None


def _raise_quantized_loader_failure_if_known(
    error: Exception,
    loader_scenario: Optional[str],
    effective_loader: str,
) -> None:
    if (
        (loader_scenario or "").strip().lower() == "quantized_transformers_native"
        or effective_loader in {"gptq", "awq", "compressed_tensors"}
    ):
        dependency_failure = classify_quantized_dependency_failure(error)
        if dependency_failure is not None:
            raise dependency_failure from error


def _quant_method_from_config(config: Any) -> str:
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        return ""
    if isinstance(quantization_config, dict):
        method = quantization_config.get("quant_method")
    else:
        method = getattr(quantization_config, "quant_method", None)
    return str(method or "").strip().lower()


def resolve_effective_loader_for_repo(
    repo_id: str,
    loader_scenario: Optional[str] = None,
    base_model_relation: Optional[str] = None,
    source_model: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    scenario = (loader_scenario or "").strip().lower()
    relation = (base_model_relation or "").strip().lower()
    repo_hint = " ".join(
        part for part in [repo_id, source_model or ""] if part
    ).lower()

    if relation in {"adapter", "lora", "peft"}:
        return resolve_adapter_effective_loader(source_model or repo_id, loader_scenario=scenario)
    if scenario in {
        "standard_causal",
        "compressed_tensors",
        "gptq",
        "awq",
        "multimodal",
        "seq2seq",
        "sequence_classification",
    }:
        return scenario
    if scenario == "gguf":
        return "gguf"
    if scenario == "seq2seq" or "seq2seq" in repo_hint:
        return "seq2seq"
    if scenario == "sequence_classification" or "text-classification" in repo_hint:
        return "sequence_classification"
    if scenario == "multimodal_transformers":
        return "multimodal"
    if scenario == "quantized_transformers_native":
        if any(marker in repo_hint for marker in ("gptq", "awq")):
            return "gptq" if "gptq" in repo_hint else "awq"

    config = None
    try:
        config = AutoConfig.from_pretrained(
            repo_id,
            token=get_hf_token(),
            revision=revision,
            trust_remote_code=True,
        )
    except Exception:
        config = None

    if config is not None:
        model_type = str(getattr(config, "model_type", "") or "").strip().lower()
        quant_method = _quant_method_from_config(config)
        architectures = [
            str(arch).strip().lower()
            for arch in getattr(config, "architectures", []) or []
            if str(arch).strip()
        ]
        if quant_method in {"gptq", "awq", "compressed-tensors"}:
            return quant_method.replace("-", "_")
        if any("forsequenceclassification" in arch for arch in architectures):
            return "sequence_classification"
        if model_type.startswith("t5") or any(
            marker in arch for arch in architectures for marker in ("conditionalgeneration", "seq2seq")
        ):
            return "seq2seq"
        if any(
            marker in model_type for marker in ("llava", "vision", "multi_modality", "multimodal")
        ) or any(
            marker in arch for arch in architectures for marker in ("image", "vision", "llava")
        ):
            return "multimodal"

    return "standard_causal"


def resolve_adapter_effective_loader(
    base_loader_or_repo: Optional[str],
    loader_scenario: Optional[str] = None,
) -> str:
    candidates = " ".join(
        part for part in [base_loader_or_repo or "", loader_scenario or ""] if part
    ).lower()
    if "gptq" in candidates:
        return "gptq"
    if "awq" in candidates:
        return "awq"
    if "compressed-tensors" in candidates or "compressed_tensors" in candidates:
        return "compressed_tensors"
    if "gguf" in candidates:
        return "gguf"
    if "seq2seq" in candidates:
        return "seq2seq"
    if "sequence_classification" in candidates:
        return "sequence_classification"
    if "multimodal" in candidates:
        return "multimodal"
    return "standard_causal"


def resolve_dense_upstream_base_reference(
    repo_id: str,
    revision: Optional[str] = None,
) -> Optional[Tuple[str, Optional[str]]]:
    api = HfApi()
    try:
        info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            token=get_hf_token(),
        )
    except Exception:
        return None

    tags = getattr(info, "tags", None) or []
    for tag in tags:
        text = str(tag or "").strip()
        if not text.startswith("base_model:") or text.startswith("base_model:quantized:"):
            continue
        return parse_model_string(text[len("base_model:"):])

    card_data = getattr(info, "cardData", None)
    if isinstance(card_data, dict):
        base_model = card_data.get("base_model")
    else:
        base_model = getattr(card_data, "base_model", None)
    if base_model:
        candidates = base_model if isinstance(base_model, (list, tuple)) else [base_model]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text or text.lower().startswith("quantized:"):
                continue
            return parse_model_string(text)

    return None


def _resolve_adapter_task_loader(
    adapter_repo: str,
    source_model: Optional[str] = None,
    loader_scenario: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    token = get_hf_token()
    try:
        cfg = PeftConfig.from_pretrained(adapter_repo, token=token, revision=revision)
        task_type = str(getattr(cfg, "task_type", "") or "").strip().upper()
        if task_type == "SEQ_2_SEQ_LM":
            return "seq2seq"
        if task_type in {"SEQ_CLS", "SEQUENCE_CLASSIFICATION"}:
            return "sequence_classification"
        if source_model:
            source_repo, source_revision = parse_model_string(str(source_model))
            return resolve_effective_loader_for_repo(
                source_repo,
                loader_scenario=loader_scenario,
                revision=source_revision or revision,
            )
        base_name = getattr(cfg, "base_model_name_or_path", None)
        if base_name:
            _, embedded_revision = parse_model_string(str(base_name))
            base_revision = _normalize_optional_revision(getattr(cfg, "revision", None)) or embedded_revision
            return resolve_effective_loader_for_repo(
                str(base_name).split("@", 1)[0],
                loader_scenario=loader_scenario,
                revision=base_revision,
            )
    except Exception:
        pass
    return resolve_adapter_effective_loader(source_model or adapter_repo, loader_scenario=loader_scenario)


def _select_auto_model_cls(effective_loader: str):
    if effective_loader == "seq2seq":
        if AutoModelForSeq2SeqLM is None:
            raise LoaderFailure(
                "load",
                "unsupported_loader_scenario",
                "Seq2seq loading requires transformers with AutoModelForSeq2SeqLM support",
            )
        return AutoModelForSeq2SeqLM
    if effective_loader == "sequence_classification":
        if AutoModelForSequenceClassification is None:
            raise LoaderFailure(
                "load",
                "unsupported_loader_scenario",
                "Sequence-classification loading requires transformers with AutoModelForSequenceClassification support",
            )
        return AutoModelForSequenceClassification
    if effective_loader == "multimodal":
        if AutoModelForImageTextToText is None:
            raise LoaderFailure(
                "load",
                "unsupported_loader_scenario",
                "Multimodal loading requires transformers with AutoModelForImageTextToText support",
            )
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def resolve_gguf_filename(repo_id: str, revision: Optional[str] = None) -> str:
    api = HfApi()
    try:
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            token=get_hf_token(),
        )
    except Exception as exc:
        raise LoaderFailure(
            "load",
            "repo_inaccessible",
            f"Could not inspect GGUF repo files for {repo_id}: {exc}",
        ) from exc
    candidates = sorted(
        path for path in files if str(path).strip().lower().endswith(".gguf")
    )
    if not candidates:
        raise LoaderFailure(
            "load",
            "missing_required_artifact",
            f"GGUF repo {repo_id} does not expose a .gguf file",
        )
    root_level = [path for path in candidates if "/" not in str(path).strip("/")]
    return (root_level or candidates)[0]


def _fallback_loader_from_error(current_loader: str, error: Exception) -> Optional[str]:
    message = str(error).lower()
    if "unrecognized configuration class" not in message:
        return None
    if current_loader == "multimodal":
        if "t5config" in message:
            return "seq2seq"
        if any(marker in message for marker in ("qwen2config", "llamaconfig", "gemmaconfig", "phiconfig", "mistralconfig")):
            return "standard_causal"
    return None


def _fallback_auto_model_cls(current_loader: str, error: Exception):
    if current_loader != "standard_causal":
        return None
    message = str(error).lower()
    if "unrecognized configuration class" not in message:
        return None
    return AutoModel


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


def _normalize_optional_revision(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def resolve_base_model_reference(
    adapter_repo: str,
    source_model: Optional[str] = None,
    adapter_revision: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Resolve base model for an adapter.
    
    Args:
        adapter_repo: The adapter repository ID
        source_model: Optional explicitly specified base model
    
    Returns:
        Tuple of (base model repository ID, base revision)
    """
    # Use explicit source_model if provided
    if source_model and isinstance(source_model, str) and source_model.strip():
        return parse_model_string(source_model.strip())
    
    # Try to infer from PeftConfig
    token = get_hf_token()
    try:
        cfg = PeftConfig.from_pretrained(adapter_repo, token=token, revision=adapter_revision)
        if hasattr(cfg, "base_model_name_or_path") and cfg.base_model_name_or_path:
            base_repo, embedded_revision = parse_model_string(str(cfg.base_model_name_or_path))
            base_revision = _normalize_optional_revision(getattr(cfg, "revision", None))
            return base_repo, base_revision or embedded_revision
    except Exception as e:
        warnings.warn(f"Could not load PeftConfig for {adapter_repo}: {e}")
    
    # Try to infer from AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(
            adapter_repo,
            token=token,
            revision=adapter_revision,
            trust_remote_code=True,
        )
        for key in ["base_model_name_or_path", "base_model", "model_name", "parent_model_name_or_path"]:
            if hasattr(cfg, key) and getattr(cfg, key):
                return parse_model_string(str(getattr(cfg, key)))
    except Exception as e:
        warnings.warn(f"Could not load AutoConfig for {adapter_repo}: {e}")
    
    raise RuntimeError(
        f"Could not resolve base model for adapter {adapter_repo}. "
        f"Please provide source_model explicitly in the model list."
    )


def resolve_base_model(adapter_repo: str, source_model: Optional[str] = None) -> str:
    base_repo, _ = resolve_base_model_reference(adapter_repo, source_model=source_model)
    return base_repo


def load_and_merge_adapter(
    adapter_repo: str,
    base_repo: Optional[str] = None,
    device_map: str = "cpu",
    torch_dtype = torch.float16,
    revision: Optional[str] = None,
    loader_scenario: Optional[str] = None,
    effective_loader: Optional[str] = None,
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
    base_revision = revision
    if base_repo is None:
        try:
            base_repo, base_revision = resolve_base_model_reference(
                adapter_repo,
                adapter_revision=revision,
            )
        except RuntimeError as exc:
            raise LoaderFailure("load", "adapter_base_unresolved", str(exc)) from exc
    else:
        base_repo, parsed_revision = parse_model_string(base_repo)
        base_revision = parsed_revision or revision
    
    effective_loader = effective_loader or _resolve_adapter_task_loader(
        adapter_repo,
        source_model=base_repo,
        loader_scenario=loader_scenario,
        revision=revision,
    )
    if effective_loader in {"gptq", "awq", "gguf", "compressed_tensors"}:
        dense_base_reference = resolve_dense_upstream_base_reference(
            base_repo,
            revision=base_revision,
        )
        if dense_base_reference is not None:
            dense_base_repo, dense_base_revision = dense_base_reference
            if dense_base_repo != base_repo or dense_base_revision != base_revision:
                base_repo = dense_base_repo
                base_revision = dense_base_revision
                effective_loader = _resolve_adapter_task_loader(
                    adapter_repo,
                    source_model=base_repo,
                    loader_scenario=loader_scenario,
                    revision=revision,
                )
    auto_model_cls = _select_auto_model_cls(effective_loader)

    print(f"Loading base model: {base_repo}")
    if effective_loader in {"awq", "gguf"}:
        raise LoaderFailure(
            "load",
            "unsupported_backend",
            f"Adapter loading does not support {effective_loader} base loaders",
        )
    if effective_loader == "gptq":
        ensure_optimum_gptq_backend_compat()
    try:
        if effective_loader == "compressed_tensors":
            ensure_compressed_tensors_backend_compat()
        base = hf_from_pretrained(
            auto_model_cls,
            base_repo,
            device_map=device_map,
            torch_dtype=torch_dtype,
            revision=base_revision,
        )
    except Exception as exc:
        if effective_loader in {"gptq", "awq", "compressed_tensors"}:
            dependency_failure = classify_quantized_dependency_failure(exc)
            if dependency_failure is not None:
                raise dependency_failure from exc
        raise
    base.eval()
    if effective_loader == "gptq":
        try:
            base = base.dequantize()
            base.eval()
        except NotImplementedError as exc:
            raise LoaderFailure(
                "load",
                "adapter_merge_unsupported",
                "GPTQ base models cannot currently be merged with adapters because GPTQ dequantization is not implemented in the active Transformers/GPTQModel stack",
            ) from exc
    
    print(f"Loading adapter: {adapter_repo}")
    token = get_hf_token()
    try:
        peft_model = PeftModel.from_pretrained(
            base,
            adapter_repo,
            is_trainable=False,
            token=token,
            revision=revision,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "size mismatch for base_model." in message:
            raise LoaderFailure(
                "load",
                "adapter_base_checkpoint_mismatch",
                "Adapter weights do not match the resolved base model checkpoint",
            ) from exc
        raise
    
    print("Merging adapter weights into base model...")
    try:
        merged = peft_model.merge_and_unload() # type: ignore
    except ValueError as exc:
        message = str(exc)
        if "cannot merge lora layers when the model is gptq quantized" in message.lower():
            raise LoaderFailure(
                "load",
                "adapter_merge_unsupported",
                "GPTQ base models cannot currently be merged with adapters because PEFT refuses merge on quantized GPTQ bases",
            ) from exc
        raise
    merged.eval()
    
    return merged


def load_model(
    repo_id: str,
    base_model_relation: Optional[str] = None,
    source_model: Optional[str] = None,
    device_map: str = "cpu",
    torch_dtype = torch.float16,
    revision: Optional[str] = None,
    loader_scenario: Optional[str] = None,
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
    scenario_failure = classify_loader_scenario_support(loader_scenario)
    if scenario_failure is not None:
        raise scenario_failure

    # Check if this is an adapter
    if is_adapter_model(repo_id, base_model_relation):
        effective_loader = _resolve_adapter_task_loader(
            repo_id,
            source_model=source_model,
            loader_scenario=loader_scenario,
            revision=revision,
        )
        print(f"[ADAPTER] Loading adapter model: {repo_id}")
        model = load_and_merge_adapter(
            adapter_repo=repo_id,
            base_repo=source_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            revision=revision,
            loader_scenario=loader_scenario,
            effective_loader=effective_loader,
        )
        return model, True
    else:
        print(f"[STANDARD] Loading standard model: {repo_id}")
        effective_loader = resolve_effective_loader_for_repo(
            repo_id,
            loader_scenario=loader_scenario,
            base_model_relation=base_model_relation,
            source_model=source_model,
            revision=revision,
        )
        auto_model_cls = _select_auto_model_cls(effective_loader)
        if effective_loader == "gptq":
            ensure_optimum_gptq_backend_compat()
        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "revision": revision,
        }
        if effective_loader == "gguf":
            load_kwargs["gguf_file"] = resolve_gguf_filename(repo_id, revision=revision)
            load_kwargs["dtype"] = torch_dtype
            load_kwargs.pop("torch_dtype", None)
        try:
            if effective_loader == "compressed_tensors":
                ensure_compressed_tensors_backend_compat()
            model = hf_from_pretrained(
                auto_model_cls,
                repo_id,
                **load_kwargs,
            )
        except Exception as exc:
            fallback_loader = _fallback_loader_from_error(effective_loader, exc)
            if fallback_loader is not None and fallback_loader != effective_loader:
                model = hf_from_pretrained(
                    _select_auto_model_cls(fallback_loader),
                    repo_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    revision=revision,
                )
            else:
                fallback_cls = _fallback_auto_model_cls(effective_loader, exc)
                if fallback_cls is not None:
                    try:
                        model = hf_from_pretrained(
                            fallback_cls,
                            repo_id,
                            **load_kwargs,
                        )
                    except Exception as fallback_exc:
                        _raise_quantized_loader_failure_if_known(
                            fallback_exc,
                            loader_scenario=loader_scenario,
                            effective_loader=effective_loader,
                        )
                        raise
                else:
                    _raise_quantized_loader_failure_if_known(
                        exc,
                        loader_scenario=loader_scenario,
                        effective_loader=effective_loader,
                    )
                    raise
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
