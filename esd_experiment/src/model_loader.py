"""
Robust model loader with PEFT adapter support.
Heavily inspired by calculate_adapters.py and run_metric.py patterns.
"""
import importlib
import hashlib
import json
import os
import re
import torch
import transformers
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig
from measurement_config import is_commit_sha
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
from peft import PeftConfig
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
    kwargs.setdefault("trust_remote_code", False)
    kwargs.setdefault("low_cpu_mem_usage", True)
    if kwargs["trust_remote_code"]:
        revision = kwargs.get("revision")
        if not Path(repo_id).is_dir() and not is_commit_sha(revision):
            raise LoaderFailure(
                "load", "remote_code_revision_unpinned",
                "Remote code requires a full commit SHA, not a mutable branch or tag",
            )
        # Read JSON without importing checkpoint code. Transformers uses the
        # model revision for same-repository code, but not for external code.
        config, _ = PretrainedConfig.get_config_dict(repo_id, token=token, revision=revision)
        auto_map = config.get("auto_map") or {}
        if not isinstance(auto_map, dict):
            raise LoaderFailure("load", "unsupported_remote_code", "Checkpoint auto_map must be a mapping")
        for references in auto_map.values():
            for reference in references if isinstance(references, (list, tuple)) else [references]:
                if isinstance(reference, str) and "--" in reference and reference.split("--", 1)[0] != repo_id:
                    raise LoaderFailure(
                        "load", "unsupported_remote_code",
                        "Cross-repository auto_map code is not pinned by this model's revision; "
                        "only same-repository custom code is supported",
                    )

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


def _json_safe_loading_value(value):
    """Transformers versions return lists, sets, tuples, or shape objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe_loading_value(item) for key, item in value.items()}
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple)):
        return [_json_safe_loading_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _load_checkpoint_checked(model_cls, repo_id: str, **kwargs):
    """Never measure parameters silently initialized or discarded during loading.

    Keep ``hf_from_pretrained``'s public return convention unchanged; only
    production loads require and validate Transformers' loading report.
    """
    loaded = hf_from_pretrained(model_cls, repo_id, output_loading_info=True, **kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], dict):
        raise LoaderFailure("load", "checkpoint_loading_info_missing", "Loader did not return a checkpoint loading report")
    model, info = loaded
    required = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if not required.issubset(info):
        raise LoaderFailure("load", "checkpoint_loading_info_missing", "Checkpoint loading report is incomplete")
    info = _json_safe_loading_value(info)
    config = getattr(model, "config", None)
    allowed_unexpected = []
    unexpected = []
    for key in info["unexpected_keys"]:
        # These scalar buffers were serialized by old GPT2 versions but are
        # absent in current checkpoints/models. Do not whitelist arbitrary
        # biases or other unexpected keys, which may be trained parameters.
        if (getattr(config, "model_type", None) == "gpt2"
                and re.fullmatch(r"transformer\.h\.\d+\.attn\.masked_bias", key)):
            allowed_unexpected.append(key)
        else:
            unexpected.append(key)
    if (info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]
            or info.get("conversion_errors") or unexpected):
        problems = {
            "missing_keys": info["missing_keys"], "mismatched_keys": info["mismatched_keys"],
            "error_msgs": info["error_msgs"], "unexpected_keys": unexpected,
            "conversion_errors": info.get("conversion_errors"),
        }
        raise LoaderFailure(
            "load", "checkpoint_weight_mismatch",
            f"Checkpoint weights do not match {type(model).__name__}: "
            + "; ".join(f"{key}={value}" for key, value in problems.items() if value),
        )
    model._model_zoo_loading_info = {
        "model_class": type(model).__name__,
        "loader_class": model_cls.__name__,
        "loading_info": info,
        "allowed_unexpected_keys": allowed_unexpected,
        "validated": True,
    }
    return model


def _declared_checkpoint_model_cls(repo_id: str, revision: Optional[str] = None):
    """Prefer the checkpoint's built-in architecture over a task-name guess.

    Looking up an installed Transformers class does not evaluate checkpoint
    code. Unknown/custom architectures keep their existing AutoModel route,
    whose loaded weights must still pass the integrity check.
    """
    try:
        config = AutoConfig.from_pretrained(
            repo_id, token=get_hf_token(), revision=revision, trust_remote_code=False,
        )
    except Exception:
        return None
    candidates = []
    for name in getattr(config, "architectures", None) or []:
        if not isinstance(name, str) or not name.isidentifier():
            continue
        try:
            candidate = getattr(transformers, name, None)
        except (ImportError, RuntimeError) as exc:
            raise LoaderFailure(
                "load", "checkpoint_architecture_unavailable",
                f"Could not import declared Transformers architecture {name}: {exc}",
            ) from exc
        if (isinstance(candidate, type)
                and issubclass(candidate, transformers.PreTrainedModel)
                and candidate.__module__.startswith("transformers.")):
            config_cls = getattr(candidate, "config_class", None)
            if config_cls is not None and not isinstance(config, config_cls):
                raise LoaderFailure(
                    "load", "checkpoint_architecture_mismatch",
                    f"Declared architecture {name} is incompatible with {type(config).__name__}",
                )
            if candidate not in candidates:
                candidates.append(candidate)
    if len(candidates) > 1:
        raise LoaderFailure(
            "load", "ambiguous_checkpoint_architecture",
            "Checkpoint declares multiple supported architectures; refusing to choose a different set of weights",
        )
    return candidates[0] if candidates else None


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
            trust_remote_code=False,
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
                revision=source_revision,
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


def hf_repo_has_prefix(repo_id: str, prefix: str, revision: Optional[str] = None) -> bool:
    """Check if HuggingFace repo has files with given prefix."""
    api = HfApi()
    try:
        token = get_hf_token()
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token, revision=revision)
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


def is_adapter_model(repo_id: str, base_model_relation: Optional[str] = None,
                     revision: Optional[str] = None) -> bool:
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
        PeftConfig.from_pretrained(repo_id, token=token, revision=revision)
        return True
    except Exception:
        pass
    
    # Check file structure: has adapter*.safetensors but no model*.safetensors
    if hf_repo_has_prefix(repo_id, "adapter", revision) and not hf_repo_has_prefix(repo_id, "model", revision):
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
            trust_remote_code=False,
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


def _load_adapter_checked(base, adapter_repo: str, revision: Optional[str] = None):
    """Validate a configured LoRA payload before it can overwrite base weights.

    PEFT's permissive loader can silently accept extra base-layer tensors.
    Its public serializer gives us the exact configured adapter key/shape
    manifest, including explicitly saved heads and biases, without cloning the
    dense base. Other adapter forms need their own integrity pilot first.
    """
    import peft
    from peft import (
        get_peft_model, get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict,
    )

    config = PeftConfig.from_pretrained(adapter_repo, token=get_hf_token(), revision=revision)
    peft_type = getattr(config.peft_type, "value", config.peft_type)
    unsupported = [
        name for name in (
            "use_dora", "use_qalora", "alora_invocation_tokens", "arrow_config",
            "layer_replication", "target_parameters", "trainable_token_indices",
            "megatron_config", "ensure_weight_tying", "lora_bias",
        ) if getattr(config, name, None)
    ]
    # PiSSA/OLoRA/LoftQ and similar initializers can modify the dense base
    # before checkpoint tensors are loaded. Only adapter-only initializers
    # have been validated here.
    if getattr(config, "init_lora_weights", True) not in (True, False, "gaussian"):
        unsupported.append("init_lora_weights")
    if getattr(config, "bias", "none") not in ("none", "all"):
        unsupported.append("bias")
    if peft_type != "LORA" or unsupported:
        raise LoaderFailure(
            "load", "unsupported_adapter_integrity",
            f"Adapter integrity is implemented for ordinary/RSLoRA only; unsupported: {peft_type}, {unsupported}",
        )
    auto_mapping = getattr(config, "auto_mapping", None) or {}
    if not isinstance(auto_mapping, dict):
        raise LoaderFailure("load", "unsupported_adapter_integrity", "Adapter auto_mapping must be a mapping")
    declared_class = auto_mapping.get("base_model_class")
    declared_library = auto_mapping.get("parent_library")
    if declared_class and not any(
        cls.__name__ == declared_class and (not declared_library or cls.__module__ == declared_library)
        for cls in type(base).__mro__
    ):
        raise LoaderFailure(
            "load", "adapter_base_architecture_mismatch",
            f"Adapter declares base class {declared_class}, but checkpoint loads as {type(base).__name__}",
        )

    checkpoint_config = _json_safe_loading_value(config.to_dict())
    config.inference_mode = True
    adapted = get_peft_model(base, config, adapter_name="default", autocast_adapter_dtype=True)
    # False deliberately excludes optional, unconfigured full embedding dumps.
    # Such payloads are rejected as extra keys, not silently ignored.
    expected = get_peft_model_state_dict(adapted, adapter_name="default", save_embedding_layers=False)
    lora_keys = [key for key in expected if ".lora_" in key]
    if not lora_keys or any(
        not key.endswith((".lora_A.weight", ".lora_B.weight")) or expected[key].ndim != 2
        for key in lora_keys
    ):
        raise LoaderFailure(
            "load", "unsupported_adapter_integrity",
            "Only dense Linear/Conv1D LoRA A/B matrix layouts have been validated",
        )

    weights = load_peft_weights(adapter_repo, device="cpu", token=get_hf_token(), revision=revision)
    missing, surplus = sorted(set(expected) - set(weights)), sorted(set(weights) - set(expected))
    mismatched = [key for key in expected.keys() & weights.keys() if expected[key].shape != weights[key].shape]
    if missing or surplus or mismatched:
        raise LoaderFailure(
            "load", "adapter_checkpoint_mismatch",
            f"Adapter tensor manifest mismatch: missing={missing}; unexpected={surplus}; shapes={mismatched}",
        )
    digest = hashlib.sha256()
    for key in sorted(weights):
        tensor = weights[key]
        if expected[key].is_floating_point() and not tensor.is_floating_point():
            raise LoaderFailure(
                "load", "unsupported_adapter_integrity",
                f"Expected floating adapter tensor {key}; packed/integer representations are not supported",
            )
        if not torch.isfinite(tensor).all():
            raise LoaderFailure("load", "adapter_nonfinite_weights", f"Adapter tensor {key} is not finite")
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
        # Hash contiguous CPU storage without making a second full tensor copy.
        digest.update(memoryview(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))

    # PEFT may rewrite dict keys for modules_to_save, so pass a shallow copy.
    # The tensors themselves are not duplicated.
    load_result = set_peft_model_state_dict(adapted, dict(weights), adapter_name="default")
    missing_adapter = [
        key for key in load_result.missing_keys
        if ".lora_" in key or ".modules_to_save.default." in key
    ]
    if missing_adapter or load_result.unexpected_keys:
        raise LoaderFailure(
            "load", "adapter_checkpoint_mismatch",
            f"PEFT did not apply the verified adapter: missing={missing_adapter}; unexpected={load_result.unexpected_keys}",
        )
    loaded = get_peft_model_state_dict(adapted, adapter_name="default", save_embedding_layers=False)
    for key, value in loaded.items():
        if not torch.isfinite(value).all():
            raise LoaderFailure("load", "adapter_nonfinite_weights", f"Loaded adapter tensor {key} is not finite")
        if not torch.equal(value.detach().cpu(), weights[key].to(dtype=value.dtype)):
            raise LoaderFailure("load", "adapter_checkpoint_mismatch", f"PEFT changed adapter tensor {key} while loading")
    adapted.eval()
    report = {
        "peft_version": peft.__version__,
        "adapter_config": checkpoint_config,
        "adapter_config_sha256": hashlib.sha256(
            json.dumps(checkpoint_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "adapter_state_sha256": digest.hexdigest(),
        "adapter_tensor_count": len(weights),
        "lora_tensor_count": len(lora_keys),
        "checkpoint_dtypes": sorted({str(value.dtype) for value in weights.values()}),
        "loaded_dtypes": sorted({str(value.dtype) for value in loaded.values()}),
        "verification_scope": "exact configured keys/shapes, finite payload, exact loaded values after recorded dtype conversion",
    }
    return adapted, report


def load_and_merge_adapter(
    adapter_repo: str,
    base_repo: Optional[str] = None,
    device_map: str = "cpu",
    torch_dtype = torch.float16,
    revision: Optional[str] = None,
    loader_scenario: Optional[str] = None,
    effective_loader: Optional[str] = None,
    trust_remote_code: bool = False,
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
    base_revision = None
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
        base_revision = parsed_revision
    requested_base_repo, requested_base_revision = base_repo, base_revision
    if is_commit_sha(revision) and not is_commit_sha(base_revision):
        raise LoaderFailure(
            "load", "adapter_base_revision_unpinned",
            "Pinned adapters require a pinned base: set source_model to repo_id@<full commit SHA>",
        )
    
    effective_loader = effective_loader or _resolve_adapter_task_loader(
        adapter_repo,
        source_model=f"{base_repo}@{base_revision}" if base_revision else base_repo,
        loader_scenario=loader_scenario,
        revision=revision,
    )
    auto_model_cls = _select_auto_model_cls(effective_loader)
    if effective_loader in {"standard_causal", "seq2seq", "sequence_classification", "multimodal"}:
        auto_model_cls = _declared_checkpoint_model_cls(base_repo, base_revision) or auto_model_cls

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
        base = _load_checkpoint_checked(
            auto_model_cls,
            base_repo,
            device_map=device_map,
            torch_dtype=torch_dtype,
            revision=base_revision,
            trust_remote_code=trust_remote_code,
        )
    except LoaderFailure:
        raise
    except Exception as exc:
        if effective_loader in {"gptq", "awq", "compressed_tensors"}:
            dependency_failure = classify_quantized_dependency_failure(exc)
            if dependency_failure is not None:
                raise dependency_failure from exc
        raise
    base.eval()
    base_loading_info = base._model_zoo_loading_info
    base_commit_hash = getattr(getattr(base, "config", None), "_commit_hash", None)
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
    try:
        peft_model, adapter_loading_info = _load_adapter_checked(base, adapter_repo, revision=revision)
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
        merged = peft_model.merge_and_unload(safe_merge=True) # type: ignore
    except ValueError as exc:
        message = str(exc)
        if "cannot merge lora layers when the model is gptq quantized" in message.lower():
            raise LoaderFailure(
                "load",
                "adapter_merge_unsupported",
                "GPTQ base models cannot currently be merged with adapters because PEFT refuses merge on quantized GPTQ bases",
            ) from exc
        if "nan" in message.lower() or "finite" in message.lower():
            raise LoaderFailure("load", "adapter_merge_nonfinite", "Adapter merge produced non-finite weights") from exc
        raise
    merged.eval()
    merged._model_zoo_loading_info = {
        "model_class": type(merged).__name__,
        "base_loading_info": base_loading_info,
        "adapter_loading_info": adapter_loading_info,
        "adapter_repo": adapter_repo,
        "adapter_requested_revision": revision or "main",
        "base_repo_requested": requested_base_repo,
        "base_revision_requested": requested_base_revision or "main",
        "base_repo_loaded": base_repo,
        "base_revision_loaded": base_revision or "main",
        "base_resolved_commit_hash": base_commit_hash,
        "adapter_merge": True,
        "adapter_weights_verified": True,
        "safe_merge": True,
    }
    
    return merged


def load_model(
    repo_id: str,
    base_model_relation: Optional[str] = None,
    source_model: Optional[str] = None,
    device_map: str = "cpu",
    torch_dtype = torch.float16,
    revision: Optional[str] = None,
    loader_scenario: Optional[str] = None,
    trust_remote_code: bool = False,
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
        trust_remote_code: Explicit permission to execute same-repository checkpoint code
    
    Returns:
        Tuple of (model, is_adapter)
    """
    scenario_failure = classify_loader_scenario_support(loader_scenario)
    if scenario_failure is not None:
        raise scenario_failure

    # Check if this is an adapter
    if is_adapter_model(repo_id, base_model_relation, revision=revision):
        print(f"[ADAPTER] Loading adapter model: {repo_id}")
        model = load_and_merge_adapter(
            adapter_repo=repo_id,
            base_repo=source_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            revision=revision,
            loader_scenario=loader_scenario,
            trust_remote_code=trust_remote_code,
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
        if (effective_loader in {"standard_causal", "seq2seq", "sequence_classification", "multimodal"}
                and (loader_scenario or "").strip().lower() != "quantized_transformers_native"):
            auto_model_cls = _declared_checkpoint_model_cls(repo_id, revision) or auto_model_cls
        if effective_loader == "gptq":
            ensure_optimum_gptq_backend_compat()
        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if effective_loader == "gguf":
            load_kwargs["gguf_file"] = resolve_gguf_filename(repo_id, revision=revision)
            load_kwargs["dtype"] = torch_dtype
            load_kwargs.pop("torch_dtype", None)
        try:
            if effective_loader == "compressed_tensors":
                ensure_compressed_tensors_backend_compat()
            model = _load_checkpoint_checked(
                auto_model_cls,
                repo_id,
                **load_kwargs,
            )
        except LoaderFailure:
            raise
        except Exception as exc:
            fallback_loader = _fallback_loader_from_error(effective_loader, exc)
            if fallback_loader is not None and fallback_loader != effective_loader:
                model = _load_checkpoint_checked(
                    _select_auto_model_cls(fallback_loader),
                    repo_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )
            else:
                fallback_cls = _fallback_auto_model_cls(effective_loader, exc)
                if fallback_cls is not None:
                    try:
                        model = _load_checkpoint_checked(
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
