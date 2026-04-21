from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Set


@dataclass(frozen=True)
class PreflightDecision:
    eligible: bool
    reason: str
    effective_loader: str = "standard_causal"


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN-safe check without importing pandas.
            return ""
    except Exception:
        pass
    return str(value).strip().lower()


def _file_set(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        candidates = [part for part in value.replace(";", ",").split(",")]
    elif isinstance(value, Mapping):
        candidates = list(value.values())
    elif isinstance(value, Iterable):
        candidates = list(value)
    else:
        candidates = [value]
    return {
        _text(item)
        for item in candidates
        if _text(item)
    }


def _has_file_named(value: Any, filename: str) -> bool:
    target = _text(filename)
    if not target:
        return False
    if value is None:
        return False
    if isinstance(value, str):
        candidates = [part for part in value.replace(";", ",").split(",")]
    elif isinstance(value, Mapping):
        candidates = list(value.values())
    elif isinstance(value, Iterable):
        candidates = list(value)
    else:
        candidates = [value]
    return any(_text(str(item).rsplit("/", 1)[-1]) == target for item in candidates)


def _has_adapter_artifact(row: Mapping[str, Any]) -> bool:
    if _text(row.get("adapter_config")):
        return True

    for source in (
        row.get("files"),
        row.get("file_names"),
        row.get("repo_file_names"),
        row.get("repo_files"),
    ):
        if "adapter_config" in _file_set(source) or _has_file_named(source, "adapter_config.json"):
            return True

    return False


def resolve_effective_loader(row: Mapping[str, Any]) -> str:
    model_type = _text(row.get("model_type"))
    config_model_type = _text(row.get("config_model_type"))
    architectures = _text(row.get("architectures"))
    config_architectures = _text(row.get("config_architectures"))
    loader_scenario = _text(row.get("loader_scenario"))
    base_model_relation = _text(row.get("base_model_relation"))

    if base_model_relation in {"adapter", "lora", "peft"}:
        return "adapter_requires_base"

    if any(
        value.startswith("t5")
        for value in (model_type, config_model_type)
        if value
    ) or "seq2seq" in loader_scenario:
        return "seq2seq"

    multimodal_markers = (
        "image-text",
        "image text",
        "multimodal",
        "imagetext",
        "visiontext",
        "image_text",
    )
    if any(marker in architectures for marker in multimodal_markers) or any(
        marker in config_architectures for marker in multimodal_markers
    ):
        return "multimodal"

    if any(marker in loader_scenario for marker in multimodal_markers):
        return "multimodal"

    return "standard_causal"


def classify_row_preflight(row: Mapping[str, Any]) -> PreflightDecision:
    effective_loader = resolve_effective_loader(row)
    base_model_relation = _text(row.get("base_model_relation"))
    loader_scenario = _text(row.get("loader_scenario"))
    backend_status = _text(row.get("backend_status"))

    if base_model_relation in {"adapter", "lora", "peft"} and not _has_adapter_artifact(row):
        return PreflightDecision(
            eligible=False,
            reason="missing_required_artifact",
            effective_loader="adapter_requires_base",
        )

    if loader_scenario == "quantized_transformers_native" or effective_loader == "gptq":
        if not backend_status or backend_status in {"missing", "absent", "none"}:
            return PreflightDecision(
                eligible=False,
                reason="unsupported_backend",
                effective_loader="gptq",
            )
        return PreflightDecision(
            eligible=True,
            reason="eligible",
            effective_loader="gptq",
        )

    return PreflightDecision(
        eligible=True,
        reason="eligible",
        effective_loader=effective_loader,
    )
