from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Set


@dataclass(frozen=True)
class PreflightDecision:
    eligible: bool
    reason: str
    effective_loader: str = "standard_transformers"


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


def resolve_effective_loader(row: Mapping[str, Any]) -> str:
    model_type = _text(row.get("model_type"))
    architectures = _text(row.get("architectures"))
    config_text = _text(row.get("config"))
    loader_scenario = _text(row.get("loader_scenario"))

    if model_type.startswith("t5") or "seq2seq" in loader_scenario:
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
        marker in config_text for marker in multimodal_markers
    ):
        return "multimodal"

    if any(marker in loader_scenario for marker in multimodal_markers):
        return "multimodal"

    return "standard_transformers"


def classify_row_preflight(row: Mapping[str, Any]) -> PreflightDecision:
    effective_loader = resolve_effective_loader(row)
    base_model_relation = _text(row.get("base_model_relation"))
    loader_scenario = _text(row.get("loader_scenario"))
    adapter_config = _text(row.get("adapter_config"))
    files = _file_set(
        row.get("files")
        or row.get("repo_files")
        or row.get("file_names")
        or row.get("repo_file_names")
    )
    backend_status = _text(row.get("backend_status"))

    if base_model_relation in {"adapter", "lora", "peft"} and not (
        adapter_config or "adapter_config" in files
    ):
        return PreflightDecision(
            eligible=False,
            reason="adapter_config_missing",
            effective_loader=effective_loader,
        )

    if effective_loader == "standard_transformers" and loader_scenario == "quantized_transformers_native":
        if not backend_status or backend_status in {"missing", "absent", "none"}:
            return PreflightDecision(
                eligible=False,
                reason="gptq_backend_missing",
                effective_loader=effective_loader,
            )

    return PreflightDecision(
        eligible=True,
        reason="eligible",
        effective_loader=effective_loader,
    )
