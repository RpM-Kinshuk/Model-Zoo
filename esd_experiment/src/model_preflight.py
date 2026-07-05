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
        if source is not None:
            if isinstance(source, str):
                candidates = [part for part in source.replace(";", ",").split(",")]
            elif isinstance(source, Mapping):
                candidates = list(source.values())
            elif isinstance(source, Iterable):
                candidates = list(source)
            else:
                candidates = [source]
            for item in candidates:
                name = _text(str(item).rsplit("/", 1)[-1])
                if name.startswith("adapter") and (
                    name.endswith(".safetensors") or name.endswith(".bin")
                ):
                    return True

    return False


def _has_explicit_repo_file_metadata(row: Mapping[str, Any]) -> bool:
    for source in (
        row.get("files"),
        row.get("file_names"),
        row.get("repo_file_names"),
        row.get("repo_files"),
    ):
        if _file_set(source):
            return True
    return False


def _has_gguf_artifact(row: Mapping[str, Any]) -> bool:
    for source in (
        row.get("files"),
        row.get("file_names"),
        row.get("repo_file_names"),
        row.get("repo_files"),
    ):
        if source is None:
            continue
        if isinstance(source, str):
            candidates = [part for part in source.replace(";", ",").split(",")]
        elif isinstance(source, Mapping):
            candidates = list(source.values())
        elif isinstance(source, Iterable):
            candidates = list(source)
        else:
            candidates = [source]
        for item in candidates:
            name = _text(str(item).rsplit("/", 1)[-1])
            if name.endswith(".gguf"):
                return True
    return False


def _blob(*values: Any) -> str:
    return " ".join(_text(value) for value in values if _text(value))


def resolve_effective_loader(row: Mapping[str, Any]) -> str:
    model_type = _text(row.get("model_type"))
    config_model_type = _text(row.get("config_model_type"))
    architectures = _blob(
        row.get("architectures"),
        row.get("config_architectures"),
        row.get("Architecture"),
        row.get("Architecture_lb"),
    )
    pipeline_tag = _blob(row.get("pipeline_tag"), row.get("pipeline_tag_lb"))
    tags = _blob(
        row.get("tags"),
        row.get("tags_lb"),
        row.get("Type"),
        row.get("Type_lb"),
        row.get("quant_method"),
    )
    loader_scenario = _text(row.get("loader_scenario"))
    base_model_relation = _text(row.get("base_model_relation"))
    model_id = _text(row.get("model_id"))
    loader_hints = _blob(model_id, tags, loader_scenario)

    if base_model_relation in {"adapter", "lora", "peft"}:
        return "adapter_requires_base"

    if "exl2" in loader_hints:
        return "quantized_alt_format"

    if loader_scenario == "gguf" or _has_gguf_artifact(row) or "gguf" in loader_hints:
        return "gguf"

    if "awq" in loader_hints:
        return "awq"

    if "gptq" in loader_hints:
        return "gptq"

    if "compressed-tensors" in loader_hints or "compressed_tensors" in loader_hints:
        return "compressed_tensors"

    if (
        "forsequenceclassification" in architectures
        or pipeline_tag == "text-classification"
    ):
        return "sequence_classification"

    if any(
        value.startswith("t5")
        for value in (model_type, config_model_type)
        if value
    ) or "t5forconditionalgeneration" in architectures or "seq2seq" in loader_scenario or "text2text-generation" in pipeline_tag:
        return "seq2seq"

    multimodal_markers = (
        "image-text",
        "image text",
        "multimodal",
        "imagetext",
        "visiontext",
        "image_text",
    )
    if any(marker in architectures for marker in multimodal_markers):
        return "multimodal"

    if any(marker in loader_scenario for marker in multimodal_markers):
        return "multimodal"

    return "standard_causal"


def classify_row_preflight(row: Mapping[str, Any]) -> PreflightDecision:
    effective_loader = resolve_effective_loader(row)
    base_model_relation = _text(row.get("base_model_relation"))
    backend_status = _text(row.get("backend_status"))
    available_on_hub = _text(row.get("Available on the hub"))

    if available_on_hub in {"false", "0", "no"}:
        return PreflightDecision(
            eligible=False,
            reason="repo_inaccessible",
            effective_loader=effective_loader,
        )

    if (
        base_model_relation in {"adapter", "lora", "peft"}
        and _has_explicit_repo_file_metadata(row)
        and not _has_adapter_artifact(row)
    ):
        return PreflightDecision(
            eligible=False,
            reason="missing_required_artifact",
            effective_loader="adapter_requires_base",
        )

    if effective_loader == "quantized_alt_format":
        return PreflightDecision(
            eligible=False,
            reason="unsupported_loader_scenario",
            effective_loader="quantized_alt_format",
        )

    if effective_loader == "gguf":
        if not backend_status or backend_status in {"missing", "absent", "none"}:
            return PreflightDecision(
                eligible=False,
                reason="unsupported_backend",
                effective_loader="gguf",
            )
        return PreflightDecision(
            eligible=True,
            reason="eligible",
            effective_loader="gguf",
        )

    if effective_loader in {"gptq", "awq", "compressed_tensors"}:
        if not backend_status or backend_status in {"missing", "absent", "none"}:
            return PreflightDecision(
                eligible=False,
                reason="unsupported_backend",
                effective_loader=effective_loader,
            )
        return PreflightDecision(
            eligible=True,
            reason="eligible",
            effective_loader=effective_loader,
        )

    return PreflightDecision(
        eligible=True,
        reason="eligible",
        effective_loader=effective_loader,
    )
