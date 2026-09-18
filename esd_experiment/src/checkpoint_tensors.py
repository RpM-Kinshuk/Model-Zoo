"""Explicit matrix descriptors from checkpoints, without constructing a model.

Checkpoint keys are identities, not inferred modules. Only stored floating 2D
tensors are measured: no reshaping, dequantization, adapter merge or alias guesses.
Downloads use the worker's existing ephemeral cache. Analysis opens one file at
a time, plus the working tensor and accumulated scalar results and spectra.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import zipfile

import torch
from safetensors import safe_open

from model_loader import LoaderFailure, checkpoint_tensor_index, get_hf_token
from measurement_config import is_commit_sha
from net_esd.constants import RESULT_KEYS
from net_esd.core import compute_esd_for_weight


FLOAT_DTYPES = {"F16", "BF16", "F32", "F64", "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
PACKED_KEY = re.compile(
    r"(^|\.)(qweight|qzeros|quant_state|weight_scale|weight_scale_inv|"
    r"weight_packed|weight_zero_point|lora_[ab]|ia3_[a-z]+)(\.|$)", re.I,
)


def inspect_checkpoint(repo_id, revision, *, base_model_relation="", loader_scenario=""):
    """Validate headers and known representation markers before any ESD work.

    Absence of markers cannot prove a tensor's role in an unknown architecture.
    Results describe the stored matrices, not reconstructed effective model weights.
    Local directories are supported for offline tests; the worker requires HF pins.
    """
    from transformers.utils.hub import cached_file

    if not Path(repo_id).is_dir() and not is_commit_sha(revision):
        raise LoaderFailure("load", "checkpoint_inspection_failed", "Checkpoint inspection requires a pinned revision")
    if base_model_relation.lower() in {"adapter", "lora", "peft"} or loader_scenario in {
        "adapter_requires_base", "quantized_transformers_native", "quantized_alt_format",
        "gguf", "gptq", "awq", "compressed_tensors",
    }:
        raise LoaderFailure("load", "unsupported_checkpoint_representation",
                            "Checkpoint mode does not merge adapters or interpret quantized weights; use model mode")

    def resolve(filename):
        return cached_file(repo_id, filename, revision=revision, token=get_hf_token(),
                           _raise_exceptions_for_missing_entries=False)

    if resolve("adapter_config.json"):
        raise LoaderFailure("load", "unsupported_checkpoint_representation",
                            "Adapter checkpoints require the pinned-base model-loading path")
    config_path = resolve("config.json")
    config_hash = None
    if config_path:
        if Path(config_path).stat().st_size > 8 * 1024 * 1024:
            raise LoaderFailure("load", "checkpoint_inspection_failed", "Config exceeds the 8 MiB inspection limit")
        config_bytes = Path(config_path).read_bytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        try:
            config = json.loads(config_bytes)
        except (ValueError, UnicodeDecodeError) as exc:
            raise LoaderFailure("load", "checkpoint_inspection_failed", "Config is not valid JSON") from exc
        if not isinstance(config, dict):
            raise LoaderFailure("load", "checkpoint_inspection_failed", "Config must be a JSON object")

        def has_quantization(value):
            if isinstance(value, dict):
                return any((key in {"quantization_config", "compression_config"} and bool(item))
                           or has_quantization(item) for key, item in value.items())
            return isinstance(value, list) and any(has_quantization(item) for item in value)

        if has_quantization(config):
            raise LoaderFailure("load", "unsupported_checkpoint_representation",
                                "Quantization/compression config requires a supported model-loading path")

    index = checkpoint_tensor_index(repo_id, revision, allow_bin=True)
    if index["format"] == "pytorch" and any(not zipfile.is_zipfile(path) for path in index["files"].values()):
        raise LoaderFailure("load", "unsupported_checkpoint_representation",
                            "Matrix-only PyTorch analysis requires mmap-compatible ZIP state dicts; "
                            "provide safetensors for older non-ZIP files")
    if any(PACKED_KEY.search(name) for name in index["tensors"]):
        raise LoaderFailure("load", "unsupported_checkpoint_representation",
                            "Packed/adapter tensor markers found; checkpoint mode does not interpret them")
    index["provenance"] = {
        "scope": "checkpoint_tensors", "config_sha256": config_hash,
        "files": sorted(index["files"]), "tensor_count": len(index["tensors"]),
        "format": index["format"],
        "tensor_access": "safetensors" if index["format"] == "safetensors" else "pytorch_mmap",
        "checkpoint_bytes": sum(Path(path).stat().st_size for path in index["files"].values()),
        "dtypes": sorted({entry["dtype"] for entry in index["tensors"].values()}),
        "selection": "stored_floating_2d_tensors", "aliases": "not_inferred",
    }
    return index


@contextmanager
def _tensor_reader(index, filename):
    """Open one inspected file; never eagerly load a legacy PyTorch shard."""
    path = index["files"][filename]
    if index["format"] == "safetensors":
        with safe_open(path, framework="pt", device="cpu") as handle:
            yield handle.get_tensor
    else:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        try:
            yield state.__getitem__
        finally:
            state.clear()  # Release tensors even if the caller retains the reader.


def analyze_checkpoint(index, measurement, *, device="cpu"):
    """Stream matrices into the same numerical core used by loaded models.

    A PyTorch shard stays memory-mapped while its matrices are analyzed; resident
    pages are managed by the OS. No model or full-shard tensor copy is constructed.
    """
    results = {key: [] for key in RESULT_KEYS}
    records = []
    with torch.no_grad():
        for filename in sorted(index["files"]):
            with _tensor_reader(index, filename) as read_tensor:
                for name in sorted(name for name, entry in index["tensors"].items() if entry["file"] == filename):
                    entry = index["tensors"][name]
                    shape = entry["shape"]
                    record = {"name": name, **entry, "status": "skipped", "reason": "",
                              "measurement_names": []}
                    records.append(record)
                    if len(shape) != 2:
                        record["reason"] = "not_a_matrix" if len(shape) < 2 else "undeclared_tensor_layout"
                    elif entry["dtype"] not in FLOAT_DTYPES:
                        record["reason"] = "unsupported_dtype"
                    elif not all(shape):
                        record["reason"] = "empty_tensor"
                    if record["reason"]:
                        continue
                    weight = read_tensor(name)
                    dtype = weight.dtype if measurement["load_dtype"] == "auto" else getattr(torch, measurement["load_dtype"])
                    weight = weight.to(device=device, dtype=dtype)
                    result = compute_esd_for_weight(
                        name, weight, measurement["evals_thresh"], measurement["bins"],
                        None if measurement["fix_fingers"] == "DKS" else measurement["fix_fingers"],
                        measurement["xmin_pos"], measurement["conv_norm"], measurement["filter_zeros"],
                        measurement["use_svd"], measurement["save_eigs"], None, weight.numel(),
                        measurement["compute_dtype"],
                    )
                    del weight  # Do not retain the previous matrix while reading the next one.
                    if result is None:
                        record["reason"] = "no_spectrum"
                        continue
                    if result["longname"] != name:
                        raise ValueError(f"ESD result identity differs from checkpoint key {name!r}")
                    # There is no known module tree. Preserve the exact key as
                    # weight_attribute as well as longname; do not parse its dots.
                    result.update(module_name="", weight_attribute=name, slice="")
                    for key in results:
                        results[key].append(result.get(key, float("nan")))
                    record.update(status="analyzed", measurement_names=[name],
                                  fit_status=result["fit_status"], source_dtype=result["source_dtype"])
    analyzed = len(results["longname"])
    coverage = {
        "scope": "checkpoint_tensors", "tensors": records,
        "counts": {
            "stored_tensors": len(records), "analyzed_tensors": analyzed,
            "skipped_tensors": len(records) - analyzed,
            "analyzed_measurements": analyzed,
            "fitted_measurements": sum(status == "fitted" for status in results["fit_status"]),
        },
    }
    return results, coverage
