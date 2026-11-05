"""Pinned model identities and the v0 database table contract."""

import hashlib
import json
from numbers import Integral
import re
from statistics import median

import pandas as pd


MODELS_COLUMNS = [
    "model_key", "repo_id", "revision", "access", "weight_format", "is_adapter",
    "license", "gated", "declared_bases", "base_model_relation", "architecture",
    "is_decision", "system_one", "decision_type", "num_labels", "labels",
    "arch_signature", "n_views", "modality", "pipeline_tag", "downloads", "params",
    "stratum", "inclusion_prob", "dup_group", "row_block_fingerprint", "full_weight_fingerprint",
    "full_weight_spectra", "coverage_status", "skipped_tensors",
]
VIEWS_COLUMNS = [
    "model_key", "view_index", "name", "geometry", "component", "n_rows", "n_cols",
    "r0", "r1", "block_sha256", "tier",
]
COVERAGE_COLUMNS = ["model_key", "tensor_name", "decode", "geometry", "status", "reason"]
LINEAGE_COLUMNS = [
    "child_key", "parent_key", "estimand", "edge_type", "evidence_tier", "evidence",
    "score", "p_value", "calibrated", "method_version",
]
SCORES_COLUMNS = [
    "model_key", "repo_id", "benchmark", "skill", "value", "metric", "source",
    "source_revision", "revision_verified", "harness", "n_items",
]
PREDICTIONS_COLUMNS = ["model_key", "skill", "value", "lower", "upper", "method", "inputs"]

LAYERS_COLUMNS = [
    "model_key", "longname", "module_name", "weight_attribute", "slice", "view_name",
    "alpha", "alpha_weighted", "D", "xmin", "xmax", "n_tail", "fit_status",
    "spectral_norm", "log_spectral_norm", "stable_rank", "entropy", "matrix_rank", "num_evals", "h5_index",
]

_CONFIG_FIELDS = (
    "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
    "num_attention_heads", "num_key_value_heads", "num_experts",
)
_PIN = re.compile(r"(?:[0-9a-fA-F]{40}|local:[0-9a-fA-F]{64})")


def model_key(repo_id: str, revision: str) -> str:
    """Keep the complete input identity; mutable revision names are rejected."""
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("A model key requires a nonempty repo_id")
    if not isinstance(revision, str) or _PIN.fullmatch(revision) is None:
        raise ValueError("A model key requires a commit SHA or local content pin")
    return f"{repo_id}@{revision}"


def split_key(key: str) -> tuple[str, str]:
    """Split at the final separator, allowing @ in explicit local paths."""
    if not isinstance(key, str) or "@" not in key:
        raise ValueError("A model key requires repo_id@revision")
    repo_id, revision = key.rsplit("@", 1)
    model_key(repo_id, revision)
    return repo_id, revision


def arch_signature(config: dict, view_shapes: list[tuple[int, int]]) -> str:
    """Hash topology facts, excluding vocabulary size and tall fallback views."""
    if config.get("hidden_size") is not None and config.get("num_hidden_layers") is not None:
        prefix, facts = "cfg:", {field: config.get(field) for field in _CONFIG_FIELDS}
    else:
        shapes = []
        for shape in view_shapes:
            if len(shape) != 2 or any(not isinstance(d, Integral) or isinstance(d, bool) or d < 0 for d in shape):
                raise ValueError("Architecture view shapes must be pairs of nonnegative integers")
            shapes.append(tuple(int(d) for d in shape))
        limit = 8 * median(shape[0] for shape in shapes) if shapes else 0
        prefix, facts = "shape:", sorted(shape for shape in shapes if shape[0] <= limit)
    payload = json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def empty_table(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))
