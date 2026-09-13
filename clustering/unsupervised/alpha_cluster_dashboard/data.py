from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import warnings

import h5py
import numpy as np
import pandas as pd

COMMON_ALPHA_MODULES = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)

CANONICAL_MODULE_ORDER = (
    "mlp_down",
    "mlp_gate",
    "mlp_up",
    "mlp_gate_up",
    "attn_k",
    "attn_o",
    "attn_q",
    "attn_qkv",
    "attn_v",
    "cross_attn_k",
    "cross_attn_o",
    "cross_attn_q",
    "cross_attn_v",
)

MODULE_ROLE_MAP = {
    "mlp.down_proj": "mlp_down",
    "output.dense": "mlp_down",
    "feed_forward.w2": "mlp_down",
    "mlp.fc2": "mlp_down",
    "mlp.gate_proj": "mlp_gate",
    "feed_forward.w1": "mlp_gate",
    "mlp.up_proj": "mlp_up",
    "intermediate.dense": "mlp_up",
    "feed_forward.w3": "mlp_up",
    "mlp.fc1": "mlp_up",
    "mlp.gate_up_proj": "mlp_gate_up",
    "self_attn.k_proj": "attn_k",
    "attention.self.key": "attn_k",
    "self_attn.o_proj": "attn_o",
    "self_attn.out_proj": "attn_o",
    "attention.output.dense": "attn_o",
    "attention.wo": "attn_o",
    "self_attn.q_proj": "attn_q",
    "attention.self.query": "attn_q",
    "attention.wqkv": "attn_qkv",
    "self_attn.v_proj": "attn_v",
    "attention.self.value": "attn_v",
    "cross_attn.k_proj": "cross_attn_k",
    "cross_attn.o_proj": "cross_attn_o",
    "cross_attn.q_proj": "cross_attn_q",
    "cross_attn.v_proj": "cross_attn_v",
}

CANONICAL_MODULE_LABELS = {
    "mlp_down": "MLP down",
    "mlp_gate": "MLP gate",
    "mlp_up": "MLP up",
    "mlp_gate_up": "MLP gate+up",
    "attn_k": "Attention K",
    "attn_o": "Attention O",
    "attn_q": "Attention Q",
    "attn_qkv": "Attention QKV",
    "attn_v": "Attention V",
    "cross_attn_k": "Cross-attn K",
    "cross_attn_o": "Cross-attn O",
    "cross_attn_q": "Cross-attn Q",
    "cross_attn_v": "Cross-attn V",
}


@dataclass(frozen=True)
class AlphaRecord:
    model_id: str
    display_name: str
    path: str
    num_layers: int
    raw_modules: tuple[str, ...]
    alpha: np.ndarray
    view_status: str = "legacy"

    @property
    def common_schema(self) -> bool:
        return self.raw_modules == COMMON_ALPHA_MODULES

    @property
    def raw_module_count(self) -> int:
        return len(self.raw_modules)


def canonicalize_module_name(module_name: str) -> str:
    return MODULE_ROLE_MAP.get(module_name, f"other::{module_name}")


def list_metric_files(metrics_dir: str | Path) -> list[Path]:
    return sorted(Path(metrics_dir).expanduser().glob("*.h5"))


def load_alpha_records(metrics_dir: str | Path) -> list[AlphaRecord]:
    """Read complete depth views, not arbitrary canonical layer collections.

    Older root-only files remain usable, but their layer coverage is unknown.
    This structural check does not validate historical numerical conventions.
    """
    records: list[AlphaRecord] = []
    for path in list_metric_files(metrics_dir):
        try:
            with h5py.File(path, "r") as handle:
                dataset = handle.get("alpha")
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError("missing /alpha dataset")
                view_status = dataset.attrs.get("view_status")
                if isinstance(view_status, bytes):
                    view_status = view_status.decode("utf-8")
                canonical = ("layers" in handle or "canonical_records" in dataset.attrs
                             or str(handle.attrs.get("format_version", "")) == "2.0")
                if canonical:
                    if not isinstance(handle.get("layers"), h5py.Group):
                        raise ValueError("missing canonical /layers records")
                    if view_status != "complete":
                        raise ValueError(f"/alpha view_status is {view_status or 'missing'}")
                    if int(dataset.attrs.get("unmapped_layer_count", 0)) != 0:
                        raise ValueError("/alpha omits canonical layers")
                elif view_status not in (None, "complete"):
                    raise ValueError(f"/alpha view_status is {view_status}")
                else:
                    view_status = "legacy"
                modules = json.loads(dataset.attrs["module_names_json"])
                if (not isinstance(modules, list)
                        or not all(isinstance(module, str) for module in modules)
                        or len(set(modules)) != len(modules)):
                    raise ValueError("invalid or duplicate /alpha module names")
                num_layers = int(dataset.attrs["num_layers"])
                if num_layers <= 0 or not modules or dataset.shape != (num_layers, len(modules)):
                    raise ValueError("/alpha shape does not match its layer/module labels")
                modules = tuple(modules)
                alpha = np.asarray(dataset[:], dtype=float)
                alpha[~np.isfinite(alpha)] = np.nan
        except (OSError, ValueError, TypeError, KeyError) as exc:
            warnings.warn(
                f"Skipping {path.name}: {exc}. The dashboard requires a complete "
                "depth-by-module /alpha view; use canonical /layers for general analysis.",
                stacklevel=2,
            )
            continue
        stem = path.stem
        records.append(
            AlphaRecord(
                model_id=stem,
                display_name=stem.replace("--", "/"),
                path=str(path),
                num_layers=num_layers,
                raw_modules=modules,
                alpha=alpha,
                view_status=view_status,
            )
        )
    legacy_count = sum(record.view_status == "legacy" for record in records)
    if legacy_count:
        warnings.warn(
            f"Loaded {legacy_count} legacy root-only alpha views. Their layer identity, "
            "coverage and numerical conventions are unverified (view_status='legacy').",
            stacklevel=2,
        )
    return records


def schema_name(record: AlphaRecord) -> str:
    if record.common_schema:
        return "common_llm_7"
    canonical_modules = sorted({canonicalize_module_name(module) for module in record.raw_modules})
    return " + ".join(canonical_modules)


def records_to_frame(records: Iterable[AlphaRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "model_id": record.model_id,
                "display_name": record.display_name,
                "path": record.path,
                "num_layers": record.num_layers,
                "raw_module_count": record.raw_module_count,
                "raw_modules": ", ".join(record.raw_modules),
                "common_schema": record.common_schema,
                "schema_name": schema_name(record),
                "view_status": record.view_status,
            }
        )
    return pd.DataFrame(rows)


def available_modules(records: Iterable[AlphaRecord], schema_mode: str) -> list[str]:
    seen: dict[str, int] = {}
    for record in records:
        if schema_mode == "common":
            if not record.common_schema:
                continue
            modules = record.raw_modules
        else:
            modules = tuple(canonicalize_module_name(module) for module in record.raw_modules)
        for module in modules:
            seen[module] = seen.get(module, 0) + 1

    if schema_mode == "common":
        ordered = [module for module in COMMON_ALPHA_MODULES if module in seen]
    else:
        ordered = [module for module in CANONICAL_MODULE_ORDER if module in seen]
        ordered.extend(sorted(module for module in seen if module not in ordered))
    return ordered


def module_label(module_name: str) -> str:
    return CANONICAL_MODULE_LABELS.get(module_name, module_name.replace("_", " ").replace("::", ": "))
