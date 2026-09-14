"""Shared measurement conventions and read-only resume checks (no torch import)."""

import csv
import json
import math
from pathlib import Path
import re

NUMERICS_VERSION = "6"
FORMAT_VERSION = "2.0"
# Version 6 extends checkpoint-shape selection to the tested encoder families.
# Loading policy is separate from the spectral formulas.
LOADER_VERSION = "6"


def is_commit_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None


def validate_model_pin(model_id, revision="", source_model="", base_model_relation="", loader_scenario=""):
    """Require explicit HF revisions before dispatch or standalone loading."""
    from huggingface_hub.utils import validate_repo_id

    for reference in (model_id, source_model):
        if reference:
            repo_id = reference.partition("@")[0]
            validate_repo_id(repo_id)
            if Path(repo_id).is_dir():
                raise ValueError(f"{repo_id}: a local directory would override the pinned HF repository")
    _, _, embedded_revision = model_id.partition("@")
    if not is_commit_sha(revision or embedded_revision):
        raise ValueError(f"{model_id}: a full commit SHA is required; prepare the model list with --prepare_only first")
    if source_model:
        _, _, base_revision = source_model.partition("@")
        if not is_commit_sha(base_revision):
            raise ValueError(f"{model_id}: source_model must be repo@<full commit SHA>")
    elif base_model_relation.lower() in {"adapter", "lora", "peft"} or loader_scenario == "adapter_requires_base":
        raise ValueError(f"{model_id}: adapters require a pinned source_model; use --prepare_only to resolve it")


def measurement_config(args, *, model_id=None, revision="", source_model="",
                       base_model_relation="", loader_scenario=""):
    """Build the same configuration in the dispatcher and standalone worker."""
    config = {
        "numerics_version": NUMERICS_VERSION,
        "loader_version": LOADER_VERSION,
        "fix_fingers": getattr(args, "fix_fingers", "xmin_mid") or "DKS",
        "evals_thresh": float(getattr(args, "evals_thresh", 1e-5)),
        "bins": int(getattr(args, "bins", 100)),
        "filter_zeros": bool(getattr(args, "filter_zeros", True)),
        "use_svd": bool(getattr(args, "use_svd", True)),
        "cuda_svd_driver": "gesvd",
        "save_eigs": bool(getattr(args, "save_eigs", True)),
        "load_dtype": getattr(args, "load_dtype", "auto"),
        "compute_dtype": getattr(args, "compute_dtype", "float32"),
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
        "xmin_pos": 2,
        "conv_norm": 0.5,
        "filter_type": True,
        "spectrum_storage": "full",
    }
    if not math.isfinite(config["evals_thresh"]) or config["evals_thresh"] < 0:
        raise ValueError("evals_thresh must be finite and >= 0")
    if config["bins"] < 1:
        raise ValueError("bins must be >= 1")
    if model_id is not None:
        repo_id, _, embedded_revision = str(model_id).partition("@")
        config.update(
            repo_id=repo_id.strip(),
            requested_revision=str(revision or embedded_revision or "main").strip(),
            source_model=str(source_model or "").strip(),
            base_model_relation=str(base_model_relation or "").strip(),
            loader_scenario=str(loader_scenario or "").strip(),
        )
    return config


def artifact_compatibility(csv_path: Path, h5_path: Path, expected=None):
    """Return (compatible, reason); never modify or accept unreadable artifacts.

    Extra recorded runtime provenance is permitted. All requested settings must
    match. This checks schema/alignment, not a checksum of every spectral value.
    """
    import h5py

    try:
        if not csv_path.is_file() or not h5_path.is_file():
            return False, "missing CSV/HDF5 pair"
        if csv_path.stat().st_size == 0:
            return False, "empty CSV"
        with h5py.File(h5_path, "r") as h5:
            if str(h5.attrs.get("numerics_version", "")) != NUMERICS_VERSION:
                return False, "different or missing numerics_version"
            if str(h5.attrs.get("format_version", "")) != FORMAT_VERSION:
                return False, "different or missing format_version"
            config = json.loads(h5.attrs.get("measurement_config_json", "null"))
            if not isinstance(config, dict) or config.get("numerics_version") != NUMERICS_VERSION:
                return False, "missing measurement configuration"
            if config.get("loader_version") != LOADER_VERSION:
                return False, "different or missing loader_version"
            for key, value in (expected or {}).items():
                if config.get(key) != value:
                    return False, f"measurement setting differs: {key}"
            layers = h5.get("layers")
            if not isinstance(layers, h5py.Group) or "longname" not in layers or "alpha" not in layers:
                return False, "missing canonical layer records"
            n_layers = len(layers["longname"])
            if n_layers == 0 or any(dset.shape != (n_layers,) for dset in layers.values()):
                return False, "unaligned canonical layer records"
            if "eigs" in h5 and h5["eigs"].shape != (n_layers,):
                return False, "unaligned saved spectra"
            if config.get("save_eigs") and "eigs" not in h5:
                return False, "missing requested spectra"
            names = layers["longname"].asstr()[:].tolist()
            if len(set(names)) != n_layers:
                return False, "duplicate canonical layer identities"
            alphas = layers["alpha"][:]
            full_name = h5.attrs.get("full_name")
            if isinstance(full_name, bytes):
                full_name = full_name.decode("utf-8")

            # CSV is streamed: only canonical HDF5 names and alphas are read,
            # never the (potentially very large) eigenvalue datasets.
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                columns = reader.fieldnames or []
                required = {"longname", "alpha"}
                if full_name is not None:
                    required.add("model_id")
                if not required.issubset(columns) or len(columns) != len(set(columns)):
                    return False, "missing or duplicate CSV identity/alpha columns"
                row_count = 0
                for index, row in enumerate(reader):
                    if index >= n_layers:
                        return False, "CSV/HDF5 row count differs"
                    if None in row or any(row.get(key) is None for key in required):
                        return False, "malformed CSV layer row"
                    if row["longname"] != names[index]:
                        return False, "CSV/HDF5 layer identities or order differ"
                    if full_name is not None and row["model_id"] != full_name:
                        return False, "CSV/HDF5 model identity differs"
                    csv_alpha = float(row["alpha"]) if row["alpha"].strip() else math.nan
                    h5_alpha = float(alphas[index])
                    if not ((math.isnan(csv_alpha) and math.isnan(h5_alpha)) or
                            math.isclose(csv_alpha, h5_alpha, rel_tol=1e-12, abs_tol=0.0)):
                        return False, "CSV/HDF5 alpha values differ"
                    row_count += 1
                if row_count != n_layers:
                    return False, "CSV/HDF5 row count differs"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error) as exc:
        return False, f"unreadable or invalid CSV/HDF5: {type(exc).__name__}"
    return True, "compatible"
