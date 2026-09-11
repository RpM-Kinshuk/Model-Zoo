#!/usr/bin/env python3
"""Small, pinned public-checkpoint measurement/storage pilot; CPU only.

This is a numerical and coverage smoke test, not predictive validation. It
uses explicit built-in HF classes, so automatic loader routing is not tested.
Only config.json and pytorch_model.bin are downloaded (7.5 MB total); model
loading is local-only with remote code disabled. Reuse a cache with --cache-dir
and --offline while publishing to a fresh --output-dir.
"""

import argparse
from datetime import datetime, timezone
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECS = [
    ("encoder", "hf-internal-testing/tiny-random-BertModel",
     "fc08ad9cc33be9aef4f55cc80e16ef5ae3d5981c", "BertModel", 384112),
    ("decoder", "sshleifer/tiny-gpt2",
     "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be", "GPT2LMHeadModel", 2514808),
    ("encoder_decoder", "hf-internal-testing/tiny-random-T5ForConditionalGeneration",
     "b12e41904f9f5c33909e8f6a21cf638eca966e74", "T5ForConditionalGeneration", 4489578),
    ("cnn", "hf-internal-testing/tiny-random-ResNetModel",
     "fafa6cdf9986c6cfbae360596b3574162430bcd3", "ResNetModel", 106926),
]
FILES = ["config.json", "pytorch_model.bin"]


def json_safe(value):
    """Write missing numerical values as standard JSON null, never NaN tokens."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        value = sorted(value)
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def download_snapshot(spec, cache_dir, offline=False):
    from huggingface_hub import HfApi, snapshot_download

    _, repo, revision, _, expected_bytes = spec
    if not offline:
        info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True, timeout=30)
        sizes = {item.rfilename: item.size for item in info.siblings}
        if info.private or info.gated or info.sha != revision:
            raise ValueError("Pilot requires the pinned public, ungated checkpoint")
        if any(name not in sizes for name in FILES) or sum(sizes[name] for name in FILES) != expected_bytes:
            raise ValueError("Pinned checkpoint file manifest changed")
    snapshot = Path(snapshot_download(
        repo, revision=revision, allow_patterns=FILES, cache_dir=str(cache_dir),
        token=False, local_files_only=offline, max_workers=1,
    ))
    if sum((snapshot / name).stat().st_size for name in FILES) != expected_bytes:
        raise ValueError("Downloaded checkpoint does not match the expected byte count")
    if json.loads((snapshot / "config.json").read_text()).get("auto_map"):
        raise ValueError("Remote-code configurations are outside this pilot")
    return snapshot


def reference_spectrum(weight, ww_svd_vals):
    """Use WW's accurate SVD on explicit float64 arrays, not its dtype adapter."""
    import numpy as np

    array = weight.detach().cpu().double().numpy()
    if weight.ndim == 2:
        matrices = [array]
    else:
        matrices = array.reshape(*array.shape[:2], -1).transpose(2, 0, 1) * math.sqrt(.5)
    return np.sort(np.concatenate([ww_svd_vals(matrix, method="accurate") ** 2 for matrix in matrices]))


def compare_weight(name, weight, ww_svd_vals):
    import numpy as np
    from net_esd.core import compute_esd_for_weight

    reference = reference_spectrum(weight, ww_svd_vals)
    denominator = max(float(reference.max()), np.finfo(np.float64).tiny)
    variants = [
        ("float32_svd", weight, "float32", True, True, "xmin_mid"),
        ("float64_svd", weight, "float64", True, True, "xmin_mid"),
        ("float32_gram", weight, "float32", False, True, "xmin_mid"),
        ("float16_roundtrip", weight.half(), "float32", True, True, "xmin_mid"),
        ("unfiltered", weight, "float32", True, False, "xmin_mid"),
        ("dks", weight, "float32", True, True, None),
        ("peak", weight, "float32", True, True, "xmin_peak"),
    ]
    comparisons = {}
    for label, tensor, dtype, svd, filtered, method in variants:
        started = time.perf_counter()
        result = compute_esd_for_weight(
            name, tensor, 1e-5, 100, method, 2, .5, filtered, svd, True,
            None, tensor.numel(), dtype,
        )
        eigs = result.pop("eigs")
        result["spectrum_linf_relative_to_ww64"] = float(np.max(np.abs(eigs - reference)) / denominator)
        result["seconds"] = time.perf_counter() - started
        comparisons[label] = result
    return {"longname": name, "shape": list(weight.shape), "variants": comparisons}


def synthetic_controls(ww_svd_vals):
    import numpy as np
    import torch

    rng = np.random.default_rng(123)
    pareto = (1 - rng.random(64)) ** (-1 / 1.5)  # Known generating density alpha=2.5.
    controls = {
        "pareto_alpha_2_5": torch.diag(torch.from_numpy(np.sqrt(pareto))),
        "lognormal": torch.diag(torch.from_numpy(np.sqrt(np.exp(rng.normal(size=64))))),
        "gaussian_matrix": torch.from_numpy(rng.normal(size=(64, 32))),
        "near_constant": torch.diag(.84 + torch.arange(8, dtype=torch.float32) * 1e-7),
        "low_rank": torch.diag(torch.tensor([0., 0., 0., 0., 0., 0., 1., 2.])),
    }
    return [compare_weight(name, weight, ww_svd_vals) for name, weight in controls.items()]


def analyze_checkpoint(spec, snapshot, output_dir, ww_svd_vals):
    import h5py
    import numpy as np
    import pandas as pd
    import transformers
    from measurement_config import artifact_compatibility, measurement_config
    from model_loader import hf_from_pretrained, safe_filename
    from net_esd import net_esd_estimator
    from net_esd.utils import iter_eligible_layers
    from worker import coverage_report, runtime_provenance, save_results

    family, repo, revision, class_name, expected_bytes = spec
    started = time.perf_counter()
    model, loading_info = hf_from_pretrained(
        getattr(transformers, class_name), str(snapshot), trust_remote_code=False,
        local_files_only=True, torch_dtype="auto", device_map="cpu", output_loading_info=True,
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise ValueError("Checkpoint loading left missing, mismatched, or erroneous weights")
    model.eval()
    loaded_at = time.perf_counter()
    records = []
    metrics = net_esd_estimator(
        model, fix_fingers="xmin_mid", filter_zeros=True, use_svd=True,
        save_eigs=True, parallel=False, compute_dtype="float32", coverage=records,
    )
    analyzed_at = time.perf_counter()
    report = coverage_report(records, metrics)
    args = SimpleNamespace(
        fix_fingers="xmin_mid", evals_thresh=1e-5, bins=100, filter_zeros=True,
        use_svd=True, save_eigs=True, load_dtype="auto", compute_dtype="float32",
        device_map="cpu", parallel_esd=False,
    )
    config = measurement_config(args, model_id=repo, revision=revision)
    config.update(runtime_provenance(model, args))
    config.update(resolved_revision=revision, explicit_loader_class=class_name,
                  trust_remote_code=False, pilot=True)
    csv_path = output_dir / "stats" / f"{safe_filename(repo)}.csv"
    h5_path = output_dir / "metrics" / f"{safe_filename(repo)}.h5"
    if csv_path.exists() or h5_path.exists():
        raise FileExistsError("Pilot will not overwrite existing model artifacts")
    save_results(metrics, csv_path, repo, False, fix_fingers="xmin_mid",
                 h5_output_path=h5_path, save_eigs=True, measurement_config=config, coverage=report)
    compatible, reason = artifact_compatibility(csv_path, h5_path, config)
    if not compatible:
        raise AssertionError(f"Published artifacts failed compatibility validation: {reason}")
    frame = pd.read_csv(csv_path, keep_default_na=False)
    with h5py.File(h5_path, "r") as h5:
        names = h5["layers/longname"].asstr()[:].tolist()
        assert names == metrics["longname"] == frame["longname"].tolist()
        assert len(names) == len(set(names))
        for index, expected in enumerate(metrics["eigs"]):
            np.testing.assert_array_equal(h5["eigs"][index], expected)

    eligible = list(iter_eligible_layers(model))
    indices = sorted({0, len(eligible) // 2, len(eligible) - 1})
    representatives = [compare_weight(eligible[i][0], eligible[i][1], ww_svd_vals) for i in indices]
    return {
        "family": family, "repo_id": repo, "revision": revision,
        "loader_class": class_name, "checkpoint_bytes": expected_bytes,
        "loading_info": loading_info,
        "coverage": report, "artifacts_checked": True,
        "csv": str(csv_path), "hdf5": str(h5_path),
        "timing_seconds": {"load": loaded_at - started, "analyze": analyzed_at - loaded_at,
                           "total": time.perf_counter() - started},
        "representative_layers": representatives,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh pilot output directory")
    parser.add_argument("--cache-dir", type=Path, help="Reuse a pilot cache; defaults to OUTPUT/cache")
    parser.add_argument("--weightwatcher-path", type=Path, default=PROJECT_ROOT.parent / "WeightWatcher")
    parser.add_argument("--offline", action="store_true", help="Require already downloaded pinned snapshots")
    parser.add_argument("--download-only", action="store_true", help="Populate cache without loading models")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    cache = args.cache_dir.resolve() if args.cache_dir else output / "cache"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "pilot_report.json"
    if report_path.exists():
        parser.error("pilot_report.json already exists; use a fresh output directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(cache)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "esd_experiment/src")]
    snapshots = [download_snapshot(spec, cache, args.offline) for spec in SPECS]
    if args.download_only:
        print(json.dumps({"downloaded_bytes": sum(spec[-1] for spec in SPECS), "snapshots": list(map(str, snapshots))}))
        return 0

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ww_path = args.weightwatcher_path.resolve()
    sys.path.insert(0, str(ww_path))
    ww = importlib.import_module("weightwatcher")
    if not Path(ww.__file__).resolve().is_relative_to(ww_path):
        raise RuntimeError("Did not import the requested local WeightWatcher checkout")
    from weightwatcher.RMT_Util import svd_vals
    ww_commit = subprocess.check_output(["git", "-C", str(ww_path), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": 123,
        "scope": "CPU numerical/coverage/storage pilot, not predictive validation or automatic loader-routing validation",
        "weightwatcher": {"path": str(ww_path), "version": ww.__version__, "commit": ww_commit,
                          "reference": "accurate SVD on explicit float64 arrays; no legacy D or entropy comparison"},
        "checkpoint_bytes_total": sum(spec[-1] for spec in SPECS),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "net_esd/core.py", "net_esd/utils.py", "net_esd/__init__.py", "net_esd/constants.py",
                "esd_experiment/src/worker.py", "esd_experiment/src/measurement_config.py",
                "esd_experiment/scripts/measurement_pilot.py",
            )
        },
        "models": [], "errors": [], "controls": [],
    }
    for spec, snapshot in zip(SPECS, snapshots):
        try:
            result = analyze_checkpoint(spec, snapshot, output, svd_vals)
            report["models"].append(result)
            print(json.dumps({"repo_id": spec[1], "coverage": result["coverage"]["counts"]}), flush=True)
        except Exception as error:
            report["errors"].append({"repo_id": spec[1], "type": type(error).__name__, "message": str(error)})
    report["controls"] = synthetic_controls(svd_vals)
    report_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(f"Pilot report: {report_path}", flush=True)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
