#!/usr/bin/env python3
"""Bounded, offline matrix-scale checks of the measurement backend.

Example: python backend_pilot.py --output-dir /tmp/backend-cpu --device cpu
Run the same command with a fresh directory and --device cuda:0 on a GPU host.
Every case runs in a separate, time-limited process with one CPU thread. There
are no checkpoint downloads. These controls test measurement, not model quality.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 123
SPECTRUM_TOLERANCE = 1e-4
CASES = {
    "attention_768": {"shape": (768, 768), "kind": "shifted_gaussian", "well_conditioned": True},
    "mlp_3072x768": {"shape": (3072, 768), "kind": "gaussian", "well_conditioned": True},
    "embedding_4096x256": {"shape": (4096, 256), "kind": "gaussian", "well_conditioned": True},
    "projection_2048": {"shape": (2048, 2048), "kind": "shifted_gaussian", "well_conditioned": True},
    "low_rank": {"shape": (128, 128), "kind": "repeated_rows", "well_conditioned": False},
    "ill_conditioned": {"shape": (128, 128), "kind": "geometric_singular_values", "well_conditioned": False},
    "near_constant": {"shape": (8, 8), "kind": "near_constant", "well_conditioned": False},
    "filter_boundary": {"shape": (4, 4), "kind": "small_diagonal", "well_conditioned": False},
}
BASE_VARIANTS = [
    ("float32_svd", "float32", True, True, "xmin_mid"),
    ("float32_gram", "float32", False, True, "xmin_mid"),
    ("float64_svd", "float64", True, True, "xmin_mid"),
]
EXTRA_VARIANTS = [
    ("unfiltered_svd", "float32", True, False, "xmin_mid"),
    ("dks_svd", "float32", True, True, None),
    ("peak_svd", "float32", True, True, "xmin_peak"),
]
METRICS = ("alpha", "D", "fit_xmin", "n_tail", "matrix_rank", "entropy", "num_evals", "norm",
           "raw_matrix_rank", "raw_entropy", "raw_num_evals", "raw_norm")


def source_hashes():
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in (
        "net_esd/core.py", "net_esd/utils.py", "net_esd/constants.py",
        "esd_experiment/src/measurement_config.py", "esd_experiment/scripts/backend_pilot.py",
    )}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_report(path, report):
    # Exclusive creation also protects child reports from accidental reuse.
    with Path(path).open("x") as output:
        json.dump(json_safe(report), output, indent=2, allow_nan=False)
        output.write("\n")


def validate_device(device, torch):
    if device == "cpu":
        return None
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("device must be cpu or cuda:N with an explicit nonnegative index")
    index = int(device[5:])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"Requested {device} is unavailable; refusing a CPU fallback")
    return index


def configure_runtime(device):
    # Set these before importing numerical libraries in a fresh child process.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch

    index = validate_device(device, torch)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    if index is not None:
        torch.cuda.set_device(index)
    return torch, index


def build_matrix(case_name):
    """All backends receive identical float32 source weights, then cast explicitly."""
    import numpy as np

    spec = CASES[case_name]
    rows, columns = spec["shape"]
    rng = np.random.default_rng(SEED)
    kind = spec["kind"]
    if kind in ("gaussian", "shifted_gaussian"):
        matrix = rng.standard_normal((rows, columns), dtype=np.float32) / math.sqrt(rows)
        if kind == "shifted_gaussian":
            matrix += 3 * np.eye(rows, dtype=np.float32)
    elif kind == "repeated_rows":
        # Repeated rows give exact algebraic rank <= 32 in the stored weights.
        matrix = np.tile(rng.standard_normal((32, columns), dtype=np.float32), (4, 1)) / math.sqrt(rows)
    elif kind == "geometric_singular_values":
        q, _ = np.linalg.qr(rng.standard_normal((rows, rows)))
        matrix = (q * np.geomspace(1, 1e-8, rows)) @ q.T
    elif kind == "near_constant":
        matrix = np.diag(np.float32(.84) + np.arange(rows, dtype=np.float32) * np.float32(1e-7))
    else:
        matrix = np.diag(np.array([.001, .002, .003, .004], dtype=np.float32))
    return np.ascontiguousarray(matrix, dtype=np.float32)


def reference_fit(eigenvalues, cutoff):
    """Independent NumPy float64 Pareto MLE/full KS; no core fitter is reused."""
    import numpy as np

    values = eigenvalues[eigenvalues > 0]
    n = len(values)
    if n < 2 or values[0] == values[-1]:
        return {"alpha": math.nan, "D": math.nan, "fit_xmin": math.nan, "n_tail": 0}
    if cutoff == "xmin_mid":
        candidates = [int(np.searchsorted(values, values[min(n // 2, n - 2)]))]
    else:
        candidates = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])[:-1]
        if cutoff == "xmin_peak":
            counts, boundaries = np.histogram(np.log10(values), bins=100)
            peak = 10 ** boundaries[np.argmax(counts)]
            candidates = [i for i in candidates if .95 * peak <= values[i] <= 1.5 * peak]
    best = {"alpha": math.nan, "D": math.nan, "fit_xmin": math.nan, "n_tail": 0}
    for start in candidates:
        tail = values[start:]
        if tail[0] == tail[-1]:
            continue
        ratios = np.log1p((tail - tail[0]) / tail[0])
        alpha = 1 + len(tail) / float(ratios.sum())
        cdf = -np.expm1((1 - alpha) * ratios)
        d = max(float(np.max(np.arange(1, len(tail) + 1) / len(tail) - cdf)),
                float(np.max(cdf - np.arange(len(tail)) / len(tail))))
        if not math.isfinite(best["D"]) or d < best["D"]:
            best = {"alpha": alpha, "D": d, "fit_xmin": float(tail[0]), "n_tail": len(tail)}
    return best


def rank_and_entropy(values, shape, dtype):
    import numpy as np

    rank = 0
    entropy = math.nan
    if len(values):
        tolerance = math.sqrt(float(values[-1])) * max(shape) * np.finfo(dtype).eps
        rank = int(np.count_nonzero(np.sqrt(values) > tolerance))
        if rank == 1:
            entropy = 0.0
        elif rank > 1:
            probabilities = values[-rank:] / values[-rank:].sum()
            entropy = float(-np.sum(probabilities * np.log(probabilities)) / math.log(rank))
    return rank, entropy


def reference_metrics(eigenvalues, shape, dtype, filtered, cutoff):
    """Use each variant's rank tolerance, separating dtype policy from SVD error."""
    values = eigenvalues[eigenvalues > 1e-5] if filtered else eigenvalues
    rank, entropy = rank_and_entropy(values, shape, dtype)
    raw_rank, raw_entropy = rank_and_entropy(eigenvalues, shape, dtype)
    return {**reference_fit(values, cutoff), "matrix_rank": rank, "entropy": entropy,
            "num_evals": len(values), "norm": float(values.sum()),
            "raw_matrix_rank": raw_rank, "raw_entropy": raw_entropy,
            "raw_num_evals": len(eigenvalues), "raw_norm": float(eigenvalues.sum())}


def metric_deltas(measured, reference):
    return {
        key: float(measured[key] - reference[key])
        if math.isfinite(measured[key]) and math.isfinite(reference[key]) else None
        for key in METRICS
    }


def sensitivity_flags(measured, reference):
    flags = []
    for key in ("n_tail", "matrix_rank", "num_evals", "raw_matrix_rank", "raw_num_evals"):
        if measured[key] != reference[key]:
            flags.append(f"{key}_changed")
    if math.isfinite(measured["alpha"]) != math.isfinite(reference["alpha"]):
        flags.append("fit_availability_changed")
    elif math.isfinite(measured["alpha"]):
        if abs(measured["alpha"] - reference["alpha"]) > .01 * abs(reference["alpha"]):
            flags.append("alpha_changed_over_one_percent")
    for key in ("entropy", "raw_entropy"):
        if math.isfinite(measured[key]) != math.isfinite(reference[key]):
            flags.append(f"{key}_availability_changed")
        elif math.isfinite(measured[key]) and abs(measured[key] - reference[key]) > 1e-3:
            flags.append(f"{key}_changed_over_0.001")
    return flags


def run_case(case_name, device, torch, device_index):
    import numpy as np
    import scipy.linalg
    from net_esd.core import compute_esd_for_weight

    started = time.perf_counter()
    source_before = source_hashes()
    array = build_matrix(case_name)
    weight = torch.from_numpy(array)
    reference_started = time.perf_counter()
    spectrum = np.sort(scipy.linalg.svdvals(array.astype(np.float64), check_finite=True) ** 2)
    reference_seconds = time.perf_counter() - reference_started
    scale = max(float(spectrum[-1]), np.finfo(np.float64).tiny)
    report = {
        "case": case_name, **CASES[case_name], "source_dtype": "float32",
        "source_sha256": source_before,
        "weight_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "reference": {"method": "scipy.linalg.svdvals on explicit float64 CPU array",
                      "seconds": reference_seconds, "eigenvalue_count": len(spectrum),
                      "lambda_min": float(spectrum[0]), "lambda_max": float(spectrum[-1])},
        "variants": {},
    }
    variants = BASE_VARIANTS + (EXTRA_VARIANTS if case_name == "attention_768" or not CASES[case_name]["well_conditioned"] else [])
    for label, dtype, svd, filtered, cutoff in variants:
        if device_index is not None:
            torch.cuda.synchronize(device_index)
            torch.cuda.reset_peak_memory_stats(device_index)
        variant_started = time.perf_counter()
        result = compute_esd_for_weight(
            case_name, weight, 1e-5, 100, cutoff, 2, .5, filtered, svd, True,
            device_index, weight.numel(), dtype,
        )
        if device_index is not None:
            torch.cuda.synchronize(device_index)
        seconds = time.perf_counter() - variant_started
        if result is None or result["compute_device"] != device:
            raise AssertionError("Core did not measure the requested matrix on the requested device")
        eigs = result.pop("eigs").astype(np.float64)
        if eigs.shape != spectrum.shape or not np.isfinite(eigs).all() or np.any(eigs < 0):
            raise AssertionError("Invalid spectrum shape, sign, or finiteness")
        reference = reference_metrics(spectrum, array.shape, dtype, filtered, cutoff)
        error = float(np.max(np.abs(eigs - spectrum)))
        result.update({
            "seconds": seconds,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device_index) if device_index is not None else None,
            "spectrum_max_abs_error": error,
            "spectrum_max_abs_error_over_reference_lambda_max": error / scale,
            "lambda_max_relative_error": abs(float(eigs[-1]) - float(spectrum[-1])) / scale,
            "reference_metrics": reference,
            "deltas_from_matching_reference": metric_deltas(result, reference),
            "reference_sensitivity_flags": sensitivity_flags(result, reference),
            "spectrum_check_passed": error / scale <= SPECTRUM_TOLERANCE if CASES[case_name]["well_conditioned"] else None,
        })
        baseline = report["variants"].get("float32_svd", result)
        result["deltas_from_float32_svd_baseline"] = metric_deltas(result, baseline)
        result["baseline_sensitivity_flags"] = sensitivity_flags(result, baseline)
        report["variants"][label] = result
    report["seconds"] = time.perf_counter() - started
    if source_hashes() != source_before:
        raise RuntimeError("Measurement source changed during this case; rerun with stable sources")
    report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    report["status"] = "failed" if any(row["spectrum_check_passed"] is False for row in report["variants"].values()) else (
        "passed_spectrum_checks" if CASES[case_name]["well_conditioned"] else "sensitivity_control_completed"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="A new directory; existing directories are refused")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N; no fallback")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--case-timeout", type=float, default=180, help="Seconds per subprocess, including imports (default: 180)")
    parser.add_argument("--_case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.case_timeout <= 0 or not math.isfinite(args.case_timeout):
        parser.error("--case-timeout must be a positive finite number")
    if args._case is None and args.output_dir is None:
        parser.error("--output-dir is required")
    if args._case is not None and args._result is None:
        parser.error("Internal case execution requires a result path")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("Output directory already exists; use a fresh path")
    try:
        torch, device_index = configure_runtime(args.device)
    except ValueError as error:
        parser.error(str(error))
    if args._case is not None:
        write_report(args._result, run_case(args._case, args.device, torch, device_index))
        return 0

    import numpy as np
    import scipy
    # Import the lightweight file directly, not src/__init__ (which loads HF).
    sys.path.insert(0, str(PROJECT_ROOT / "esd_experiment/src"))
    from measurement_config import measurement_config

    conventions = measurement_config(SimpleNamespace())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline matrix-scale measurement checks; not checkpoint coverage or predictive validation",
        "configuration": {"device": args.device, "seed": SEED, "cpu_threads": 1, "cases": args.cases,
                          "numerics_version": conventions["numerics_version"],
                          "cuda_svd_driver": conventions["cuda_svd_driver"],
                          "case_timeout_seconds": args.case_timeout, "spectrum_tolerance": SPECTRUM_TOLERANCE,
                          "evals_thresh": 1e-5, "bins": 100, "xmin_pos": 2,
                          "sensitivity_alpha_relative_threshold": .01, "sensitivity_entropy_absolute_threshold": 1e-3},
        "interpretation": [
            "Only well-conditioned controls have pass/fail spectrum tolerances; sensitivity flags are not automatic failures.",
            "The spectrum reference is independent SciPy float64 CPU SVD of the identical float32 source weights.",
            "Reference rank/entropy use the compared variant's dtype tolerance; float64 versus float32 rank policy can differ.",
            "Peak fitting uses each backend's histogram precision; cutoff changes are reported, not forced equal.",
            "GPU reports compare GPU measurements with the CPU float64 reference; a CPU run does not verify GPU behavior.",
            "Peak process RSS includes imports; GPU bytes measure allocator high-water marks, not total device/driver usage.",
        ],
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__, "torch_build": torch.__config__.show(),
                    "cuda_build": torch.version.cuda, "cuda_executed": device_index is not None,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device_name": torch.cuda.get_device_name(device_index) if device_index is not None else platform.machine(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "source_sha256": source_hashes(),
        "cases": [], "errors": [],
    }
    for name in dict.fromkeys(args.cases):
        case_path = output / f"{name}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--device", args.device,
                   "--_case", name, "--_result", str(case_path)]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=args.case_timeout, check=True)
            result = json.loads(case_path.read_text())
            report["cases"].append(result)
            print(json.dumps({"case": name, "status": result["status"], "seconds": result["seconds"]}), flush=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, ValueError) as error:
            detail = error.stderr[-4000:] if isinstance(error, subprocess.CalledProcessError) else str(error)
            report["errors"].append({"case": name, "type": type(error).__name__, "detail": detail})
            print(json.dumps({"case": name, "error": type(error).__name__}), flush=True)
    report["sources_consistent"] = all(case["source_sha256"] == report["source_sha256"] for case in report["cases"]) and source_hashes() == report["source_sha256"]
    report["passed"] = not report["errors"] and report["sources_consistent"] and all(case["status"] != "failed" for case in report["cases"])
    write_report(output / "backend_report.json", report)
    print(f"Backend report: {output / 'backend_report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
