#!/usr/bin/env python3
"""Bounded pilot of actual public, pre-quantized Hugging Face checkpoints.

Only pinned config.json and safetensors files are fetched (under 5 MB total).
Each production-loader/measurement case runs offline in a separate subprocess,
with at most 300 seconds. Packed weights remain subject to the production skip
policy. Explicit backend dequantization is a separate diagnostic: reconstructed
quantized weights are not the original unquantized checkpoint.

Example: CUDA_VISIBLE_DEVICES=5 python quantized_checkpoint_pilot.py \
    --output-dir analysis_runs/validation/quantized-public-new --device cuda:0
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILES = ("config.json", "model.safetensors")
MAX_DOWNLOAD_BYTES = 256 * 1024**2
SPECTRUM_TOLERANCE = 1e-4
CASES = {
    "gptq": {
        "repo_id": "yujiepan/llama-3-tiny-random-gptq-w4",
        "revision": "3ac28aac2b279d11e8e8c34844dfd3dab25686e9",
        "files": {"config.json": 1289, "model.safetensors": 2057704},
        "quant_method": "gptq",
        "role": "negative_control_empty_packed_weights",
        "file_sha256": {
            "config.json": "02a32ebc8734863328d9d133335bed7657a9da6d2341e5fa692580a18f9772ce",
            "model.safetensors": "8b5fc398732bbef4391239c53091d5f3d703c85b3ea15be0a3e8f47a4a50e95a",
        },
    },
    "nf4": {
        "repo_id": "amanpatkar/tiny-random-llama-2",
        "revision": "21e8d02fd722d940454de780a30e4ce7a57d713e",
        "files": {"config.json": 1177, "model.safetensors": 431742},
        "quant_method": "bitsandbytes",
        "quant_type": "nf4",
        "role": "negative_control_undeclared_adapter_wrappers",
        "file_sha256": {
            "config.json": "b270d68d070719173ff85f1d143f90c03192a9ff8c6b7896747680ae83bf12a7",
            "model.safetensors": "6895bec13aafe7557c8fd8293a86300ae70aee033d7adf4aaa3d987f9e7385a4",
        },
    },
    "bnb_fp4": {
        "repo_id": "RichardErkhov/trl-internal-testing_-_tiny-random-LlamaForCausalLM-4bits",
        "revision": "e8f56d61771489cb76c1703c7655afb96c6b1db2",
        "files": {"config.json": 1183, "model.safetensors": 2061254},
        "quant_method": "bitsandbytes",
        "quant_type": "fp4",
        "role": "positive_loading_attempt",
        "file_sha256": {
            "config.json": "01795a91e9b958c3c175fa2505dfa07bc2463eba412a259f65e7c25ba8ef0531",
            "model.safetensors": "f3546997d8defeecede0ed28ba5865d759432b59e7b728bde729e8bf57048911",
        },
    },
}


def source_hashes():
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in (
        "esd_experiment/scripts/quantized_checkpoint_pilot.py",
        "esd_experiment/scripts/backend_pilot.py",
        "esd_experiment/src/model_loader.py", "esd_experiment/src/worker.py",
        "esd_experiment/src/measurement_config.py", "net_esd/core.py",
        "net_esd/utils.py", "net_esd/__init__.py", "net_esd/constants.py",
    )}


def validate_metadata(info, case):
    if info.private or info.gated or info.sha != case["revision"]:
        raise ValueError("Pilot requires the pinned public, ungated revision")
    sizes = {item.rfilename: item.size for item in info.siblings}
    if any(sizes.get(name) != size for name, size in case["files"].items()):
        raise ValueError("Pinned checkpoint file manifest changed")
    if sum(case["files"].values()) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Checkpoint exceeds download budget")


def validate_config(config, case):
    if config.get("auto_map") or config.get("model_type") != "llama":
        raise ValueError("Remote-code or undeclared architecture configuration")
    if config.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError("Unexpected checkpoint architecture")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != case["quant_method"]:
        raise ValueError("Checkpoint is not the declared pre-quantized representation")
    if case["quant_method"] == "bitsandbytes" and (
        quant.get("bnb_4bit_quant_type") != case["quant_type"] or not quant.get("load_in_4bit")
    ):
        raise ValueError("Checkpoint is not the declared pre-quantized 4-bit type")
    return quant


def quantized_module_names(keys, method, quant_type="nf4"):
    suffix = ".qweight" if method == "gptq" else f".weight.quant_state.bitsandbytes__{quant_type}"
    names = sorted(key[:-len(suffix)] for key in keys if key.endswith(suffix))
    if not names:
        raise ValueError("Safe checkpoint contains no declared packed weight tensors")
    return names


def download_case(case, cache, offline):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from safetensors import safe_open

    info = None
    if not offline:
        info = HfApi(token=False).model_info(case["repo_id"], revision=case["revision"],
                                          files_metadata=True, timeout=30)
        validate_metadata(info, case)
    common = dict(revision=case["revision"], cache_dir=str(cache), token=False,
                  local_files_only=offline)
    config_path = Path(hf_hub_download(case["repo_id"], "config.json", **common))
    if config_path.stat().st_size != case["files"]["config.json"]:
        raise ValueError("Config byte count changed")
    config = json.loads(config_path.read_text())
    validate_config(config, case)  # Before fetching any weight files.
    snapshot = Path(snapshot_download(case["repo_id"], allow_patterns=list(FILES),
                                     max_workers=1, **common))
    hashes = {}
    for filename, expected_bytes in case["files"].items():
        path = snapshot / filename
        if path.stat().st_size != expected_bytes:
            raise ValueError("Downloaded file byte count changed")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = case.get("file_sha256", {}).get(filename)
        if expected_hash is not None and hashes[filename] != expected_hash:
            raise ValueError("Pinned checkpoint checksum changed")
        if info is not None:
            entry = next(item for item in info.siblings if item.rfilename == filename)
            if entry.lfs is not None and hashes[filename] != entry.lfs.sha256:
                raise ValueError("Downloaded weight checksum differs from the HF LFS manifest")
    with safe_open(snapshot / "model.safetensors", framework="numpy") as handle:
        packed = quantized_module_names(handle.keys(), case["quant_method"], case.get("quant_type", ""))
        header_evidence = {
            "empty_tensor_names": [name for name in handle.keys() if 0 in handle.get_slice(name).get_shape()],
            "wrapped_adapter_tensor_names": [name for name in handle.keys() if ".base_layer." in name or ".lora_" in name],
        }
    return {**case, "snapshot": str(snapshot), "config": config, "file_sha256": hashes,
            "quantized_module_names": packed, "checkpoint_header_evidence": header_evidence,
            "public_metadata_checked_online": not offline}


def reconstructed_diagnostics(model, method, quant_type, torch):
    """Backend APIs only; never interpret packed bytes as an ordinary matrix."""
    if method != "bitsandbytes":
        return {"status": "not_attempted", "reason": "no generic GPTQ unpacking adapter"}
    import bitsandbytes as bnb
    import numpy as np
    from scipy.linalg import svdvals
    from net_esd.core import compute_esd_for_weight

    rows = []
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
    if not candidates:
        return {"status": "not_attempted", "reason": "no supported bitsandbytes layers"}
    for index in sorted({0, len(candidates) // 2, len(candidates) - 1}):
        name, module = candidates[index]
        if int(np.prod(module.weight.quant_state.shape)) > 4 * 1024**2:
            rows.append({"module_name": name, "status": "skipped_diagnostic_element_budget"})
            continue
        dense = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state).detach().cpu()
        if type(dense) is not torch.Tensor or dense.ndim != 2:
            raise ValueError("Backend did not reconstruct an ordinary dense matrix")
        result = compute_esd_for_weight(name, dense, 1e-5, 100, "xmin_mid", 2, .5,
                                       False, True, True, None, dense.numel(), "float64")
        reference = np.sort(svdvals(dense.double().numpy()) ** 2)
        error = float(np.max(np.abs(result.pop("eigs") - reference)) / max(float(reference[-1]), np.finfo(float).tiny))
        rows.append({"module_name": name, "reconstructed_dtype": str(dense.dtype),
                     "shape": list(dense.shape), "n_evals": result["num_evals"],
                     "fit_status": result["fit_status"], "alpha": result["alpha"],
                     "spectrum_error_over_reference_lambda_max": error,
                     "within_spectrum_tolerance": bool(np.isfinite(error) and error <= SPECTRUM_TOLERANCE)})
    checked = [row["within_spectrum_tolerance"] for row in rows if "within_spectrum_tolerance" in row]
    return {"status": "completed", "representation": f"reconstructed {quant_type.upper()} weights, not original FP32 weights",
            "backend_api": "bitsandbytes.functional.dequantize_4bit",
            "reference": "SciPy float64 CPU SVD of the identical reconstructed matrix",
            "spectrum_tolerance": SPECTRUM_TOLERANCE,
            "all_measured_spectra_within_tolerance": bool(checked) and all(checked), "layers": rows}


def run_case(manifest, output, device):
    from backend_pilot import configure_runtime

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(output.parent / "cache")
    for variable in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(variable, None)
    torch, index = configure_runtime(device)
    sys.path.insert(0, str(PROJECT_ROOT / "esd_experiment/src"))
    from importlib.metadata import version
    from measurement_config import artifact_compatibility, measurement_config
    from model_loader import load_model
    from net_esd import net_esd_estimator
    from worker import coverage_report, runtime_provenance, save_results
    import h5py
    import numpy as np
    from scipy.linalg import svdvals

    report = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
              "role": manifest["role"], "checkpoint_header_evidence": manifest["checkpoint_header_evidence"],
              "checkpoint_file_sha256": manifest["file_sha256"],
              "checkpoint_snapshot": manifest["snapshot"],
              "quantization_config": manifest["config"]["quantization_config"],
              "source_sha256": source_hashes(), "status": "failed", "stage": "load",
              "device": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "device_name": torch.cuda.get_device_name(index) if index is not None else "CPU",
              "versions": {name: version(name) for name in ("torch", "transformers", "bitsandbytes", "gptqmodel", "optimum", "safetensors", "datasets", "fsspec")},
              "no_silent_success": False, "complete_quantized_layer_analysis": False}
    started = time.perf_counter()
    try:
        model, is_adapter = load_model(manifest["snapshot"], device_map=device, torch_dtype="auto",
                                       loader_scenario="quantized_transformers_native")
        if is_adapter:
            raise AssertionError("Full quantized checkpoint was mistaken for an adapter")
        report["stage"] = "analyze"
        records = []
        metrics = net_esd_estimator(model, fix_fingers="xmin_mid", filter_zeros=True,
                                    save_eigs=True, parallel=False, coverage=records)
        coverage = coverage_report(records, metrics)
        report["coverage"] = coverage
        by_name = {row["module_name"]: row for row in records}
        missing = sorted(set(manifest["quantized_module_names"]) - by_name.keys())
        unexpected_analysis = [name for name in manifest["quantized_module_names"]
                               if name in by_name and by_name[name]["status"] != "skipped"]
        report["packed_checkpoint_modules_missing_from_coverage"] = missing
        report["packed_checkpoint_modules_unexpectedly_analyzed"] = unexpected_analysis
        if missing or unexpected_analysis:
            raise AssertionError("Packed checkpoint layer coverage is missing or inconsistent with the production skip policy")
        if not metrics["longname"] or set(metrics["compute_device"]) != {device}:
            raise AssertionError("No ordinary measurements on the requested device")
        args = SimpleNamespace(fix_fingers="xmin_mid", save_eigs=True, device_map=device, parallel_esd=False)
        config = measurement_config(args, model_id=manifest["repo_id"], revision=manifest["revision"],
                                    loader_scenario="quantized_transformers_native")
        config.update(runtime=runtime_provenance(model, args), checkpoint_file_sha256=manifest["file_sha256"],
                      quantization_config=report["quantization_config"], pilot=True)
        report["runtime"] = config["runtime"]
        report["stage"] = "save"
        csv_path, h5_path = output / "stats.csv", output / "metrics.h5"
        if csv_path.exists() or h5_path.exists():
            raise FileExistsError("Refusing to overwrite pilot artifacts")
        save_results(metrics, csv_path, manifest["repo_id"], False, h5_output_path=h5_path,
                     fix_fingers="xmin_mid", save_eigs=True, measurement_config=config, coverage=coverage)
        compatible, reason = artifact_compatibility(csv_path, h5_path, config)
        if not compatible:
            raise AssertionError(f"Saved artifacts failed integrity checks: {reason}")
        errors = []
        modules = dict(model.named_modules())
        with h5py.File(h5_path, "r") as handle:
            if handle.attrs["fix_fingers"] != config["fix_fingers"]:
                raise AssertionError("Saved cutoff method differs from the measurement configuration")
            if handle["layers/longname"].asstr()[:].tolist() != metrics["longname"]:
                raise AssertionError("Saved module identity changed")
            for i, name in enumerate(metrics["module_name"]):
                np.testing.assert_array_equal(handle["eigs"][i], metrics["eigs"][i])
                matrix = modules[name].weight.detach().cpu().double().numpy()
                reference = np.sort(svdvals(matrix) ** 2)
                errors.append(float(np.max(np.abs(handle["eigs"][i] - reference)) / max(float(reference[-1]), np.finfo(float).tiny)))
        report["saved_spectrum_errors_over_reference_lambda_max"] = errors
        if not all(np.isfinite(value) and value <= SPECTRUM_TOLERANCE for value in errors):
            raise AssertionError("Actual saved ordinary spectra exceed the predeclared tolerance")
        report["artifacts"] = {"csv": str(csv_path), "hdf5": str(h5_path), "roundtrip_checked": True}
        report["stage"] = "reconstructed_diagnostic"
        report["reconstructed_diagnostic"] = reconstructed_diagnostics(model, manifest["quant_method"], manifest.get("quant_type", ""), torch)
        report["status"] = "loaded_partial_coverage"
        report["no_silent_success"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "reason": getattr(error, "reason", None),
                           "message": str(error), "traceback": traceback.format_exc()}
    report["seconds"] = time.perf_counter() - started
    report["sources_consistent"] = source_hashes() == report["source_sha256"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0", help="cpu or visible cuda:N; no CPU fallback")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--case-timeout", type=float, default=300)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--_manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.case_timeout <= 300:
        parser.error("Case timeout must be positive and at most 300 seconds")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a fresh path")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from backend_pilot import write_report
    if args._manifest:
        write_report(output / "case_report.json", run_case(json.loads(args._manifest.read_text()), output, args.device))
        return 0
    cases = list(dict.fromkeys(args.cases))
    if sum(sum(CASES[name]["files"].values()) for name in cases) > MAX_DOWNLOAD_BYTES:
        parser.error("Combined download budget exceeded")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    cache = (args.cache_dir or output / "cache").resolve()
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes(),
              "scope": "Actual pre-quantized checkpoint loading, coverage, and storage pilot; not full quantized analysis or model-quality validation",
              "checkpoint_bytes": sum(sum(CASES[name]["files"].values()) for name in cases),
              "network_downloads_enabled": not args.offline,
              "spectrum_tolerance": SPECTRUM_TOLERANCE, "cases": [], "errors": []}
    for name in cases:
        try:
            manifest = download_case(CASES[name], cache, args.offline)
            manifest_path = output / f"{name}_manifest.json"
            write_report(manifest_path, manifest)
            case_output = output / name
            command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(case_output),
                       "--device", args.device, "--_manifest", str(manifest_path)]
            with (output / f"{name}.log").open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=args.case_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            case_path = case_output / "case_report.json"
            if process.returncode or not case_path.is_file():
                raise RuntimeError(f"Case exited {process.returncode}; see {name}.log")
            result = json.loads(case_path.read_text())
            report["cases"].append(result)
            print(json.dumps({"case": name, "status": result["status"],
                              "error": {key: value for key, value in result.get("error", {}).items() if key != "traceback"}}), flush=True)
        except Exception as error:
            report["errors"].append({"case": name, "type": type(error).__name__, "message": str(error)})
    report["sources_consistent"] = source_hashes() == report["source_sha256"] and all(
        case["source_sha256"] == report["source_sha256"] and case["sources_consistent"] for case in report["cases"])
    report["all_cases_loaded"] = not report["errors"] and len(report["cases"]) == len(cases) and all(
        case["status"] == "loaded_partial_coverage" for case in report["cases"])
    write_report(output / "quantized_checkpoint_report.json", report)
    print(f"Report: {output / 'quantized_checkpoint_report.json'}", flush=True)
    return 0 if report["all_cases_loaded"] and report["sources_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
