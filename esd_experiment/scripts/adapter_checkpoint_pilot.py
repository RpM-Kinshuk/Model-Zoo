#!/usr/bin/env python3
"""Bounded public LoRA loading/merge pilot, not a dataset-scale experiment.

Pin both repositories independently. Compare merged weights to explicit float64
W + (alpha/r) B A, then compare every saved spectrum to the manual model. Only
ordinary matrix LoRA is a positive reference here; unsupported variants fail
explicitly instead of being interpreted with the wrong formula.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECTRUM_TOLERANCE = 1e-4
MERGE_RTOL, MERGE_ATOL = 5e-5, 1e-7
SPECS = {
    "opt_lora": {
        "adapter": {"repo_id": "peft-internal-testing/tiny-OPTForCausalLM-lora",
                    "revision": "14e64b8ba522284138bfc22e76002ab6c0ce31e2",
                    "files": {"adapter_config.json": 443, "adapter_model.bin": 17261},
                    "sha256": {"adapter_model.bin": "80c26b0b017f458860b7ef169139fbd43e8f48d9375455a7527de3b33b79c8b3"}},
        "base": {"repo_id": "hf-internal-testing/tiny-random-OPTForCausalLM",
                 "revision": "0abea37ca0a786ba455967e799b7b3d67f86541f",
                 "files": {"config.json": 684, "model.safetensors": 3258784}},
        "base_class": "OPTForCausalLM",
        "control_role": "zero_delta_preservation",
    },
    "gpt2_lora": {
        "adapter": {"repo_id": "Teja00000000001/miniclay-3c7a9189",
                    "revision": "1acd0a065fda7601a212a8337a84042a8a19cf66",
                    "files": {"adapter_config.json": 557, "adapter_model.safetensors": 1536},
                    "sha256": {"adapter_model.safetensors": "771e3bb675d1b8dabf7e9bdc2a3730888bfd5f6358c75ddb4db60e82a1dcf0a4"}},
        "base": {"repo_id": "sshleifer/tiny-gpt2",
                 "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
                 "files": {"config.json": 662, "pytorch_model.bin": 2514146}},
        "base_class": "GPT2LMHeadModel",
        "control_role": "nonzero_merge",
        "allowed_base_unexpected": ["transformer.h.0.attn.masked_bias", "transformer.h.1.attn.masked_bias"],
    },
}
SOURCE_FILES = (
    "esd_experiment/scripts/adapter_checkpoint_pilot.py",
    "esd_experiment/scripts/measurement_pilot.py",
    "esd_experiment/src/model_loader.py", "esd_experiment/src/measurement_config.py",
    "esd_experiment/src/worker.py", "net_esd/core.py", "net_esd/utils.py",
    "net_esd/constants.py", "net_esd/__init__.py",
)


def source_hashes():
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def download_pinned(entry, cache, offline):
    from huggingface_hub import HfApi, snapshot_download

    files = entry["files"]
    if not offline:
        info = HfApi(token=False).model_info(entry["repo_id"], revision=entry["revision"],
                                           files_metadata=True, timeout=30)
        sizes = {item.rfilename: item.size for item in info.siblings}
        if info.private or info.gated or info.sha != entry["revision"]:
            raise ValueError("Pilot requires the pinned public, ungated repository")
        if any(sizes.get(name) != size for name, size in files.items()):
            raise ValueError("Pinned file manifest changed; refusing download")
    snapshot = Path(snapshot_download(entry["repo_id"], revision=entry["revision"],
                    allow_patterns=list(files), cache_dir=str(cache), token=False,
                    local_files_only=offline, max_workers=1))
    for name, size in files.items():
        if (snapshot / name).stat().st_size != size:
            raise ValueError("Cached file size does not match pinned manifest")
    for name, expected_hash in entry.get("sha256", {}).items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Checkpoint content does not match pinned SHA256")
    for name in files:
        if name.endswith(".json") and json.loads((snapshot / name).read_text()).get("auto_map"):
            raise ValueError("Remote-code configurations are outside this pilot")
    return snapshot


def manual_lora_state(base_state, adapter_state, config):
    """Independent ordinary matrix-LoRA reference; never calls PEFT merge helpers."""
    import torch

    if (config.get("peft_type") != "LORA" or config.get("bias", "none") != "none"
            or config.get("use_dora") or config.get("modules_to_save")
            or config.get("rank_pattern") or config.get("alpha_pattern")
            or config.get("layer_replication") or config.get("lora_bias")):
        raise ValueError("Adapter variant is outside the ordinary matrix-LoRA reference")
    rank = config["r"]
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be positive")
    scaling = config["lora_alpha"] / (math.sqrt(rank) if config.get("use_rslora") else rank)
    if not math.isfinite(scaling):
        raise ValueError("LoRA scale must be finite")
    expected = {name: value.detach().cpu().double().clone() for name, value in base_state.items()}
    used, updates = set(), []
    for a_key, a_value in adapter_state.items():
        if not a_key.startswith("base_model.model.") or not a_key.endswith(".lora_A.weight"):
            continue
        module = a_key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        b_key = a_key.removesuffix(".lora_A.weight") + ".lora_B.weight"
        base_key = module + ".weight"
        if b_key not in adapter_state or base_key not in expected:
            raise ValueError("Adapter matrix pair or original base matrix is missing")
        a, b = a_value.detach().cpu().double(), adapter_state[b_key].detach().cpu().double()
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError("Adapter matrix dimensions disagree with declared rank")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Non-finite adapter matrix")
        delta = scaling * (b @ a)
        if config.get("fan_in_fan_out"):
            delta = delta.T
        if delta.shape != expected[base_key].shape:
            raise ValueError("Adapter update shape disagrees with base weight layout")
        expected[base_key] += delta
        used.update((a_key, b_key))
        updates.append({"module_name": module, "scaling": scaling,
                        "max_abs_delta": delta.abs().max().item()})
    if not updates or used != set(adapter_state):
        raise ValueError("Unpaired or unsupported adapter tensors cannot enter the manual reference")
    return expected, updates


def validate_control(updates, role, coverage=None):
    count = sum(row["max_abs_delta"] > 0 for row in updates)
    if (role == "nonzero_merge" and count == 0) or (role == "zero_delta_preservation" and count != 0):
        raise AssertionError("Adapter does not exercise the declared control role")
    if coverage is not None:
        by_name = {row["module_name"]: row for row in coverage}
        for update in updates:
            row = by_name.get(update["module_name"], {})
            if row.get("status") != "analyzed" or not row.get("measurement_names"):
                raise AssertionError("An adapted module is missing from analyzed coverage")
    return count


def analyze_case(name, spec, adapter_path, base_path, output, device, ww_svd):
    import h5py
    import numpy as np
    import torch
    import transformers
    from safetensors.torch import load_file
    from measurement_pilot import reference_spectrum, tied_parameter_groups
    from model_loader import hf_from_pretrained, load_model
    from measurement_config import measurement_config, artifact_compatibility
    from net_esd import net_esd_estimator
    from net_esd.utils import iter_eligible_layers
    from worker import coverage_report, runtime_provenance, save_results

    config = json.loads((adapter_path / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != spec["base"]["repo_id"]:
        raise ValueError("Adapter declared a different base repository")
    model, base_info = hf_from_pretrained(getattr(transformers, spec["base_class"]), str(base_path),
        torch_dtype=torch.float32, device_map="cpu", local_files_only=True,
        trust_remote_code=False, output_loading_info=True)
    if (any(base_info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs"))
            or set(base_info.get("unexpected_keys", [])) - set(spec.get("allowed_base_unexpected", []))):
        raise ValueError("Explicit base baseline has unexplained loading differences")
    weights_path = next(adapter_path / file for file in spec["adapter"]["files"]
                        if file.endswith((".safetensors", ".bin")))
    adapter_state = (load_file(str(weights_path), device="cpu") if weights_path.suffix == ".safetensors"
                     else torch.load(weights_path, map_location="cpu", weights_only=True))
    expected, updates = manual_lora_state(model.state_dict(), adapter_state, config)
    nonzero_count = validate_control(updates, spec["control_role"])
    manual = copy.deepcopy(model).double()
    manual.load_state_dict(expected, strict=True)
    started = time.perf_counter()
    # Local snapshots pin adapter/base independently. The adapter SHA must not
    # leak into the base lookup merely because source_model has no @ suffix.
    merged, is_adapter = load_model(str(adapter_path), source_model=str(base_path),
        base_model_relation="adapter", revision=spec["adapter"]["revision"],
        device_map=device, torch_dtype=torch.float32, loader_scenario="adapter_requires_base")
    if not is_adapter or type(merged) is not type(model):
        raise AssertionError("Adapter loading changed the base architecture")
    actual = merged.state_dict()
    if actual.keys() != expected.keys() or tied_parameter_groups(merged) != tied_parameter_groups(model):
        raise AssertionError("Merge changed base tensor identities or shared-Parameter ties")
    maximum_error = 0.
    updated_keys = {row["module_name"] + ".weight" for row in updates}
    for key, reference in expected.items():
        value = actual[key].detach().cpu().double()
        if key not in updated_keys:
            torch.testing.assert_close(actual[key].cpu(), model.state_dict()[key].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(value, reference, rtol=MERGE_RTOL, atol=MERGE_ATOL)
        maximum_error = max(maximum_error, (value - reference).abs().max().item())
    records = []
    metrics = net_esd_estimator(merged, fix_fingers="xmin_mid", filter_zeros=True,
        use_svd=True, save_eigs=True, parallel=False, compute_dtype="float32", coverage=records)
    validate_control(updates, spec["control_role"], records)
    reference_layers = {layer: weight for layer, weight, _ in iter_eligible_layers(manual)}
    if metrics["longname"] != list(reference_layers):
        raise AssertionError("Merged measurement identities do not match the manual model")
    discrepancies = []
    for layer, eigs, executed_device in zip(metrics["longname"], metrics["eigs"], metrics["compute_device"]):
        if executed_device != device:
            raise AssertionError("Measurement silently used a different device")
        reference = reference_spectrum(reference_layers[layer], ww_svd)
        if eigs.shape != reference.shape or not np.isfinite(eigs).all():
            raise AssertionError("Invalid saved spectrum")
        error = float(np.max(np.abs(eigs - reference)) / max(float(reference.max()), np.finfo(float).tiny))
        discrepancies.append({"longname": layer, "spectrum_error_over_reference_max": error})
    coverage = coverage_report(records, metrics)
    args = SimpleNamespace(save_eigs=True, load_dtype="float32", compute_dtype="float32",
                           device_map=device, parallel_esd=False)
    measurement = measurement_config(args, model_id=spec["adapter"]["repo_id"],
        revision=spec["adapter"]["revision"], base_model_relation="adapter",
        source_model=spec["base"]["repo_id"] + "@" + spec["base"]["revision"],
        loader_scenario="adapter_requires_base")
    file_hashes = {
        role: {filename: hashlib.sha256((path / filename).read_bytes()).hexdigest()
               for filename in spec[role]["files"]}
        for role, path in (("adapter", adapter_path), ("base", base_path))
    }
    measurement.update(runtime=runtime_provenance(merged, args), pilot=True,
                       pinned_base_revision=spec["base"]["revision"], checkpoint_file_sha256=file_hashes)
    csv_path, h5_path = output / "stats" / f"{name}.csv", output / "metrics" / f"{name}.h5"
    save_results(metrics, csv_path, spec["adapter"]["repo_id"], True, save_eigs=True,
                 h5_output_path=h5_path, measurement_config=measurement, coverage=coverage,
                 source_model=measurement["source_model"], base_model_relation="adapter",
                 fix_fingers="xmin_mid")
    compatible, reason = artifact_compatibility(csv_path, h5_path, measurement)
    if not compatible:
        raise AssertionError(reason)
    with h5py.File(h5_path) as h5:
        assert h5.attrs["source_model"] == measurement["source_model"]
        assert h5.attrs["base_model_relation"] == measurement["base_model_relation"]
        assert h5.attrs["fix_fingers"] == "xmin_mid"
        for stored, original in zip(h5["eigs"], metrics["eigs"]):
            np.testing.assert_array_equal(stored, original)
    return {"case": name, "specification": spec, "adapter_config": config,
            "base_revision_provenance": "explicit pilot pin; not proof of the historical training revision",
            "adapter_tensor_count": len(adapter_state), "base_loading_info": base_info, "updates": updates,
            "nonzero_update_count": nonzero_count, "control_role": spec["control_role"],
            "checkpoint_file_sha256": file_hashes,
            "max_merge_abs_error": maximum_error, "runtime": measurement["runtime"],
            "coverage": coverage, "spectral_comparisons": discrepancies,
            "passed": bool(discrepancies) and all(row["spectrum_error_over_reference_max"] <= SPECTRUM_TOLERANCE for row in discrepancies),
            "artifacts_checked": True, "csv": str(csv_path), "hdf5": str(h5_path),
            "seconds": time.perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu or visible cuda:N; no fallback")
    parser.add_argument("--weightwatcher-path", type=Path, default=PROJECT_ROOT.parent / "WeightWatcher")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    output = args.output_dir.resolve()
    cache = args.cache_dir.resolve() if args.cache_dir else output / "cache"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(HF_HOME=str(cache), HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      HF_HUB_DISABLE_XET="1", MPLCONFIGDIR=str(output / "matplotlib"))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "esd_experiment/src"), str(Path(__file__).resolve().parent)]
    import torch
    from measurement_pilot import json_safe, validate_device
    validate_device(args.device, False, torch)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "device": args.device,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "scope": "Pinned ordinary LoRA merge and saved-spectrum checks, not predictive validation",
              "spectrum_tolerance": SPECTRUM_TOLERANCE, "merge_rtol": MERGE_RTOL, "merge_atol": MERGE_ATOL,
              "source_sha256": source_hashes(), "cases": [], "errors": []}
    ww_svd = None
    if not args.download_only:
        ww_path = args.weightwatcher_path.resolve()
        sys.path.insert(0, str(ww_path))
        ww = importlib.import_module("weightwatcher")
        if not Path(ww.__file__).resolve().is_relative_to(ww_path):
            raise RuntimeError("Wrong WeightWatcher checkout imported")
        from weightwatcher.RMT_Util import svd_vals
        ww_svd = svd_vals
        report["weightwatcher"] = {"version": ww.__version__, "commit": subprocess.check_output(
            ["git", "-C", str(ww_path), "rev-parse", "HEAD"], text=True).strip()}
    for name, spec in SPECS.items():
        try:
            adapter_path = download_pinned(spec["adapter"], cache, args.offline)
            base_path = download_pinned(spec["base"], cache, args.offline)
            result = ({"case": name, "specification": spec, "downloaded": True}
                      if args.download_only else analyze_case(name, spec, adapter_path, base_path, output, args.device, ww_svd))
            report["cases"].append(result)
            if not args.download_only and not result["passed"]:
                report["errors"].append({"case": name, "type": "SpectrumToleranceFailure"})
            print(json.dumps({"case": name, "passed": result.get("passed"), "downloaded": args.download_only}), flush=True)
        except Exception as error:
            report["errors"].append({"case": name, "type": type(error).__name__, "message": str(error)})
    report["sources_consistent"] = source_hashes() == report["source_sha256"]
    report["passed"] = not args.download_only and not report["errors"] and report["sources_consistent"]
    with (output / "adapter_report.json").open("x") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Adapter report: {output / 'adapter_report.json'}", flush=True)
    return int(bool(report["errors"]) or not report["sources_consistent"])


if __name__ == "__main__":
    raise SystemExit(main())
