"""Architecture-independent descriptors must not imply model reconstruction."""

import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import checkpoint_tensors as checkpoint
from measurement_config import measurement_config
from test_worker import load_worker_module


def settings(**kwargs):
    return measurement_config(SimpleNamespace(analysis_source="checkpoint", **kwargs))


def saved_tensors(path):
    path.mkdir(exist_ok=True)
    tensors = {
        "encoder.layers.0.q.weight": torch.diag(torch.arange(1., 5.)).double(),
        "decoder.layers.0.q.weight": torch.diag(torch.arange(2., 6.)).bfloat16(),
        "direct_matrix": torch.arange(24.).reshape(6, 4),
        "features.0.weight": torch.ones(3, 3, 2, 2),
        "bias": torch.ones(4), "counter": torch.tensor(3),
        "integer_matrix": torch.ones(2, 2, dtype=torch.int32),
        "empty": torch.empty(0, 2), "zero_matrix": torch.zeros(2, 2),
    }
    save_file(tensors, path / "model.safetensors")
    return tensors


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    monkeypatch.setattr(checkpoint, "get_hf_token", lambda: None)
    monkeypatch.setattr(sys.modules[checkpoint.checkpoint_tensor_index.__module__], "get_hf_token", lambda: None)


def test_exact_keys_native_precision_and_explicit_skips(tmp_path):
    tensors = saved_tensors(tmp_path)
    index = checkpoint.inspect_checkpoint(str(tmp_path), None)
    metrics, coverage = checkpoint.analyze_checkpoint(index, settings(compute_dtype="float64"))

    expected = sorted(name for name, tensor in tensors.items()
                      if tensor.ndim == 2 and tensor.is_floating_point() and tensor.numel())
    assert metrics["longname"] == expected
    assert metrics["weight_attribute"] == expected and metrics["module_name"] == [""] * len(expected)
    for i, name in enumerate(expected):
        reference = torch.linalg.svdvals(tensors[name].double()).square().sort().values.numpy()
        np.testing.assert_allclose(metrics["eigs"][i], reference, rtol=1e-12, atol=1e-12)
        assert metrics["source_dtype"][i] == str(tensors[name].dtype).removeprefix("torch.")
    records = {row["name"]: row for row in coverage["tensors"]}
    assert records.keys() == tensors.keys()
    assert records["features.0.weight"]["reason"] == "undeclared_tensor_layout"
    assert records["integer_matrix"]["reason"] == "unsupported_dtype"
    assert records["empty"]["reason"] == "empty_tensor"
    assert records["bias"]["reason"] == "not_a_matrix"
    assert coverage["scope"] == "checkpoint_tensors" and "modules" not in coverage
    assert coverage["counts"]["stored_tensors"] == len(tensors)
    assert coverage["counts"]["analyzed_tensors"] == len(expected)
    assert index["provenance"]["aliases"] == "not_inferred"


def test_unknown_architecture_is_not_constructed(tmp_path, monkeypatch):
    import transformers
    saved_tensors(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "not_installed", "architectures": ["UnusableModel"],
        "auto_map": {"AutoModel": "untrusted.CustomModel"},
    }))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: pytest.fail("No model config construction"))
    index = checkpoint.inspect_checkpoint(str(tmp_path), None)
    assert index["provenance"]["config_sha256"]
    assert checkpoint.analyze_checkpoint(index, settings())[0]["longname"]


@pytest.mark.parametrize("defect", ["quantization", "nested_compression", "adapter", "packed_key", "lora_key"])
def test_known_encoded_weights_are_not_silently_measured(tmp_path, defect):
    saved_tensors(tmp_path)
    if defect == "quantization":
        (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
    elif defect == "nested_compression":
        (tmp_path / "config.json").write_text(json.dumps({"text_config": {"compression_config": {"format": "packed"}}}))
    elif defect == "adapter":
        (tmp_path / "adapter_config.json").write_text("{}")
    else:
        name = "proj.qweight" if defect == "packed_key" else "proj.lora_A.weight"
        save_file({name: torch.ones(3, 3)}, tmp_path / "model.safetensors")
    with pytest.raises(checkpoint.LoaderFailure) as exc:
        checkpoint.inspect_checkpoint(str(tmp_path), None)
    assert exc.value.reason == "unsupported_checkpoint_representation"


def test_streaming_releases_inputs_and_does_not_read_skipped_tensors(tmp_path, monkeypatch):
    tensors = saved_tensors(tmp_path)
    index = checkpoint.inspect_checkpoint(str(tmp_path), None)
    compute = checkpoint.compute_esd_for_weight
    previous = None
    read_names = []
    original_open = checkpoint.safe_open

    class CheckedOpen:
        def __init__(self, *args, **kwargs):
            self.handle = original_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_tensor(self, name):
            assert previous is None or previous() is None
            read_names.append(name)
            return self.handle.get_tensor(name)

    def checked_compute(name, weight, *args):
        nonlocal previous
        previous = weakref.ref(weight)
        return compute(name, weight, *args)

    monkeypatch.setattr(checkpoint, "safe_open", CheckedOpen)
    monkeypatch.setattr(checkpoint, "compute_esd_for_weight", checked_compute)
    metrics, _ = checkpoint.analyze_checkpoint(index, settings())
    assert read_names == metrics["longname"]
    assert previous() is None
    assert "features.0.weight" not in read_names
    assert all(tensors[name].ndim == 2 for name in read_names)


def test_checkpoint_worker_writer_summary_and_resume(tmp_path, monkeypatch):
    from esd_experiment.utils.analyze_results import read_model_summary
    tensors = saved_tensors(tmp_path / "checkpoint")
    inspect = checkpoint.inspect_checkpoint
    calls = []

    def local_inspection(repo_id, revision, **kwargs):
        assert repo_id == "org/control" and revision == "a" * 40
        calls.append(repo_id)
        return inspect(str(tmp_path / "checkpoint"), None, **kwargs)

    monkeypatch.setattr(checkpoint, "inspect_checkpoint", local_inspection)
    worker = load_worker_module()
    monkeypatch.setattr(worker, "torch", torch)
    monkeypatch.setattr(worker, "load_model", lambda **kwargs: pytest.fail("No architecture load"))
    output = tmp_path / "run"
    arguments = ["worker.py", "--model_id", "org/control", "--revision", "a" * 40,
                 "--output_dir", str(output), "--device_map", "cpu", "--analysis_source", "checkpoint",
                 "--compute_dtype", "float64"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert worker.main() == 0
    csv_path, h5_path = output / "stats/org--control.csv", output / "metrics/org--control.h5"
    with worker.h5py.File(h5_path) as h5:
        assert h5["alpha"].attrs["view_status"] == "unavailable"
        assert h5["alpha"].shape == (0, 0)
        config = json.loads(h5.attrs["measurement_config_json"])
        assert config["analysis_source"] == "checkpoint"
        assert config["filter_type"] is False
        assert config["runtime"]["model_class"] is None
        assert config["runtime"]["parallel_esd"] is False
        assert "loading_info" not in config["runtime"] or config["runtime"]["loading_info"] is None
        for i, name in enumerate(h5["layers/longname"].asstr()[:]):
            reference = torch.linalg.svdvals(tensors[name].double()).square().sort().values.numpy()
            np.testing.assert_allclose(h5["eigs"][i], reference, rtol=1e-12, atol=1e-12)
    summary = read_model_summary(csv_path, h5_path)
    assert summary["analysis_source"] == "checkpoint"
    assert summary["stored_tensors"] == len(tensors)
    assert summary["coverage_status"] == "checkpoint_tensors"
    assert summary["candidate_modules"] is None and summary["measured_modules"] is None
    assert summary["registered_tensors"] is None and summary["weight_usage_status"] == "not_applicable"
    assert worker.main() == 0 and len(calls) == 1
    before = [path.read_bytes() for path in (csv_path, h5_path)]
    monkeypatch.setattr(sys, "argv", arguments + ["--analysis_source", "model"])
    assert worker.main() == 1
    assert [path.read_bytes() for path in (csv_path, h5_path)] == before


def test_runner_routes_checkpoint_mode_without_backend_probes(tmp_path, monkeypatch):
    from test_run_experiment import run_experiment as runner
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--model_list", "models.csv",
                                    "--output_dir", str(tmp_path), "--analysis_source", "checkpoint"])
    args = runner.parse_args()
    monkeypatch.setattr(runner, "_available_backends", lambda: pytest.fail("No quantized backend probes"))
    rows = pd.DataFrame([
        {"model_id": "org/unknown", "loader_scenario": "custom", "source_model": "", "base_model_relation": ""},
        {"model_id": "org/quantized", "loader_scenario": "quantized_transformers_native"},
        {"model_id": "org/adapter", "base_model_relation": "adapter"},
        {"model_id": "org/gone", "Available on the hub": False},
    ])
    runnable, blocked = runner.apply_preflight(rows, args.analysis_source)
    assert runnable["model_id"].tolist() == ["org/unknown"]
    assert len(blocked) == 3
    command = shlex.split(runner.generate_commands(runnable, tmp_path, args)[0])
    assert command[command.index("--analysis_source") + 1] == "checkpoint"
    assert command[command.index("--loader_scenario") + 1] == "checkpoint_tensors"


def test_checkpoint_mode_rejects_remote_code_flag():
    with pytest.raises(ValueError, match="does not execute model code"):
        settings(trust_remote_code=True)


def test_sharded_bert_matches_model_mode_for_the_same_stored_matrices(tmp_path):
    from test_loader_checkpoint_integrity import tiny_bert
    from net_esd import net_esd_estimator

    model = tiny_bert()
    model.save_pretrained(tmp_path, max_shard_size="2KB")
    index = checkpoint.inspect_checkpoint(str(tmp_path), None)
    assert len(index["files"]) > 1
    direct, coverage = checkpoint.analyze_checkpoint(index, settings(compute_dtype="float64"))
    loaded = net_esd_estimator(model, parallel=False, filter_type=False, filter_zeros=True,
                               fix_fingers="xmin_mid", compute_dtype="float64")
    direct_eigs = dict(zip(direct["longname"], direct["eigs"]))
    assert set(direct_eigs) == {name + ".weight" for name in loaded["longname"]}
    for name, eigs in zip(loaded["longname"], loaded["eigs"]):
        np.testing.assert_array_equal(direct_eigs[name + ".weight"], eigs)
    assert coverage["counts"]["stored_tensors"] == len(model.state_dict())


def test_saved_alias_omissions_are_not_reconstructed(tmp_path):
    from safetensors.torch import save_model
    layer = torch.nn.Linear(4, 4)
    model = torch.nn.ModuleDict({"first": layer, "shared": layer})
    save_model(model, tmp_path / "model.safetensors")
    index = checkpoint.inspect_checkpoint(str(tmp_path), None)
    metrics, coverage = checkpoint.analyze_checkpoint(index, settings())
    assert metrics["longname"] == ["first.weight"]
    assert coverage["counts"]["stored_tensors"] == 2
    assert index["provenance"]["aliases"] == "not_inferred"


@pytest.mark.parametrize("load_dtype,expected", [("auto", "float64"), ("float32", "float32"), ("float16", "float16")])
def test_loading_cast_is_explicit_and_separate_from_checkpoint_dtype(tmp_path, load_dtype, expected):
    save_file({"matrix": torch.diag(torch.arange(1., 5.)).double()}, tmp_path / "model.safetensors")
    index = checkpoint.inspect_checkpoint(str(tmp_path), None)
    metrics, coverage = checkpoint.analyze_checkpoint(index, settings(load_dtype=load_dtype))
    assert metrics["source_dtype"] == [expected]
    assert coverage["tensors"][0]["dtype"] == "F64"
    assert coverage["tensors"][0]["source_dtype"] == expected


def test_every_remote_fetch_is_pinned_including_config_probes(tmp_path, monkeypatch):
    import transformers.utils.hub as hub
    saved_tensors(tmp_path)
    calls = []

    def resolve(repo_id, filename, **kwargs):
        assert repo_id == "org/model" and kwargs["revision"] == "a" * 40
        calls.append(filename)
        path = tmp_path / filename
        return str(path) if path.is_file() else None

    monkeypatch.setattr(hub, "cached_file", resolve)
    checkpoint.inspect_checkpoint("org/model", "a" * 40)
    assert calls == ["adapter_config.json", "config.json", "model.safetensors"]
    calls.clear()
    with pytest.raises(checkpoint.LoaderFailure, match="pinned revision"):
        checkpoint.inspect_checkpoint("org/model", "main")
    assert calls == []


def test_no_matrices_keeps_coverage_without_success_artifacts(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    save_file({"kernel": torch.ones(2, 2, 3), "bias": torch.ones(2)}, weights / "model.safetensors")
    inspect = checkpoint.inspect_checkpoint
    monkeypatch.setattr(checkpoint, "inspect_checkpoint", lambda *a, **k: inspect(str(weights), None))
    worker = load_worker_module()
    monkeypatch.setattr(worker, "torch", torch)
    monkeypatch.setattr(worker, "load_model", lambda **kwargs: pytest.fail("No model fallback"))
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["worker.py", "--model_id", "org/unsupported", "--revision", "a" * 40,
                                    "--output_dir", str(output), "--analysis_source", "checkpoint", "--device_map", "cpu"])
    assert worker.main() == 1
    coverage = json.loads((output / "logs/coverage/org--unsupported.json").read_text())
    assert coverage["counts"]["stored_tensors"] == 2 and coverage["counts"]["analyzed_tensors"] == 0
    terminal = json.loads((output / "logs/terminal_status/org--unsupported.json").read_text())
    assert terminal["reason"] == "analysis_empty"
    assert not list(output.glob("metrics/*.h5")) and not list(output.glob("stats/*.csv"))


def test_checkpoint_device_constraints_are_checked_before_launch(tmp_path, monkeypatch):
    from test_run_experiment import run_experiment as runner
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--model_list", "models.csv",
                                    "--output_dir", str(tmp_path), "--analysis_source", "checkpoint",
                                    "--num_gpus_per_job", "2"])
    with pytest.raises(SystemExit):
        runner.parse_args()
    worker = load_worker_module()
    monkeypatch.setattr(sys, "argv", ["worker.py", "--model_id", "org/model", "--revision", "a" * 40,
                                    "--output_dir", str(tmp_path), "--analysis_source", "checkpoint",
                                    "--device_map", "balanced"])
    with pytest.raises(SystemExit):
        worker.parse_args()
