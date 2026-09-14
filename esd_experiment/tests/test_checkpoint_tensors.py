"""Architecture-independent descriptors must not imply model reconstruction."""

import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import Mock
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
        assert config["analysis_policy"] == "checkpoint"
        assert config["runtime"]["analysis_source"] == "checkpoint"
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


@pytest.fixture
def auto_worker(tmp_path, monkeypatch):
    """Exercise the real loader/core/writer; replace only remote file resolution."""
    import net_esd
    from net_esd.utils import weight_usage_report
    loader = sys.modules[checkpoint.checkpoint_tensor_index.__module__]
    weights = tmp_path / "weights"
    weights.mkdir()
    output = tmp_path / "run"
    worker = load_worker_module()
    monkeypatch.setattr(worker, "torch", torch)
    monkeypatch.setattr(worker, "LoaderFailure", loader.LoaderFailure)
    monkeypatch.setattr(worker, "net_esd_estimator", net_esd.net_esd_estimator)
    monkeypatch.setattr(worker, "weight_usage_report", weight_usage_report)
    monkeypatch.setattr(loader, "hf_repo_has_prefix", lambda *args: False)

    def local_load(repo_id, revision, **kwargs):
        assert repo_id == "org/control" and revision == "a" * 40
        return loader.load_model(str(weights), revision=revision, **kwargs)

    inspect = checkpoint.inspect_checkpoint

    def local_inspect(repo_id, revision, **kwargs):
        assert repo_id == "org/control" and revision == "a" * 40
        return inspect(str(weights), revision, **kwargs)

    load = Mock(side_effect=local_load)
    inspection = Mock(side_effect=local_inspect)
    monkeypatch.setattr(worker, "load_model", load)
    monkeypatch.setattr(checkpoint, "inspect_checkpoint", inspection)
    arguments = ["worker.py", "--model_id", "org/control", "--revision", "a" * 40,
                 "--output_dir", str(output), "--device_map", "cpu", "--compute_dtype", "float64"]
    monkeypatch.setattr(sys, "argv", arguments)
    return SimpleNamespace(worker=worker, weights=weights, output=output, arguments=arguments,
                           load=load, inspection=inspection)


def unknown_config(path, *, custom_code=False):
    saved_tensors(path)
    config = {"model_type": "not_an_installed_model", "architectures": ["UnknownModel"]}
    if custom_code:
        config["auto_map"] = {"AutoConfig": "custom.Config", "AutoModel": "custom.Model"}
        (path / "custom.py").write_text("raise RuntimeError('Repository code must not execute')\n")
    (path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("kind,reason", [
    ("unsupported", "checkpoint_config_unsupported"),
    ("custom_code", "checkpoint_config_requires_code"),
    ("ambiguous", "checkpoint_layout_ambiguous"),
])
def test_auto_fallback_records_source_and_resumes_without_reinterpreting(auto_worker, monkeypatch, kind, reason):
    from esd_experiment.utils.analyze_results import read_model_summary
    case = auto_worker
    if kind == "ambiguous":
        import transformers
        from test_loader_checkpoint_integrity import tiny_bert
        config = tiny_bert().config
        config.num_labels = 1
        model = transformers.BertForSequenceClassification(config)
        model.save_pretrained(case.weights)
        config.architectures = None
        config.save_pretrained(case.weights)
    else:
        unknown_config(case.weights, custom_code=kind == "custom_code")
    assert case.worker.parse_args().analysis_source == "auto"
    assert case.worker.main() == 0
    csv_path, h5_path = case.output / "stats/org--control.csv", case.output / "metrics/org--control.h5"
    with case.worker.h5py.File(h5_path) as h5:
        config = json.loads(h5.attrs["measurement_config_json"])
        assert config["analysis_policy"] == "auto" and config["filter_type"] is True
        assert config["runtime"]["analysis_source"] == "checkpoint"
        assert config["runtime"]["filter_type"] is False
        assert config["runtime"]["model_class"] is None
        assert config["runtime"]["fallback"]["reason"] == reason
        assert h5["alpha"].attrs["view_status"] == "unavailable"
    summary = read_model_summary(csv_path, h5_path)
    assert summary["analysis_policy"] == "auto" and summary["analysis_source"] == "checkpoint"
    assert summary["effective_filter_type"] is False and summary["fallback_reason"] == reason
    assert case.load.call_count == case.inspection.call_count == 1
    # A later environment capable of model loading must not promote old tensor results.
    case.load.side_effect = lambda **kwargs: pytest.fail("Resume must not try a different source")
    assert case.worker.main() == 0
    before = [path.read_bytes() for path in (csv_path, h5_path)]
    monkeypatch.setattr(sys, "argv", case.arguments + ["--analysis_source", "model"])
    assert case.worker.main() == 1
    assert [path.read_bytes() for path in (csv_path, h5_path)] == before
    assert case.load.call_count == case.inspection.call_count == 1


def test_explicit_model_mode_does_not_fall_back(auto_worker, monkeypatch):
    case = auto_worker
    unknown_config(case.weights)
    monkeypatch.setattr(sys, "argv", case.arguments + ["--analysis_source", "model"])
    assert case.worker.main() == 1
    case.inspection.assert_not_called()


def test_auto_uses_a_supported_model_without_inspecting_raw_tensors(auto_worker):
    from test_loader_checkpoint_integrity import tiny_bert
    case = auto_worker
    tiny_bert().save_pretrained(case.weights)
    assert case.worker.main() == 0
    case.inspection.assert_not_called()
    with case.worker.h5py.File(case.output / "metrics/org--control.h5") as h5:
        config = json.loads(h5.attrs["measurement_config_json"])
        assert config["analysis_policy"] == "auto"
        assert config["runtime"]["analysis_source"] == "model"
        assert config["runtime"]["fallback"] is None
        assert config["runtime"]["loading_info"]["validated"]


@pytest.mark.parametrize("declared", [True, False])
@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
def test_auto_never_salvages_broken_bert_weights(auto_worker, declared, defect):
    from test_loader_checkpoint_integrity import tiny_bert
    case = auto_worker
    model = tiny_bert()
    state = model.state_dict()
    key = "encoder.layer.0.attention.self.query.weight"
    if defect == "missing":
        del state[key]
    elif defect == "unexpected":
        state["unexplained.weight"] = torch.ones(2, 2)
    else:
        state[key] = torch.ones(3, 3)
    model.save_pretrained(case.weights, state_dict=state)
    if not declared:
        model.config.architectures = None
        model.config.save_pretrained(case.weights)
    assert case.worker.main() == 1
    case.inspection.assert_not_called()
    assert not list(case.output.glob("metrics/*.h5"))


@pytest.mark.parametrize("failure", [
    ("load", "checkpoint_architecture_unresolved"),
    ("load", "ambiguous_checkpoint_architecture"),  # Conflicting declarations, not verified layouts.
    ("load", "checkpoint_architecture_unavailable"),
    ("load", "checkpoint_loading_info_missing"),
    ("load", "checkpoint_weight_mismatch"),
    ("load", "quantized_dependency_missing"),
    ("load", "adapter_checkpoint_mismatch"),
    ("load", "cuda_oom"),
    ("analyze", "checkpoint_config_unsupported"),  # The allowed reason at the wrong stage.
])
def test_auto_only_handles_typed_architecture_failures_at_load(auto_worker, failure):
    case = auto_worker
    case.load.side_effect = checkpoint.LoaderFailure(*failure, "Control failure")
    assert case.worker.main() == 1
    case.inspection.assert_not_called()


@pytest.mark.parametrize("error", [TimeoutError("config request timed out"), ConnectionError("unreachable"),
                                   RuntimeError("CUDA out of memory"), ValueError("unsupported model type")])
def test_auto_does_not_guess_from_generic_error_messages(auto_worker, error):
    case = auto_worker
    case.load.side_effect = error
    assert case.worker.main() == 1
    case.inspection.assert_not_called()


@pytest.mark.parametrize("keep_standard", [True, False])
def test_auto_does_not_fill_in_partial_or_empty_module_analysis(auto_worker, keep_standard):
    case = auto_worker
    model = torch.nn.Module()
    model.unknown = torch.nn.Module()
    model.unknown.weight = torch.nn.Parameter(torch.eye(4))
    if keep_standard:
        model.standard = torch.nn.Linear(4, 4)
    case.load.side_effect = None
    case.load.return_value = (model, False)
    assert case.worker.main() == (0 if keep_standard else 1)
    case.inspection.assert_not_called()
    coverage = json.loads((case.output / "logs/coverage/org--control.json").read_text())
    assert coverage["counts"]["skipped_modules"] == 1


@pytest.mark.parametrize("defect", ["file", "quantization", "analysis"])
def test_failed_fallback_is_attempted_once_and_keeps_original_reason(auto_worker, monkeypatch, defect):
    case = auto_worker
    unknown_config(case.weights)
    if defect == "file":
        (case.weights / "model.safetensors").write_bytes(b"truncated")
        expected_reason = "checkpoint_inspection_failed"
    elif defect == "quantization":
        config_path = case.weights / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization_config"] = {"quant_method": "unknown"}
        config_path.write_text(json.dumps(config))
        expected_reason = "unsupported_checkpoint_representation"
    else:
        def failed_analysis(*args, **kwargs):
            raise RuntimeError("CUDA out of memory during ESD")
        monkeypatch.setattr(checkpoint, "analyze_checkpoint", failed_analysis)
        expected_reason = "cuda_oom"
    monkeypatch.setattr(sys, "argv", case.arguments + ["--max_retries", "3"])
    assert case.worker.main() == 1
    assert case.load.call_count == case.inspection.call_count == 1
    terminal = json.loads((case.output / "logs/terminal_status/org--control.json").read_text())
    assert terminal["reason"] == expected_reason
    assert "checkpoint_config_unsupported" in terminal["message"]
    assert not list(case.output.glob("metrics/*.h5"))


@pytest.mark.parametrize("contents", ["{broken-json", '{"model_type": ["bert"]}',
                                     '{"model_type": "bert", "hidden_size": "not-a-number"}'])
def test_bad_config_is_not_treated_as_missing_architecture_support(auto_worker, contents):
    case = auto_worker
    saved_tensors(case.weights)
    (case.weights / "config.json").write_text(contents)
    assert case.worker.main() == 1
    case.inspection.assert_not_called()


def test_new_config_probe_passes_the_requested_pin(monkeypatch):
    loader = sys.modules[checkpoint.checkpoint_tensor_index.__module__]
    calls = []

    def no_config(*args, **kwargs):
        raise ValueError("unrecognized config")

    def config_dict(repo_id, **kwargs):
        calls.append((repo_id, kwargs["revision"]))
        return {"model_type": "not_an_installed_model"}, {}

    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", no_config)
    monkeypatch.setattr(loader.PretrainedConfig, "get_config_dict", config_dict)
    with pytest.raises(loader.LoaderFailure) as exc:
        loader._checkpoint_model_cls("org/control", "a" * 40)
    assert exc.value.reason == "checkpoint_config_unsupported"
    assert calls == [("org/control", "a" * 40)]
    calls.clear()
    assert loader._checkpoint_model_cls("org/control", "a" * 40, trust_remote_code=True) == (None, None)
    assert not calls  # Explicitly trusted custom config can reach the normal code-loading path.
