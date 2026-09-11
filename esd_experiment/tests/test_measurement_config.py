"""Small persistence/configuration checks; no models, network or GPUs."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from measurement_config import (
    FORMAT_VERSION, NUMERICS_VERSION, artifact_compatibility, measurement_config,
)


def write_pair(tmp_path, config):
    csv_path = tmp_path / "stats" / "org--model.csv"
    h5_path = tmp_path / "metrics" / "org--model.h5"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("longname,alpha\nclassifier,2\n")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["numerics_version"] = NUMERICS_VERSION
        h5.attrs["measurement_config_json"] = json.dumps(config)
        h5.create_dataset("layers/longname", data=["classifier"], dtype=h5py.string_dtype())
        h5.create_dataset("layers/alpha", data=[2.0])
    return csv_path, h5_path


def test_current_configuration_and_extra_provenance_are_compatible(tmp_path):
    expected = measurement_config(SimpleNamespace(), model_id="org/model@old", revision="pinned")
    assert expected["requested_revision"] == "pinned"
    csv_path, h5_path = write_pair(tmp_path, dict(expected, torch_version="test", resolved_revision="sha"))
    assert artifact_compatibility(csv_path, h5_path, expected) == (True, "compatible")


@pytest.mark.parametrize("key,value", [
    ("load_dtype", "float16"), ("compute_dtype", "float64"),
    ("evals_thresh", 0.001), ("filter_zeros", False), ("use_svd", False),
    ("fix_fingers", "DKS"), ("save_eigs", True), ("requested_revision", "other"),
    ("source_model", "org/other"),
])
def test_changed_measurement_setting_does_not_resume(tmp_path, key, value):
    expected = measurement_config(SimpleNamespace(), model_id="org/model")
    csv_path, h5_path = write_pair(tmp_path, dict(expected, **{key: value}))
    compatible, reason = artifact_compatibility(csv_path, h5_path, expected)
    assert not compatible
    assert key in reason


@pytest.mark.parametrize("mutation", ["legacy", "corrupt", "unaligned", "missing_config", "missing_eigs"])
def test_incomplete_or_old_artifacts_are_not_completed(tmp_path, mutation):
    config = measurement_config(SimpleNamespace())
    if mutation == "missing_eigs":
        config["save_eigs"] = True
    csv_path, h5_path = write_pair(tmp_path, config)
    if mutation == "corrupt":
        h5_path.write_text("not HDF5")
    else:
        with h5py.File(h5_path, "a") as h5:
            if mutation == "legacy":
                h5.attrs["numerics_version"] = "3"
            elif mutation == "unaligned":
                h5.create_dataset("layers/n_tail", data=[1, 2])
            elif mutation == "missing_config":
                del h5.attrs["measurement_config_json"]
    assert not artifact_compatibility(csv_path, h5_path, config)[0]


def test_runner_refuses_mismatched_pair_without_modifying_it(tmp_path):
    spec = importlib.util.spec_from_file_location("runner_config_test", SRC / "run_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = SimpleNamespace()
    stored = measurement_config(args, model_id="org/model")
    stored["load_dtype"] = "float16"
    csv_path, h5_path = write_pair(tmp_path, stored)
    before = h5_path.read_bytes()
    df = pd.DataFrame([{"model_id": "org/model", "source_model": "", "base_model_relation": ""}])
    with pytest.raises(ValueError, match="fresh output directory"):
        runner.filter_models_to_run(df, tmp_path, args=args)
    assert h5_path.read_bytes() == before
    assert len(runner.filter_models_to_run(df, tmp_path, args=args, overwrite=True)) == 1
    assert h5_path.read_bytes() == before


@pytest.mark.parametrize("args", [SimpleNamespace(evals_thresh=float("nan")), SimpleNamespace(evals_thresh=-1), SimpleNamespace(bins=0)])
def test_invalid_measurement_configuration_is_rejected(args):
    with pytest.raises(ValueError):
        measurement_config(args)


def test_runner_and_worker_share_defaults_and_explicit_negative_flags(monkeypatch, tmp_path):
    import shlex
    spec = importlib.util.spec_from_file_location("runner_cli_config_test", SRC / "run_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import worker

    monkeypatch.setattr(sys, "argv", ["runner", "--model_list", "models.csv", "--output_dir", str(tmp_path)])
    runner_args = runner.parse_args()
    monkeypatch.setattr(sys, "argv", ["worker", "--model_id", "org/model", "--output_dir", str(tmp_path)])
    worker_args = worker.parse_args()
    assert measurement_config(runner_args) == measurement_config(worker_args)
    assert runner_args.load_dtype == "auto"
    assert runner_args.use_svd is True

    runner_args.filter_zeros = runner_args.use_svd = runner_args.parallel_esd = False
    runner_args.load_dtype, runner_args.compute_dtype = "bfloat16", "float64"
    df = pd.DataFrame([{"model_id": "org/model", "base_model_relation": "", "source_model": ""}])
    command = shlex.split(runner.generate_commands(df, tmp_path, runner_args)[0])
    monkeypatch.setattr(sys, "argv", command[1:])
    worker_args = worker.parse_args()
    assert not worker_args.filter_zeros and not worker_args.use_svd and not worker_args.parallel_esd
    assert measurement_config(runner_args) == measurement_config(worker_args)


def test_standalone_worker_keeps_incompatible_outputs(monkeypatch, tmp_path):
    import worker
    args = SimpleNamespace(model_id="org/model", revision="", source_model="", base_model_relation="",
                           loader_scenario="", output_dir=str(tmp_path), overwrite=False)
    config = measurement_config(args, model_id="org/model")
    csv_path, h5_path = write_pair(tmp_path, dict(config, load_dtype="float16"))
    before = (csv_path.read_bytes(), h5_path.read_bytes())
    monkeypatch.setattr(worker, "parse_args", lambda: args)
    monkeypatch.setattr(worker, "load_model", lambda **kwargs: pytest.fail("Must not load an incompatible completed model"))
    assert worker.main() == 1
    assert (csv_path.read_bytes(), h5_path.read_bytes()) == before
    status = json.loads((tmp_path / "logs/terminal_status/org--model.json").read_text())
    assert status["reason"] == "incompatible_results"


def test_standalone_worker_skips_matching_configuration(monkeypatch, tmp_path):
    import worker
    args = SimpleNamespace(model_id="org/model", revision="", source_model="", base_model_relation="",
                           loader_scenario="", output_dir=str(tmp_path), overwrite=False)
    write_pair(tmp_path, measurement_config(args, model_id="org/model"))
    monkeypatch.setattr(worker, "parse_args", lambda: args)
    monkeypatch.setattr(worker, "load_model", lambda **kwargs: pytest.fail("Compatible results should resume without loading"))
    assert worker.main() == 0


@pytest.mark.parametrize("missing", ["csv", "h5"])
def test_standalone_worker_preserves_incomplete_artifacts(monkeypatch, tmp_path, missing):
    import worker
    args = SimpleNamespace(model_id="org/model", revision="", source_model="", base_model_relation="",
                           loader_scenario="", output_dir=str(tmp_path), overwrite=False)
    csv_path, h5_path = write_pair(tmp_path, measurement_config(args, model_id="org/model"))
    (csv_path if missing == "csv" else h5_path).unlink()
    remaining = h5_path if missing == "csv" else csv_path
    before = remaining.read_bytes()
    monkeypatch.setattr(worker, "parse_args", lambda: args)
    monkeypatch.setattr(worker, "load_model", lambda **kwargs: pytest.fail("Must not replace existing artifacts implicitly"))
    assert worker.main() == 1
    assert remaining.read_bytes() == before


def test_derived_view_does_not_merge_exotic_structural_namespaces():
    import worker
    names = ["encoder.layers.0.q_proj", "decoder.layers.0.q_proj", "model.layers.1.encoder.layers.q_proj"]
    mat, columns, n_layers = worker.build_tensor_from_pairs(names, [2, 3, 4])
    assert n_layers == 2
    assert len(columns) == 3
    assert sorted(mat[~pd.isna(mat)].tolist()) == [2, 3, 4]


def test_sparse_huge_depth_keeps_canonical_records_without_dense_allocation(tmp_path):
    import worker
    names = ["model.layers.999999999.proj"]
    csv_path = tmp_path / "stats" / "model.csv"
    h5_path = tmp_path / "metrics" / "model.h5"
    worker.save_results({"longname": names, "alpha": [2.]}, csv_path, "org/model", False, h5_output_path=h5_path)
    with h5py.File(h5_path) as h5:
        assert h5["layers/longname"].asstr()[:].tolist() == names
        assert h5["alpha"].shape == (0, 0)
        assert h5["alpha"].attrs["view_status"] == "unavailable"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hdf5_preserves_computed_spectrum_precision_without_upcasting(tmp_path, dtype):
    import numpy as np
    import worker
    eigs = np.array([0, 1e-8, 2.3], dtype=dtype)
    path = tmp_path / "model.h5"
    worker.save_results({"longname": [""], "alpha": [float("nan")], "slice": [""], "eigs": [eigs]},
                        tmp_path / "model.csv", "root-model", False, save_eigs=True, h5_output_path=path)
    with h5py.File(path) as h5:
        assert h5["eigs"][0].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(h5["eigs"][0], eigs)
        assert "missing_value" not in h5["layers/longname"].attrs
        assert "missing_value" not in h5["layers/slice"].attrs
