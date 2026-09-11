import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


@contextmanager
def _worker_module_context():
    fake_torch = ModuleType("torch")
    fake_torch.float16 = object()
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        empty_cache=lambda: None,
        get_device_name=lambda index: "cpu",
    )
    fake_torch.nn = SimpleNamespace(Module=object)

    @dataclass
    class _FakeLoaderFailure(Exception):
        stage: str
        reason: str
        message: str

        def __post_init__(self):
            super().__init__(self.message)

        def __str__(self):
            return self.message

    fake_model_loader = ModuleType("model_loader")
    fake_model_loader.LoaderFailure = _FakeLoaderFailure
    fake_model_loader.load_model = lambda *args, **kwargs: None
    fake_model_loader.parse_model_string = lambda model_id: (
        model_id.split("@", 1)[0],
        model_id.split("@", 1)[1] if "@" in model_id else "",
    )
    fake_model_loader.safe_filename = lambda value: value.replace("/", "--").replace("@", "__")

    fake_net_esd = ModuleType("net_esd")
    fake_net_esd.net_esd_estimator = lambda *args, **kwargs: None

    original_modules = {
        "torch": sys.modules.get("torch"),
        "model_loader": sys.modules.get("model_loader"),
        "net_esd": sys.modules.get("net_esd"),
    }
    try:
        sys.modules["torch"] = fake_torch
        sys.modules["model_loader"] = fake_model_loader
        sys.modules["net_esd"] = fake_net_esd
        yield
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_worker_module():
    with _worker_module_context():
        module_path = PROJECT_ROOT / "src" / "worker.py"
        spec = importlib.util.spec_from_file_location("worker_under_test", module_path)
        worker = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(worker)
    return worker


def test_temp_output_path_is_hidden_sidecar(tmp_path: Path):
    worker = load_worker_module()

    final_path = tmp_path / "stats" / "org--model.csv"

    temp_path = worker.temp_output_path(final_path)

    assert temp_path.name == ".org--model.csv.tmp"


def test_finalize_output_path_renames_temp_file(tmp_path: Path):
    worker = load_worker_module()

    final_path = tmp_path / "stats" / "org--model.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = worker.temp_output_path(final_path)
    temp_path.write_text("alpha\n1.0\n")

    worker.finalize_output_path(temp_path, final_path)

    assert final_path.exists()
    assert final_path.read_text() == "alpha\n1.0\n"
    assert not temp_path.exists()


def test_terminal_status_path_uses_per_model_json_file(tmp_path: Path):
    worker = load_worker_module()

    path = worker._terminal_status_path(tmp_path, "org/model")

    assert path == tmp_path / "logs" / "terminal_status" / "org--model.json"


def test_record_terminal_status_writes_one_json_file_per_model(tmp_path: Path):
    worker = load_worker_module()
    output_dir = tmp_path / "results"

    worker.record_terminal_status(
        output_dir=output_dir,
        model_id="org/model",
        status="success",
        stage="save",
        reason="completed",
        message="done",
    )

    status_file = output_dir / "logs" / "terminal_status" / "org--model.json"
    payload = json.loads(status_file.read_text())

    assert payload["model_id"] == "org/model"
    assert payload["status"] == "success"
    assert payload["stage"] == "save"
    assert payload["reason"] == "completed"
    assert payload["message"] == "done"


def test_record_failure_writes_jsonl_and_text_summary(tmp_path: Path):
    worker = load_worker_module()
    output_dir = tmp_path / "results"

    worker.record_failure(
        output_dir=output_dir,
        model_id="org/model",
        stage="load",
        reason="repo_gated",
        message="access denied",
        attempt=1,
    )

    jsonl_path = output_dir / "logs" / "failure_records.jsonl"
    text_path = output_dir / "logs" / "failed_models.txt"

    record = json.loads(jsonl_path.read_text().strip())
    assert record["model_id"] == "org/model"
    assert record["stage"] == "load"
    assert record["reason"] == "repo_gated"
    assert record["message"] == "access denied"
    assert record["attempt"] == 1
    assert "org/model\tload\trepo_gated\taccess denied" in text_path.read_text()


def test_record_failure_appends_one_valid_json_line_per_call(tmp_path: Path):
    worker = load_worker_module()
    output_dir = tmp_path / "results"

    worker.record_failure(
        output_dir=output_dir,
        model_id="org/model-a",
        stage="load",
        reason="repo_gated",
        message="access denied",
        attempt=1,
    )
    worker.record_failure(
        output_dir=output_dir,
        model_id="org/model-b",
        stage="save",
        reason="save_error",
        message="disk full",
        attempt=2,
    )

    jsonl_path = output_dir / "logs" / "failure_records.jsonl"
    lines = jsonl_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]

    assert len(lines) == 2
    assert records[0]["model_id"] == "org/model-a"
    assert records[1]["model_id"] == "org/model-b"
    assert records[0]["attempt"] == 1
    assert records[1]["attempt"] == 2


def test_validate_metrics_output_rejects_empty_longname_rows():
    worker = load_worker_module()

    metrics = {"longname": [], "alpha": []}

    assert worker.validate_metrics_output(metrics) == ("analyze", "analysis_empty")


def test_validate_metrics_output_rejects_longnames_without_usable_alpha_values():
    worker = load_worker_module()

    metrics = {"longname": ["model.layers.0.mlp.up_proj"], "alpha": [None]}

    assert worker.validate_metrics_output(metrics) == ("analyze", "analysis_empty")


@pytest.mark.parametrize("alpha", [-1.0, 0.0, 1.0, float("nan"), float("inf"), -float("inf")])
def test_validate_metrics_output_rejects_invalid_power_law_fits(alpha):
    worker = load_worker_module()

    metrics = {"longname": ["model.layers.0.proj"], "alpha": [alpha]}

    assert worker.validate_metrics_output(metrics) == ("analyze", "analysis_empty")


@pytest.mark.parametrize("alphas", [[float("nan"), 2.5], [2.5, float("nan")]])
def test_partial_fits_preserve_rows_and_missing_values_in_saved_results(tmp_path: Path, capsys, alphas):
    worker = load_worker_module()
    output_path = tmp_path / "stats" / "org--model.csv"
    h5_path = tmp_path / "metrics" / "org--model.h5"
    metrics = {
        "longname": ["model.layers.0.proj", "model.layers.1.proj"],
        "alpha": alphas,
        "norm": [4.0, 30.0],
    }

    assert worker.validate_metrics_output(metrics) is None
    worker.save_results(metrics, output_path, "org/model", False, h5_output_path=h5_path)

    saved = worker.pd.read_csv(output_path)
    assert len(saved) == 2
    worker.np.testing.assert_allclose(saved["alpha"], alphas, equal_nan=True)
    assert saved["norm"].tolist() == [4.0, 30.0]
    with worker.h5py.File(h5_path, "r") as h5:
        assert h5["alpha"].shape == (2, 1)
        worker.np.testing.assert_allclose(h5["alpha"][:, 0], alphas, equal_nan=True)
        assert h5.attrs["numerics_version"] == worker.NUMERICS_VERSION
        assert h5.attrs["format_version"] == "2.0"
        assert h5["layers/longname"].asstr()[:].tolist() == metrics["longname"]
        worker.np.testing.assert_allclose(h5["layers/alpha"][:], alphas, equal_nan=True)
    assert "Finite power-law fits: 1/2 layers" in capsys.readouterr().out


def test_alpha_matrix_keeps_a_module_with_no_fits():
    worker = load_worker_module()
    mat, modules, layers = worker.build_tensor_from_pairs(
        ["model.layers.0.q_proj", "model.layers.0.k_proj", "model.layers.1.k_proj"],
        [2.5, float("nan"), float("nan")],
    )

    assert layers == 2
    assert modules == ["k_proj", "q_proj"]
    assert mat.shape == (2, 2)
    assert worker.np.isnan(mat[:, 0]).all()
    assert mat[0, 1] == 2.5
    assert worker.np.isnan(mat[1, 1])


def test_spectra_without_any_fits_can_be_saved(tmp_path: Path, capsys):
    worker = load_worker_module()
    output_path = tmp_path / "stats" / "org--model.csv"
    h5_path = tmp_path / "metrics" / "org--model.h5"
    metrics = {
        "longname": ["model.layers.0.proj", "model.layers.1.proj"],
        "alpha": [float("nan"), float("nan")],
        "norm": [4.0, 0.0],
        "num_evals": [1, 0],
        "fit_xmin": [float("nan"), float("nan")],
        "n_tail": [0, 0],
        "eigs": [worker.np.array([4.0]), worker.np.array([])],
    }

    assert worker.validate_metrics_output(metrics) is None
    worker.save_results(metrics, output_path, "org/model", False, h5_output_path=h5_path, save_eigs=True)

    saved = worker.pd.read_csv(output_path)
    assert len(saved) == 2
    assert saved["alpha"].isna().all()
    assert saved["fit_xmin"].isna().all()
    assert saved["n_tail"].tolist() == [0, 0]
    with worker.h5py.File(h5_path, "r") as h5:
        assert h5["alpha"].shape == (2, 1)
        assert worker.np.isnan(h5["alpha"][:]).all()
        assert h5["eigs"][0].tolist() == [4.0]
        assert h5["eigs"][1].tolist() == []
    assert "Finite power-law fits: 0/2 layers" in capsys.readouterr().out


@pytest.mark.parametrize("norm, num_evals", [
    (float("nan"), 4), (float("inf"), 4), (-1., 4), (None, 4), (0., -1), (0., None),
])
def test_missing_fits_do_not_make_invalid_spectral_metrics_usable(norm, num_evals):
    worker = load_worker_module()
    metrics = {
        "longname": ["model.layers.0.proj"], "alpha": [float("nan")],
        "norm": [norm], "num_evals": [num_evals],
    }

    assert worker.validate_metrics_output(metrics) == ("analyze", "analysis_empty")


@pytest.mark.parametrize("alphas", [[], [float("nan"), float("nan")]])
def test_spectral_measurements_still_require_aligned_fit_rows(alphas):
    worker = load_worker_module()
    metrics = {
        "longname": ["model.layers.0.proj"], "alpha": alphas,
        "norm": [4.0], "num_evals": [1],
    }

    assert worker.validate_metrics_output(metrics) == ("analyze", "analysis_empty")


def test_save_results_keeps_eigenvalues_only_in_h5_when_requested(tmp_path: Path):
    worker = load_worker_module()
    output_path = tmp_path / "stats" / "org--model.csv"
    h5_path = tmp_path / "metrics" / "org--model.h5"
    metrics = {
        "longname": ["model.layers.0.mlp.up_proj", "model.layers.1.mlp.up_proj"],
        "alpha": [1.5, 2.5],
        "eigs": [
            worker.np.array([1.0, 2.0], dtype=float),
            worker.np.array([3.0, 4.0, 5.0], dtype=float),
        ],
    }

    worker.save_results(
        metrics,
        output_path,
        model_id="org/model",
        is_adapter=False,
        save_eigs=True,
        h5_output_path=h5_path,
    )

    csv_text = output_path.read_text()
    assert "eigs" not in csv_text
    assert "[1. 2.]" not in csv_text
    with worker.h5py.File(h5_path, "r") as h5:
        assert "eigs" in h5
        assert h5["eigs"][0].tolist() == [1.0, 2.0]
        assert h5["eigs"][1].tolist() == [3.0, 4.0, 5.0]
        assert h5["eigs"].attrs["aligned_with"] == "/layers/longname"
        assert h5["layers/longname"].asstr()[:].tolist() == metrics["longname"]


def test_alpha_matrix_separates_encoder_and_decoder_even_at_different_depths():
    worker = load_worker_module()
    mat, modules, layers = worker.build_tensor_from_pairs(
        ["encoder.layers.0.self_attn.q_proj", "decoder.layers.0.self_attn.q_proj",
         "decoder.layers.1.self_attn.q_proj"],
        [2.0, 4.0, float("nan")],
    )

    assert layers == 2
    assert modules == ["decoder.layers.self_attn.q_proj", "encoder.layers.self_attn.q_proj"]
    worker.np.testing.assert_allclose(mat, [[4.0, 2.0], [float("nan"), float("nan")]], equal_nan=True)


@pytest.mark.parametrize("longnames, expected_status", [
    (["features.0", "classifier", ""], "unavailable"),
    (["model.layers.0.proj", "classifier", ""], "partial"),
])
def test_arbitrary_layer_names_remain_canonical_when_depth_view_is_incomplete(
    tmp_path: Path, longnames, expected_status,
):
    worker = load_worker_module()
    output_path = tmp_path / "stats" / "org--model.csv"
    h5_path = tmp_path / "metrics" / "org--model.h5"
    metrics = {
        "longname": longnames,
        "alpha": [2.0, float("nan"), 4.0],
        "raw_norm": [30.0, 1e-8, 5.0],
        "fit_status": ["estimated", "insufficient_tail", "estimated"],
        "weight_shape": [(4, 4), (1, 1), (2, 2)],
        "eigs": [worker.np.array([1., 4., 9., 16.]), worker.np.array([1e-8]), worker.np.array([1., 4.])],
    }

    worker.save_results(metrics, output_path, "org/model", False, save_eigs=True, h5_output_path=h5_path)

    with worker.h5py.File(h5_path, "r") as h5:
        assert h5["layers/longname"].asstr()[:].tolist() == longnames
        worker.np.testing.assert_allclose(h5["layers/alpha"][:], metrics["alpha"], equal_nan=True)
        worker.np.testing.assert_allclose(h5["layers/raw_norm"][:], metrics["raw_norm"])
        assert h5["layers/fit_status"].asstr()[:].tolist() == metrics["fit_status"]
        assert json.loads(h5["layers/weight_shape"].asstr()[0]) == [4, 4]
        assert h5["alpha"].attrs["view_status"] == expected_status
        assert h5["alpha"].attrs["canonical_records"] == "/layers"
        expected_unmapped = [name for name in longnames if worker.parse_longname(name)[0] is None]
        assert h5["alpha_unmapped_longname"].asstr()[:].tolist() == expected_unmapped
        assert h5["alpha"].shape == ((0, 0) if expected_status == "unavailable" else (1, 1))
        for i, values in enumerate(metrics["eigs"]):
            worker.np.testing.assert_array_equal(h5["eigs"][i], values)


@pytest.mark.parametrize("metrics, error", [
    ({"longname": ["a", "a"], "alpha": [2.0, 4.0]}, "Duplicate canonical"),
    ({"longname": ["a", "b"], "alpha": [2.0]}, "expected 2"),
    ({"longname": ["a"], "alpha": [2.0], "eigs": [[], []]}, "expected 1"),
    ({"longname": ["a"], "alpha": [2.0], "norm": [1.0, 2.0]}, "expected 1"),
    ({"longname": [None, "a"], "alpha": [2.0, 4.0]}, "must be strings"),
])
def test_save_results_rejects_ambiguous_or_misaligned_rows_before_writing(tmp_path: Path, metrics, error):
    worker = load_worker_module()
    output_path = tmp_path / "model.csv"
    h5_path = tmp_path / "model.h5"

    with pytest.raises(ValueError, match=error):
        worker.save_results(metrics, output_path, "org/model", False, h5_output_path=h5_path)

    assert not output_path.exists()
    assert not h5_path.exists()


def test_save_results_removes_only_explicit_aligned_legacy_summary(tmp_path: Path):
    worker = load_worker_module()
    metrics = {
        "longname": ["classifier", "features.0", None],
        "alpha": [2.0, None, 2.0],
        "norm": [1.0, 0.0, 0.5],
        "eigs": [[1.0], [], None],
    }
    output_path, h5_path = tmp_path / "model.csv", tmp_path / "model.h5"

    worker.save_results(metrics, output_path, "org/model", False, h5_output_path=h5_path, save_eigs=True)

    assert len(worker.pd.read_csv(output_path)) == 2
    with worker.h5py.File(h5_path, "r") as h5:
        assert h5["layers/longname"].asstr()[:].tolist() == ["classifier", "features.0"]
        assert h5["layers/alpha"][0] == 2.0
        assert worker.np.isnan(h5["layers/alpha"][1])
        assert len(h5["eigs"]) == 2


def test_save_results_records_measurement_configuration_and_coverage(tmp_path: Path):
    worker = load_worker_module()
    metrics = {"longname": ["classifier"], "alpha": [float("nan")], "raw_norm": [0.0]}
    measurement_config = {"numerics_version": worker.NUMERICS_VERSION, "load_dtype": "auto", "filter_zeros": False}
    coverage = {
        "counts": {"eligible": 2, "analyzed": 1, "skipped": 1},
        "modules": [{"longname": "classifier", "status": "analyzed"},
                    {"longname": "packed", "status": "skipped", "reason": "unsupported_dtype"}],
    }
    h5_path = tmp_path / "model.h5"

    worker.save_results(metrics, tmp_path / "model.csv", "org/model", False,
                        h5_output_path=h5_path, measurement_config=measurement_config, coverage=coverage)

    with worker.h5py.File(h5_path, "r") as h5:
        assert json.loads(h5.attrs["measurement_config_json"]) == measurement_config
        assert json.loads(h5["coverage"].asstr()[()]) == coverage
        assert h5.attrs["coverage_analyzed"] == 1
        assert h5.attrs["coverage_skipped"] == 1
        assert "eigs" not in h5


def test_classify_retryable_failure_marks_only_transient_cases_retryable():
    worker = load_worker_module()

    assert worker.classify_retryable_failure(stage="load", reason="unsupported_loader_scenario") is False
    assert worker.classify_retryable_failure(stage="analyze", reason="analysis_empty") is False
    assert worker.classify_retryable_failure(stage="load", reason="model_load_error") is True


def test_classify_retryable_failure_requires_stage_specific_reason_matches():
    worker = load_worker_module()

    assert worker.classify_retryable_failure(stage="load", reason="model_load_error") is True
    assert worker.classify_retryable_failure(stage="analyze", reason="model_load_error") is False
    assert worker.classify_retryable_failure(stage="save", reason="save_error") is True
    assert worker.classify_retryable_failure(stage="load", reason="save_error") is False


def test_main_regenerates_incomplete_outputs_when_overwrite_is_requested(tmp_path: Path):
    worker = load_worker_module()

    class _FakeParam:
        def numel(self):
            return 1

        @property
        def device(self):
            return "cpu"

    class _FakeModel:
        def parameters(self):
            return [_FakeParam()]

    output_file = tmp_path / "stats" / "org--model.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("stale\n")

    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="standard_transformers",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=True,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )
    worker.load_model = Mock(return_value=(_FakeModel(), False))
    worker.net_esd_estimator = Mock(
        return_value={
            "longname": ["model.layers.0.mlp.up_proj"],
            "alpha": [2.0],
        }
    )

    def fake_save_results(metrics, output_path, *args, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("alpha\n1.0\n")
        h5_output_path = kwargs["h5_output_path"]
        h5_output_path.parent.mkdir(parents=True, exist_ok=True)
        h5_output_path.write_text("h5-temp")

    worker.save_results = fake_save_results

    exit_code = worker.main()

    metrics_file = tmp_path / "metrics" / "org--model.h5"
    terminal_status_file = tmp_path / "logs" / "terminal_status" / "org--model.json"

    assert exit_code == 0
    assert worker.load_model.call_count == 1
    assert worker.net_esd_estimator.call_count == 1
    assert output_file.read_text() == "alpha\n1.0\n"
    assert metrics_file.exists()
    assert metrics_file.read_text() == "h5-temp"
    assert json.loads(terminal_status_file.read_text())["status"] == "success"


def test_main_clears_partial_outputs_before_explicit_overwrite_regeneration(tmp_path: Path):
    worker = load_worker_module()

    class _FakeLoaderFailure(Exception):
        def __init__(self, stage: str, reason: str, message: str):
            self.stage = stage
            self.reason = reason
            self.message = message
            super().__init__(message)

        def __str__(self):
            return self.message

    output_file = tmp_path / "stats" / "org--model.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("stale\n")

    worker.LoaderFailure = _FakeLoaderFailure
    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="quantized_alt_format",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=True,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )
    worker.load_model = Mock(
        side_effect=_FakeLoaderFailure("load", "unsupported_loader_scenario", "unsupported")
    )
    worker.net_esd_estimator = Mock()
    worker.save_results = Mock()

    exit_code = worker.main()

    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())
    terminal_status_file = tmp_path / "logs" / "terminal_status" / "org--model.json"
    terminal_record = json.loads(terminal_status_file.read_text())

    assert exit_code == 1
    assert not output_file.exists()
    assert failure_record["stage"] == "load"
    assert failure_record["reason"] == "unsupported_loader_scenario"
    assert terminal_record["status"] == "failed"
    assert terminal_record["reason"] == "unsupported_loader_scenario"


def test_main_clears_existing_outputs_when_overwrite_is_requested(tmp_path: Path):
    worker = load_worker_module()

    class _FakeLoaderFailure(Exception):
        def __init__(self, stage: str, reason: str, message: str):
            self.stage = stage
            self.reason = reason
            self.message = message
            super().__init__(message)

        def __str__(self):
            return self.message

    output_file = tmp_path / "stats" / "org--model.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("old\n")
    metrics_file = tmp_path / "metrics" / "org--model.h5"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text("old-h5\n")

    worker.LoaderFailure = _FakeLoaderFailure
    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="quantized_alt_format",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=True,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )
    worker.load_model = Mock(
        side_effect=_FakeLoaderFailure("load", "unsupported_loader_scenario", "unsupported")
    )
    worker.net_esd_estimator = Mock()
    worker.save_results = Mock()

    exit_code = worker.main()

    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert not output_file.exists()
    assert not metrics_file.exists()
    assert failure_record["stage"] == "load"
    assert failure_record["reason"] == "unsupported_loader_scenario"


def test_main_records_preflight_cleanup_failure(tmp_path: Path):
    worker = load_worker_module()

    output_file = tmp_path / "stats" / "org--model.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("old\n")

    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="standard_transformers",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=True,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )

    original_cleanup_output_artifacts = worker.cleanup_output_artifacts

    def failing_cleanup_output_artifacts(*paths):
        raise RuntimeError("cleanup failed")

    worker.cleanup_output_artifacts = failing_cleanup_output_artifacts
    worker.load_model = Mock()
    worker.net_esd_estimator = Mock()
    worker.save_results = Mock()

    try:
        exit_code = worker.main()
    finally:
        worker.cleanup_output_artifacts = original_cleanup_output_artifacts

    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert worker.load_model.call_count == 0
    assert failure_record["attempt"] == 0
    assert failure_record["stage"] == "save"
    assert failure_record["reason"] == "save_error"
    assert "cleanup failed" in failure_record["message"]


def test_main_records_permanent_loader_failure_without_retry_or_final_outputs(tmp_path: Path):
    worker = load_worker_module()

    class _FakeLoaderFailure(Exception):
        def __init__(self, stage: str, reason: str, message: str):
            self.stage = stage
            self.reason = reason
            self.message = message
            super().__init__(message)

        def __str__(self):
            return self.message

    worker.LoaderFailure = _FakeLoaderFailure
    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="quantized_alt_format",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=False,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=2,
    )
    worker.load_model = Mock(side_effect=_FakeLoaderFailure("load", "unsupported_loader_scenario", "unsupported"))
    worker.net_esd_estimator = Mock()
    worker.save_results = Mock()

    exit_code = worker.main()

    output_file = tmp_path / "stats" / "org--model.csv"
    metrics_file = tmp_path / "metrics" / "org--model.h5"
    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert worker.load_model.call_count == 1
    assert worker.net_esd_estimator.call_count == 0
    assert worker.save_results.call_count == 0
    assert not output_file.exists()
    assert not metrics_file.exists()
    assert failure_record["stage"] == "load"
    assert failure_record["reason"] == "unsupported_loader_scenario"


@pytest.mark.parametrize("metrics", [
    {"longname": [], "alpha": []},
    {"longname": ["model.layers.0.proj"], "alpha": [float("nan")]},
])
def test_main_rejects_empty_metrics_as_failure(tmp_path: Path, metrics):
    worker = load_worker_module()

    class _FakeParam:
        def numel(self):
            return 1

        @property
        def device(self):
            return "cpu"

    class _FakeModel:
        def parameters(self):
            return [_FakeParam()]

    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="standard_transformers",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=False,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=1,
    )
    worker.load_model = Mock(return_value=(_FakeModel(), False))
    worker.net_esd_estimator = Mock(return_value=metrics)
    worker.save_results = Mock()

    exit_code = worker.main()

    output_file = tmp_path / "stats" / "org--model.csv"
    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert worker.load_model.call_count == 1
    assert worker.net_esd_estimator.call_count == 1
    assert worker.save_results.call_count == 0
    assert not output_file.exists()
    assert failure_record["stage"] == "analyze"
    assert failure_record["reason"] == "analysis_empty"


def test_main_classifies_unwrapped_post_load_exception_as_analyze_failure(tmp_path: Path):
    worker = load_worker_module()

    class _BrokenModel:
        def parameters(self):
            raise RuntimeError("parameter inspection failed")

    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="standard_transformers",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=False,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )
    worker.load_model = Mock(return_value=(_BrokenModel(), False))
    worker.net_esd_estimator = Mock()
    worker.save_results = Mock()

    exit_code = worker.main()

    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert worker.net_esd_estimator.call_count == 0
    assert worker.save_results.call_count == 0
    assert failure_record["stage"] == "analyze"
    assert failure_record["reason"] == "analysis_exception"
    assert "parameter inspection failed" in failure_record["message"]


def test_main_treats_h5_write_failure_as_failed_run_without_final_csv(tmp_path: Path):
    worker = load_worker_module()

    class _FakeParam:
        def numel(self):
            return 1

        @property
        def device(self):
            return "cpu"

    class _FakeModel:
        def parameters(self):
            return [_FakeParam()]

    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="standard_transformers",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=False,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )
    worker.load_model = Mock(return_value=(_FakeModel(), False))
    worker.net_esd_estimator = Mock(
        return_value={
            "longname": ["model.layers.0.mlp.up_proj"],
            "alpha": [2.0],
        }
    )

    worker.net_esd_estimator.return_value["eigs"] = [worker.np.array([1., 2.])]
    original_save_h5 = worker.save_h5

    def failing_save_h5(*args, **kwargs):
        raise RuntimeError("disk full")

    worker.save_h5 = failing_save_h5

    try:
        exit_code = worker.main()
    finally:
        worker.save_h5 = original_save_h5

    output_file = tmp_path / "stats" / "org--model.csv"
    temp_output_file = tmp_path / "stats" / ".org--model.csv.tmp"
    metrics_file = tmp_path / "metrics" / "org--model.h5"
    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert not output_file.exists()
    assert not temp_output_file.exists()
    assert not metrics_file.exists()
    assert failure_record["stage"] == "save"
    assert failure_record["reason"] == "save_error"
    assert "disk full" in failure_record["message"]


def test_main_classifies_unwrapped_pre_save_exception_as_save_failure(tmp_path: Path):
    worker = load_worker_module()

    class _FakeParam:
        def numel(self):
            return 1

        @property
        def device(self):
            return "cpu"

    class _FakeModel:
        def parameters(self):
            return [_FakeParam()]

    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="standard_transformers",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=False,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )
    worker.load_model = Mock(return_value=(_FakeModel(), False))
    worker.net_esd_estimator = Mock(
        return_value={
            "longname": ["model.layers.0.mlp.up_proj"],
            "alpha": [2.0],
        }
    )
    worker.save_results = Mock()

    original_cleanup_temp_path = worker.cleanup_temp_path
    cleanup_calls = {"count": 0}

    def flaky_cleanup_temp_path(path):
        if path.suffix == ".tmp" and cleanup_calls["count"] == 0:
            cleanup_calls["count"] += 1
            raise RuntimeError("temp cleanup failed")
        original_cleanup_temp_path(path)

    worker.cleanup_temp_path = flaky_cleanup_temp_path

    try:
        exit_code = worker.main()
    finally:
        worker.cleanup_temp_path = original_cleanup_temp_path

    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert worker.save_results.call_count == 0
    assert failure_record["stage"] == "save"
    assert failure_record["reason"] == "save_error"
    assert "temp cleanup failed" in failure_record["message"]


def test_main_cleans_up_final_csv_if_h5_finalize_fails_after_csv_finalize(tmp_path: Path):
    worker = load_worker_module()

    class _FakeParam:
        def numel(self):
            return 1

        @property
        def device(self):
            return "cpu"

    class _FakeModel:
        def parameters(self):
            return [_FakeParam()]

    worker.parse_args = lambda: SimpleNamespace(
        model_id="org/model",
        revision="",
        base_model_relation="",
        source_model="",
        loader_scenario="standard_transformers",
        primary_type_bucket="",
        output_dir=str(tmp_path),
        overwrite=False,
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        parallel_esd=True,
        use_svd=False,
        device_map="cpu",
        max_retries=0,
    )
    worker.load_model = Mock(return_value=(_FakeModel(), False))
    worker.net_esd_estimator = Mock(
        return_value={
            "longname": ["model.layers.0.mlp.up_proj"],
            "alpha": [2.0],
        }
    )

    def fake_save_results(metrics, output_path, *args, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("alpha\n1.0\n")
        h5_output_path = kwargs["h5_output_path"]
        h5_output_path.parent.mkdir(parents=True, exist_ok=True)
        h5_output_path.write_text("h5-temp")

    original_finalize_output_path = worker.finalize_output_path
    finalize_calls = []

    def flaky_finalize_output_path(temp_path, final_path):
        finalize_calls.append(final_path.name)
        original_finalize_output_path(temp_path, final_path)
        if final_path.suffix == ".h5":
            raise RuntimeError("rename failed")

    worker.save_results = fake_save_results
    worker.finalize_output_path = flaky_finalize_output_path

    try:
        exit_code = worker.main()
    finally:
        worker.finalize_output_path = original_finalize_output_path

    output_file = tmp_path / "stats" / "org--model.csv"
    metrics_file = tmp_path / "metrics" / "org--model.h5"
    failure_record = json.loads((tmp_path / "logs" / "failure_records.jsonl").read_text().strip())

    assert exit_code == 1
    assert finalize_calls == ["org--model.csv", "org--model.h5"]
    assert not output_file.exists()
    assert not metrics_file.exists()
    assert failure_record["stage"] == "save"
    assert failure_record["reason"] == "save_error"
    assert "rename failed" in failure_record["message"]


def test_write_worker_heartbeat_records_generic_state(tmp_path: Path):
    worker = load_worker_module()
    heartbeat_path = tmp_path / "worker-heartbeat.json"

    worker.write_worker_heartbeat(
        heartbeat_path,
        model_id="org/model",
        state="running",
        stage="loading",
        pid=123,
    )

    payload = json.loads(heartbeat_path.read_text())
    assert payload["model_id"] == "org/model"
    assert payload["state"] == "running"
    assert payload["stage"] == "loading"
    assert payload["pid"] == 123
    assert "updated_at" in payload
    assert "stage_entered_at" in payload


def test_heartbeat_reporter_keeps_stage_entered_at_until_stage_changes(tmp_path: Path):
    worker = load_worker_module()
    heartbeat_path = tmp_path / "worker-heartbeat.json"
    reporter = worker.HeartbeatReporter(str(heartbeat_path), "org/model", interval_seconds=60)

    reporter.start(stage="load")
    first = json.loads(heartbeat_path.read_text())
    reporter.update(stage="load")
    second = json.loads(heartbeat_path.read_text())
    time.sleep(0.001)
    reporter.update(stage="analyze")
    third = json.loads(heartbeat_path.read_text())
    reporter.stop(state="success", stage="analyze")

    assert first["stage"] == "load"
    assert second["stage_entered_at"] == first["stage_entered_at"]
    assert third["stage"] == "analyze"
    assert third["stage_entered_at"] != first["stage_entered_at"]
