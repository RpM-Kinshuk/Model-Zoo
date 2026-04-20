import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


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


def test_validate_metrics_output_rejects_empty_longname_rows():
    worker = load_worker_module()

    metrics = {"longname": [], "alpha": []}

    assert worker.validate_metrics_output(metrics) == ("analyze", "analysis_empty")


def test_validate_metrics_output_rejects_longnames_without_usable_alpha_values():
    worker = load_worker_module()

    metrics = {"longname": ["model.layers.0.mlp.up_proj"], "alpha": [None]}

    assert worker.validate_metrics_output(metrics) == ("analyze", "analysis_empty")


def test_classify_retryable_failure_marks_only_transient_cases_retryable():
    worker = load_worker_module()

    assert worker.classify_retryable_failure(stage="load", reason="unsupported_loader_scenario") is False
    assert worker.classify_retryable_failure(stage="analyze", reason="analysis_empty") is False
    assert worker.classify_retryable_failure(stage="load", reason="model_load_error") is True


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


def test_main_rejects_empty_metrics_as_failure(tmp_path: Path):
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
    worker.net_esd_estimator = Mock(return_value={"longname": [], "alpha": []})
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
            "alpha": [1.0],
        }
    )

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
