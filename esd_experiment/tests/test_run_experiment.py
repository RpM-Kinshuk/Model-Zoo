import sys
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "src" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_experiment_under_test", MODULE_PATH)
run_experiment = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(run_experiment)

generate_commands = run_experiment.generate_commands
get_completed_models = run_experiment.get_completed_models
load_model_list = run_experiment.load_model_list
apply_preflight = run_experiment.apply_preflight
collect_run_outcomes = run_experiment.collect_run_outcomes


@contextmanager
def _worker_module_context():
    fake_torch = ModuleType("torch")
    fake_torch.float16 = object()
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        empty_cache=lambda: None,
        get_device_name=lambda index: "cpu",
        get_device_properties=lambda index: SimpleNamespace(total_memory=0, major=0, minor=0),
    )
    fake_torch.nn = SimpleNamespace(Module=object)

    fake_numpy = ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_numpy.full = lambda *args, **kwargs: None
    fake_numpy.nan = float("nan")

    fake_h5py = ModuleType("h5py")
    fake_h5py.File = lambda *args, **kwargs: None

    @dataclass
    class _FakeLoaderFailure(Exception):
        stage: str
        reason: str
        message: str

        def __post_init__(self) -> None:
            super().__init__(self.message)

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
        "numpy": sys.modules.get("numpy"),
        "h5py": sys.modules.get("h5py"),
        "model_loader": sys.modules.get("model_loader"),
        "net_esd": sys.modules.get("net_esd"),
        "pandas": sys.modules.get("pandas"),
    }
    try:
        sys.modules["torch"] = fake_torch
        sys.modules["numpy"] = fake_numpy
        sys.modules["h5py"] = fake_h5py
        sys.modules["model_loader"] = fake_model_loader
        sys.modules["net_esd"] = fake_net_esd
        yield
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_available_backends_includes_compressed_tensors(monkeypatch):
    def fake_find_spec(name):
        if name in {"gptqmodel", "compressed_tensors"}:
            return object()
        return None

    def fake_run(cmd, stdout, stderr, env, timeout, check):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_experiment.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(run_experiment.subprocess, "run", fake_run)

    assert run_experiment._available_backends() == {"gptq", "compressed_tensors"}


def test_parse_args_accepts_max_concurrent_jobs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--model_list",
            "models.csv",
            "--output_dir",
            "results",
            "--max_concurrent_jobs",
            "3",
        ],
    )

    args = run_experiment.parse_args()

    assert args.max_concurrent_jobs == 3


def test_parse_args_rejects_non_positive_max_concurrent_jobs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--model_list",
            "models.csv",
            "--output_dir",
            "results",
            "--max_concurrent_jobs",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        run_experiment.parse_args()


def test_create_runtime_config_includes_max_concurrent_jobs(tmp_path):
    config_path = tmp_path / "gpu_config.json"
    args = SimpleNamespace(
        gpus=[0, 1, 2, 3],
        max_check=5,
        gpu_memory_threshold=500,
        max_concurrent_jobs=3,
    )

    run_experiment.create_runtime_config(args, config_path)

    assert config_path.read_text()
    config = run_experiment.json.loads(config_path.read_text())
    assert config["max_concurrent_jobs"] == 3


def test_available_backends_uses_cuda_hidden_subprocess_not_parent_import(monkeypatch):
    calls = []

    def fake_find_spec(name):
        if name == "gptqmodel":
            return object()
        return None

    def fake_run(cmd, stdout, stderr, env, timeout, check):
        calls.append((cmd, env.copy(), timeout, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_experiment.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        run_experiment.importlib,
        "import_module",
        Mock(side_effect=AssertionError("parent process must not import GPU backends")),
    )
    monkeypatch.setattr(run_experiment.subprocess, "run", fake_run)

    assert run_experiment._available_backends() == {"gptq"}
    assert calls
    assert all(env.get("CUDA_VISIBLE_DEVICES") == "" for _, env, _, _ in calls)


def test_available_backends_skips_broken_compressed_tensors_without_retry_path(monkeypatch):
    def fake_find_spec(name):
        if name == "compressed_tensors":
            return object()
        return None

    def fake_run(cmd, stdout, stderr, env, timeout, check):
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(run_experiment.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(run_experiment.subprocess, "run", fake_run)

    assert run_experiment._available_backends() == set()


def test_available_backends_retries_compressed_tensors_after_gptq_side_effect(monkeypatch):
    probe_sequences = []

    def fake_find_spec(name):
        if name in {"gptqmodel", "compressed_tensors"}:
            return object()
        return None

    def fake_run(cmd, stdout, stderr, env, timeout, check):
        code = cmd[-1]
        sequence = tuple(
            name for name in ("gptqmodel", "compressed_tensors") if name in code
        )
        probe_sequences.append(sequence)
        return SimpleNamespace(
            returncode=0 if sequence == ("gptqmodel", "compressed_tensors") else 1
        )

    monkeypatch.setattr(run_experiment.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(run_experiment.subprocess, "run", fake_run)

    assert run_experiment._available_backends() == {"gptq", "compressed_tensors"}
    assert probe_sequences == [
        ("gptqmodel",),
        ("compressed_tensors",),
        ("gptqmodel", "compressed_tensors"),
    ]


def test_row_backend_status_marks_compressed_tensors_from_tags():
    row = pd.Series(
        {
            "model_id": "org/compressed-model",
            "tags": "['compressed-tensors']",
            "loader_scenario": "quantized_transformers_native",
        }
    )

    status = run_experiment._row_backend_status(
        row,
        {"compressed_tensors"},
        effective_loader="compressed_tensors",
    )

    assert status == "available"


def test_row_backend_status_overrides_stale_available_status_when_backend_missing():
    row = pd.Series(
        {
            "model_id": "org/compressed-model",
            "tags": "['compressed-tensors']",
            "loader_scenario": "quantized_transformers_native",
            "backend_status": "available",
        }
    )

    status = run_experiment._row_backend_status(
        row,
        set(),
        effective_loader="compressed_tensors",
    )

    assert status == "missing"


def test_row_backend_status_checks_adapter_source_backend_before_stale_status():
    row = pd.Series(
        {
            "model_id": "org/adapter",
            "source_model": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
            "base_model": "",
            "base_model_name_or_path": "",
            "loader_scenario": "adapter_requires_base",
            "backend_status": "available",
        }
    )

    status = run_experiment._row_backend_status(
        row,
        set(),
        effective_loader="adapter_requires_base",
    )

    assert status == "missing"


def test_load_model_list_preserves_curated_columns_and_normalizes_values(tmp_path: Path):
    csv_path = tmp_path / "curated.csv"
    pd.DataFrame(
        [
            {
                "model_id": "org/model-a",
                "revision_norm": " rev-a ",
                "source_model": "base/model",
                "base_model_relation": "adapter",
                "loader_scenario": "adapter_requires_base",
                "primary_type_bucket": "adapter",
                "lineage_status": " curated ",
                "candidate_source": "seed-a",
            },
            {
                "model_id": "org/model-b",
                "revision_norm": None,
                "source_model": "",
                "base_model_relation": "source",
                "loader_scenario": None,
                "primary_type_bucket": None,
                "lineage_status": None,
                "candidate_source": None,
            },
        ]
    ).to_csv(csv_path, index=False)

    df = load_model_list(str(csv_path))

    assert list(df["model_id"]) == ["org/model-a", "org/model-b"]
    assert df.loc[0, "revision_norm"] == "rev-a"
    assert df.loc[1, "revision_norm"] == ""
    assert df.loc[0, "loader_scenario"] == "adapter_requires_base"
    assert df.loc[1, "loader_scenario"] == ""
    assert df.loc[0, "primary_type_bucket"] == "adapter"
    assert df.loc[1, "primary_type_bucket"] == ""
    assert df.loc[0, "lineage_status"] == "curated"
    assert df.loc[1, "lineage_status"] == ""
    assert df.loc[0, "candidate_source"] == "seed-a"
    assert df.loc[1, "candidate_source"] == ""


def test_load_model_list_adds_curated_fields_for_legacy_three_column_format(tmp_path: Path):
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "model_id": "org/model-a",
                "base_model_relation": "adapter",
                "source_model": "base/model",
            }
        ]
    ).to_csv(csv_path, index=False)

    df = load_model_list(str(csv_path))

    assert df.loc[0, "model_id"] == "org/model-a"
    assert df.loc[0, "base_model_relation"] == "adapter"
    assert df.loc[0, "source_model"] == "base/model"
    assert df.loc[0, "revision_norm"] == ""
    assert df.loc[0, "loader_scenario"] == ""
    assert df.loc[0, "primary_type_bucket"] == ""
    assert df.loc[0, "lineage_status"] == ""
    assert df.loc[0, "candidate_source"] == ""


def test_generate_commands_passes_curated_loader_fields(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "model_id": "org/model-a",
                "revision_norm": "rev-a",
                "source_model": "base/model",
                "base_model_relation": "adapter",
                "loader_scenario": "adapter_requires_base",
                "primary_type_bucket": "adapter",
            }
        ]
    )
    args = SimpleNamespace(
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        use_svd=False,
        parallel_esd=True,
        overwrite=False,
    )

    commands = generate_commands(df, tmp_path, args)

    assert "--revision 'rev-a'" in commands[0]
    assert "--loader_scenario 'adapter_requires_base'" in commands[0]
    assert "--primary_type_bucket 'adapter'" in commands[0]


def test_generate_commands_prefers_preflight_effective_loader_when_present(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "model_id": "org/model-a",
                "revision_norm": "",
                "source_model": "",
                "base_model_relation": "",
                "loader_scenario": "multimodal_transformers",
                "preflight_effective_loader": "seq2seq",
                "primary_type_bucket": "multimodal",
            }
        ]
    )
    args = SimpleNamespace(
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        use_svd=False,
        parallel_esd=True,
        overwrite=False,
    )

    commands = generate_commands(df, tmp_path, args)

    assert "--loader_scenario 'seq2seq'" in commands[0]
    assert "--loader_scenario 'multimodal_transformers'" not in commands[0]


def test_get_completed_models_uses_stats_and_metrics_pairs(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    metrics_dir = tmp_path / "metrics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (stats_dir / "org--model-a.csv").write_text("alpha\n1.0\n")
    (metrics_dir / "org--model-a.h5").write_text("ok")
    (stats_dir / "org--model-b.csv").write_text("alpha\n2.0\n")

    completed = get_completed_models(tmp_path, skip_failed=False)

    assert completed == {"org/model-a"}


def test_get_completed_models_parses_tab_separated_failure_summary(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    metrics_dir = tmp_path / "metrics"
    logs_dir = tmp_path / "logs"
    stats_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    (stats_dir / "org--model-a.csv").write_text("alpha\n1.0\n")
    (metrics_dir / "org--model-a.h5").write_text("ok")
    (logs_dir / "failed_models.txt").write_text(
        "org/model-a\tload\tunsupported_loader_scenario\tunsupported\n"
    )

    completed = get_completed_models(tmp_path, skip_failed=True)

    assert completed == set()


def test_apply_preflight_separates_runnable_and_blocked_rows():
    model_df = pd.DataFrame(
        [
            {
                "model_id": "org/text-model",
                "base_model_relation": "",
                "loader_scenario": "standard_transformers",
                "adapter_config": "",
            },
            {
                "model_id": "org/adapter-model",
                "base_model_relation": "adapter",
                "loader_scenario": "adapter_requires_base",
                "adapter_config": "",
                "files": ["README.md"],
            },
        ]
    )

    runnable_df, blocked_df = apply_preflight(model_df)

    assert list(runnable_df["model_id"]) == ["org/text-model"]
    assert list(blocked_df["model_id"]) == ["org/adapter-model"]
    assert blocked_df.iloc[0]["preflight_reason"] == "missing_required_artifact"
    assert blocked_df.iloc[0]["preflight_effective_loader"] == "adapter_requires_base"


def test_apply_preflight_does_not_block_generic_quantized_native_rows_without_backend_hint():
    model_df = pd.DataFrame(
        [
            {
                "model_id": "org/quantized-model",
                "base_model_relation": "",
                "loader_scenario": "quantized_transformers_native",
                "tags": "",
            },
        ]
    )

    runnable_df, blocked_df = apply_preflight(model_df)

    assert list(runnable_df["model_id"]) == ["org/quantized-model"]
    assert blocked_df.empty
    assert runnable_df.iloc[0]["preflight_effective_loader"] == "standard_causal"


def test_apply_preflight_uses_loader_resolution_metadata_for_backend_status():
    model_df = pd.DataFrame(
        [
            {
                "model_id": "org/quantized-model",
                "base_model_relation": "",
                "loader_scenario": "quantized_transformers_native",
                "tags": "",
                "tags_lb": "awq",
            },
        ]
    )

    runnable_df, blocked_df = apply_preflight(model_df)

    assert runnable_df.empty
    assert list(blocked_df["model_id"]) == ["org/quantized-model"]
    assert blocked_df.iloc[0]["preflight_reason"] == "unsupported_backend"
    assert blocked_df.iloc[0]["preflight_effective_loader"] == "awq"


def test_collect_run_outcomes_counts_success_artifacts_and_terminal_statuses(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    metrics_dir = tmp_path / "metrics"
    terminal_dir = tmp_path / "logs" / "terminal_status"
    logs_dir = tmp_path / "logs"
    stats_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    terminal_dir.mkdir(parents=True, exist_ok=True)

    (stats_dir / "org--model-a.csv").write_text("alpha\n1.0\n")
    (metrics_dir / "org--model-a.h5").write_text("ok")
    (stats_dir / "org--model-b.csv").write_text("alpha\n2.0\n")
    (terminal_dir / "org--model-b.json").write_text(
        "{\"model_id\": \"org/model-b\", \"status\": \"success\"}\n"
    )
    (logs_dir / "failed_models.txt").write_text(
        "org/model-c\tload\tunsupported_backend\tlegacy failure\n"
    )

    outcomes = collect_run_outcomes(tmp_path)

    assert outcomes.success_count == 1
    assert outcomes.failure_count == 1
    assert outcomes.completed_models == {"org/model-a"}
    assert outcomes.failed_models == {"org/model-c"}


def test_collect_run_outcomes_prefers_success_over_stale_failure_history(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    metrics_dir = tmp_path / "metrics"
    terminal_dir = tmp_path / "logs" / "terminal_status"
    logs_dir = tmp_path / "logs"
    stats_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    terminal_dir.mkdir(parents=True, exist_ok=True)

    (stats_dir / "org--model-a.csv").write_text("alpha\n1.0\n")
    (metrics_dir / "org--model-a.h5").write_text("ok")
    (logs_dir / "failed_models.txt").write_text(
        "org/model-a\tload\tunsupported_backend\tstale failure\n"
        "org/model-b\tload\tunsupported_backend\tfresh failure\n"
    )
    (terminal_dir / "org--model-b.json").write_text(
        "{\"model_id\": \"org/model-b\", \"status\": \"failed\", \"reason\": \"unsupported_backend\"}\n"
    )

    outcomes = collect_run_outcomes(tmp_path)

    assert outcomes.success_count == 1
    assert outcomes.failure_count == 1
    assert outcomes.completed_models == {"org/model-a"}
    assert outcomes.failed_models == {"org/model-b"}


def test_worker_honors_revision_override_when_model_id_contains_revision():
    with _worker_module_context():
        module_path = PROJECT_ROOT / "src" / "worker.py"
        spec = importlib.util.spec_from_file_location("worker_under_test", module_path)
        worker = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(worker)

    repo_id, revision = worker.resolve_model_revision("org/model@legacy", "curated-rev")

    assert repo_id == "org/model"
    assert revision == "curated-rev"


def test_worker_threads_loader_scenario_into_load_model(tmp_path: Path):
    class _FakeParam:
        def numel(self):
            return 1

        @property
        def device(self):
            return "cpu"

    class _FakeModel:
        def parameters(self):
            return [_FakeParam(), _FakeParam()]

    with _worker_module_context():
        module_path = PROJECT_ROOT / "src" / "worker.py"
        spec = importlib.util.spec_from_file_location("worker_under_test", module_path)
        worker = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(worker)

        worker.parse_args = lambda: SimpleNamespace(
            model_id="org/model",
            revision="rev-a",
            base_model_relation="",
            source_model="",
            loader_scenario="quantized_transformers_native",
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
        worker.net_esd_estimator = Mock(return_value={"longname": ["layer.0"], "alpha": [1.0]})
        worker.save_results = Mock(
            side_effect=lambda metrics, output_path, *args, **kwargs: (
                output_path.parent.mkdir(parents=True, exist_ok=True),
                output_path.write_text("alpha\n1.0\n"),
            )
        )
        worker.record_failure = Mock()

        exit_code = worker.main()

    assert exit_code == 0
    assert worker.load_model.call_args.kwargs["loader_scenario"] == "quantized_transformers_native"
