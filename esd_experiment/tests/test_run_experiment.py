import sys
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import Mock

import pandas as pd

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
load_model_list = run_experiment.load_model_list


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

    fake_model_loader = ModuleType("model_loader")
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
        worker.net_esd_estimator = Mock(return_value={})
        worker.record_failure = Mock()

        exit_code = worker.main()

    assert exit_code == 0
    assert worker.load_model.call_args.kwargs["loader_scenario"] == "quantized_transformers_native"
