import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
            },
            {
                "model_id": "org/model-b",
                "revision_norm": None,
                "source_model": "",
                "base_model_relation": "source",
                "loader_scenario": None,
                "primary_type_bucket": None,
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
