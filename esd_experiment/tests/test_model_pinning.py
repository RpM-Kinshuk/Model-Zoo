"""Offline checks for the small metadata-only preparation step."""

import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location(
    "run_experiment_pinning_test", Path(__file__).parents[1] / "src/run_experiment.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

MODEL_SHA = "a" * 40
BASE_SHA = "b" * 40
OTHER_SHA = "c" * 40


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Only model metadata and the adapter JSON may be requested."""
    responses = {}
    configs = {}

    def model_info(repo_id, *, revision, timeout, expand):
        assert timeout > 0
        assert set(expand) == {"sha", "siblings", "config"}
        response = responses[(repo_id, revision)]
        if isinstance(response, Exception):
            raise response
        return response

    def download(repo_id, filename, *, revision, token):
        assert filename == "adapter_config.json"
        assert len(revision) == 40
        path = tmp_path / f"adapter-{len(download_mock.call_args_list)}.json"
        path.write_text(json.dumps(configs[(repo_id, revision)]), encoding="utf-8")
        return str(path)

    info_mock = Mock(side_effect=model_info)
    download_mock = Mock(side_effect=download)
    api = SimpleNamespace(model_info=info_mock)
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.HfApi = Mock(return_value=api)
    fake_hub.hf_hub_download = download_mock
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(runner, "get_hf_token", lambda: None)
    return SimpleNamespace(responses=responses, configs=configs, info=info_mock,
                           download=download_mock)


def model_info(sha=MODEL_SHA, *, adapter=False, config=None):
    files = [SimpleNamespace(rfilename="adapter_config.json")] if adapter else []
    return SimpleNamespace(sha=sha, siblings=files, config=config)


@pytest.mark.parametrize("row,requested", [
    ({"model_id": "org/model@embedded", "revision_norm": "explicit", "Modelsha": OTHER_SHA}, "explicit"),
    ({"model_id": "org/model@embedded", "Modelsha": OTHER_SHA}, "embedded"),
    ({"model_id": "org/model", "Modelsha": OTHER_SHA}, OTHER_SHA),
    ({"model_id": "org/model", "Modelsha": "short-or-invalid"}, "main"),
    ({"model_id": "org/model"}, "main"),
])
def test_revision_precedence_preserves_model_identity(hub, row, requested):
    resolved_sha = requested if requested == OTHER_SHA else MODEL_SHA
    hub.responses[("org/model", requested)] = model_info(resolved_sha)

    pinned = runner.pin_model_revisions(pd.DataFrame([row])).iloc[0]

    assert pinned["model_id"] == row["model_id"]
    assert pinned["revision_requested"] == requested
    assert pinned["revision_norm"] == resolved_sha
    assert pinned["pin_status"] == "pinned"
    assert pinned["pin_error"] == ""
    hub.info.assert_called_once_with("org/model", revision=requested, timeout=30, expand=["sha", "siblings", "config"])
    hub.download.assert_not_called()


@pytest.mark.parametrize("architectures,status,expected", [
    (["BertModel", "BertForMaskedLM"], "recorded", ["BertModel", "BertForMaskedLM"]),
    ("BertForMaskedLM", "recorded", ["BertForMaskedLM"]),
    (None, "recorded", []),
    (["B"], "invalid", []),
    ({"name": "BertModel"}, "invalid", []),
])
def test_preparation_records_complete_config_labels_without_overwriting_input(hub, architectures, status, expected):
    hub.responses[("org/model", "main")] = model_info(config={"model_type": "bert", "architectures": architectures})
    original = {"model_id": "org/model", "Architecture": "B", "Architecture_lb": "WrongForSequenceClassification"}
    pinned = runner.pin_model_revisions(pd.DataFrame([original])).iloc[0]
    assert pinned["pin_status"] == "pinned"
    assert pinned["config_metadata_status"] == status
    assert json.loads(pinned["config_architectures"]) == expected
    assert pinned["config_model_type"] == ("bert" if status == "recorded" else "")
    assert pinned["config_revision"] == pinned["revision_norm"] == MODEL_SHA
    assert pinned["Architecture"] == "B" and pinned["Architecture_lb"] == original["Architecture_lb"]
    hub.info.assert_called_once()
    hub.download.assert_not_called()


def test_repreparation_clears_old_config_metadata_on_failure_or_missing_config(hub):
    rows = [{"model_id": name, "config_model_type": "old", "config_architectures": '["OldModel"]',
             "config_metadata_status": "recorded", "config_revision": OTHER_SHA}
            for name in ("org/failed", "org/missing")]
    hub.responses[("org/failed", "main")] = OSError("unavailable")
    hub.responses[("org/missing", "main")] = model_info()
    pinned = runner.pin_model_revisions(pd.DataFrame(rows))
    assert pinned["config_model_type"].tolist() == ["", ""]
    assert pinned["config_architectures"].tolist() == ["[]", "[]"]
    assert pinned["config_metadata_status"].tolist() == ["missing", "missing"]
    assert pinned["config_revision"].tolist() == ["", MODEL_SHA]


def test_adapter_base_is_inferred_and_shared_resolutions_are_cached(hub):
    rows = []
    for name in ("org/first", "org/second"):
        hub.responses[(name, "main")] = model_info(adapter=True)
        hub.configs[(name, MODEL_SHA)] = {
            "base_model_name_or_path": "org/base@embedded-base",
            "revision": "trained-base",
        }
        rows.append({"model_id": name})
    hub.responses[("org/base", "trained-base")] = model_info(BASE_SHA)

    pinned = runner.pin_model_revisions(pd.DataFrame(rows))

    assert pinned["pin_status"].tolist() == ["pinned", "pinned"]
    assert pinned["base_model_relation"].tolist() == ["adapter", "adapter"]
    assert pinned["source_model"].tolist() == [f"org/base@{BASE_SHA}"] * 2
    assert pinned["source_model_requested"].tolist() == ["org/base@trained-base"] * 2
    assert hub.info.call_count == 3
    assert hub.download.call_count == 2
    runner.validate_model_list_pins(pinned)


def test_supplied_base_overrides_adapter_config(hub):
    hub.responses[("org/adapter", "main")] = model_info(adapter=True)
    hub.responses[("org/chosen", "release")] = model_info(BASE_SHA)
    hub.configs[("org/adapter", MODEL_SHA)] = {"base_model_name_or_path": "org/other"}

    pinned = runner.pin_model_revisions(pd.DataFrame([
        {"model_id": "org/adapter", "source_model": "org/chosen@release"}
    ])).iloc[0]

    assert pinned["pin_status"] == "pinned"
    assert pinned["source_model"] == f"org/chosen@{BASE_SHA}"
    assert pinned["source_model_requested"] == "org/chosen@release"
    assert [call.args[0] for call in hub.info.call_args_list] == ["org/adapter", "org/chosen"]


def test_repreparing_edited_rows_records_current_requested_references(hub):
    hub.responses[("org/model", "fixed-tag")] = model_info()
    hub.responses[("org/new-base", "release")] = model_info(BASE_SHA)
    row = {"model_id": "org/model", "revision_norm": "fixed-tag",
           "source_model": "org/new-base@release", "pin_status": "error",
           "pin_error": "old failure", "revision_requested": "missing-tag",
           "source_model_requested": "org/old-base@main"}

    pinned = runner.pin_model_revisions(pd.DataFrame([row])).iloc[0]

    assert pinned["pin_status"] == "pinned"
    assert pinned["pin_error"] == ""
    assert pinned["revision_requested"] == "fixed-tag"
    assert pinned["source_model_requested"] == "org/new-base@release"
    assert pinned["revision_norm"] == MODEL_SHA
    assert pinned["source_model"] == f"org/new-base@{BASE_SHA}"


@pytest.mark.parametrize("failure", ["unavailable", "malformed_sha", "different_pin", "missing_base"])
def test_unresolved_rows_are_retained_and_refused_at_launch(hub, failure):
    failed_row = {"model_id": "org/bad", "notes": "keep this row"}
    revision = "main"
    response = model_info()
    if failure == "unavailable":
        response = OSError("repository unavailable")
    elif failure == "malformed_sha":
        response = model_info("abc123")
    elif failure == "different_pin":
        failed_row["revision_norm"] = BASE_SHA
        revision = BASE_SHA
    else:
        response = model_info(adapter=True)
        hub.configs[("org/bad", MODEL_SHA)] = {}
    hub.responses[("org/bad", revision)] = response
    hub.responses[("org/good", "main")] = model_info(OTHER_SHA)

    pinned = runner.pin_model_revisions(pd.DataFrame([failed_row, {"model_id": "org/good"}]))

    assert pinned["model_id"].tolist() == ["org/bad", "org/good"]
    assert pinned["pin_status"].tolist() == ["error", "pinned"]
    assert pinned.iloc[0]["notes"] == "keep this row"
    assert pinned.iloc[0]["pin_error"]
    with pytest.raises(ValueError, match="unresolved"):
        runner.validate_model_list_pins(pinned)


@pytest.mark.parametrize("row", [
    {"model_id": "org/model", "revision_norm": "main"},
    {"model_id": "org/model@abc123"},
    {"model_id": "org/model", "revision_norm": MODEL_SHA, "source_model": "org/base@main"},
    {"model_id": "org/model", "revision_norm": MODEL_SHA, "base_model_relation": "adapter"},
    {"model_id": "org/model", "revision_norm": MODEL_SHA, "loader_scenario": "adapter_requires_base"},
])
def test_manual_lists_also_require_model_and_base_pins(row):
    with pytest.raises(ValueError):
        runner.validate_model_list_pins(pd.DataFrame([row]))


def test_pinned_hf_repository_cannot_be_shadowed_by_local_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org/model").mkdir(parents=True)

    with pytest.raises(ValueError, match="local directory"):
        runner.validate_model_list_pins(pd.DataFrame([
            {"model_id": "org/model", "revision_norm": MODEL_SHA}
        ]))


def test_csv_preserves_numeric_sha_and_literal_na_metadata(tmp_path):
    sha = "0" * 39 + "1"
    path = tmp_path / "models.csv"
    pd.DataFrame([{"model_id": "org/model", "revision_norm": sha, "Modelsha": sha,
                   "source_model": f"org/base@{sha}", "Architecture": "NA"}]).to_csv(path, index=False)

    loaded = runner.load_model_list(path)

    assert loaded.iloc[0]["revision_norm"] == sha
    assert loaded.iloc[0]["Modelsha"] == sha
    assert loaded.iloc[0]["Architecture"] == "NA"
    runner.validate_model_list_pins(loaded)


@pytest.mark.parametrize("names", [["org/model", "org/model"], ["org/model", "org--model"]])
def test_duplicate_artifact_names_are_rejected(tmp_path, names):
    path = tmp_path / "models.csv"
    pd.DataFrame({"model_id": names}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="output filename"):
        runner.load_model_list(path)


def test_publishing_never_replaces_existing_pins(tmp_path):
    path = tmp_path / "models.csv"
    original = pd.DataFrame([{"model_id": "org/model", "revision_norm": MODEL_SHA}])
    runner.write_pinned_model_list(original, path)
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        runner.write_pinned_model_list(pd.DataFrame([{"model_id": "org/different"}]), path)

    assert path.read_bytes() == before
    assert list(tmp_path.glob(".models.*.tmp")) == []


@pytest.mark.parametrize("existing", [False, True])
def test_failed_csv_write_does_not_publish_partial_pins(tmp_path, monkeypatch, existing):
    path = tmp_path / "models.csv"
    if existing:
        path.write_text("original pins\n", encoding="utf-8")

    def partial_write(self, handle, **kwargs):
        handle.write("incomplete output")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_write)
    with pytest.raises(OSError, match="disk full"):
        runner.write_pinned_model_list(pd.DataFrame([{"model_id": "org/model"}]), path)

    if existing:
        assert path.read_text() == "original pins\n"
    else:
        assert not path.exists()
    assert list(tmp_path.glob(".models.*.tmp")) == []


def forbid_runtime(monkeypatch):
    for name in ("create_runtime_config", "_available_backends", "GPUDispatcher",
                 "DispatchThread", "apply_preflight", "generate_worker_jobs"):
        monkeypatch.setattr(runner, name, Mock(side_effect=AssertionError(f"unexpected {name}")))


@pytest.mark.parametrize("failed", [False, True])
def test_prepare_only_stops_before_gpu_backends_or_dispatch(tmp_path, monkeypatch, hub, failed):
    source = tmp_path / "input.csv"
    pd.DataFrame([{"model_id": "org/model"}]).to_csv(source, index=False)
    output = tmp_path / "run"
    hub.responses[("org/model", "main")] = OSError("unavailable") if failed else model_info()
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--model_list", str(source),
                                     "--output_dir", str(output), "--prepare_only"])
    forbid_runtime(monkeypatch)

    if failed:
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 1
    else:
        runner.main()

    pinned = runner.load_model_list(output / "models.csv")
    assert pinned.iloc[0]["pin_status"] == ("error" if failed else "pinned")
    assert {path.name for path in output.iterdir()} == {"models.csv"}
    hub.download.assert_not_called()


def test_prepare_refuses_existing_manifest_even_with_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    pd.DataFrame([{"model_id": "org/model"}]).to_csv(source, index=False)
    manifest = tmp_path / "models.csv"
    manifest.write_text("do not replace\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--model_list", str(source),
                                     "--output_dir", str(tmp_path), "--prepare_only", "--overwrite"])
    monkeypatch.setattr(runner, "pin_model_revisions", Mock(side_effect=AssertionError("unexpected Hub lookup")))
    forbid_runtime(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2
    assert manifest.read_text() == "do not replace\n"


def test_unpinned_launch_stops_before_gpu_work(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    pd.DataFrame([{"model_id": "org/model"}]).to_csv(source, index=False)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--model_list", str(source),
                                     "--output_dir", str(output)])
    forbid_runtime(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2
    assert not output.exists()


@pytest.fixture
def standalone_worker():
    spec = importlib.util.spec_from_file_location(
        "standalone_worker_pinning_test", Path(__file__).parents[1] / "src/worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def test_standalone_worker_refuses_unpinned_overwrite_before_touching_output(
    tmp_path, monkeypatch, standalone_worker
):
    marker = tmp_path / "stats/org--model.csv"
    marker.parent.mkdir()
    marker.write_text("existing results\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker.py", "--model_id", "org/model",
                                     "--output_dir", str(tmp_path), "--overwrite"])
    monkeypatch.setattr(standalone_worker, "load_model", Mock(side_effect=AssertionError("unexpected model load")))

    with pytest.raises(SystemExit) as exc:
        standalone_worker.main()

    assert exc.value.code == 2
    assert marker.read_text() == "existing results\n"
    assert list(tmp_path.rglob("*.csv")) == [marker]
    assert {path.name for path in tmp_path.iterdir()} == {"stats"}


def test_standalone_worker_accepts_pin_and_explicit_code_permission(
    tmp_path, monkeypatch, standalone_worker
):
    monkeypatch.setattr(sys, "argv", ["worker.py", "--model_id", "org/model",
                                     "--revision", MODEL_SHA, "--output_dir", str(tmp_path),
                                     "--trust_remote_code"])

    args = standalone_worker.parse_args()

    assert args.revision == MODEL_SHA
    assert args.trust_remote_code is True


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_worker_command_roundtrips_metadata_and_explicit_code_permission(tmp_path, monkeypatch, trust_remote_code):
    output = tmp_path / "run 'quoted'; $(never-execute)"
    flags = ["--trust_remote_code"] if trust_remote_code else []
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--model_list", "models.csv",
                                     "--output_dir", str(output), *flags])
    args = runner.parse_args()
    bucket = "classification; $(never-execute) 'quoted'\nsecond line"
    row = {"model_id": "org/model@branch", "revision_norm": MODEL_SHA,
           "base_model_relation": "adapter", "source_model": f"org/base@{BASE_SHA}",
           "primary_type_bucket": bucket}

    command = runner.generate_commands(pd.DataFrame([row]), output, args)[0]
    words = shlex.split(command)

    assert words[words.index("--output_dir") + 1] == str(output)
    assert words[words.index("--primary_type_bucket") + 1] == bucket
    assert words[words.index("--revision") + 1] == MODEL_SHA
    assert words[words.index("--model_id") + 1] == row["model_id"]
    assert words[words.index("--source_model") + 1] == row["source_model"]
    assert ("--trust_remote_code" in words) is trust_remote_code
    assert runner.measurement_config(args)["trust_remote_code"] is trust_remote_code
