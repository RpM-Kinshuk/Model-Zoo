"""Exercise the default launcher journeys without downloading weights."""

import json
from types import SimpleNamespace

import pytest

from zoo import tui


def test_prepare_uses_one_workspace_and_no_questions(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "My Zoo"
    workspace.mkdir()
    (workspace / "models.csv").write_text("model_id\norg/model\n")
    commands = []
    monkeypatch.setattr(tui, "_ask", lambda *args, **kwargs: pytest.fail("unexpected prompt"))
    def capture(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tui.subprocess, "run", capture)
    assert tui.main(["--workspace", str(workspace), "prepare"]) == 0
    command, kwargs = commands[0]
    assert command[command.index("--model_list") + 1] == str(workspace / "models.csv")
    assert command[command.index("--output_dir") + 1] == str(workspace / "runs" / "default")
    assert kwargs["env"]["ZOO_DATA"] == str(workspace)
    assert "--prepare_only" in command
    assert "1 model entry" in capsys.readouterr().out


def test_init_preserves_config_and_can_use_relative_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "New Zoo"
    assert tui.main(["--workspace", str(workspace), "init"]) == 0
    config_path = workspace / "model-zoo.json"
    config = {"models": "lists/short list.csv", "run": "runs/my run", "prepare_limit": 3}
    config_path.write_text(json.dumps(config))
    assert tui.main(["--workspace", str(workspace), "init"]) == 0
    assert json.loads(config_path.read_text()) == config
    monkeypatch.chdir(tmp_path)
    assert tui.Workspace.load(workspace).models == workspace / "lists" / "short list.csv"


def test_auto_gpu_uses_single_visible_device_and_never_assumes_gpu_zero(monkeypatch):
    devices = {
        0: tui.Gpu("A", "GPU-aaaa", 12000, 11000, "0"),
        2: tui.Gpu("B", "GPU-bbbb", 16000, 13000, "8"),
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert tui._gpu_choices(devices) == [2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
    assert tui._gpu_choices(devices) == [2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,-1,0")
    assert tui._gpu_choices(devices) == [2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-unknown")
    assert tui._gpu_choices(devices) == []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert tui._gpu_choices({}) == []


def test_powershell_preview_quotes_paths_and_literal_arguments(monkeypatch):
    monkeypatch.setattr(tui.os, "name", "nt")
    assert tui._shell_command([r"C:\User Files\python.exe", "Zoo's $data"]) == (
        r"& 'C:\User Files\python.exe' 'Zoo''s $data'"
    )


@pytest.mark.skipif(tui.os.name == "nt", reason="The GPU supervisor uses Linux process groups")
def test_measure_uses_detected_gpu_and_can_be_cancelled(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "Zoo"
    run = workspace / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "models.csv").write_text("model_id,preparation_status\norg/ready,ready\norg/held,held\n")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(tui, "_gpu_inventory", lambda: {2: tui.Gpu("GPU", "GPU-bbbb", 1000, 800, "1")})
    monkeypatch.setattr(tui, "_confirm", lambda: False)
    monkeypatch.setattr(tui.subprocess, "run", lambda *args, **kwargs: pytest.fail("cancelled"))
    assert tui.main(["--workspace", str(workspace), "measure"]) == 0
    output = capsys.readouterr().out
    assert "GPU pool  2" in output
    assert "1 held, 1 ready" in output
    assert "Cancelled." in output


def test_build_requires_packages_and_returns_actionable_error(tmp_path, capsys):
    assert tui.main(["--workspace", str(tmp_path), "build"]) == 2
    assert "Publish completed evidence first" in capsys.readouterr().out.replace("\n  ", " ")


def test_summarize_discovers_runs_with_only_worker_records(tmp_path, monkeypatch):
    logs = tmp_path / "runs" / "review" / "logs" / "terminal_status"
    logs.mkdir(parents=True)
    (logs / "example.json").write_text("{}")
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tui.subprocess, "run", capture)
    assert tui.main(["--workspace", str(tmp_path), "summarize"]) == 0
    assert commands[0][-1] == str(tmp_path / "runs" / "review")


def test_publish_skips_metadata_only_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name in ("default", "complete"):
        evidence = runs / name / "evidence" / "model"
        evidence.mkdir(parents=True)
        (evidence / "model.manifest.json").touch()
        if name == "complete":
            (evidence / "model.row_blocks.fingerprint.safetensors").touch()
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tui.subprocess, "run", capture)
    assert tui.main(["--workspace", str(tmp_path), "publish"]) == 0
    assert commands[0][commands[0].index("--runs") + 1] == str(runs / "complete")


def test_explicit_run_never_switches_to_another_run(tmp_path, monkeypatch, capsys):
    other = tmp_path / "runs" / "other"
    other.mkdir(parents=True)
    (other / "models.csv").write_text("model_id\norg/model\n")
    monkeypatch.setattr(tui.subprocess, "run", lambda *args, **kwargs: pytest.fail("wrong run"))
    assert tui.main(["--workspace", str(tmp_path), "--run", "runs/missing", "summarize"]) == 2
    assert "missing" in capsys.readouterr().out


@pytest.mark.skipif(tui.os.name == "nt", reason="The GPU supervisor uses Linux process groups")
def test_measure_refuses_uuid_allocation_before_starting_work(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
    monkeypatch.setattr(tui.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected process"))
    assert tui.main(["--workspace", str(tmp_path), "measure"]) == 2
    assert "numeric GPU IDs" in capsys.readouterr().out
