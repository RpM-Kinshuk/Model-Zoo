"""Build command plumbing and real artifact-only publication."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from zoo.cli import main
from zoo.testing import SHA2, SHA10


def test_cli_build(universe, tmp_path):
    assert main(["build", "--manifests", str(universe.manifests), "--fingerprints", str(universe.fingerprints),
                 "--data_root", str(universe.root), "--out", str(tmp_path / "zoo")]) == 0
    assert (tmp_path / "zoo" / "zoo.json").exists()


def test_defaults_and_multiple_fingerprint_directories(universe, tmp_path, monkeypatch):
    monkeypatch.setenv("ZOO_DATA", str(universe.root))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["build", "--manifests", str(universe.manifests), "--fingerprints",
                 str(universe.fingerprints), str(empty)]) == 0
    metadata = json.loads((universe.root / "builds/current/zoo.json").read_text())
    assert metadata["row_counts"]["models"] == 12


def test_measured_score_imports_keep_revision_verification(universe, tmp_path):
    leaderboard, scores, mapping = (tmp_path / name for name in ("leaderboard.csv", "scores.csv", "mapping.json"))
    pd.DataFrame([{"fullname": "org/child", "Model sha": SHA2, "IFEval Raw": .75}]).to_csv(leaderboard, index=False)
    pd.DataFrame([{"model": "org/child", "commit": SHA10, "bench": "heldout", "skill": "accuracy", "value": .5}]).to_csv(scores, index=False)
    mapping.write_text(json.dumps(dict(repo_id="model", revision="commit", benchmark="bench", skill="skill", value="value", source="local_eval")))
    frame = tmp_path / "frame.csv"
    pd.DataFrame([dict(repo_id="org/child", revision=SHA2, downloads=42, params=1000, modality="text")]).to_csv(frame, index=False)
    out = tmp_path / "scored-zoo"
    assert main(["build", "--manifests", str(universe.manifests), "--fingerprints", str(universe.fingerprints),
                 "--data_root", str(universe.root), "--out", str(out), "--frame", str(frame),
                 "--leaderboard_v2", str(leaderboard), "--scores_csv", str(scores), "--scores_mapping", str(mapping)]) == 0
    measured = pd.read_parquet(out / "scores.parquet")
    assert len(measured) == 4 and measured.revision_verified.sum() == 2
    assert set(measured.source) == {"open_llm_leaderboard_v2", "local_eval"}
    assert len(pd.read_parquet(out / "predictions.parquet")) == 0
    models = pd.read_parquet(out / "models.parquet")
    assert models.set_index("model_key").loc["org/child@" + SHA2, "downloads"] == 42


def test_optional_edge_and_calibration_paths_are_forwarded(universe, tmp_path, monkeypatch):
    import zoo.cli as cli
    captured = {}
    def build(manifests, fingerprints, out, data_root, **kwargs):
        captured.update(manifests=manifests, fingerprints=fingerprints, out=out, data_root=data_root, **kwargs)
        return out
    monkeypatch.setattr(cli, "build_zoo", build)
    edges, null = tmp_path / "edges.parquet", tmp_path / "null.json"
    out = tmp_path / "out"
    assert main(["build", "--manifests", str(universe.manifests), "--fingerprints", str(universe.fingerprints),
                 "--data_root", str(universe.root), "--out", str(out), "--lineage_edges", str(edges), "--null", str(null)]) == 0
    assert captured["lineage_edges"] == edges and captured["null"] == null
    assert captured["manifests"] == universe.manifests
    assert captured["fingerprints"] == [universe.fingerprints]


@pytest.mark.parametrize("flag", ["--scores_csv", "--scores_mapping"])
def test_score_pair_is_required_before_publication(universe, tmp_path, capsys, flag):
    out = tmp_path / "unchanged"
    out.mkdir()
    marker = out / "keep"
    marker.write_text("original")
    assert main(["build", "--manifests", str(universe.manifests), "--fingerprints", str(universe.fingerprints),
                 "--data_root", str(universe.root), "--out", str(out), flag, str(tmp_path / "input")]) == 2
    captured = capsys.readouterr()
    assert not captured.out and "--scores_csv" in captured.err and "--scores_mapping" in captured.err
    assert marker.read_text() == "original"


def test_missing_root_and_bad_mapping_are_clean_errors(universe, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ZOO_DATA", raising=False)
    args = ["build", "--manifests", str(universe.manifests), "--fingerprints", str(universe.fingerprints)]
    assert main(args) == 2
    assert "data_root" in capsys.readouterr().err
    mapping = tmp_path / "mapping.json"
    mapping.write_text("[]")
    assert main(args + ["--data_root", str(universe.root), "--scores_csv", str(tmp_path / "scores.csv"),
                        "--scores_mapping", str(mapping)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and "object" in captured.err


def test_module_entrypoint_help():
    result = subprocess.run([sys.executable, "-m", "zoo.cli", "--help"],
                            cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True,
                            env=dict(os.environ, HF_HUB_OFFLINE="1"))
    assert result.returncode == 0 and "build" in result.stdout and not result.stderr
