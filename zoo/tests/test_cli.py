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


def test_build_without_arguments_uses_packages_and_current_conventions(package_run, monkeypatch):
    from zoo.publish import publish_package, current_conventions_id
    r = package_run
    publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert main(["build"]) == 0
    meta = json.loads((r.root / "builds/current/zoo.json").read_text())
    assert meta["conventions_id"] == current_conventions_id()
    assert meta["row_counts"]["models"] == 1 and meta["row_counts"]["layers"] == 2


def test_build_reports_skipped_conventions(package_run):
    from zoo.publish import publish_package
    import h5py
    r = package_run
    publish_package(r.run, "lab/tiny", r.packages, r.root)
    for source in [r.run / "metrics/lab--tiny.h5", next(r.folder.glob("*.spectra.h5"))]:
        replacement = source.with_suffix(".new")
        replacement.write_bytes(source.read_bytes())
        with h5py.File(replacement, "r+") as h:
            cfg = json.loads(h.attrs["measurement_config_json"])
            cfg["fix_fingers"] = "xmin_peak"
            h.attrs["measurement_config_json"] = json.dumps(cfg)
        os.replace(replacement, source)
    publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert main(["build"]) == 0
    meta = json.loads((r.root / "builds/current/zoo.json").read_text())
    assert meta["skipped_conventions"] == 1 and meta["row_counts"]["models"] == 1


def test_deltas_defaults_to_current_zoo_and_packages(tmp_path, monkeypatch, capsys):
    import zoo.deltas
    monkeypatch.setenv("ZOO_DATA", str(tmp_path))
    calls = []
    monkeypatch.setattr(zoo.deltas, "run_deltas", lambda *args, **kw: calls.append((args, kw)) or {"measured": 0})
    assert main(["deltas"]) == 0
    assert calls[0][0] == (tmp_path / "builds/current", tmp_path / "packages", tmp_path)
    assert calls[0][1]["parent_cache"] == tmp_path / "cache/parent-blocks"
    assert "rebuild" in json.loads(capsys.readouterr().out)


def test_build_reads_pair_delta_supplements(package_run):
    from evidence.pair_delta import pair_deltas
    from evidence.hub import HubReader
    from zoo.publish import publish_package
    from zoo.build import build_zoo
    r = package_run
    package = publish_package(r.run, "lab/tiny", r.packages, r.root)
    core = (package / "package.json").read_bytes()
    delta = pair_deltas(r.manifest, r.manifest, HubReader(), package)
    built = build_zoo(packages=r.packages, data_root=r.root, out=r.root / "built")
    meta = json.loads((built / "zoo.json").read_text())
    assert any(str(delta.relative_to(r.root)) in str(item) for item in meta["inputs"])
    assert (package / "package.json").read_bytes() == core


def test_deltas_measures_detected_package_pair_and_resumes(package_run, capsys):
    from zoo.publish import publish_package
    from zoo.build import build_zoo, open_tables
    from evidence.fingerprint import fingerprint_model
    from evidence.hub import HubReader
    from evidence.manifest import write_manifest
    from measurement_config import safe_filename
    r = package_run
    publish_package(r.run, "lab/tiny", r.packages, r.root)
    parent = dict(r.manifest, repo_id="lab/other")
    folder = r.run / "evidence" / safe_filename(parent["repo_id"] + "@" + parent["revision"])
    write_manifest(parent, folder)
    fingerprint_model(parent, HubReader(), "row_blocks", folder)
    publish_package(r.run, "lab/other", r.packages, r.root)
    built = build_zoo(packages=r.packages, data_root=r.root, out=r.root / "builds/current")
    capsys.readouterr()
    assert main(["deltas", "--workers", "1", "--compute_device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["measured"] == 1
    assert main(["deltas", "--workers", "1", "--compute_device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["measured"] == 0
    build_zoo(packages=r.packages, data_root=r.root, out=built)
    lineage = open_tables(built)["lineage"]
    assert lineage.evidence.str.contains("row_block_delta").any()
    assert not lineage.calibrated.any()
