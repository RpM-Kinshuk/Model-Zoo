import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from zoo.build import build_zoo, open_tables
from zoo.schema import (COVERAGE_COLUMNS, LINEAGE_COLUMNS, MODELS_COLUMNS,
                        PREDICTIONS_COLUMNS, SCORES_COLUMNS, VIEWS_COLUMNS)
from zoo.testing import SHA1, SHA11


COLUMNS = dict(models=MODELS_COLUMNS, views=VIEWS_COLUMNS, coverage=COVERAGE_COLUMNS,
               lineage=LINEAGE_COLUMNS, scores=SCORES_COLUMNS, predictions=PREDICTIONS_COLUMNS)


def test_build_writes_all_tables(built_zoo):
    t = open_tables(built_zoo)
    assert set(t) == {"models", "views", "coverage", "lineage", "scores", "predictions"}
    assert json.loads((built_zoo / "zoo.json").read_text())["zoo_version"] == "0"
    assert list((built_zoo / "retrieval").glob("*.npy"))


def test_deterministic_rebuild(universe, tmp_path):
    a = open_tables(build_zoo(universe.manifests, [universe.fingerprints], tmp_path / "a", data_root=universe.root))
    b = open_tables(build_zoo(universe.manifests, [universe.fingerprints], tmp_path / "b", data_root=universe.root))
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_interrupted_build_keeps_previous(universe, tmp_path, monkeypatch):
    zoo = build_zoo(universe.manifests, [universe.fingerprints], tmp_path / "zoo", data_root=universe.root)
    before = (zoo / "models.parquet").read_bytes()
    monkeypatch.setattr("zoo.build.build_retrieval", Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        build_zoo(universe.manifests, [universe.fingerprints], zoo, data_root=universe.root)
    assert (zoo / "models.parquet").read_bytes() == before and not (tmp_path / "zoo.building").exists()


def _build(universe, out, **kwargs):
    return build_zoo(universe.manifests, [universe.fingerprints], out, universe.root, **kwargs)


def _snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_nested_values_and_typed_empty_tables_roundtrip(built_zoo):
    t = open_tables(built_zoo)
    for name, columns in COLUMNS.items():
        assert list(t[name].columns) == columns
    models = t["models"].set_index("model_key")
    assert models.loc["org/decider@" + SHA11, "labels"] == {"0": "no", "1": "yes"}
    assert models.loc["org/parent@" + SHA1, "labels"] == {}
    assert models.loc["org/parent@" + SHA1, "declared_bases"] == []
    assert isinstance(models.loc["org/parent@" + SHA1, "skipped_tensors"], list)
    assert t["scores"].empty and t["predictions"].empty
    schema = pq.read_schema(built_zoo / "models.parquet")
    assert pa.types.is_map(schema.field("labels").type)
    assert pa.types.is_list(schema.field("declared_bases").type)
    assert pa.types.is_list(schema.field("skipped_tensors").type)
    assert pa.types.is_floating(pq.read_schema(built_zoo / "scores.parquet").field("value").type)
    assert pa.types.is_list(pq.read_schema(built_zoo / "predictions.parquet").field("inputs").type)


def test_all_empty_zoo_serializes_all_six_tables(tmp_path):
    manifests, fingerprints = tmp_path / "manifests", tmp_path / "fingerprints"
    manifests.mkdir()
    fingerprints.mkdir()
    out = build_zoo(manifests, [fingerprints], tmp_path / "empty", tmp_path)
    t = open_tables(out)
    assert set(t) == set(COLUMNS)
    assert all(table.empty and list(table.columns) == COLUMNS[name] for name, table in t.items())


def test_all_empty_label_maps_and_skipped_lists_serialize(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = dict(repo_id="org/only", revision=SHA1, access="t0_only", declared_bases=[],
                    decision=dict(labels={}, is_decision=False), tensors=[])
    (manifests / "only.manifest.json").write_text(json.dumps(manifest))
    out = build_zoo(manifests, [], tmp_path / "only", tmp_path)
    model = open_tables(out)["models"].iloc[0]
    assert model.labels == {} and model.skipped_tensors == [] and model.declared_bases == []


def test_selected_input_provenance_excludes_raw_weights_and_is_deterministic(universe, tmp_path, null_dict):
    null_path = tmp_path / "null.json"
    null_path.write_text(json.dumps(null_dict))
    frame = tmp_path / "frame.csv"
    frame.write_text("repo_id,downloads\norg/parent,10\n")
    unrelated = tmp_path / "irrelevant.bin"
    unrelated.write_bytes(b"must not hash arbitrary data_root files")
    scores = pd.DataFrame([dict(model_key="org/parent@" + SHA1, repo_id="org/parent", benchmark="test",
        skill=None, value=.7, metric="accuracy", source="local", source_revision=SHA1,
        revision_verified=True, harness=None, n_items=10)], columns=SCORES_COLUMNS)
    external = tmp_path / "edges.parquet"
    pd.DataFrame(columns=LINEAGE_COLUMNS).to_parquet(external, index=False)
    out = _build(universe, tmp_path / "zoo", frame=frame, null=null_path, lineage_edges=external, score_tables=[scores])
    metadata = json.loads((out / "zoo.json").read_text())
    inputs = metadata["inputs"]
    files = {entry["path"]: entry["sha256"] for entry in inputs if entry["kind"] == "file"}
    expected = list(universe.manifests.glob("*.manifest.json")) + list(universe.fingerprints.glob("*.t1.fingerprint.safetensors")) + [frame, null_path, external]
    assert set(files) == {p.relative_to(universe.root).as_posix() for p in expected}
    for path in expected:
        assert files[path.relative_to(universe.root).as_posix()] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len([entry for entry in inputs if entry["kind"] == "score_table"]) == 1
    assert metadata["row_counts"] == {name: len(table) for name, table in open_tables(out).items()}
    assert open_tables(out)["scores"].value.tolist() == [.7]
    _build(universe, out, frame=frame, null=null_path, lineage_edges=external, score_tables=[scores.copy()])
    second = json.loads((out / "zoo.json").read_text())
    metadata.pop("build_time")
    second.pop("build_time")
    assert metadata == second
    scores.loc[0, "value"] = .8
    _build(universe, out, frame=frame, null=null_path, lineage_edges=external, score_tables=[scores])
    changed = json.loads((out / "zoo.json").read_text())
    assert [x["sha256"] for x in changed["inputs"] if x["kind"] == "score_table"] != [x["sha256"] for x in inputs if x["kind"] == "score_table"]


def test_successful_publication_keeps_complete_old_generation(built_zoo, universe):
    assert built_zoo.is_symlink() and not Path(os.readlink(built_zoo)).is_absolute()
    old = built_zoo.resolve()
    before = _snapshot(old)
    _build(universe, built_zoo)
    assert built_zoo.resolve() != old and _snapshot(old) == before
    assert not list(built_zoo.parent.glob(f".{built_zoo.name}.building-*"))
    assert not list(built_zoo.parent.glob(f".{built_zoo.name}.publish-*"))


def test_pointer_publication_failure_preserves_output_sources_and_unowned_staging(built_zoo, universe, monkeypatch):
    old = built_zoo.resolve()
    before = _snapshot(old)
    evidence = _snapshot(universe.manifests) | {"fp/" + k: v for k, v in _snapshot(universe.fingerprints).items()}
    siblings = set(built_zoo.parent.iterdir())
    unowned = built_zoo.parent / f".{built_zoo.name}.building-someone-else"
    unowned.mkdir()
    (unowned / "keep").write_text("unowned")
    previous = built_zoo.with_name(built_zoo.name + ".previous")
    previous.mkdir()
    (previous / "keep").write_text("legacy-unowned")
    replace = os.replace
    def fail_pointer(source, destination):
        if Path(destination) == built_zoo:
            raise OSError("publication failed")
        return replace(source, destination)
    monkeypatch.setattr("zoo.build.os.replace", fail_pointer)
    with pytest.raises(OSError, match="publication failed"):
        _build(universe, built_zoo)
    assert built_zoo.resolve() == old and _snapshot(old) == before
    assert _snapshot(universe.manifests) | {"fp/" + k: v for k, v in _snapshot(universe.fingerprints).items()} == evidence
    assert (unowned / "keep").read_text() == "unowned"
    assert (previous / "keep").read_text() == "legacy-unowned"
    expected = siblings | {unowned, previous}
    remaining = set(built_zoo.parent.iterdir()) - expected
    assert len(remaining) == 1
    orphan = remaining.pop()
    assert orphan.name.startswith(f".{built_zoo.name}.generation-")
    assert set(open_tables(orphan)) == set(COLUMNS)
    assert expected <= set(built_zoo.parent.iterdir())


def test_post_replacement_interrupt_keeps_public_generation_readable(built_zoo, universe, monkeypatch):
    old = built_zoo.resolve()
    before = _snapshot(old)
    expected = open_tables(built_zoo)
    replace = os.replace
    def interrupt_after_replace(source, destination):
        replace(source, destination)
        if Path(destination) == built_zoo:
            raise KeyboardInterrupt("after publication")
    monkeypatch.setattr("zoo.build.os.replace", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt, match="after publication"):
        _build(universe, built_zoo)
    assert built_zoo.is_symlink() and built_zoo.resolve().is_dir()
    assert built_zoo.resolve() != old and _snapshot(old) == before
    assert set(_snapshot(built_zoo.resolve())) == set(before)
    actual = open_tables(built_zoo)
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name])
    assert json.loads((built_zoo / "zoo.json").read_text())["row_counts"] == {
        name: len(table) for name, table in actual.items()}
    assert not list(built_zoo.parent.glob(f".{built_zoo.name}.building-*"))
    assert not list(built_zoo.parent.glob(f".{built_zoo.name}.publish-*"))


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "broken_symlink"])
def test_unowned_outputs_are_rejected_without_changes(universe, tmp_path, kind):
    out = tmp_path / "unowned"
    target = tmp_path / "external-directory"
    target.mkdir()
    (target / "keep").write_text("keep")
    if kind == "directory":
        out.mkdir()
        (out / "keep").write_text("keep")
    elif kind == "file":
        out.write_text("keep")
    else:
        out.symlink_to(target if kind == "symlink" else tmp_path / "missing")
    with pytest.raises(ValueError, match="owned|managed|output"):
        _build(universe, out)
    assert (target / "keep").read_text() == "keep"
    assert out.is_symlink() if "symlink" in kind else out.exists()


@pytest.mark.parametrize("location", ["data_root", "ancestor", "manifests", "inside_manifests", "fingerprints", "source_file"])
def test_output_may_not_overlap_source_evidence(universe, location):
    choices = dict(data_root=universe.root, ancestor=universe.root.parent,
                   manifests=universe.manifests, inside_manifests=universe.manifests / "zoo",
                   fingerprints=universe.fingerprints, source_file=next(universe.manifests.glob("*.manifest.json")))
    before = _snapshot(universe.manifests)
    with pytest.raises(ValueError, match="overlap|data_root|source|input"):
        _build(universe, choices[location])
    assert _snapshot(universe.manifests) == before


def test_open_tables_resolves_generation_once(built_zoo, universe, monkeypatch):
    old = built_zoo.resolve()
    expected = open_tables(built_zoo)
    _build(universe, built_zoo)
    newest = built_zoo.resolve()
    built_zoo.unlink()
    built_zoo.symlink_to(old.name)
    read = pq.read_table
    calls = []
    def switch_during_read(path, *args, **kwargs):
        calls.append(Path(path).parent)
        if len(calls) == 1:
            link = built_zoo.parent / "reader-switch"
            link.symlink_to(newest.name)
            os.replace(link, built_zoo)
        return read(path, *args, **kwargs)
    monkeypatch.setattr("zoo.build.pq.read_table", switch_during_read)
    actual = open_tables(built_zoo)
    assert len(calls) == 6 and set(calls) == {old}
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name])
