import hashlib
import csv
import io
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
                        PREDICTIONS_COLUMNS, SCORES_COLUMNS, VIEWS_COLUMNS, LAYERS_COLUMNS)
from zoo.testing import SHA1, SHA11


COLUMNS = dict(models=MODELS_COLUMNS, views=VIEWS_COLUMNS, coverage=COVERAGE_COLUMNS,
               lineage=LINEAGE_COLUMNS, scores=SCORES_COLUMNS, predictions=PREDICTIONS_COLUMNS, layers=LAYERS_COLUMNS)


def test_build_writes_all_tables(built_zoo):
    t = open_tables(built_zoo)
    assert set(t) == {"models", "views", "coverage", "lineage", "scores", "predictions", "layers"}
    assert json.loads((built_zoo / "zoo.json").read_text())["zoo_version"] == "3"
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
    assert t["scores"].empty and t["predictions"].empty
    schema = pq.read_schema(built_zoo / "models.parquet")
    assert pa.types.is_map(schema.field("labels").type)
    assert pa.types.is_list(schema.field("declared_bases").type)
    assert "skipped_tensors" not in schema.names  # Skipped tensors live only in coverage.
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


def test_all_empty_label_maps_and_base_lists_serialize(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = dict(repo_id="org/only", revision=SHA1, access="metadata_only", declared_bases=[],
                    decision=dict(labels={}, is_decision=False), tensors=[])
    (manifests / "only.manifest.json").write_text(json.dumps(manifest))
    out = build_zoo(manifests, [], tmp_path / "only", tmp_path)
    model = open_tables(out)["models"].iloc[0]
    assert model.labels == {} and model.declared_bases == []


def test_hub_gating_values_roundtrip_as_nullable_booleans_without_manifest_changes(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    values = {"auto": "auto", "manual": "manual", "true": True, "false": False, "unknown": None}
    for name, gated in values.items():
        manifest = dict(repo_id=f"org/{name}", revision=SHA1, access="metadata_only", tensors=[], gated=gated)
        (manifests / f"{name}.manifest.json").write_text(json.dumps(manifest))
    before = _snapshot(manifests)
    out = build_zoo(manifests, [], tmp_path / "zoo", tmp_path)
    gated = pq.read_table(out / "models.parquet").select(["repo_id", "gated"])
    assert pa.types.is_boolean(gated.schema.field("gated").type)
    assert {row["repo_id"]: row["gated"] for row in gated.to_pylist()} == {
        "org/auto": True, "org/manual": True, "org/true": True, "org/false": False, "org/unknown": None}
    assert _snapshot(manifests) == before


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
    expected = list(universe.manifests.glob("*.manifest.json")) + list(universe.manifests.glob("*.source-metadata.json.gz")) + list(universe.fingerprints.glob("*.row_blocks.fingerprint.safetensors")) + [frame, null_path, external]
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
    assert len(calls) == 7 and set(calls) == {old}
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name])


def test_build_retains_raw_score_source_and_exact_row_bindings(universe, tmp_path):
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    raw = ("repo,bench,value,metric,harness,n,commit,unmapped_context\n"
           f"org/parent,task,.5,acc,eval:bf16,10,{SHA1},keep-this\n"
           "not/in-zoo,task,.3,acc,eval:bf16,,,unbound\n").encode()
    source.write_bytes(raw)
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value",
        source="local", revision="commit", metric="metric", harness="harness", n_items="n"), models)
    out = _build(universe, tmp_path / "with-scores", score_tables=[scores])
    source.unlink()  # Only this pytest-created CSV; the generation must stand alone.
    assert not source.exists()
    context = json.loads((out / "score-sources/0.json").read_text())
    assert (out / context["snapshot"]).read_bytes() == raw
    recovered = list(csv.DictReader(io.StringIO((out / context["snapshot"]).read_text())))
    assert len(recovered) == 2
    assert recovered[0]["unmapped_context"] == "keep-this"
    assert recovered[1]["unmapped_context"] == "unbound"
    assert context["sha256"] == hashlib.sha256(raw).hexdigest()
    assert context["rows"][0]["scores"][0]["n_items"] == 10
    bound = context["rows"][0]["scores"][0]
    assert bound["model_key"] == recovered[0]["repo"] + "@" + recovered[0]["commit"]
    assert bound["value"] == float(recovered[0]["value"])
    assert context["rows"][1]["status"] == "unmatched_model"
    stored = open_tables(out)["scores"]
    assert stored.metric.tolist() == ["acc"] and stored.n_items.tolist() == [10]
    assert stored.loc[stored.model_key.eq(bound["model_key"]), "value"].tolist() == [bound["value"]]
    metadata = json.loads((out / "zoo.json").read_text())
    files = {r["path"]:r["sha256"] for r in metadata["inputs"] if r["kind"] == "file"}
    assert files[source.relative_to(universe.root).as_posix()] == context["sha256"]


def test_changed_score_source_is_rejected_before_publication(universe, tmp_path):
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    source.write_text("repo,bench,value\norg/parent,task,.5\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value", source="local"), models)
    source.write_text("repo,bench,value\norg/parent,task,.9\n")
    out = tmp_path / "changed"
    with pytest.raises(ValueError, match="source.*changed|hash"):
        _build(universe, out, score_tables=[scores])
    assert not out.exists()


def test_programmatic_score_table_marks_unavailable_source_context(universe, tmp_path):
    scores = pd.DataFrame([dict(model_key="org/parent@" + SHA1, repo_id="org/parent", benchmark="test",
        skill=None, value=.7, metric="accuracy", source="local", source_revision=SHA1,
        revision_verified=True, harness=None, n_items=None)], columns=SCORES_COLUMNS)
    out = _build(universe, tmp_path / "scores-only", score_tables=[scores])
    context = json.loads((out / "score-sources/0.json").read_text())
    assert context["status"] == "source_context_unavailable"
    assert context["snapshot"] is None and context["rows"] is None


def test_score_extension_preserves_inherited_evidence_and_context_only_scores(universe, tmp_path):
    from zoo.build import extend_scores
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    source.write_text(f"repo,bench,value,revision\norg/parent,task,.5,{SHA1}\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value",
        source="local", revision="revision"), models)
    parent = _build(universe, tmp_path / "parent", score_tables=[scores])
    before = _snapshot(parent.resolve())
    out = extend_scores(parent, [scores], tmp_path / "extended", universe.root,
                        source_tables=[scores])
    assert len(open_tables(out)["scores"]) == 2
    for name, raw in before.items():
        if name not in ["scores.parquet", "zoo.json", "collection.json"]:
            assert (out / name).read_bytes() == raw
    assert _snapshot(parent.resolve()) == before
    next_index = max(int(p.stem) for p in (parent / "score-sources").glob("*.json")) + 1
    assert (out / f"score-sources/{next_index}.csv").read_bytes() == source.read_bytes()
    assert (out / f"score-sources/{next_index + 1}.csv").read_bytes() == source.read_bytes()
    metadata = json.loads((out / "zoo.json").read_text())
    assert metadata["inherited_generation"]["zoo_json_sha256"] == hashlib.sha256(before["zoo.json"]).hexdigest()


def test_score_extension_rejects_context_not_in_parent_and_keeps_old_output(universe, tmp_path):
    from zoo.build import extend_scores
    from zoo.scores import load_scores_csv
    parent = _build(universe, tmp_path / "parent")
    out = _build(universe, tmp_path / "out")
    before = _snapshot(out.resolve())
    source = tmp_path / "scores.csv"
    source.write_text("repo,bench,value\norg/parent,task,.5\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value",
        source="local"), open_tables(parent)["models"])
    with pytest.raises(ValueError, match="inherited|context"):
        extend_scores(parent, [], out, universe.root, source_tables=[scores])
    assert _snapshot(out.resolve()) == before


def test_stale_score_binding_attributes_cannot_describe_changed_scores(universe, tmp_path):
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    source.write_text("repo,bench,value\norg/parent,task,.5\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value", source="local"), models)
    scores.loc[0, "value"] = .9
    with pytest.raises(ValueError, match="bindings"):
        _build(universe, tmp_path / "stale", score_tables=[scores])


@pytest.mark.parametrize("extension", [False, True])
def test_score_source_mutation_during_build_prevents_publication(universe, tmp_path, monkeypatch, extension):
    import zoo.build as builder
    from zoo.scores import load_scores_csv
    parent = _build(universe, tmp_path / "parent")
    source = tmp_path / "scores.csv"
    source.write_text("repo,bench,value\norg/parent,task,.5\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value", source="local"), open_tables(parent)["models"])
    original = builder._provenance
    def mutate_after_hash(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_text("repo,bench,value\norg/parent,task,.9\n")
        return result
    monkeypatch.setattr(builder, "_provenance", mutate_after_hash)
    out = tmp_path / "changed"
    with pytest.raises(ValueError, match="source.*changed|hash"):
        if extension:
            builder.extend_scores(parent, [scores], out, universe.root)
        else:
            _build(universe, out, score_tables=[scores])
    assert not out.exists()


def test_parent_mutation_while_reading_is_rejected(universe, tmp_path, monkeypatch):
    import zoo.build as builder
    parent = _build(universe, tmp_path / "parent")
    original = builder.open_tables
    def mutate_after_read(path):
        result = original(path)
        changed = parent / "models.parquet"
        changed.write_bytes(changed.read_bytes() + b"changed-during-read")
        return result
    monkeypatch.setattr(builder, "open_tables", mutate_after_read)
    with pytest.raises(ValueError, match="Inherited generation changed"):
        builder.extend_scores(parent, [], tmp_path / "changed", universe.root)


def test_large_item_count_stays_exact_beside_unknown_through_build_and_read(universe, tmp_path):
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    count = 2**53 + 1
    source.write_text(f"repo,bench,value,n\norg/parent,known,.5,{count}\norg/parent,unknown,.6,\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value",
        source="local", n_items="n"), models)
    out = _build(universe, tmp_path / "exact-count", score_tables=[scores])
    stored = pq.read_table(out / "scores.parquet").to_pylist()
    assert next(r["n_items"] for r in stored if r["benchmark"] == "known") == count
    assert next(r["n_items"] for r in stored if r["benchmark"] == "unknown") is None
    read = open_tables(out)["scores"]
    assert int(read.loc[read.benchmark.eq("known"), "n_items"].iloc[0]) == count


def test_default_rebuild_retains_scores_sources_and_unbound_rows_without_original(universe, tmp_path):
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    source.write_text(f"key,bench,value\norg/parent@{SHA1},task,0\norg/parent@{SHA1},task,0\nmissing@{'9'*40},task,.3\n")
    mapping = dict(model_key="key", benchmark="bench", value="value", source="local")
    scores = load_scores_csv(source, mapping, models)
    out = _build(universe, tmp_path / "retained", score_tables=[scores])
    before = pq.read_table(out / "scores.parquet").to_pylist()
    source_count = len(list((out / "score-sources").glob("*.json")))
    source.unlink()
    _build(universe, out)
    _build(universe, out)
    assert pq.read_table(out / "scores.parquet").to_pylist() == before
    assert len(list((out / "score-sources").glob("*.json"))) == source_count
    descriptors = [p for p in (out / "score-sources").glob("*.json")
                   if json.loads(p.read_text()).get("adapter") == "scores_csv"]
    assert len(descriptors) == 1
    context = json.loads(descriptors[0].read_text())
    assert len(context["rows"]) == 3 and context["rows"][2]["status"] == "unmatched_model"
    assert (out / context["snapshot"]).read_text().count("task,0") == 2
    coverage = json.loads((out / "score-coverage.json").read_text())
    assert any(r["model_key"] == "org/parent@" + SHA1 and r["status"] == "available" for r in coverage["records"])


def test_same_source_is_idempotent_but_distinct_sources_remain_separate(universe, tmp_path):
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    source.write_text("repo,bench,value,unused\norg/parent,task,.5,a\n")
    mapping = dict(repo_id="repo", benchmark="bench", value="value", source="local")
    scores = load_scores_csv(source, mapping, models)
    out = _build(universe, tmp_path / "idempotent", score_tables=[scores])
    source_count = len(list((out / "score-sources").glob("*.json")))
    _build(universe, out, score_tables=[scores])
    assert len(open_tables(out)["scores"]) == 1
    assert len(list((out / "score-sources").glob("*.json"))) == source_count
    source.write_text("repo,bench,value,unused\norg/parent,task,.5,b\n")
    _build(universe, out, score_tables=[load_scores_csv(source, mapping, models)])
    assert len(open_tables(out)["scores"]) == 2
    assert len(list((out / "score-sources").glob("*.json"))) == source_count + 1


def test_default_retains_programmatic_flags_and_explicit_reset_discards(universe, tmp_path):
    scores = pd.DataFrame([dict(model_key="org/parent@" + SHA1, repo_id="org/parent", benchmark="test",
        skill=None, value=.7, metric="accuracy", source="local", source_revision=SHA1,
        revision_verified=False, harness=None, n_items=None)], columns=SCORES_COLUMNS)
    out = _build(universe, tmp_path / "retained", score_tables=[scores])
    _build(universe, out)
    assert len(open_tables(out)["scores"]) == 1
    assert not open_tables(out)["scores"].revision_verified.any()
    # Simulate a legacy generation lacking source descriptors, using this
    # test-created programmatic table only; no scientific source is removed.
    for path in (out / "score-sources").glob("*.json"):
        descriptor = json.loads(path.read_text())
        if descriptor.get("status") == "source_context_unavailable":
            path.unlink()
    _build(universe, out)
    descriptors = [json.loads(p.read_text()) for p in (out / "score-sources").glob("*.json")]
    assert any(d.get("status") == "source_context_unavailable" for d in descriptors)
    _build(universe, out, retain_scores=False)
    assert open_tables(out)["scores"].empty


def test_hash_pinned_default_source_configuration_survives_source_deletion(universe, tmp_path):
    source = tmp_path / "scores.csv"
    source.write_text(f"key,bench,value\norg/parent@{SHA1},task,.25\n")
    config = tmp_path / "score-sources.json"
    config.write_text(json.dumps(dict(sources=[dict(adapter="scores_csv", path="scores.csv",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mapping=dict(model_key="key", benchmark="bench", value="value", source="local"))])))
    out = _build(universe, tmp_path / "configured")
    assert len(open_tables(out)["scores"]) == 1
    source.unlink()
    _build(universe, out)
    assert len(open_tables(out)["scores"]) == 1
    with pytest.raises(OSError):
        _build(universe, tmp_path / "fresh-without-source")


def test_context_only_unpinned_source_never_activates_on_default_rebuild(universe, tmp_path):
    from zoo.build import extend_scores
    from zoo.scores import load_scores_csv
    parent = _build(universe, tmp_path / "parent")
    source = tmp_path / "evidence-only.csv"
    source.write_text("repo,bench,value\norg/parent,unbound,.5\n")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    models = open_tables(parent)["models"].iloc[:0]
    table = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value", source="publisher"), models)
    out = extend_scores(parent, [], tmp_path / "context-only", universe.root, source_tables=[table])
    source.unlink()
    _build(universe, out)
    assert open_tables(out)["scores"].empty
    context = next(json.loads(p.read_text()) for p in (out / "score-sources").glob("*.json")
                   if json.loads(p.read_text()).get("adapter") == "scores_csv")
    assert context["role"] == "inherited_context_only"
    assert context["rows"][0]["scores"] == []
    # A deliberate hash-pinned local declaration activates that retained source.
    (tmp_path / "score-sources.json").write_text(json.dumps(dict(sources=[dict(
        adapter="scores_csv", path="evidence-only.csv", sha256=source_hash,
        mapping=dict(repo_id="repo", benchmark="bench", value="value", source="publisher"))])))
    _build(universe, out)
    active = open_tables(out)["scores"]
    assert len(active) == 1 and not active.revision_verified.any()
    _build(universe, out)
    assert len(open_tables(out)["scores"]) == 1


def test_retained_source_rebinds_new_exact_identity_without_duplicate_old_rows(universe, tmp_path):
    from zoo.scores import load_scores_csv
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    source = tmp_path / "scores.csv"
    source.write_text(f"key,bench,value\norg/parent@{SHA1},task,.5\norg/future@{SHA1},task,.6\n")
    scores = load_scores_csv(source, dict(model_key="key", benchmark="bench", value="value", source="local"), models)
    out = _build(universe, tmp_path / "expanding", score_tables=[scores])
    source.unlink()
    (universe.manifests / "future.manifest.json").write_text(json.dumps(dict(repo_id="org/future", revision=SHA1, access="metadata_only", tensors=[])))
    _build(universe, out)
    assert len(open_tables(out)["scores"]) == 2
    for path in universe.manifests.glob("*.manifest.json"):
        if json.loads(path.read_text())["repo_id"] == "org/parent":
            path.unlink()
    for path in universe.fingerprints.glob("org--parent__" + SHA1 + ".*"):
        path.unlink()
    _build(universe, out)
    assert open_tables(out)["scores"].model_key.tolist() == ["org/future@" + SHA1]
    context = next(json.loads(p.read_text()) for p in (out / "score-sources").glob("*.json")
                   if json.loads(p.read_text()).get("adapter") == "scores_csv")
    assert context["rows"][0]["scores"] == [] and context["rows"][0]["inactive_scores"]


def test_filling_unrelated_metadata_does_not_duplicate_card_annotations(tmp_path):
    import gzip
    from evidence.source_metadata import SourceMetadataCapture, canonical_json
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "one.manifest.json").write_text(json.dumps(dict(repo_id="org/parent", revision=SHA1,
        access="metadata_only", tensors=[])))
    capture = SourceMetadataCapture("org/parent", SHA1)
    capture.record_json("cardData.json", {"model-index":[{"results":[{
        "dataset":{"name":"task"},"metrics":[{"type":"accuracy","value":0}]}]}]})
    source = manifests / "one.source-metadata.json.gz"
    source.write_bytes(gzip.compress(canonical_json(capture.snapshot()), mtime=0))
    out = build_zoo(manifests, [], tmp_path / "out", tmp_path)
    assert len(open_tables(out)["scores"]) == 1
    capture.record_bytes("config.json", b'{"unrelated":true}')
    source.write_bytes(gzip.compress(canonical_json(capture.snapshot()), mtime=0))
    build_zoo(manifests, [], out, tmp_path)
    assert len(open_tables(out)["scores"]) == 1
    descriptors = list((out / "score-sources").glob("*.json"))
    assert len(descriptors) == 1
    context = json.loads(descriptors[0].read_text())
    assert context["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    retained = json.loads(gzip.decompress((out / context["snapshot"]).read_bytes()))
    assert retained["documents"]["config.json"]["content"] == '{"unrelated":true}'


def test_source_configuration_mutation_keeps_previous_generation(universe, tmp_path, monkeypatch):
    import zoo.build as builder
    out = _build(universe, tmp_path / "out")
    before = _snapshot(out.resolve())
    source = tmp_path / "source.csv"
    source.write_text(f"key,bench,value\norg/parent@{SHA1},task,.5\n")
    config = tmp_path / "score-sources.json"
    config.write_text(json.dumps(dict(sources=[dict(adapter="scores_csv", path="source.csv",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mapping=dict(model_key="key", benchmark="bench", value="value", source="local"))])))
    original = builder._provenance
    def mutate_after_hash(*args, **kwargs):
        result = original(*args, **kwargs)
        config.write_text('{"sources":[]}')
        return result
    monkeypatch.setattr(builder, "_provenance", mutate_after_hash)
    with pytest.raises(ValueError, match="configuration.*changed|configuration.*hash"):
        _build(universe, out)
    assert _snapshot(out.resolve()) == before


def test_every_missing_value_is_null_and_tables_use_zstd(built_zoo):
    for name in COLUMNS:
        path = built_zoo / f"{name}.parquet"
        assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
        table = pq.read_table(path)
        for field in table.schema:
            if pa.types.is_string(field.type):
                assert "" not in table.column(field.name).to_pylist(), (name, field.name)
    models = open_tables(built_zoo)["models"].set_index("model_key")
    assert models.loc["org/t0only@" + "8" * 40, ["row_block_fingerprint", "full_weight_fingerprint",
                                                    "full_weight_spectra"]].isna().all()


def test_score_extension_shares_inherited_files_and_rewrites_only_its_own(universe, tmp_path):
    from zoo.build import extend_scores
    from zoo.scores import load_scores_csv
    parent = _build(universe, tmp_path / "parent").resolve()
    before = _snapshot(parent)
    source = tmp_path / "scores.csv"
    source.write_text(f"repo,bench,value,revision\norg/parent,task,.5,{SHA1}\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value",
        source="local", revision="revision"), open_tables(parent)["models"])
    out = extend_scores(parent, [scores], tmp_path / "extended", universe.root).resolve()
    for name in before:
        shared = (out / name).stat().st_ino == (parent / name).stat().st_ino
        assert shared == (name not in {"scores.parquet", "zoo.json", "collection.json"}), name
    assert _snapshot(parent) == before and len(open_tables(out)["scores"]) == 1


def test_version2_generation_reads_in_current_columns_and_extends_as_version2(universe, tmp_path):
    from zoo.build import extend_scores
    from zoo.scores import load_scores_csv
    generation = _build(universe, tmp_path / "parent").resolve()
    old = tmp_path / "v2"
    import shutil
    shutil.copytree(generation, old)
    models = pq.read_table(old / "models.parquet")
    skipped = pa.array([[{"name": "x.bias", "reason": ""}]] * len(models),
                       pa.list_(pa.struct([("name", pa.string()), ("reason", pa.string())])))
    pq.write_table(models.append_column("skipped_tensors", skipped), old / "models.parquet")
    metadata = json.loads((old / "zoo.json").read_text())
    metadata["zoo_version"] = "2"
    (old / "zoo.json").write_text(json.dumps(metadata))
    tables = open_tables(old)
    assert list(tables["models"].columns) == MODELS_COLUMNS
    assert list(tables["views"].columns) == VIEWS_COLUMNS
    source = tmp_path / "scores.csv"
    source.write_text(f"repo,bench,value,revision\norg/parent,task,.5,{SHA1}\n")
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value",
        source="local", revision="revision"), tables["models"])
    out = extend_scores(old, [scores], tmp_path / "extended", universe.root)
    assert json.loads((out / "zoo.json").read_text())["zoo_version"] == "2"
    assert "skipped_tensors" in pq.read_schema(out / "models.parquet").names


def test_rebuild_recognizes_a_programmatic_table_recorded_with_empty_text(universe, tmp_path):
    """A version <=2 build stored "" (and its own table id); v3 resupplies null."""
    from zoo.build import _arrow
    from zoo.schema import SCORES_COLUMNS
    key = "org/parent@" + SHA1
    row = dict(model_key=key, repo_id="org/parent", benchmark="task", skill=None, value=.5, metric="accuracy",
               source="local", source_revision=None, revision_verified=False, harness=None, n_items=None)
    table = pd.DataFrame([row], columns=SCORES_COLUMNS)
    out = _build(universe, tmp_path / "zoo", score_tables=[table])
    generation = out.resolve()
    descriptor_path = next(p for p in (generation / "score-sources").glob("*.json")
                           if json.loads(p.read_text()).get("status") == "source_context_unavailable")
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["table_id"] = "recorded-by-an-older-build"
    descriptor["scores"] = [dict(score, source_revision="", skill="", harness="") for score in descriptor["scores"]]
    descriptor_path.write_text(json.dumps(descriptor))
    rebuilt = _build(universe, tmp_path / "zoo", score_tables=[table])
    assert len(open_tables(rebuilt)["scores"]) == 1


def test_score_extension_of_version2_parent_keeps_inherited_text_and_parent_files(universe, tmp_path):
    from zoo.build import extend_scores
    from zoo.scores import load_scores_csv
    source = tmp_path / "scores.csv"
    source.write_text("repo,bench,value\norg/parent,task,.5\n")  # No revision: source_revision missing.
    models = open_tables(_build(universe, tmp_path / "baseline"))["models"]
    scores = load_scores_csv(source, dict(repo_id="repo", benchmark="bench", value="value", source="local"), models)
    parent = _build(universe, tmp_path / "parent", score_tables=[scores]).resolve()
    # Present the parent as a version 2 generation that stored "" for missing text.
    inherited = pq.read_table(parent / "scores.parquet").to_pylist()
    rows = [dict(r, source_revision="", harness="") for r in inherited]
    parent_scores = parent / "scores.parquet"
    parent_scores.unlink()
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(tmp_path / "baseline" / "scores.parquet")), parent_scores)
    metadata = json.loads((parent / "zoo.json").read_text())
    metadata["zoo_version"] = "2"
    (parent / "zoo.json").unlink()
    (parent / "zoo.json").write_text(json.dumps(metadata))
    orphan = parent / "score-sources" / "40.csv"  # A snapshot without a descriptor.
    orphan.write_bytes(b"orphan\n")
    before = _snapshot(parent)
    out = extend_scores(parent, [scores], tmp_path / "extended", universe.root).resolve()
    extended = pq.read_table(out / "scores.parquet").to_pylist()
    assert extended[:len(rows)] == rows and json.loads((out / "zoo.json").read_text())["zoo_version"] == "2"
    assert _snapshot(parent) == before
    assert min(int(p.name.split(".")[0]) for p in out.joinpath("score-sources").glob("*")
               if not (parent / "score-sources" / p.name).exists()) == 41
