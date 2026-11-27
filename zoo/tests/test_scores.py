import pandas as pd
import pytest

from zoo.schema import SCORES_COLUMNS
from zoo.scores import LEADERBOARD_V2_SKILLS, load_leaderboard_v2, load_scores_csv
from zoo.testing import SHA1, SHA2, SHA10


@pytest.fixture
def lb_csv(tmp_path):
    path = tmp_path / "leaderboard.csv"
    rows = [
        {"fullname": "org/child", "Model sha": SHA2, "Precision": "bfloat16"},
        {"fullname": "org/child", "Model sha": SHA2, "Precision": "float16"},
        {"fullname": "ORG/PARENT", "Model sha": SHA1, "Precision": "bfloat16"},
    ]
    for i, row in enumerate(rows):
        row.update({benchmark: .4 + .1 * i for benchmark in LEADERBOARD_V2_SKILLS})
        row["IFEval"] = 99.0  # Normalized values are never read as Raw scores.
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_leaderboard_verified_join(zoo_models, lb_csv):
    s = load_leaderboard_v2(lb_csv, zoo_models)
    assert set(s.columns) == set(SCORES_COLUMNS)
    row = s[(s.model_key == "org/child@" + SHA2) & (s.benchmark == "IFEval Raw")].iloc[0]
    assert row.skill == "instruction_following" and 0 <= row.value <= 1 and bool(row.revision_verified)


def test_other_revision_unverified_not_merged(zoo_models, lb_csv):
    s = load_leaderboard_v2(lb_csv, zoo_models)
    other = s[s.model_key == "org/child@" + SHA10]
    assert len(other) and not other.revision_verified.any()


def test_generic_csv_without_revision(zoo_models, tmp_path):
    p = tmp_path / "eee.csv"; p.write_text("model,bench,score\norg/parent,arc,0.5\n")
    s = load_scores_csv(p, {"repo_id": "model", "benchmark": "bench", "value": "score", "source": "every_eval_ever"}, zoo_models)
    assert (~s.revision_verified).all() and s.source.eq("every_eval_ever").all()


def test_leaderboard_preserves_precision_and_casefolds_repositories(zoo_models, lb_csv):
    result = load_leaderboard_v2(lb_csv, zoo_models)
    child = result[result.model_key == "org/child@" + SHA2]
    assert len(child) == 12
    assert set(child.harness) == {"open_llm_leaderboard_v2:bfloat16", "open_llm_leaderboard_v2:float16"}
    assert set(child[child.benchmark == "IFEval Raw"].value) == {.4, .5}
    parent = result[result.model_key == "org/parent@" + SHA1]
    assert len(parent) == 6 and parent.revision_verified.all()
    assert set(parent.repo_id) == {"org/parent"}
    assert result.metric.eq("raw_accuracy").all()
    assert result.source.eq("open_llm_leaderboard_v2").all()
    assert result.value.max() <= 1


def test_missing_pins_scores_and_unmatched_repositories(zoo_models, tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame([
        {"fullname": "org/child", "Model sha": "", "IFEval Raw": .7},
        {"fullname": "org/child", "Model sha": SHA2, "IFEval Raw": ""},
        {"fullname": "not/in-zoo", "Model sha": SHA2, "IFEval Raw": .2},
    ]).to_csv(path, index=False)
    result = load_leaderboard_v2(path, zoo_models)
    assert len(result) == 2 and not result.revision_verified.any()
    assert result.source_revision.isna().all()
    assert set(result.model_key) == {"org/child@" + SHA2, "org/child@" + SHA10}
    assert result.value.eq(.7).all()


def test_generic_mapping_keeps_source_columns_and_revisions(zoo_models, tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame([
        {"model": "ORG/CHILD", "commit": SHA2, "bench": "speed", "value": 15.5, "source_name": "measured-a", "ability": "speed"},
        {"model": "org/child", "commit": SHA10, "bench": "speed", "value": 16.5, "source_name": "measured-b", "ability": "speed"},
    ]).to_csv(path, index=False)
    result = load_scores_csv(path, {"repo_id": "model", "revision": "commit", "benchmark": "bench",
                                   "value": "value", "source": "source_name", "skill": "ability"}, zoo_models)
    assert len(result) == 4
    verified = result[result.revision_verified]
    assert dict(zip(verified.model_key, verified.value)) == {"org/child@" + SHA2: 15.5, "org/child@" + SHA10: 16.5}
    assert set(result.source) == {"measured-a", "measured-b"} and result.skill.eq("speed").all()
    assert set(result.columns) == set(SCORES_COLUMNS)


def test_raw_invalid_values_are_not_normalized_or_substituted(zoo_models, tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame([{"fullname": "org/parent", "Model sha": SHA1, "IFEval Raw": 87.0, "IFEval": .87}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Raw|accuracy"):
        load_leaderboard_v2(path, zoo_models)
    path.write_text("model,bench,value\norg/parent,acc,inf\n")
    with pytest.raises(ValueError, match="finite"):
        load_scores_csv(path, {"repo_id": "model", "benchmark": "bench", "value": "value", "source": "x"}, zoo_models)


@pytest.mark.parametrize("adapter", ["leaderboard", "generic"])
def test_leading_zero_identity_text_is_preserved(adapter, tmp_path):
    pin = "0" * 39 + "1"
    repo = "org/leading-zero" if adapter == "leaderboard" else "000123"
    models = pd.DataFrame([dict(model_key=repo + "@" + pin, repo_id=repo, revision=pin)])
    path = tmp_path / "text-identity.csv"
    if adapter == "leaderboard":
        pd.DataFrame([{"fullname": repo, "Model sha": pin, "IFEval Raw": .5}]).to_csv(path, index=False)
        result = load_leaderboard_v2(path, models)
    else:
        pd.DataFrame([{"repo": repo, "commit": pin, "bench": "acc", "value": .5}]).to_csv(path, index=False)
        result = load_scores_csv(path, {"repo_id": "repo", "revision": "commit", "benchmark": "bench",
                                        "value": "value", "source": "fixture"}, models)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.source_revision == pin and row.repo_id == repo and bool(row.revision_verified)


def test_generic_preserves_optional_context_and_duplicate_evaluations(zoo_models, tmp_path):
    path = tmp_path / "contexts.csv"
    path.write_text("model,bench,value,metric,harness,n,commit,template\n"
                    f"org/parent,task,.5,acc,eval:bf16,42,{SHA1},a\n"
                    f"org/parent,task,.5,exact_match,eval:fp16,0,{SHA1},b\n"
                    f"org/parent,task,.5,acc,eval:bf16,42,{SHA1},a\n")
    mapping = dict(repo_id="model", benchmark="bench", value="value", source="local",
                   revision="commit", metric="metric", harness="harness", n_items="n")
    result = load_scores_csv(path, mapping, zoo_models)
    assert len(result) == 3
    assert sorted(zip(result.metric, result.harness, result.n_items)) == [
        ("acc", "eval:bf16", 42), ("acc", "eval:bf16", 42),
        ("exact_match", "eval:fp16", 0)]
    assert result.revision_verified.all()
    context = result.attrs["score_source"]
    assert context["mapping"] == mapping
    assert len(context["rows"]) == 3
    assert [r["source_record"] for r in context["rows"]] == [1, 2, 3]


@pytest.mark.parametrize("n", ["-1", "1.5", "inf", "nan", "abc", "9223372036854775808"])
def test_generic_rejects_invalid_mapped_item_count(zoo_models, tmp_path, n):
    path = tmp_path / "count.csv"
    path.write_text(f"model,bench,value,n\norg/parent,task,.5,{n}\n")
    with pytest.raises(ValueError, match="n_items"):
        load_scores_csv(path, dict(repo_id="model", benchmark="bench", value="value",
                                  source="local", n_items="n"), zoo_models)


@pytest.mark.parametrize("field", ["metric", "harness", "n_items"])
def test_generic_requires_explicitly_mapped_context_column(zoo_models, tmp_path, field):
    path = tmp_path / "missing-context.csv"
    path.write_text("model,bench,value\norg/parent,task,.5\n")
    mapping = dict(repo_id="model", benchmark="bench", value="value", source="local")
    mapping[field] = "missing"
    with pytest.raises(ValueError, match=field):
        load_scores_csv(path, mapping, zoo_models)


def test_generic_missing_context_remains_unknown(zoo_models, tmp_path):
    path = tmp_path / "unknown.csv"
    path.write_text("model,bench,value,metric,harness,n\norg/parent,task,.5,,,\n")
    base = dict(repo_id="model", benchmark="bench", value="value", source="local")
    legacy = load_scores_csv(path, base, zoo_models)
    assert legacy.metric.eq("score").all()
    assert legacy.harness.isna().all() and legacy.n_items.isna().all()
    mapped = load_scores_csv(path, dict(base, metric="metric", harness="harness", n_items="n"), zoo_models)
    assert mapped.metric.isna().all() and mapped.harness.isna().all() and mapped.n_items.isna().all()


def test_source_binding_retains_unmatched_and_missing_score_records(zoo_models, tmp_path):
    path = tmp_path / "all-contexts.csv"
    path.write_text(f"fullname,Model sha,Precision,BBH Raw,BBH,Chat Template,other\n"
                    f"org/parent,{SHA1},bfloat16,.5,99,true,retain-me\n"
                    f"not/in-zoo,,float16,.3,88,false,unbound\n"
                    f"org/parent,{SHA1},bfloat16,,77,true,no-raw\n")
    result = load_leaderboard_v2(path, zoo_models)
    assert len(result) == 1 and result.value.tolist() == [.5]
    context = result.attrs["score_source"]
    assert [r["status"] for r in context["rows"]] == ["imported", "unmatched_model", "missing_score"]
    assert context["rows"][0]["scores"][0]["model_key"] == "org/parent@" + SHA1
    assert context["rows"][1]["matched_model_keys"] == []
    assert context["rows"][2]["scores"] == []


def test_generic_exact_model_key_mapping_never_falls_back_to_repository(zoo_models, tmp_path):
    path = tmp_path / "exact.csv"
    path.write_text(f"key,task,accuracy\norg/child@{SHA2},task,.5\n"
                    f"org/child@{'9' * 40},task,.6\n")
    result = load_scores_csv(path, dict(model_key="key", benchmark="task", value="accuracy",
                                        source="local"), zoo_models)
    assert len(result) == 1 and result.model_key.tolist() == ["org/child@" + SHA2]
    assert result.source_revision.tolist() == [SHA2] and result.revision_verified.all()
    assert result.attrs["score_source"]["rows"][1]["status"] == "unmatched_model"


@pytest.mark.parametrize("repo,revision", [("org/parent", SHA2), ("org/child", SHA1)])
def test_exact_key_mapping_rejects_contradictory_identity_columns(zoo_models, tmp_path, repo, revision):
    path = tmp_path / "contradictory.csv"
    path.write_text(f"key,repo,revision,task,value\norg/child@{SHA2},{repo},{revision},task,.5\n")
    with pytest.raises(ValueError, match="disagree|contradict"):
        load_scores_csv(path, dict(model_key="key", repo_id="repo", revision="revision",
                                  benchmark="task", value="value", source="local"), zoo_models)


def test_model_card_annotations_are_declared_zero_is_available_and_context_retained(tmp_path):
    import gzip
    import hashlib
    import json
    from zoo.scores import load_model_card_scores
    from evidence.source_metadata import SourceMetadataCapture
    zoo_models = pd.DataFrame([dict(model_key="org/parent@" + SHA1, repo_id="org/parent", revision=SHA1)])
    content = {"model-index": [{"name": "declared", "results": [{
        "task": {"type": "text-classification"},
        "dataset": {"name": "benchmark", "config": "subset", "split": "test"},
        "metrics": [{"type": "accuracy", "value": 0}, {"type": "accuracy", "value": "unsupported +/- value"},
                    {"type": "accuracy", "value": True}, {"type": "accuracy", "value": False}]}]}]}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    capture = SourceMetadataCapture("org/parent", SHA1, source_kind="hub")
    capture.record_json("cardData.json", content)
    snapshot = capture.snapshot()
    assert snapshot["documents"]["cardData.json"]["sha256"] == hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "card.source-metadata.json.gz"
    path.write_bytes(gzip.compress(json.dumps(snapshot).encode(), mtime=0))
    result = load_model_card_scores(path, zoo_models)
    assert len(result) == 1 and result.value.tolist() == [0]
    assert result.metric.tolist() == ["accuracy"] and result.benchmark.tolist() == ["benchmark"]
    assert result.skill.isna().all() and not result.revision_verified.any() and result.source_revision.isna().all()
    context = result.attrs["score_source"]
    assert context["snapshot_suffix"] == ".json.gz"
    assert context["rows"][0]["json_pointer"].endswith("/metrics/0")
    assert context["rows"][1]["status"] == "unsupported"
    assert context["rows"][2]["status"] == context["rows"][3]["status"] == "unsupported"
    # Only a hash-bound complete immutable package can share its source inode.
    (tmp_path / "package.json").write_text(json.dumps(dict(complete=True,
        model_key="org/parent@" + SHA1,
        files={path.name: hashlib.sha256(path.read_bytes()).hexdigest()})))
    packaged = load_model_card_scores(path, zoo_models)
    assert packaged.attrs["score_source"]["immutable_snapshot"] is True
    from zoo.scores import write_score_sources
    output = tmp_path / "generation"
    output.mkdir()
    write_score_sources(output, [packaged])
    assert (output / "score-sources/0.json.gz").stat().st_ino == path.stat().st_ino


@pytest.mark.parametrize("document_status,expected", [
    ("not_captured", "not_attempted"), ("missing", "missing"),
    ("error", "failed"), ("oversize", "unsupported")])
def test_model_card_coverage_preserves_capture_states(tmp_path, document_status, expected):
    import gzip
    from evidence.source_metadata import SourceMetadataCapture, canonical_json
    from zoo.scores import load_model_card_scores
    models = pd.DataFrame([dict(model_key="org/parent@" + SHA1, repo_id="org/parent", revision=SHA1)])
    capture = SourceMetadataCapture("org/parent", SHA1)
    capture.record_status("cardData.json", document_status)
    path = tmp_path / "capture.json.gz"
    path.write_bytes(gzip.compress(canonical_json(capture.snapshot()), mtime=0))
    table = load_model_card_scores(path, models)
    assert table.empty and table.attrs["score_source"]["collection_status"] == expected


def test_default_card_loading_validates_target_index_once(tmp_path, monkeypatch):
    import gzip
    import zoo.scores as scores
    from evidence.source_metadata import SourceMetadataCapture, canonical_json
    models = pd.DataFrame([dict(model_key="org/parent@" + SHA1, repo_id="org/parent", revision=SHA1),
                           dict(model_key="org/child@" + SHA2, repo_id="org/child", revision=SHA2)])
    paths = []
    for index, row in enumerate(models.to_dict("records")):
        capture = SourceMetadataCapture(row["repo_id"], row["revision"])
        path = tmp_path / f"{index}.json.gz"
        path.write_bytes(gzip.compress(canonical_json(capture.snapshot()), mtime=0))
        paths.append(path)
    original, calls = scores._models, []
    def counted(table):
        calls.append(len(table))
        return original(table)
    monkeypatch.setattr(scores, "_models", counted)
    scores.prepare_score_inputs(models, (), card_paths=paths)
    assert calls == [2]
