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
    assert result.source_revision.eq("").all()
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
