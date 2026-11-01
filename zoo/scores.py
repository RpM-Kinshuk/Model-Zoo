"""Measured score imports with explicit, revision-specific provenance."""

import math
from pathlib import Path

import pandas as pd

from .schema import SCORES_COLUMNS, empty_table, model_key


LEADERBOARD_V2_SKILLS = {
    "IFEval Raw": "instruction_following",
    "BBH Raw": "reasoning",
    "MATH Lvl 5 Raw": "math",
    "GPQA Raw": "science_qa",
    "MUSR Raw": "reasoning",
    "MMLU-PRO Raw": "knowledge",
}


def _models(zoo_models):
    required = {"model_key", "repo_id", "revision"}
    if not required <= set(zoo_models):
        raise ValueError("Zoo models need model_key, repo_id and revision")
    if zoo_models.model_key.duplicated().any():
        raise ValueError("Duplicate zoo model keys")
    result = {}
    for row in zoo_models.to_dict("records"):
        if row["model_key"] != model_key(row["repo_id"], row["revision"]):
            raise ValueError("Zoo model key disagrees with its pinned identity")
        result.setdefault(row["repo_id"].casefold(), []).append(row)
    return result


def _value(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Score must be numeric, got {value!r}") from error
    if math.isnan(number):
        return None
    if not math.isfinite(number):
        raise ValueError("Scores must be finite")
    return number


def _record(target, benchmark, value, source, source_revision, skill=None,
            metric="score", harness=None):
    return dict(model_key=target["model_key"], repo_id=target["repo_id"],
                benchmark=benchmark, skill=skill, value=value, metric=metric,
                source=source, source_revision=source_revision,
                revision_verified=bool(source_revision and source_revision == target["revision"]),
                harness=harness, n_items=None)


def _table(records):
    if not records:
        return empty_table(SCORES_COLUMNS)
    return pd.DataFrame(records, columns=SCORES_COLUMNS).sort_values(
        ["model_key", "benchmark", "source", "source_revision", "harness"],
        kind="stable", na_position="last",
    ).reset_index(drop=True)


def load_leaderboard_v2(path: Path, zoo_models: pd.DataFrame) -> pd.DataFrame:
    """Read Raw accuracies without rescaling, averaging or normalized fallback.

    Each evaluation is retained for every matching repository revision. The
    source pin decides verification separately for each resulting model key.
    Precision remains part of the evaluation's harness identity.
    """
    source = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "fullname" not in source:
        raise ValueError("Leaderboard export requires fullname")
    targets = _models(zoo_models)
    records = []
    for row in source.to_dict("records"):
        matches = targets.get(str(row["fullname"]).casefold(), [])
        if not matches:
            continue
        revision = str(row.get("Model sha", "") or "")
        precision = str(row.get("Precision", "") or "")
        for benchmark, skill in LEADERBOARD_V2_SKILLS.items():
            value = _value(row.get(benchmark))
            if value is None:
                continue
            if not 0 <= value <= 1:
                raise ValueError(f"{benchmark} must be a Raw accuracy in [0, 1]")
            records.extend(_record(target, benchmark, value, "open_llm_leaderboard_v2",
                                   revision, skill, "raw_accuracy",
                                   "open_llm_leaderboard_v2:" + precision)
                           for target in matches)
    return _table(records)


def load_scores_csv(path: Path, mapping: dict, zoo_models: pd.DataFrame) -> pd.DataFrame:
    """Import a generic numeric score source without inventing revision pins.

    Mapping values name CSV columns. For source only, a value that does not
    name a column is treated as the constant source label.
    """
    source = pd.read_csv(path, dtype=str, keep_default_na=False)
    for key in ("repo_id", "benchmark", "value", "source"):
        if not mapping.get(key):
            raise ValueError(f"Score mapping requires {key}")
    for key in ("repo_id", "benchmark", "value", "revision", "skill"):
        if mapping.get(key) and mapping[key] not in source:
            raise ValueError(f"Mapped {key} column is missing: {mapping[key]}")
    targets = _models(zoo_models)
    records = []
    for row in source.to_dict("records"):
        matches = targets.get(str(row[mapping["repo_id"]]).casefold(), [])
        if not matches:
            continue
        value = _value(row[mapping["value"]])
        if value is None:
            continue
        revision = str(row[mapping["revision"]] or "") if mapping.get("revision") else ""
        skill = row[mapping["skill"]] if mapping.get("skill") else None
        label = row[mapping["source"]] if mapping["source"] in source else mapping["source"]
        records.extend(_record(target, row[mapping["benchmark"]], value, label,
                               revision, skill) for target in matches)
    return _table(records)
