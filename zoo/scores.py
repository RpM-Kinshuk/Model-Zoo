"""Measured score imports with explicit, revision-specific provenance."""

from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
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
            metric="score", harness=None, n_items=None):
    text = lambda item: None if item == "" else item  # Missing text is null.
    return dict(model_key=target["model_key"], repo_id=target["repo_id"],
                benchmark=text(benchmark), skill=text(skill), value=value, metric=text(metric),
                source=text(source), source_revision=text(source_revision),
                revision_verified=bool(source_revision and source_revision == target["revision"]),
                harness=text(harness), n_items=n_items)


def _table(records):
    if not records:
        return empty_table(SCORES_COLUMNS)
    table = pd.DataFrame(records, columns=SCORES_COLUMNS)
    table["n_items"] = pd.array([r["n_items"] for r in records], dtype="Int64")
    return table.sort_values(
        ["model_key", "benchmark", "source", "source_revision", "harness"],
        kind="stable", na_position="last",
    ).reset_index(drop=True)


def _read_source(path, adapter, mapping=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    source = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    context = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                   adapter=adapter, mapping=mapping, rows=[])
    return source, context


def _binding(context, index, matches, records):
    context["rows"].append(dict(source_record=index + 1,
        matched_model_keys=[r["model_key"] for r in matches], scores=records,
        status="imported" if records else "missing_score" if matches else "unmatched_model"))


def _with_source(records, context):
    table = _table(records)
    table.attrs["score_source"] = context
    return table


def source_identity(context):
    """Source identity excludes mutable paths and current target bindings."""
    if context.get("adapter") == "model_card":
        fields = {key: context.get(key) for key in ["adapter", "metadata_identity", "card_document_sha256"]}
    else:
        fields = {key: context.get(key) for key in ["sha256", "adapter", "mapping"]}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _item_count(value):
    if value is None or value == "":
        return None
    try:
        number = Decimal(value)
        if not number.is_finite() or number < 0 or number > 2**63 - 1 or number != number.to_integral_value():
            raise ValueError("n_items must be a finite nonnegative int64 integer or empty")
        return int(number)
    except (InvalidOperation, TypeError) as error:
        raise ValueError("n_items must be a finite nonnegative int64 integer or empty") from error


def _canonical_scores(records, *, empty_text_as_null=True):
    """Score identities for retention; an empty string and null both mean missing.

    Zoo version 3 writes missing text as null. Older generations and retained
    source bindings may hold "" for the same score, which must stay one identity.
    """
    if empty_text_as_null:
        records = ({key: None if value == "" else value for key, value in row.items()} for row in records)
    return Counter(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                   for row in records)


def _table_identity(records, **options):
    return hashlib.sha256(json.dumps(sorted(_canonical_scores(records, **options).items())).encode()).hexdigest()


def score_source_paths(tables):
    """Original source files participating in normal output-overlap guards."""
    return [Path(table.attrs["score_source"]["path"]).resolve()
            for table in tables if "score_source" in table.attrs]


def check_score_sources(tables):
    """Recheck at publication after parsing/building/provenance hashing."""
    for table in tables:
        context = table.attrs.get("score_source")
        if context is not None and hashlib.sha256(Path(context["path"]).read_bytes()).hexdigest() != context["sha256"]:
            raise ValueError("Score source changed during build (hash mismatch)")


def write_score_sources(directory, tables, start_index=0, roles=None):
    """Retain byte-exact source snapshots and bindings inside a private generation.

    Importer attrs are checked against both raw source hashes and full table-row
    multiplicity, so later edits cannot silently retain stale source bindings.
    Bare programmatic tables explicitly lack recoverable source context.
    """
    from .build import _arrow
    directory = Path(directory)
    if tables:
        (directory / "score-sources").mkdir(exist_ok=True)
    for offset, table in enumerate(tables):
        index = start_index + offset
        context = table.attrs.get("score_source")
        descriptor = dict(status="source_context_unavailable", snapshot=None, rows=None)
        if context is not None:
            raw = Path(context["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != context["sha256"]:
                raise ValueError("Score source changed after import (hash mismatch)")
            bound = [score for row in context["rows"] for score in row["scores"]]
            if _canonical_scores(bound) != _canonical_scores(_arrow("scores", table).to_pylist()):
                raise ValueError("Score source bindings disagree with score table")
            suffix = context.get("snapshot_suffix", ".csv")
            if suffix not in {".csv", ".json.gz"}:
                raise ValueError("Unsupported score snapshot suffix")
            descriptor = dict(context, status="retained", snapshot=f"score-sources/{index}{suffix}",
                              source_id=source_identity(context))
            destination = directory / descriptor["snapshot"]
            if context.get("immutable_snapshot"):
                immutable = Path(context.get("immutable_snapshot_path", context["path"]))
                if hashlib.sha256(immutable.read_bytes()).hexdigest() != context["sha256"]:
                    raise ValueError("Immutable score snapshot hash mismatch")
                try:
                    os.link(immutable, destination)
                except FileExistsError:
                    raise
                except OSError:
                    _write_new(destination, raw)
            else:
                _write_new(destination, raw)
        elif "score_source_unavailable" in table.attrs:
            descriptor.update(table.attrs["score_source_unavailable"])
        descriptor["role"] = roles[offset] if roles is not None else descriptor.get("role", "imported_scores")
        _write_new(directory / f"score-sources/{index}.json",
                   (json.dumps(descriptor, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _write_new(path, data):
    """Create a new generation file; an existing name may be a link to an older generation."""
    with open(path, "xb") as handle:
        handle.write(data)


def prepare_score_inputs(models, explicit_tables, prior=None, config=None, card_paths=()):
    """Resolve local defaults and retain old values/flags without duplicate imports."""
    from .build import _arrow
    import pyarrow.parquet as pq
    selected = set(models.model_key)
    exact_targets = {r["model_key"]: r for group in _models(models).values() for r in group}
    previous, contexts, context_ids, extra_paths = [], [], {}, []
    configuration_pin = None
    old_keys = set()
    if prior is not None:
        previous = pq.read_table(prior / "scores.parquet").to_pylist()
        previous = [r for r in previous if r["model_key"] in selected]
        old_keys = set(pq.read_table(prior / "models.parquet", columns=["model_key"]).column("model_key").to_pylist())
        for path in sorted((prior / "score-sources").glob("*.json"), key=lambda p:int(p.stem)):
            descriptor = json.loads(path.read_text())
            if descriptor.get("snapshot") is None:
                table = _table([])
                table.attrs["score_source_unavailable"] = descriptor
                contexts.append(table)
                continue
            snapshot = (prior / descriptor["snapshot"]).resolve(strict=True)
            if not snapshot.is_relative_to(prior):
                raise ValueError("Retained score source escapes generation")
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != descriptor["sha256"]:
                raise ValueError("Retained score source hash mismatch")
            context = dict(descriptor, path=str(snapshot), immutable_snapshot=True, immutable_snapshot_path=str(snapshot),
                           original_path=descriptor.get("original_path", descriptor.get("path")))
            if context.get("adapter") == "model_card" and "card_document_sha256" not in context:
                from evidence.source_metadata import read_source_metadata
                captured = read_source_metadata(snapshot)
                context["card_document_sha256"] = captured.get("documents", {}).get("cardData.json", {}).get("sha256")
            context.pop("snapshot", None)
            active = []
            for row in context["rows"]:
                historical = row["scores"] + row.get("inactive_scores", [])
                row["scores"] = [s for s in historical if s["model_key"] in selected]
                row["inactive_scores"] = [s for s in historical if s["model_key"] not in selected]
                row["matched_model_keys"] = [k for k in row["matched_model_keys"] if k in selected]
                row["status"] = ("imported" if row["scores"] else "inactive" if row["inactive_scores"]
                                 else row.get("status", "missing_score"))
                active.extend(row["scores"])
                previous.extend(s for s in row["scores"] if s["model_key"] not in old_keys)
            table = _with_source(active, context)
            identity = source_identity(context)
            context_ids[identity] = len(contexts)
            contexts.append(table)

    represented = []
    for table in contexts:
        context = table.attrs.get("score_source")
        represented.extend([s for row in context["rows"] for s in row["scores"]]
                           if context else table.attrs.get("score_source_unavailable", {}).get("scores", []))
    unknown = _canonical_scores(previous) - _canonical_scores(represented)
    if unknown:
        records = [json.loads(row) for row in unknown.elements()]
        table = _table(records)
        identity = _table_identity(records)
        table.attrs["score_source_unavailable"] = dict(table_id=identity, scores=records)
        contexts.append(table)

    def admit(table, activate=False):
        nonlocal previous
        context = table.attrs.get("score_source")
        records = _arrow("scores", table).to_pylist()
        if context is None:
            identity = _table_identity(records)
            # Match retained tables by content, not only their stored id: a zoo
            # version <=2 build hashed the same scores with "" where v3 has null.
            retained = [t.attrs["score_source_unavailable"] for t in contexts
                        if "score_source_unavailable" in t.attrs]
            if any(identity == item.get("table_id") or identity == _table_identity(item.get("scores") or [])
                   for item in retained):
                return
            table.attrs["score_source_unavailable"] = dict(table_id=identity, scores=records)
            previous.extend(records)
            contexts.append(table)
            return
        identity = source_identity(context)
        if identity in context_ids:
            old = contexts[context_ids[identity]]
            old_context = old.attrs["score_source"]
            old_context["path"] = context["path"]
            if context.get("adapter") == "model_card" and context["sha256"] != old_context["sha256"]:
                # Metadata enrichment is not another benchmark evaluation.
                old_context["sha256"] = context["sha256"]
                for field in ["immutable_snapshot", "immutable_snapshot_path"]:
                    old_context.pop(field, None)
                    if field in context:
                        old_context[field] = context[field]
            if old_context.get("role") == "inherited_context_only":
                if not activate:
                    return
                old_context["role"] = "imported_scores"
            # Same source may gain selected identities; preserve every old binding/flag.
            for row, fresh in zip(old_context["rows"], context["rows"]):
                known = {s["model_key"] for s in row["scores"] + row.get("inactive_scores", [])}
                added = [s for s in fresh["scores"] if s["model_key"] not in known]
                row["scores"].extend(added)
                row["matched_model_keys"] = fresh["matched_model_keys"]
                row["status"] = "imported" if row["scores"] else fresh["status"]
                previous.extend(added)
            contexts[context_ids[identity]] = _with_source(
                [s for row in old_context["rows"] for s in row["scores"]], old_context)
            return
        context_ids[identity] = len(contexts)
        contexts.append(table)
        previous.extend(records)

    # Rebind retained snapshots to newly selected identities without original paths.
    for table in list(contexts):
        context = table.attrs.get("score_source")
        if context is not None:
            if context.get("role") == "inherited_context_only":
                continue  # Evidence-only sources were deliberately not activated.
            if context["adapter"] == "scores_csv":
                admit(load_scores_csv(Path(context["path"]), context["mapping"], models))
            elif context["adapter"] == "leaderboard_v2":
                admit(load_leaderboard_v2(Path(context["path"]), models))
            elif context["adapter"] == "model_card":
                admit(load_model_card_scores(Path(context["path"]), models, _targets=exact_targets))
    if config is not None and config.exists():
        configuration_bytes = config.read_bytes()
        configuration_pin = dict(path=str(config), sha256=hashlib.sha256(configuration_bytes).hexdigest())
        configuration = json.loads(configuration_bytes)
        if not isinstance(configuration, dict) or not isinstance(configuration.get("sources"), list):
            raise ValueError("Score source configuration requires a sources list")
        extra_paths.append(config)
        for source in configuration.get("sources", []):
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise ValueError("Configured score source requires path and adapter")
            if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in source["sha256"]):
                raise ValueError("Configured score sources require explicit sha256")
            identity = source_identity(source)
            if identity in context_ids:
                retained = contexts[context_ids[identity]].attrs["score_source"]
                if retained.get("role") != "inherited_context_only":
                    continue
                path = Path(retained["path"])
            else:
                path = (config.parent / source["path"]).resolve(strict=True)
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Configured score source hash mismatch")
            if source["adapter"] == "scores_csv":
                admit(load_scores_csv(path, source["mapping"], models), activate=True)
            elif source["adapter"] == "leaderboard_v2":
                admit(load_leaderboard_v2(path, models), activate=True)
            else:
                raise ValueError("Unsupported configured score adapter")
    for table in explicit_tables:
        admit(table, activate=True)
    for path in card_paths:
        admit(load_model_card_scores(path, models, _targets=exact_targets))
    coverage = []
    card_keys = set()
    all_keys = list(models.model_key)
    for table in contexts:
        context = table.attrs.get("score_source")
        source_id = source_identity(context) if context is not None else table.attrs.get("score_source_unavailable", {}).get("table_id", "unknown")
        if context and context.get("adapter") == "model_card":
            identity = context["metadata_identity"]
            key = identity["repo_id"] + "@" + identity["revision"]
            keys = [key] if key in selected else []
            card_keys.update(keys)
        else:
            keys = all_keys
        active_rows = ([s for row in context["rows"] for s in row["scores"]]
                       if context is not None else table.attrs.get("score_source_unavailable", {}).get("scores", []))
        counts = Counter(row["model_key"] for row in active_rows)
        for key in keys:
            count = counts[key]
            state = "available" if count else context.get("collection_status", "missing") if context else "unknown"
            coverage.append(dict(model_key=key, source_id=source_id, status=state, score_rows=count,
                                 context_available=context is not None))
    for key in models.model_key:
        if key not in card_keys:
            coverage.append(dict(model_key=key, source_id="model_card", status="not_attempted", score_rows=0))
    result = dict(records=coverage)
    if configuration_pin is not None:
        result["configuration"] = configuration_pin
    return _table(previous), contexts, result, extra_paths


def load_model_card_scores(path, zoo_models, *, _targets=None):
    """Read declared structured annotations from an already-collected snapshot."""
    from evidence.source_metadata import read_source_metadata
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = path.with_name(path.name.replace(".source-metadata.json.gz", ".manifest.json"))
    expected = None
    if manifest != path and manifest.is_file():
        source_manifest = json.loads(manifest.read_text())
        expected = {key: source_manifest[key] for key in ["repo_id", "revision"]}
    snapshot = read_source_metadata(path, identity=expected)
    identity = snapshot["identity"]
    key = identity["repo_id"] + "@" + identity["revision"]
    targets = (_targets if _targets is not None else
               {r["model_key"]: r for group in _models(zoo_models).values() for r in group})
    context = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), adapter="model_card",
        mapping=None, snapshot_suffix=".json.gz", metadata_identity=identity,
        source_format="source-metadata-v1/model-index", rows=[], collection_status="missing")
    package_path = path.parent / "package.json"
    if package_path.is_file():
        package = json.loads(package_path.read_text())
        if (package.get("complete") is True and package.get("model_key") == key
                and package.get("files", {}).get(path.name) == context["sha256"]):
            context.update(immutable_snapshot=True, immutable_snapshot_path=str(path))
    document = snapshot.get("documents", {}).get("cardData.json", {})
    context["card_document_sha256"] = document.get("sha256")
    if document.get("status") != "captured":
        status = document.get("status", "not_attempted")
        context["collection_status"] = ("failed" if status in {"error", "failed", "denied"} else
            "not_attempted" if status in {"not_attempted", "offline", "not_captured", "not_applicable"} else
            "missing" if status in {"missing", "absent"} else "unsupported")
        context["document_status"] = status
        return _with_source([], context)
    card = document.get("content")
    if not isinstance(card, dict):
        context["collection_status"] = "unsupported"
        return _with_source([], context)
    annotations = card.get("model-index")
    if annotations is None:
        return _with_source([], context)
    if not isinstance(annotations, list):
        context["collection_status"] = "unsupported"
        return _with_source([], context)
    records = []
    unsupported_shape = False
    for i, annotation in enumerate(annotations):
        results = annotation.get("results", []) if isinstance(annotation, dict) else None
        if not isinstance(results, list):
            unsupported_shape = True
            continue
        for j, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            dataset = result.get("dataset", {})
            benchmark = dataset.get("name") or dataset.get("type") if isinstance(dataset, dict) else None
            metrics = result.get("metrics", [])
            if not isinstance(metrics, list):
                unsupported_shape = True
                continue
            for k, metric in enumerate(metrics):
                emitted, state = [], "unsupported"
                if isinstance(metric, dict) and benchmark and isinstance(benchmark, str):
                    try:
                        value = None if isinstance(metric.get("value"), bool) else _value(metric.get("value"))
                    except ValueError:
                        value = None
                    if value is not None and key in targets:
                        name = metric.get("type") or metric.get("name")
                        if name is None or isinstance(name, str):
                            emitted = [_record(targets[key], benchmark, value,
                                "model_card:model-index", "", metric=name)]
                            records.extend(emitted)
                            state = "imported"
                    elif key not in targets:
                        state = "unmatched_model"
                    elif metric.get("value") in (None, ""):
                        state = "missing_score"
                context["rows"].append(dict(source_record=len(context["rows"]) + 1,
                    json_pointer=f"/documents/cardData.json/content/model-index/{i}/results/{j}/metrics/{k}",
                    matched_model_keys=[key] if key in targets else [], scores=emitted, status=state))
    context["collection_status"] = "available" if records else "unsupported" if context["rows"] or unsupported_shape else "missing"
    return _with_source(records, context)


def load_leaderboard_v2(path: Path, zoo_models: pd.DataFrame) -> pd.DataFrame:
    """Read Raw accuracies without rescaling, averaging or normalized fallback.

    Each evaluation is retained for every matching repository revision. The
    source pin decides verification separately for each resulting model key.
    Precision remains part of the evaluation's harness identity.
    """
    source, context = _read_source(path, "leaderboard_v2")
    if "fullname" not in source:
        raise ValueError("Leaderboard export requires fullname")
    targets = _models(zoo_models)
    records = []
    for index, row in enumerate(source.to_dict("records")):
        matches = targets.get(str(row["fullname"]).casefold(), [])
        if not matches:
            _binding(context, index, [], [])
            continue
        emitted = []
        revision = str(row.get("Model sha", "") or "")
        precision = str(row.get("Precision", "") or "")
        for benchmark, skill in LEADERBOARD_V2_SKILLS.items():
            value = _value(row.get(benchmark))
            if value is None:
                continue
            if not 0 <= value <= 1:
                raise ValueError(f"{benchmark} must be a Raw accuracy in [0, 1]")
            emitted.extend(_record(target, benchmark, value, "open_llm_leaderboard_v2",
                                   revision, skill, "raw_accuracy",
                                   "open_llm_leaderboard_v2:" + precision)
                           for target in matches)
        records.extend(emitted)
        _binding(context, index, matches, emitted)
    return _with_source(records, context)


def load_scores_csv(path: Path, mapping: dict, zoo_models: pd.DataFrame) -> pd.DataFrame:
    """Import a generic numeric score source without inventing revision pins.

    Mapping values name CSV columns. For source only, a value that does not
    name a column is treated as the constant source label.
    """
    source, context = _read_source(path, "scores_csv", dict(mapping))
    for key in ("benchmark", "value", "source"):
        if not mapping.get(key):
            raise ValueError(f"Score mapping requires {key}")
    if not mapping.get("repo_id") and not mapping.get("model_key"):
        raise ValueError("Score mapping requires repo_id or model_key")
    for key in ("repo_id", "model_key", "benchmark", "value", "revision", "skill", "metric", "harness", "n_items"):
        if mapping.get(key) and mapping[key] not in source:
            raise ValueError(f"Mapped {key} column is missing: {mapping[key]}")
    targets = _models(zoo_models)
    exact_targets = {r["model_key"]: r for group in targets.values() for r in group}
    records = []
    for index, row in enumerate(source.to_dict("records")):
        source_key = row[mapping["model_key"]] if mapping.get("model_key") else None
        if source_key is not None:
            if source_key:
                repo, separator, pin = source_key.rpartition("@")
                if mapping.get("repo_id") and row[mapping["repo_id"]] and row[mapping["repo_id"]].casefold() != repo.casefold():
                    raise ValueError("Mapped repository contradicts model_key")
                if mapping.get("revision") and row[mapping["revision"]] and row[mapping["revision"]] != pin:
                    raise ValueError("Mapped revision contradicts model_key")
            matches = [exact_targets[source_key]] if source_key in exact_targets else []
        else:
            matches = targets.get(str(row[mapping["repo_id"]]).casefold(), [])
        if not matches:
            _binding(context, index, [], [])
            continue
        value = _value(row[mapping["value"]])
        if value is None:
            _binding(context, index, matches, [])
            continue
        revision = (source_key.rsplit("@", 1)[1] if source_key is not None else
                    str(row[mapping["revision"]] or "") if mapping.get("revision") else "")
        skill = row[mapping["skill"]] if mapping.get("skill") else None
        label = row[mapping["source"]] if mapping["source"] in source else mapping["source"]
        metric = row[mapping["metric"]] or None if mapping.get("metric") else "score"
        harness = row[mapping["harness"]] or None if mapping.get("harness") else None
        n_items = _item_count(row[mapping["n_items"]]) if mapping.get("n_items") else None
        emitted = [_record(target, row[mapping["benchmark"]], value, label,
                           revision, skill, metric, harness, n_items) for target in matches]
        records.extend(emitted)
        _binding(context, index, matches, emitted)
    return _with_source(records, context)
