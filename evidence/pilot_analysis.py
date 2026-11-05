"""Offline pilot diagnostics over measured fingerprints and lab truth.

Run layout: manifests/, row_blocks/, full_weights/, and pairs.csv (the output of lineage_pairs).
Alternatively lab/edges.csv and adjacent models.csv can supply pairs. Summaries
are read from row_blocks/summary.json; its current rows supersede historical failures.csv.
The requested spectral cohort is full_weights/summary.json's current model identities;
without that declaration, spectral metrics are descriptive with unavailable
coverage. Default row-block companions in sibling full_weights/ supply comparisons/calibration,
but their full-read counters never supply standalone row-block throughput.
Missing lab pins use unmapped:lab/<id> endpoints. Source annotations of the form
|missing_manifest_ids=<JSON list> persist inventory mapping gaps through CSV.
Missing measurements stay explicit. Planned inventory rows never become evidence.
"""

from collections import Counter, defaultdict, deque
from itertools import combinations
import json
import math
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import torch

from net_esd.core import _fit_tail_chunk, _KS_CHUNK_SIZE
from measurement_config import safe_filename
from .compare import pair_similarity
from .compatibility import canonical_mode, companion_spectra_path, measurement_directory
from .fingerprint import load_fingerprint
from .sketch import SKETCH_SEED


def _key(manifest):
    return f"{manifest['repo_id']}@{manifest['revision']}"


def _json(path):
    return json.loads(Path(path).read_text())


def _atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _shape_signature(manifest):
    return tuple(sorted((t["name"], tuple(t["shape"])) for t in manifest.get("tensors", [])))


def lineage_pairs(lab_edges: Path, manifests_dir: Path) -> pd.DataFrame:
    """Map exact lab graph and certified reference pairs to pinned identities.

    Merge role=base is a reference, not a construction parent. Hub declarations
    are retained as sanity-only rows. A bare kind=independent is insufficient:
    frozen L11 puts known siblings in that category too.
    """
    lab_edges = Path(lab_edges)
    models = pd.read_csv(lab_edges.with_name("models.csv"), keep_default_na=False)
    edges = pd.read_csv(lab_edges, keep_default_na=False)
    if models.model_id.duplicated().any():
        raise ValueError("Duplicate lab model IDs")
    manifests = [_json(p) for p in sorted(Path(manifests_dir).glob("*.manifest.json"))]
    by_key = {}
    for manifest in manifests:
        key = _key(manifest)
        if key in by_key:
            raise ValueError(f"Duplicate manifest identity: {key}")
        by_key[key] = manifest
    mapped = {}
    for model in models.itertuples():
        external = str(model.path).rsplit("@", 1)
        if len(external) == 2:
            matches = [m for m in manifests if _key(m) == str(model.path)]
        else:
            matches = [m for m in manifests if m["repo_id"] in
                       (f"lab/{model.model_id}", str(model.path))]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous manifest revisions for {model.model_id}")
        if matches:
            mapped[model.model_id] = matches[0]
    parents = defaultdict(set)
    model_ids = set(models.model_id)
    for edge in edges.itertuples():
        if edge.child not in model_ids or edge.parent not in model_ids:
            raise ValueError("Lab edge refers to an unknown model")
        if getattr(edge, "role", "parent") != "base":
            parents[edge.child].add(edge.parent)
    ancestors = {}
    for child in sorted(model_ids):
        distances, queue = {}, deque((p, 1) for p in sorted(parents[child]))
        while queue:
            parent, depth = queue.popleft()
            if parent == child:
                raise ValueError("Cycle in lab construction edges")
            if parent in distances:
                continue
            distances[parent] = depth
            queue.extend((p, depth + 1) for p in sorted(parents[parent]))
        ancestors[child] = distances
    records = []
    missing_models = sorted(model_ids - mapped.keys())

    def model_key(model_id):
        if model_id in mapped:
            return _key(mapped[model_id])
        path = str(models.loc[models.model_id == model_id, "path"].iloc[0])
        return path if "@" in path else f"unmapped:lab/{model_id}"

    def add(a, b, label, source):
        if a != b:
            records.append(dict(a=a, b=b, label=label, source=source))

    for child, distances in ancestors.items():
        for parent, depth in distances.items():
            if depth <= 2:
                add(model_key(child), model_key(parent), "parent" if depth == 1 else "grandparent", "lab")
    for a, b in combinations(sorted(model_ids), 2):
        if (parents[a] & parents[b] and b not in ancestors[a] and a not in ancestors[b]):
            add(model_key(a), model_key(b), "sibling", "lab")
    reference_path = lab_edges.with_name("reference_pairs.json")
    if reference_path.exists():
        reference_pairs = _json(reference_path)["pairs"]
    else:
        pins_path = lab_edges.with_name("pins.json")
        pins = _json(pins_path) if pins_path.exists() else {}
        certified = sorted(m for m in model_ids if pins.get(m, {}).get("reference_type") == "polypythias")
        reference_pairs = [dict(model_ids=[a, b], label="independent", source="certified_pretrain")
                           for a, b in combinations(certified, 2)]
    for pair in reference_pairs:
        if pair["label"] not in ("independent", "sibling"):
            raise ValueError("Invalid reference-pair label")
        ids = pair.get("model_ids", [])
        if len(ids) != 2 or any(i not in model_ids for i in ids):
            raise ValueError("Reference pair has unknown lab IDs")
        both_mapped = all(i in mapped for i in ids)
        if both_mapped and "a" in pair and {model_key(i) for i in ids} != {pair["a"], pair["b"]}:
            raise ValueError("Reference-pair pin does not match mapped manifests")
        if pair["label"] == "independent":
            if both_mapped:
                a, b = (mapped[i] for i in ids)
                signature = _shape_signature(a)
                if not signature or signature != _shape_signature(b):
                    continue
            if ids[1] in ancestors[ids[0]] or ids[0] in ancestors[ids[1]] or parents[ids[0]] & parents[ids[1]]:
                raise ValueError("Independent reference contradicts lab ancestry")
        add(model_key(ids[0]), model_key(ids[1]), pair["label"], pair.get("source", "certified_reference"))
    known = {frozenset((r["a"], r["b"])) for r in records}
    for manifest in manifests:
        for base in manifest.get("declared_bases", []):
            if not isinstance(base, str):
                continue
            matches = [m for m in manifests if _key(m) == base or m["repo_id"] == base]
            if len(matches) == 1 and frozenset((_key(manifest), _key(matches[0]))) not in known:
                add(_key(manifest), _key(matches[0]), "parent", "hub_declared_base")
    result = pd.DataFrame(records, columns=["a", "b", "label", "source"]).drop_duplicates()
    if missing_models:
        result["source"] = result["source"] + "|missing_manifest_ids=" + json.dumps(missing_models, separators=(",", ":"))
    result = result.sort_values(["source", "label", "a", "b"]).reset_index(drop=True)
    result.attrs.update(n_inventory_models=len(models), n_mapped_models=len(mapped),
                        missing_model_ids=sorted(model_ids - mapped.keys()))
    return result


def _missing_mapping_ids(pairs):
    missing = set()
    for source in pairs.get("source", []):
        if "|missing_manifest_ids=" in str(source):
            encoded = str(source).split("|missing_manifest_ids=", 1)[1]
            value = json.loads(encoded)
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError("Invalid persisted manifest mapping coverage")
            missing.update(value)
    return sorted(missing)


def _fingerprints(directory, tier):
    result = {}
    for path in sorted(Path(directory).glob("*.fingerprint.safetensors")):
        try:
            fp = load_fingerprint(path)
        except FileNotFoundError:
            # A full-weight fingerprint without its required spectra companion is an
            # incomplete measurement. Requested-pair/coverage counts expose it.
            continue
        if fp.meta["tier"] != tier:
            continue
        key = _key(fp.meta)
        if key in result:
            raise ValueError(f"Duplicate fingerprint identity at tier {tier}: {key}")
        result[key] = fp
    return result


def _merge_fingerprints(primary, secondary, tier):
    """Primary measured files take precedence only over identical companions."""
    result = _fingerprints(secondary, tier) if Path(secondary) != Path(primary) else {}
    for key, fp in _fingerprints(primary, tier).items():
        if key in result:
            other = result[key]
            settings = ("tier", "row_block", "sketch_seed", "sketch_k", "evidence_version")
            if (any(fp.meta.get(name) != other.meta.get(name) for name in settings)
                    or fp.names != other.names or fp.geometry != other.geometry or fp.sha256 != other.sha256
                    or not np.array_equal(fp.rows, other.rows) or not np.array_equal(fp.shape, other.shape)
                    or not np.array_equal(fp.sketch, other.sketch)):
                raise ValueError(f"Incompatible duplicate {tier} fingerprint: {key}")
        result[key] = fp
    return result


def _baseline_row_blocks(directory, full_weights_directory=None):
    return _merge_fingerprints(directory, full_weights_directory or measurement_directory(Path(directory).parent, "full_weights"), "row_blocks")


def _truth(pairs):
    required = {"a", "b", "label", "source"}
    if not required <= set(pairs):
        raise ValueError("Pairs need a, b, label, source columns")
    result = pairs[~pairs.source.str.startswith("hub_declared", na=False)].copy()
    result = result[result.a != result.b]
    result["_pair"] = [((str(a), str(b)) if label in ("parent", "grandparent") else tuple(sorted((str(a), str(b)))))
                       for a, b, label in zip(result.a, result.b, result.label)]
    if any(group.label.nunique() > 1 for _, group in result.groupby("_pair")):
        raise ValueError("Conflicting pair labels")
    return result.drop_duplicates("_pair")


def _score(a, b, k=64):
    if a.meta["sketch_k"] < k or b.meta["sketch_k"] < k:
        return None
    right = {name: j for j, name in enumerate(b.names)}
    for i, name in enumerate(a.names):
        if name in right:
            j = right[name]
            if (a.geometry[i] != b.geometry[j] or not np.array_equal(a.rows[i], b.rows[j])
                    or a.shape[i, 1] != b.shape[j, 1]):
                return None
    value = pair_similarity(a, b, k)["median_cosine"]
    return float(value) if value is not None and math.isfinite(value) else None


def _pair_scores(pairs, fps, k=64):
    records = []
    for row in _truth(pairs).itertuples():
        if row.a in fps and row.b in fps:
            value = _score(fps[row.a], fps[row.b], k)
            if value is not None:
                records.append(dict(a=row.a, b=row.b, label=row.label, score=value))
    return pd.DataFrame(records, columns=["a", "b", "label", "score"])


def _auroc(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        return None
    ranks = rankdata(np.concatenate((positive, negative)), method="average")
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) /
                 (len(positive) * len(negative)))


def _spearman(a, b):
    if len(a) < 2 or len(b) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    result = float(spearmanr(a, b).statistic)
    return result if math.isfinite(result) else None


def _threshold(positive):
    # Largest observed threshold achieving empirical TPR >= .9 (ties included).
    positive = np.sort(np.asarray(positive, dtype=float))
    return float(positive[len(positive) - math.ceil(.9 * len(positive))]) if len(positive) else None


def _false_matches(pairs, fps, threshold):
    truth = _truth(pairs)
    queries = sorted(set(truth.loc[truth.label == "parent", "a"]) & fps.keys())
    parent_pairs = {(r.a, r.b) for r in truth.itertuples() if r.label == "parent"}
    other_labels = {tuple(sorted((r.a, r.b))): r.label for r in truth.itertuples() if r.label != "parent"}
    false_matches, unknown, pool_sizes = [], 0, []
    if threshold is None:
        return dict(false_matches_per_query=None, candidate_pool_size=0, n_unlabelled_candidates=0, n_queries=0)
    for query in queries:
        count, compatible = 0, 0
        for key, fp in fps.items():
            if key == query:
                continue
            score = _score(fps[query], fp)
            if score is None:
                continue
            compatible += 1
            if (query, key) in parent_pairs:
                continue
            known_negative = ((key, query) in parent_pairs or tuple(sorted((query, key))) in other_labels)
            if not known_negative:
                unknown += 1
            elif score >= threshold:
                count += 1
        false_matches.append(count)
        pool_sizes.append(compatible)
    return dict(false_matches_per_query=float(np.mean(false_matches)) if false_matches else None,
                candidate_pool_size=min(pool_sizes) if pool_sizes else 0,
                n_unlabelled_candidates=unknown, n_queries=len(queries),
                false_matches_note="Directional immediate-parent retrieval: sibling/grandparent/independent and reverse descendants are negatives; unlabelled candidates are unverified, so cannot certify G0.")


def tier_agreement(pairs, row_blocks_dir, full_weights_dir) -> dict:
    """Compare the same measured truth pairs at nested k=64, without sklearn."""
    truth = _truth(pairs)
    fps1, fps2 = _baseline_row_blocks(row_blocks_dir, full_weights_dir), _fingerprints(full_weights_dir, "full_weights")
    s1, s2 = _pair_scores(truth, fps1), _pair_scores(truth, fps2)
    joined = s1.merge(s2, on=["a", "b", "label"], suffixes=("_row_blocks", "_full_weights"))
    output = dict(n_pairs_requested=len(truth), n_pairs_measured=len(joined),
                  n_missing_pairs=len(truth) - len(joined), sketch_k=64,
                  n_missing_manifest_models=len(_missing_mapping_ids(pairs)),
                  missing_manifest_model_ids=_missing_mapping_ids(pairs))
    for tier in ("row_blocks", "full_weights"):
        scores = joined[f"score_{tier}"]
        positives = scores[joined.label == "parent"]
        output[f"auroc_{tier}"] = _auroc(positives, scores[joined.label == "independent"])
        for negative in ("sibling", "grandparent"):
            output[f"auroc_parent_vs_{negative}_{tier}"] = _auroc(positives, scores[joined.label == negative])
    related = joined[joined.label != "independent"]
    output["spearman_within_related"] = _spearman(related.score_row_blocks, related.score_full_weights)
    threshold = _threshold(joined.loc[joined.label == "parent", "score_full_weights"])
    output["threshold_full_weights_tpr_0_9"] = threshold
    output["decision_agreement"] = (float(np.mean((joined.score_row_blocks >= threshold) == (joined.score_full_weights >= threshold)))
                                    if threshold is not None and len(joined) else None)
    independent = joined.loc[joined.label == "independent", "score_row_blocks"].to_numpy()
    output.update(null_mean=float(np.mean(independent)) if len(independent) else None,
                  null_sd=float(np.std(independent, ddof=1)) if len(independent) > 1 else None,
                  null_quantiles=np.quantile(independent, [0, .5, .9, .95, .99, 1]).tolist() if len(independent) else [],
                  n_independent_pairs=len(independent))
    threshold_row_blocks = _threshold(joined.loc[joined.label == "parent", "score_row_blocks"])
    output["threshold_row_blocks_tpr_0_9"] = threshold_row_blocks
    output["achieved_tpr_row_blocks"] = (float(np.mean(joined.loc[joined.label == "parent", "score_row_blocks"] >= threshold_row_blocks))
                                  if threshold_row_blocks is not None else None)
    output.update(_false_matches(truth, fps1, threshold_row_blocks))
    curve, curve_counts = {}, {}
    for height in (128, 256, 512):
        if height == 256:
            scores = s1
        else:
            alternate = _merge_fingerprints(row_blocks_dir, full_weights_dir, f"row_blocks_{height}")
            scores = _pair_scores(truth, alternate)
        curve_counts[height] = dict(requested=len(truth), measured=len(scores), missing=len(truth) - len(scores))
        curve[height] = (_auroc(scores.loc[scores.label == "parent", "score"],
                                scores.loc[scores.label == "independent", "score"])
                         if len(scores) == len(truth) and not output["n_missing_manifest_models"] else None)
    output["byte_curve"] = curve
    output["byte_curve_counts"] = curve_counts
    return output


def _alpha(spectrum, config):
    """The collector's cutoff search on stored eigenvalues; no second SVD."""
    values = torch.as_tensor(np.asarray(spectrum), dtype=torch.float64)
    if values.numel() < 2 or not torch.isfinite(values).all() or (values < 0).any():
        return None
    values = torch.sort(values).values
    if config["filter_zeros"]:
        values = values[values > config["evals_thresh"]]
    values = values[values > 0]
    n = len(values)
    if n < 2 or values[0] == values[-1]:
        return None
    mode = config["fix_fingers"]
    if mode == "xmin_mid":
        index = min(n // config["xmin_pos"], n - 2)
        index = torch.searchsorted(values, values[index]).item()
        starts = torch.tensor([index])
    else:
        valid = values[:-1] < values[-1]
        valid[1:] &= values[1:-1] > values[:-2]
        if mode == "xmin_peak":
            # The collector histogram uses its compute dtype, then fits in f64.
            logs = torch.log10(values.to(getattr(torch, config["compute_dtype"])))
            counts = torch.histc(logs, config["bins"], min=logs.min().item(), max=logs.max().item())
            boundaries = torch.linspace(logs.min(), logs.max(), config["bins"] + 1, dtype=logs.dtype)
            cutoff = 10 ** boundaries[torch.argmax(counts)]
            valid &= (values[:-1] >= .95 * cutoff) & (values[:-1] <= 1.5 * cutoff)
        starts = torch.where(valid)[0]
    best_alpha, best_distance = None, math.inf
    for chunk in starts.split(_KS_CHUNK_SIZE):
        if not chunk.numel():
            continue
        alphas, distances = _fit_tail_chunk(values, chunk)
        valid = torch.isfinite(alphas) & (alphas > 1) & torch.isfinite(distances)
        if valid.any():
            distances = torch.where(valid, distances, math.inf)
            winner = torch.argmin(distances)
            if distances[winner] < best_distance:
                best_alpha, best_distance = alphas[winner].item(), distances[winner].item()
    return best_alpha


def spectra_agreement(row_blocks_dir, full_weights_dir) -> dict:
    first, second = _baseline_row_blocks(row_blocks_dir, full_weights_dir), _fingerprints(full_weights_dir, "full_weights")
    summary_path = Path(full_weights_dir) / "summary.json"
    declared = summary_path.exists()
    requested = {_key(row) for row in _json(summary_path).get("models", [])} if declared else set(second)
    cohort_source = "full_weights/summary.json" if declared else "available full-weight outputs only; requested cohort undeclared"
    alphas, correlations, skipped = [], [], Counter()
    n_requested = sum(len(first[key].names) for key in requested & first.keys())
    matched = requested & first.keys() & second.keys()
    for key in sorted(matched):
        a, b = first[key], second[key]
        # full-weight stores the fit configuration with the full spectra artifact.
        spectra_path = companion_spectra_path(b.path)
        with h5py.File(spectra_path, "r") as handle:
            config = json.loads(handle.attrs["measurement_config_json"])
        right = {name: j for j, name in enumerate(b.names)}
        for i, name in enumerate(a.names):
            if name not in right:
                skipped["missing_view"] += 1
                continue
            j = right[name]
            if a.geometry[i] != b.geometry[j] or not np.array_equal(a.shape[i], b.shape[j]):
                skipped["incompatible_view"] += 1
                continue
            x, y = (np.sort(np.asarray(s, dtype=float)) for s in (a.spectra[i], b.spectra[j]))
            if (len(x) < 2 or len(y) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or (x < 0).any() or (y < 0).any()):
                skipped["invalid_spectrum"] += 1
                continue
            alpha1, alpha2 = _alpha(x, config), _alpha(y, config)
            if alpha1 is not None and alpha2 is not None:
                alphas.append(abs(alpha1 - alpha2))
            else:
                skipped["unfittable_alpha"] += 1
            probabilities = np.linspace(0, 1, min(len(x), len(y)))
            correlation = _spearman(np.quantile(x, probabilities), np.quantile(y, probabilities))
            if correlation is not None:
                correlations.append(correlation)
            else:
                skipped["constant_spectrum"] += 1
    return dict(median_abs_alpha_diff=float(np.median(alphas)) if alphas else None,
                median_per_matrix_spearman=float(np.median(correlations)) if correlations else None,
                n_matrices=len(correlations), n_alpha_pairs=len(alphas), skipped=dict(skipped),
                n_models_requested=len(requested), n_models_matched=len(matched),
                n_row_blocks_only_models=len(first.keys() - requested),
                cohort_declared=declared, cohort_source=cohort_source,
                n_matrices_requested=n_requested,
                measurement_coverage_complete=(declared and len(requested) > 0 and len(matched) == len(requested)
                                               and n_requested > 0 and len(alphas) == n_requested
                                               and len(correlations) == n_requested),
                spearman_note="Quantile-aligned sorted-spectrum ranks are largely automatic and do not establish spectral-shape agreement.")


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def throughput_report(run_dir) -> pd.DataFrame:
    root = Path(run_dir)
    summary = measurement_directory(root, "row_blocks") / "summary.json"
    records = _json(summary).get("models", []) if summary.exists() else []
    manifests = {_key(m): m for m in (_json(p) for p in (root / "manifests").glob("*.manifest.json"))}
    if not records:
        for fp in _fingerprints(measurement_directory(root, "row_blocks"), "row_blocks").values():
            records.append(dict(fp.meta, status="complete", stratum="unknown", reason=""))
        failures = measurement_directory(root, "row_blocks") / "failures.csv"
        if failures.exists():
            records.extend(pd.read_csv(failures, keep_default_na=False).assign(status="failed", stratum="unknown").to_dict("records"))
    indexed = {}
    for record in records:
        identity = (record["repo_id"], record.get("revision", record.get("requested_revision", "")))
        if identity in indexed:
            raise ValueError("Duplicate identities in current row-block summary")
        indexed[identity] = record
    groups = defaultdict(list)
    for record in indexed.values():
        groups[record.get("stratum") or "unknown"].append(record)
    output = []
    for stratum, rows in sorted(groups.items()):
        complete = [r for r in rows if r.get("status") in ("complete", "skipped")
                    and canonical_mode(r.get("reason", "")) in ("", "already_complete")]
        excluded = [r for r in rows if r.get("reason") in
                    ("no_access", "packed_format", "unsupported_packed", "unsupported_quantization", "metadata_only")]
        eligible = [r for r in rows if r not in excluded]
        failed = [r for r in eligible if r.get("status") == "failed"]
        result = dict(stratum=stratum, n_models=len(rows), n_completed=len(complete), n_eligible=len(eligible),
                      n_excluded=len(excluded), failure_rate=len(failed) / len(eligible) if eligible else None,
                      failure_rate_by_reason={k: v / len(eligible) for k, v in sorted(Counter(r.get("reason", "error") for r in failed).items())},
                      unhandled_429=sum("429" in str(r.get("message", "")) or r.get("reason") == "rate_limited" for r in failed))
        for name in ("bytes_read", "elapsed_seconds", "resolves", "range_reads"):
            values = [v for r in complete if (v := _number(r.get(name))) is not None]
            result[f"{name}_median"] = float(np.median(values)) if values else None
            result[f"{name}_p90"] = float(np.quantile(values, .9)) if values else None
            result[f"n_missing_{'elapsed' if name == 'elapsed_seconds' else name}"] = len(complete) - len(values)
            if name == "elapsed_seconds":
                result["elapsed_seconds_mean"] = float(np.mean(values)) if values else None
        fractions, elapsed_small, n_small, n_unknown_params = [], [], 0, 0
        for row in complete:
            manifest = manifests.get(f"{row['repo_id']}@{row.get('revision', '')}")
            tensor_bytes = sum(t["end"] - t["start"] for t in manifest.get("tensors", [])) if manifest else 0
            bytes_read = _number(row.get("bytes_read"))
            if tensor_bytes > 0 and bytes_read is not None:
                fractions.append(bytes_read / tensor_bytes)
            params = _number(row.get("params"))
            if params is None and manifest:
                shapes = [t.get("shape") for t in manifest.get("tensors", [])]
                if shapes and all(isinstance(s, list) and all(type(d) is int and d >= 0 for d in s) for s in shapes):
                    params = sum(math.prod(s) for s in shapes) or None
            elapsed = _number(row.get("elapsed_seconds"))
            if params is None:
                n_unknown_params += 1
            elif params <= 9e9:
                n_small += 1
                if elapsed is not None:
                    elapsed_small.append(elapsed)
        result.update(fraction_of_model_bytes=float(np.median(fractions)) if fractions else None,
                      fraction_of_model_bytes_p90=float(np.quantile(fractions, .9)) if fractions else None,
                      n_fraction_measured=len(fractions),
                      elapsed_seconds_p90_le_9b=float(np.quantile(elapsed_small, .9)) if elapsed_small else None,
                      n_timed_le_9b=len(elapsed_small),
                      n_models_le_9b=n_small, n_unknown_params=n_unknown_params,
                      measurement_coverage_complete=(bool(complete) and len(fractions) == len(complete)
                          and not n_unknown_params and all(result[f"n_missing_{name}"] == 0
                              for name in ("elapsed", "bytes_read", "resolves", "range_reads"))),
                      projected_5000_seconds=(result["elapsed_seconds_mean"] * 5000
                                              if result["elapsed_seconds_mean"] is not None else None))
        output.append(result)
    return pd.DataFrame(output)


def write_calibration(pairs, row_blocks_dir, out: Path, sketch_k: int = 64) -> None:
    """Write every independent observation as an equal-mass empirical quantile."""
    if type(sketch_k) is not int or sketch_k < 1:
        raise ValueError("Invalid calibration sketch_k")
    if _missing_mapping_ids(pairs):
        raise ValueError("Calibration requires complete lab manifest mapping")
    truth = _truth(pairs)
    independent = truth[truth.label == "independent"]
    fps = _baseline_row_blocks(row_blocks_dir)
    scores = _pair_scores(independent, fps, sketch_k)
    if len(scores) < 2 or len(scores) != len(independent):
        raise ValueError("Calibration needs at least two fully measured independent pairs")
    identities = set(scores.a) | set(scores.b)
    settings = {(fps[key].meta["tier"], fps[key].meta["row_block"], fps[key].meta["sketch_seed"]) for key in identities}
    if len(settings) != 1:
        raise ValueError("Calibration requires homogeneous fingerprint settings")
    positives = _pair_scores(truth[truth.label == "parent"], fps, sketch_k)
    if len(positives) != sum(truth.label == "parent"):
        raise ValueError("Calibration requires all requested parent pairs to be measured")
    parent_ids = set(positives.a) | set(positives.b)
    if any((fps[key].meta["tier"], fps[key].meta["row_block"], fps[key].meta["sketch_seed"]) not in settings
           for key in parent_ids):
        raise ValueError("Parent threshold and independent null require matching settings")
    threshold = _threshold(positives.score)
    if threshold is None:
        raise ValueError("Calibration related_threshold requires measured parent pairs")
    tier, height, seed = settings.pop()
    payload = dict(statistic="median_cosine", sketch_k=sketch_k, row_block=height,
                   tier=tier, sketch_seed=seed, n_pairs=len(scores),
                   null_quantiles=sorted(scores.score.tolist()), related_threshold=threshold,
                   note="Empirical pair-level null; no per-query error-control claim.")
    _atomic(out, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_g0_report(run_dir: Path, out: Path) -> None:
    root = Path(run_dir)
    pairs_path = root / "pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path, keep_default_na=False)
    elif (root / "lab" / "edges.csv").exists():
        pairs = lineage_pairs(root / "lab" / "edges.csv", root / "manifests")
    else:
        pairs = pd.DataFrame(columns=["a", "b", "label", "source"])
    agreement = tier_agreement(pairs, measurement_directory(root, "row_blocks"), measurement_directory(root, "full_weights"))
    spectral = spectra_agreement(measurement_directory(root, "row_blocks"), measurement_directory(root, "full_weights"))
    throughput = throughput_report(root)
    lines = ["# G0 pilot report", "", "Missing evidence is FAIL (unavailable), not an observed scientific failure.",
             "Synthetic fixtures/planned inventory cannot authorize G0; the person decides the gate.", ""]

    def check(name, value, condition, threshold):
        available = value is not None and not (isinstance(value, float) and not math.isfinite(value))
        passed = available and condition(value)
        lines.append(f"- {'PASS' if passed else 'FAIL'} {name}: {value if available else 'unavailable'}; {threshold}")

    complete_pairs = (agreement["n_pairs_requested"] > 0 and agreement["n_missing_pairs"] == 0
                      and agreement["n_missing_manifest_models"] == 0)
    check("paired_measurement_coverage", agreement["n_pairs_measured"], lambda _: complete_pairs,
          f"all {agreement['n_pairs_requested']} truth pairs measured at both tiers")
    lines.append(f"  Missing lab manifest mappings: {agreement['n_missing_manifest_models']} "
                 f"{agreement['missing_manifest_model_ids']}")
    for name, minimum in (("auroc_row_blocks", .98), ("spearman_within_related", .9), ("decision_agreement", .9)):
        check(name, agreement[name], lambda x, m=minimum: x >= m and complete_pairs, f">= {minimum}")
    gap = agreement["auroc_row_blocks"] - agreement["auroc_full_weights"] if agreement["auroc_row_blocks"] is not None and agreement["auroc_full_weights"] is not None else None
    check("auroc_row_blocks_minus_full_weights", gap, lambda x: x >= -.01 and complete_pairs, ">= -0.01")
    check("false_matches_per_query", agreement["false_matches_per_query"],
          lambda x: x <= .05 and agreement["candidate_pool_size"] >= 5000 and agreement["n_unlabelled_candidates"] == 0 and complete_pairs,
          f"<= 0.05 at TPR0.9; requires 5000 verified candidates/query; actual min pool={agreement['candidate_pool_size']}, unlabelled={agreement['n_unlabelled_candidates']}")
    lines.append(f"  row-block threshold={agreement['threshold_row_blocks_tpr_0_9']}; achieved row-block parent TPR={agreement['achieved_tpr_row_blocks']}; "
                 f"decision-agreement common full-weight threshold={agreement['threshold_full_weights_tpr_0_9']}.")
    for name in ("median_abs_alpha_diff", "median_per_matrix_spearman"):
        check(name, spectral[name], (lambda x: x <= .1 and spectral["measurement_coverage_complete"])
              if name == "median_abs_alpha_diff" else (lambda x: x >= .8 and spectral["measurement_coverage_complete"]),
              "<= 0.1" if name == "median_abs_alpha_diff" else ">= 0.8")
    check("spectral_measurement_coverage", spectral["n_matrices"],
          lambda _: spectral["measurement_coverage_complete"],
          f"all {spectral['n_matrices_requested']} requested matrices have valid paired spectra/fits")
    lines.append(f"  Spectral cohort: {spectral['cohort_source']}; {spectral['n_models_matched']}/{spectral['n_models_requested']} "
                 f"requested models matched, {spectral['n_row_blocks_only_models']} models outside full-weight cohort.")
    if throughput.empty:
        for name in ("fraction_of_model_bytes", "elapsed_seconds_p90_le_9b", "failure_rate", "unhandled_429", "projected_5000_seconds"):
            check(name, None, bool, "requires measured row-block summaries")
    for row in throughput.to_dict("records"):
        stratum = row["stratum"]
        coverage = row["measurement_coverage_complete"] and stratum != "unknown"
        for name, bound in (("fraction_of_model_bytes", .1), ("elapsed_seconds_p90_le_9b", 600),
                            ("failure_rate", .05), ("unhandled_429", 0), ("projected_5000_seconds", 4 * 86400)):
            if name == "elapsed_seconds_p90_le_9b" and row["n_completed"] > 0 and not row["n_unknown_params"] and not row["n_models_le_9b"]:
                lines.append(f"- N/A {stratum}/{name}: known-empty <=9B population; all measured models are larger.")
                continue
            value = row[name]
            check(f"{stratum}/{name}", value, lambda x, b=bound: x <= b and coverage,
                  f"<= {bound}; requires known stratum and complete cost evidence")
        lines.append(f"  Cost coverage: {row['n_completed']} completed, {row['n_fraction_measured']} byte fractions, "
                     f"{row['n_timed_le_9b']}/{row['n_models_le_9b']} timed <=9B, {row['n_unknown_params']} unknown sizes.")
    lines += ["", "## Additional diagnostics (not a gate or an error-control guarantee)"]
    mean, sd = agreement["null_mean"], agreement["null_sd"]
    check("null_abs_mean", abs(mean) if mean is not None else None,
          lambda x: sd is not None and sd > 0 and x <= sd, "<= one null sd; pivot diagnostic only")
    check("null_sd", sd, lambda x: .7 * 1.12 / 64 <= x <= 1.5 * 1.12 / 64,
          "0.7--1.5 times 1.12/k; pivot diagnostic only")
    lines += ["", "Parent-vs-sibling/grandparent AUROCs and byte curve:",
              json.dumps({k: v for k, v in agreement.items() if "vs_" in k or k.startswith("byte_curve")}, sort_keys=True),
              "", spectral["spearman_note"], agreement.get("false_matches_note", ""),
              f"Spectral matrices: {spectral['n_matrices']}; fitted alpha pairs: {spectral['n_alpha_pairs']}; skipped: {spectral['skipped']}."]
    _atomic(out, "\n".join(lines) + "\n")
