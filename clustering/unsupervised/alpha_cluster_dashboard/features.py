from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import AlphaRecord, available_modules as list_available_modules, canonicalize_module_name

SUMMARY_STAT_FUNCS = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
}


@dataclass
class PreparedDataset:
    metadata: pd.DataFrame
    module_names: list[str]
    profiles: np.ndarray
    profile_grid: np.ndarray
    representation: str
    feature_matrix: np.ndarray | None = None
    feature_names: list[str] | None = None
    distance_matrix: np.ndarray | None = None


def filter_records(
    records: Iterable[AlphaRecord],
    schema_mode: str,
    min_layers: int,
    max_layers: int,
    model_query: str = "",
    max_models: int | None = None,
    random_state: int = 0,
) -> list[AlphaRecord]:
    query = model_query.strip().lower()
    filtered: list[AlphaRecord] = []
    for record in records:
        if schema_mode == "common" and not record.common_schema:
            continue
        if not (min_layers <= record.num_layers <= max_layers):
            continue
        if query and query not in record.model_id.lower() and query not in record.display_name.lower():
            continue
        filtered.append(record)
    if max_models and len(filtered) > max_models:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(len(filtered), size=max_models, replace=False)
        filtered = [filtered[index] for index in sorted(selected)]
    return filtered


def _sanitize_trace(
    trace: np.ndarray | None,
    sanitize_non_positive: bool,
    log_transform: bool,
    zscore_within_trace: bool,
) -> np.ndarray:
    if trace is None:
        return np.asarray([], dtype=float)
    cleaned = np.asarray(trace, dtype=float).copy()
    cleaned[~np.isfinite(cleaned)] = np.nan
    if sanitize_non_positive:
        cleaned[cleaned <= 0] = np.nan
    if log_transform:
        valid = np.isfinite(cleaned)
        cleaned[valid] = np.log1p(cleaned[valid])
    if zscore_within_trace:
        valid = np.isfinite(cleaned)
        if np.count_nonzero(valid) >= 2:
            mean = float(np.nanmean(cleaned))
            std = float(np.nanstd(cleaned))
            if std > 1e-8:
                cleaned[valid] = (cleaned[valid] - mean) / std
            else:
                cleaned[valid] = 0.0
    return cleaned


def _record_module_traces(record: AlphaRecord, schema_mode: str) -> dict[str, np.ndarray]:
    if schema_mode == "common":
        return {module: record.alpha[:, index] for index, module in enumerate(record.raw_modules)}

    grouped: dict[str, list[np.ndarray]] = {}
    for index, raw_module in enumerate(record.raw_modules):
        module = canonicalize_module_name(raw_module)
        grouped.setdefault(module, []).append(record.alpha[:, index])

    combined: dict[str, np.ndarray] = {}
    for module, traces in grouped.items():
        if len(traces) == 1:
            combined[module] = traces[0]
            continue
        stacked = np.stack(traces, axis=0)
        with np.errstate(all="ignore"):
            combined[module] = np.nanmean(stacked, axis=0)
    return combined


def resample_trace(trace: np.ndarray, num_bins: int) -> np.ndarray:
    valid = np.isfinite(trace)
    grid = np.linspace(0.0, 1.0, num_bins)
    if np.count_nonzero(valid) == 0:
        return np.full(num_bins, np.nan, dtype=float)
    x = np.linspace(0.0, 1.0, trace.shape[0])[valid]
    y = trace[valid]
    if y.shape[0] == 1:
        return np.full(num_bins, float(y[0]), dtype=float)
    return np.interp(grid, x, y)


def _trace_summary(trace: np.ndarray) -> dict[str, float]:
    summary: dict[str, float] = {}
    valid = np.isfinite(trace)
    if np.count_nonzero(valid) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "start": np.nan,
            "end": np.nan,
            "delta": np.nan,
            "slope": np.nan,
        }

    summary["mean"] = float(np.nanmean(trace))
    summary["std"] = float(np.nanstd(trace))
    summary["min"] = float(np.nanmin(trace))
    summary["max"] = float(np.nanmax(trace))
    summary["median"] = float(np.nanmedian(trace))
    summary["q25"] = float(np.nanpercentile(trace, 25))
    summary["q75"] = float(np.nanpercentile(trace, 75))

    valid_values = trace[valid]
    summary["start"] = float(valid_values[0])
    summary["end"] = float(valid_values[-1])
    summary["delta"] = float(valid_values[-1] - valid_values[0])

    if valid_values.shape[0] >= 2:
        x = np.linspace(0.0, 1.0, trace.shape[0])[valid]
        slope, _ = np.polyfit(x, valid_values, 1)
        summary["slope"] = float(slope)
    else:
        summary["slope"] = 0.0
    return summary


def build_prepared_dataset(
    records: list[AlphaRecord],
    schema_mode: str,
    selected_modules: list[str] | None,
    representation: str,
    num_bins: int,
    summary_stats: list[str],
    include_presence_features: bool,
    sanitize_non_positive: bool,
    log_transform: bool,
    zscore_within_trace: bool,
    dtw_missing_penalty: float = 2.5,
) -> PreparedDataset:
    module_names = list_available_modules(records, schema_mode) if selected_modules is None else selected_modules
    if not module_names:
        raise ValueError("No modules are available for the current schema mode and filters.")

    metadata = pd.DataFrame(
        {
            "model_id": [record.model_id for record in records],
            "display_name": [record.display_name for record in records],
            "num_layers": [record.num_layers for record in records],
            "raw_module_count": [record.raw_module_count for record in records],
            "common_schema": [record.common_schema for record in records],
            "path": [record.path for record in records],
        }
    )
    metadata["schema_mode"] = schema_mode

    profiles = np.full((len(records), len(module_names), num_bins), np.nan, dtype=float)
    presence = np.zeros((len(records), len(module_names)), dtype=float)

    for row_index, record in enumerate(records):
        traces = _record_module_traces(record, schema_mode)
        for column_index, module in enumerate(module_names):
            trace = traces.get(module)
            cleaned = _sanitize_trace(trace, sanitize_non_positive, log_transform, zscore_within_trace)
            if cleaned.size == 0 or np.count_nonzero(np.isfinite(cleaned)) == 0:
                continue
            presence[row_index, column_index] = 1.0
            profiles[row_index, column_index, :] = resample_trace(cleaned, num_bins)

    profile_grid = np.linspace(0.0, 1.0, num_bins)

    if representation == "dtw_profile":
        distance_matrix = pairwise_dtw_distance_matrix(profiles, missing_penalty=dtw_missing_penalty)
        return PreparedDataset(
            metadata=metadata,
            module_names=module_names,
            profiles=profiles,
            profile_grid=profile_grid,
            representation=representation,
            distance_matrix=distance_matrix,
        )

    feature_rows: list[np.ndarray] = []
    feature_names: list[str] = []

    if representation == "summary":
        if not summary_stats:
            raise ValueError("At least one summary statistic must be selected.")
        for module in module_names:
            for stat in summary_stats:
                feature_names.append(f"{module}__{stat}")
        if include_presence_features:
            feature_names.extend(f"{module}__present" for module in module_names)

        for row_index in range(profiles.shape[0]):
            features: list[float] = []
            for column_index, module in enumerate(module_names):
                trace = profiles[row_index, column_index, :]
                summary = _trace_summary(trace)
                features.extend(summary[stat] for stat in summary_stats)
            if include_presence_features:
                features.extend(presence[row_index, :].tolist())
            feature_rows.append(np.asarray(features, dtype=float))
    elif representation == "profile":
        for module in module_names:
            for bin_index in range(num_bins):
                feature_names.append(f"{module}__depth_{bin_index + 1:02d}")
        if include_presence_features:
            feature_names.extend(f"{module}__present" for module in module_names)

        flattened = profiles.reshape(profiles.shape[0], -1)
        if include_presence_features:
            flattened = np.hstack([flattened, presence])
        feature_rows = [row.astype(float, copy=False) for row in flattened]
    else:
        raise ValueError(f"Unsupported representation: {representation}")

    feature_matrix = np.vstack(feature_rows) if feature_rows else np.empty((0, len(feature_names)))
    return PreparedDataset(
        metadata=metadata,
        module_names=module_names,
        profiles=profiles,
        profile_grid=profile_grid,
        representation=representation,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
    )


def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray) -> float:
    valid_a = np.isfinite(sequence_a)
    valid_b = np.isfinite(sequence_b)
    if np.count_nonzero(valid_a) == 0 or np.count_nonzero(valid_b) == 0:
        return np.nan
    a = sequence_a[valid_a]
    b = sequence_b[valid_b]

    previous = np.full(b.shape[0] + 1, np.inf, dtype=float)
    current = np.full(b.shape[0] + 1, np.inf, dtype=float)
    previous[0] = 0.0

    for i in range(1, a.shape[0] + 1):
        current.fill(np.inf)
        for j in range(1, b.shape[0] + 1):
            cost = abs(a[i - 1] - b[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous
    return float(previous[-1] / (a.shape[0] + b.shape[0]))


def pairwise_dtw_distance_matrix(profiles: np.ndarray, missing_penalty: float = 2.5) -> np.ndarray:
    num_models = profiles.shape[0]
    num_modules = profiles.shape[1]
    distances = np.zeros((num_models, num_models), dtype=float)

    # Strip NaNs once up front so pairwise DTW does not repeatedly rescan every trace.
    compact_traces = np.empty((num_models, num_modules), dtype=object)
    valid_mask = np.zeros((num_models, num_modules), dtype=bool)
    for model_index in range(num_models):
        for module_index in range(num_modules):
            compact = profiles[model_index, module_index, :]
            compact = compact[np.isfinite(compact)]
            compact_traces[model_index, module_index] = compact
            valid_mask[model_index, module_index] = compact.size > 0

    for left_index in range(num_models):
        for right_index in range(left_index + 1, num_models):
            distance_total = 0.0
            distance_count = 0
            for module_index in range(num_modules):
                left_valid = bool(valid_mask[left_index, module_index])
                right_valid = bool(valid_mask[right_index, module_index])
                if not left_valid and not right_valid:
                    continue
                if left_valid != right_valid:
                    distance_total += missing_penalty
                    distance_count += 1
                    continue
                left_trace = compact_traces[left_index, module_index]
                right_trace = compact_traces[right_index, module_index]
                distance = dtw_distance(left_trace, right_trace)
                if np.isfinite(distance):
                    distance_total += float(distance)
                    distance_count += 1
            pair_distance = float(distance_total / distance_count) if distance_count else missing_penalty
            distances[left_index, right_index] = pair_distance
            distances[right_index, left_index] = pair_distance
    return distances
