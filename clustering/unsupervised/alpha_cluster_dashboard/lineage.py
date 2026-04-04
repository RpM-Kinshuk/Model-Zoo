from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, v_measure_score

REQUIRED_LINEAGE_COLUMNS = {"model_id", "source_model", "base_model_relation"}


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _known_model_name(value: object) -> str | None:
    text = _clean_text(value)
    if text is None or text.lower() == "unknown":
        return None
    return text


def _clean_relation(value: object) -> str:
    text = _clean_text(value)
    return text or "unknown"


def _infer_base_model(model_id: str | None, source_model: str | None, relation: str) -> str | None:
    if source_model is not None:
        return source_model
    if relation == "source":
        return model_id
    return None


def load_lineage_metadata(path: str) -> pd.DataFrame:
    lineage = pd.read_csv(path)
    missing = REQUIRED_LINEAGE_COLUMNS - set(lineage.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Lineage metadata is missing required columns: {missing_str}")

    lineage = lineage.copy()
    lineage["model_display_name"] = lineage["model_id"].map(_clean_text)
    lineage["source_model_raw"] = lineage["source_model"].map(_clean_relation)
    lineage["source_model_display_name"] = lineage["source_model"].map(_known_model_name)
    lineage["base_model_relation"] = lineage["base_model_relation"].map(_clean_relation)
    lineage["base_model_display_name"] = [
        _infer_base_model(model_id, source_model, relation)
        for model_id, source_model, relation in zip(
            lineage["model_display_name"],
            lineage["source_model_display_name"],
            lineage["base_model_relation"],
            strict=False,
        )
    ]
    lineage["base_model_known"] = lineage["base_model_display_name"].notna()
    return lineage[
        [
            "model_display_name",
            "source_model_raw",
            "source_model_display_name",
            "base_model_display_name",
            "base_model_relation",
            "base_model_known",
        ]
    ].drop_duplicates(subset=["model_display_name"])


def merge_lineage_metadata(metadata: pd.DataFrame, lineage: pd.DataFrame | None) -> pd.DataFrame:
    enriched = metadata.copy()
    if lineage is None or lineage.empty:
        enriched["source_model"] = "unknown"
        enriched["base_model"] = "unknown"
        enriched["base_model_relation"] = "unknown"
        enriched["base_model_known"] = False
        enriched["is_base_model_anchor"] = False
        return enriched

    enriched = enriched.merge(
        lineage,
        left_on="display_name",
        right_on="model_display_name",
        how="left",
    )
    enriched["source_model"] = enriched["source_model_raw"].fillna("unknown")
    enriched["base_model"] = enriched["base_model_display_name"].fillna("unknown")
    enriched["base_model_relation"] = enriched["base_model_relation"].fillna("unknown")
    enriched["base_model_known"] = enriched["base_model_known"].fillna(False).astype(bool)
    enriched["is_base_model_anchor"] = enriched["base_model_known"] & enriched["display_name"].eq(enriched["base_model"])
    return enriched.drop(
        columns=[
            "model_display_name",
            "source_model_raw",
            "source_model_display_name",
            "base_model_display_name",
        ],
        errors="ignore",
    )


def _known_lineage_frame(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = metadata.copy()
    frame["cluster"] = labels
    frame["cluster_label"] = frame["cluster"].astype(str)
    return frame[frame["base_model_known"]].copy()


def _assigned_frame(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = metadata.copy()
    frame["cluster"] = labels
    frame["cluster_label"] = frame["cluster"].astype(str)
    return frame[frame["cluster"] != -1].copy()


def compute_lineage_cluster_purity(known_lineage: pd.DataFrame) -> float | None:
    assigned = known_lineage[known_lineage["cluster"] != -1]
    if assigned.empty:
        return None

    dominant = 0
    total = 0
    for _, cluster_frame in assigned.groupby("cluster", sort=True):
        counts = cluster_frame["base_model"].value_counts()
        dominant += int(counts.iloc[0])
        total += int(cluster_frame.shape[0])
    if total == 0:
        return None
    return dominant / total


def compute_anchor_cluster_recall(known_lineage: pd.DataFrame) -> tuple[float | None, int]:
    recalls = []
    anchored_family_count = 0

    for base_model, family_frame in known_lineage.groupby("base_model", sort=False):
        anchor_rows = family_frame[(family_frame["display_name"] == base_model) & (family_frame["cluster"] != -1)]
        if anchor_rows.empty:
            continue
        anchored_family_count += 1
        anchor_cluster = anchor_rows.iloc[0]["cluster"]
        recalls.append(float((family_frame["cluster"] == anchor_cluster).mean()))

    if not recalls:
        return None, 0
    return float(np.mean(recalls)), anchored_family_count


def _safe_external_metric(
    metric_fn,
    known_lineage: pd.DataFrame,
) -> float | None:
    assigned = known_lineage[known_lineage["cluster"] != -1]
    if assigned.empty:
        return None
    if assigned["base_model"].nunique() < 2 or assigned["cluster"].nunique() < 2:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples.*",
            category=UserWarning,
        )
        return float(metric_fn(assigned["base_model"], assigned["cluster"]))


def _quality_metric_value(quality_metrics: pd.DataFrame | None, metric_key: str) -> float | None:
    if quality_metrics is None or quality_metrics.empty:
        return None
    row = quality_metrics.loc[quality_metrics["metric_key"] == metric_key, "value"]
    if row.empty:
        return None
    value = row.iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _clamp_01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalized_silhouette(silhouette: float | None) -> float | None:
    if silhouette is None:
        return None
    return _clamp_01((silhouette + 1.0) / 2.0)


def _normalized_dunn_index(dunn_index: float | None) -> float | None:
    if dunn_index is None:
        return None
    if dunn_index <= 0:
        return 0.0
    return _clamp_01(dunn_index / (1.0 + dunn_index))


def compute_lineage_run_score(
    known_assigned_fraction: float | None,
    lineage_cluster_purity: float | None,
    lineage_v_measure: float | None,
    anchor_cluster_recall: float | None,
    anchored_family_count: int,
    quality_metrics: pd.DataFrame | None,
) -> float:
    # Prioritize lineage agreement while keeping geometric cluster quality as a guardrail.
    coverage = _clamp_01(known_assigned_fraction or 0.0)
    purity = _clamp_01(lineage_cluster_purity or 0.0)
    v_measure = _clamp_01(lineage_v_measure or 0.0)
    anchor_recall = _clamp_01(anchor_cluster_recall or 0.0)
    anchor_weight = _clamp_01(anchored_family_count / 20.0)

    noise_fraction = _quality_metric_value(quality_metrics, "noise_fraction")
    largest_cluster_fraction = _quality_metric_value(quality_metrics, "largest_cluster_fraction")
    silhouette = _quality_metric_value(quality_metrics, "silhouette")
    dunn_index = _quality_metric_value(quality_metrics, "dunn_index")

    noise_penalty = 1.0 - _clamp_01(noise_fraction if noise_fraction is not None else 0.5)
    lineage_score = (0.5 + 0.5 * coverage) * (
        0.45 * v_measure
        + 0.30 * purity
        + 0.20 * anchor_recall * anchor_weight
        + 0.05 * noise_penalty
    )

    geometry_components: list[tuple[float, float]] = []
    normalized_silhouette = _normalized_silhouette(silhouette)
    if normalized_silhouette is not None:
        geometry_components.append((0.40, normalized_silhouette))

    normalized_dunn = _normalized_dunn_index(dunn_index)
    if normalized_dunn is not None:
        geometry_components.append((0.30, normalized_dunn))

    if noise_fraction is not None:
        geometry_components.append((0.20, 1.0 - _clamp_01(noise_fraction)))
    if largest_cluster_fraction is not None:
        geometry_components.append((0.10, 1.0 - _clamp_01(largest_cluster_fraction)))

    if geometry_components:
        total_weight = float(sum(weight for weight, _ in geometry_components))
        geometry_score = float(sum(weight * value for weight, value in geometry_components) / total_weight)
    else:
        geometry_score = 0.5

    return _clamp_01(0.80 * lineage_score + 0.20 * geometry_score)


def build_lineage_quality_metrics(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    quality_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    known_lineage = _known_lineage_frame(metadata, labels)
    known_fraction = float(np.mean(metadata["base_model_known"])) if len(metadata) else None
    known_assigned_fraction = float(np.mean(metadata["base_model_known"] & (labels != -1))) if len(metadata) else None
    anchor_cluster_recall, anchored_family_count = compute_anchor_cluster_recall(known_lineage)
    lineage_cluster_purity = compute_lineage_cluster_purity(known_lineage)
    lineage_homogeneity = _safe_external_metric(homogeneity_score, known_lineage)
    lineage_completeness = _safe_external_metric(completeness_score, known_lineage)
    lineage_v_measure = _safe_external_metric(v_measure_score, known_lineage)
    lineage_adjusted_rand = _safe_external_metric(adjusted_rand_score, known_lineage)
    lineage_run_score = compute_lineage_run_score(
        known_assigned_fraction=known_assigned_fraction,
        lineage_cluster_purity=lineage_cluster_purity,
        lineage_v_measure=lineage_v_measure,
        anchor_cluster_recall=anchor_cluster_recall,
        anchored_family_count=anchored_family_count,
        quality_metrics=quality_metrics,
    )

    rows = [
        {
            "metric_key": "lineage_run_score",
            "metric": "Lineage run score",
            "value": lineage_run_score,
            "optimize": "Higher",
            "description": "Composite score (80% lineage recovery, 20% geometric quality guardrails).",
        },
        {
            "metric_key": "known_lineage_fraction",
            "metric": "Known base-model fraction",
            "value": known_fraction,
            "optimize": "Higher",
            "description": "Share of displayed models with a resolved base model from sampled_metadata.csv.",
        },
        {
            "metric_key": "known_assigned_fraction",
            "metric": "Known+assigned fraction",
            "value": known_assigned_fraction,
            "optimize": "Higher",
            "description": "Share of displayed models with known lineage that also landed in a non-noise cluster.",
        },
        {
            "metric_key": "lineage_cluster_purity",
            "metric": "Base-model cluster purity",
            "value": lineage_cluster_purity,
            "optimize": "Higher",
            "description": "Weighted dominant base-model share inside each assigned cluster.",
        },
        {
            "metric_key": "lineage_homogeneity",
            "metric": "Lineage homogeneity",
            "value": lineage_homogeneity,
            "optimize": "Higher",
            "description": "How often predicted clusters contain a single dominant base-model family.",
        },
        {
            "metric_key": "lineage_completeness",
            "metric": "Lineage completeness",
            "value": lineage_completeness,
            "optimize": "Higher",
            "description": "How often members of the same base-model family stay together.",
        },
        {
            "metric_key": "lineage_v_measure",
            "metric": "Lineage V-measure",
            "value": lineage_v_measure,
            "optimize": "Higher",
            "description": "Balanced lineage agreement across homogeneity and completeness.",
        },
        {
            "metric_key": "lineage_adjusted_rand",
            "metric": "Lineage adjusted Rand",
            "value": lineage_adjusted_rand,
            "optimize": "Higher",
            "description": "Pairwise agreement between clusters and base-model families.",
        },
        {
            "metric_key": "anchor_cluster_recall",
            "metric": "Anchor cluster recall",
            "value": anchor_cluster_recall,
            "optimize": "Higher",
            "description": "For families whose base model is present, how much of the family lands with that anchor.",
        },
        {
            "metric_key": "anchored_family_count",
            "metric": "Families with visible anchor",
            "value": float(anchored_family_count),
            "optimize": "Context",
            "description": "Known base-model families where the anchor model is present in the current view.",
        },
    ]
    return pd.DataFrame(rows)


def build_lineage_family_summary(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    known_lineage = _known_lineage_frame(metadata, labels)
    if known_lineage.empty:
        return pd.DataFrame(
            columns=[
                "base_model",
                "family_size",
                "anchor_present",
                "clusters_hit",
                "dominant_cluster",
                "dominant_cluster_share",
                "relation_mix",
            ]
        )

    rows = []
    for base_model, family_frame in known_lineage.groupby("base_model", sort=False):
        cluster_counts = family_frame["cluster_label"].value_counts()
        dominant_cluster = cluster_counts.index[0]
        relation_counts = family_frame["base_model_relation"].value_counts()
        relation_mix = ", ".join(
            f"{relation}:{count}" for relation, count in relation_counts.head(3).items()
        )
        rows.append(
            {
                "base_model": base_model,
                "family_size": int(family_frame.shape[0]),
                "anchor_present": bool((family_frame["display_name"] == base_model).any()),
                "clusters_hit": int(family_frame["cluster_label"].nunique()),
                "dominant_cluster": dominant_cluster,
                "dominant_cluster_share": float(cluster_counts.iloc[0] / family_frame.shape[0]),
                "relation_mix": relation_mix,
            }
        )
    return pd.DataFrame(rows).sort_values(["family_size", "dominant_cluster_share"], ascending=[False, False]).reset_index(drop=True)


def build_lineage_cluster_summary(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    assigned = _assigned_frame(metadata, labels)
    if assigned.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "cluster_size",
                "known_lineage_share",
                "dominant_base_model",
                "dominant_base_share",
                "center_role",
                "anchor_present",
                "dominant_relation",
            ]
        )

    rows = []
    for cluster_id, cluster_frame in assigned.groupby("cluster", sort=True):
        known = cluster_frame[cluster_frame["base_model_known"]]
        known_lineage_share = float(known.shape[0] / cluster_frame.shape[0])
        dominant_base_model = "unknown"
        dominant_base_share = np.nan
        dominant_relation = "unknown"
        anchor_present = False
        center_role = "synthetic_cluster_center"

        if not known.empty:
            counts = known["base_model"].value_counts()
            dominant_base_model = str(counts.index[0])
            dominant_base_share = float(counts.iloc[0] / cluster_frame.shape[0])
            dominant_family = known[known["base_model"] == dominant_base_model]
            relation_counts = dominant_family["base_model_relation"].value_counts()
            dominant_relation = str(relation_counts.index[0]) if not relation_counts.empty else "unknown"
            anchor_present = bool((cluster_frame["display_name"] == dominant_base_model).any())
            center_role = "anchor_model" if anchor_present else "synthetic_base_center"

        rows.append(
            {
                "cluster": int(cluster_id),
                "cluster_size": int(cluster_frame.shape[0]),
                "known_lineage_share": known_lineage_share,
                "dominant_base_model": dominant_base_model,
                "dominant_base_share": dominant_base_share,
                "center_role": center_role,
                "anchor_present": anchor_present,
                "dominant_relation": dominant_relation,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["known_lineage_share", "cluster_size", "cluster"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_lineage_constellation(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    min_cluster_size: int = 2,
    max_clusters: int = 24,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if "embedding_x" not in metadata.columns or "embedding_y" not in metadata.columns:
        raise ValueError("Lineage constellation requires embedding_x and embedding_y columns.")

    assigned = _assigned_frame(metadata, labels)
    cluster_summary = build_lineage_cluster_summary(metadata, labels)
    eligible_clusters = cluster_summary[cluster_summary["cluster_size"] >= min_cluster_size].copy()
    selected_clusters = eligible_clusters.head(max_clusters)

    if assigned.empty or selected_clusters.empty:
        return pd.DataFrame(), {
            "assigned_models": int(assigned.shape[0]),
            "shown_models": 0,
            "shown_clusters": 0,
            "omitted_clusters": int(cluster_summary.shape[0]),
            "clusters_with_known_center": 0,
            "clusters_with_anchor_center": 0,
        }

    selected_cluster_ids = selected_clusters["cluster"].tolist()
    selected_frame = assigned[assigned["cluster"].isin(selected_cluster_ids)].copy()
    centroids = (
        selected_frame.groupby("cluster", sort=False)[["embedding_x", "embedding_y"]]
        .mean()
        .reindex(selected_cluster_ids)
    )
    centered = centroids - centroids.mean(axis=0)
    scales = centroids.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    global_centers = centered.divide(scales, axis=1) * 11.0
    if len(selected_cluster_ids) == 1:
        global_centers.iloc[0] = [0.0, 0.0]

    rows: list[dict[str, object]] = []
    clusters_with_known_center = 0
    clusters_with_anchor_center = 0
    target_radius = 3.2

    for cluster_id in selected_cluster_ids:
        cluster_frame = selected_frame[selected_frame["cluster"] == cluster_id].copy().sort_values("display_name")
        summary_row = selected_clusters[selected_clusters["cluster"] == cluster_id].iloc[0]
        center_x = float(global_centers.loc[cluster_id, "embedding_x"])
        center_y = float(global_centers.loc[cluster_id, "embedding_y"])
        cluster_size = int(cluster_frame.shape[0])
        dominant_base_model = summary_row["dominant_base_model"]
        center_role = str(summary_row["center_role"])

        if dominant_base_model != "unknown":
            dominant_family = cluster_frame[cluster_frame["base_model"] == dominant_base_model]
        else:
            dominant_family = cluster_frame.iloc[0:0]

        if not dominant_family.empty:
            clusters_with_known_center += 1

        anchor_candidates = cluster_frame[cluster_frame["display_name"] == dominant_base_model]

        if center_role == "anchor_model" and not anchor_candidates.empty:
            anchor_row = anchor_candidates.iloc[0]
            origin = anchor_row[["embedding_x", "embedding_y"]].to_numpy(dtype=float)
            clusters_with_anchor_center += 1
            anchor_name = dominant_base_model
        else:
            if not dominant_family.empty:
                origin = dominant_family[["embedding_x", "embedding_y"]].mean().to_numpy(dtype=float)
                anchor_name = dominant_base_model
            else:
                origin = cluster_frame[["embedding_x", "embedding_y"]].mean().to_numpy(dtype=float)
                anchor_name = f"Cluster {cluster_id} center"

            rows.append(
                {
                    "display_name": anchor_name,
                    "model_id": str(anchor_name).replace("/", "--"),
                    "path": "",
                    "num_layers": np.nan,
                    "raw_module_count": np.nan,
                    "cluster": cluster_id,
                    "cluster_label": str(cluster_id),
                    "source_model": dominant_base_model if dominant_base_model != "unknown" else "unknown",
                    "base_model": dominant_base_model if dominant_base_model != "unknown" else "unknown",
                    "base_model_relation": summary_row["dominant_relation"],
                    "base_model_known": dominant_base_model != "unknown",
                    "is_base_model_anchor": dominant_base_model != "unknown",
                    "lineage_role": center_role,
                    "is_synthetic": True,
                    "cluster_size": cluster_size,
                    "dominant_base_model": dominant_base_model,
                    "dominant_base_share": summary_row["dominant_base_share"],
                    "plot_x": center_x,
                    "plot_y": center_y,
                }
            )

        local_coords = cluster_frame[["embedding_x", "embedding_y"]].to_numpy(dtype=float) - origin
        norms = np.sqrt(np.sum(local_coords**2, axis=1))
        max_norm = float(np.max(norms)) if norms.size else 0.0
        local_scale = target_radius / max_norm if max_norm > 1e-8 else 1.0

        for row_index, record in enumerate(cluster_frame.to_dict(orient="records")):
            local_x, local_y = local_coords[row_index]
            record_role = "cluster_member"
            if (
                center_role == "anchor_model"
                and record["display_name"] == dominant_base_model
            ):
                record_role = "anchor_model"
            record.update(
                {
                    "plot_x": center_x + float(local_x * local_scale),
                    "plot_y": center_y + float(local_y * local_scale),
                    "lineage_role": record_role,
                    "is_synthetic": False,
                    "cluster_size": cluster_size,
                    "dominant_base_model": dominant_base_model,
                    "dominant_base_share": summary_row["dominant_base_share"],
                }
            )
            rows.append(record)

    plot_frame = pd.DataFrame(rows)
    if not plot_frame.empty:
        plot_frame["cluster_label"] = plot_frame["cluster_label"].astype(str)
    return plot_frame, {
        "assigned_models": int(assigned.shape[0]),
        "shown_models": int(selected_frame.shape[0]),
        "shown_clusters": int(len(selected_cluster_ids)),
        "omitted_clusters": int(max(0, cluster_summary.shape[0] - len(selected_cluster_ids))),
        "clusters_with_known_center": int(clusters_with_known_center),
        "clusters_with_anchor_center": int(clusters_with_anchor_center),
    }
