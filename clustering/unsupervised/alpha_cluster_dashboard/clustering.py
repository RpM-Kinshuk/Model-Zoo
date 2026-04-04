from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    labels: np.ndarray
    embedding: np.ndarray
    embedding_name: str
    fitted_features: np.ndarray | None
    silhouette: float | None
    quality_metrics: pd.DataFrame


def cluster_feature_matrix(
    features: np.ndarray,
    algorithm: str,
    n_clusters: int,
    linkage: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    projection: str,
    random_state: int,
) -> ClusteringResult:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    fitted = pipeline.fit_transform(features)

    if algorithm == "kmeans":
        estimator = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif algorithm == "agglomerative":
        estimator = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage=linkage)
    elif algorithm == "dbscan":
        estimator = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    else:
        raise ValueError(f"Unsupported vector clustering algorithm: {algorithm}")

    labels = estimator.fit_predict(fitted)
    if projection == "tsne" and fitted.shape[0] >= 4:
        perplexity = min(30, max(2, fitted.shape[0] // 4), fitted.shape[0] - 1)
        embedding = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(fitted)
        embedding_name = "t-SNE"
    else:
        embedding = PCA(n_components=2, random_state=random_state).fit_transform(fitted)
        embedding_name = "PCA"

    distances = pairwise_distances(fitted, metric="euclidean")
    score = compute_silhouette(labels, fitted, metric="euclidean")
    return ClusteringResult(
        labels=labels,
        embedding=embedding,
        embedding_name=embedding_name,
        fitted_features=fitted,
        silhouette=score,
        quality_metrics=build_quality_metrics(labels=labels, distances=distances, features=fitted),
    )


def cluster_distance_matrix(
    distances: np.ndarray,
    algorithm: str,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> ClusteringResult:
    if algorithm == "agglomerative":
        estimator = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    elif algorithm == "dbscan":
        estimator = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="precomputed")
    else:
        raise ValueError(f"Unsupported distance clustering algorithm: {algorithm}")

    labels = estimator.fit_predict(distances)
    embedding = classical_mds(distances, n_components=2)
    score = compute_silhouette(labels, distances, metric="precomputed")
    return ClusteringResult(
        labels=labels,
        embedding=embedding,
        embedding_name="Classical MDS",
        fitted_features=None,
        silhouette=score,
        quality_metrics=build_quality_metrics(labels=labels, distances=distances, features=None),
    )


def classical_mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    squared = distances ** 2
    n_samples = squared.shape[0]
    centering = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.maximum(eigenvalues[:n_components], 0.0)
    return eigenvectors[:, :n_components] * np.sqrt(positive)


def compute_silhouette(labels: np.ndarray, data: np.ndarray, metric: str) -> float | None:
    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2:
        return None
    non_noise = labels != -1
    filtered_labels = labels[non_noise]
    if len(set(filtered_labels.tolist())) < 2:
        return None
    if metric == "precomputed":
        filtered_data = data[np.ix_(non_noise, non_noise)]
    else:
        filtered_data = data[non_noise]
    return float(silhouette_score(filtered_data, filtered_labels, metric=metric))


def _filter_noise_labels(labels: np.ndarray, data: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    non_noise = labels != -1
    filtered_labels = labels[non_noise]
    if data is None:
        return filtered_labels, None
    if data.ndim == 1:
        return filtered_labels, data[non_noise]
    if data.shape[0] != data.shape[1]:
        return filtered_labels, data[non_noise]
    return filtered_labels, data[np.ix_(non_noise, non_noise)]


def _cluster_index_groups(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == cluster_id) for cluster_id in sorted(set(labels.tolist())) if cluster_id != -1]


def compute_calinski_harabasz(labels: np.ndarray, features: np.ndarray) -> float | None:
    filtered_labels, filtered_features = _filter_noise_labels(labels, features)
    if filtered_features is None or len(set(filtered_labels.tolist())) < 2:
        return None
    if filtered_features.shape[0] <= len(set(filtered_labels.tolist())):
        return None
    return float(calinski_harabasz_score(filtered_features, filtered_labels))


def compute_davies_bouldin(labels: np.ndarray, features: np.ndarray) -> float | None:
    filtered_labels, filtered_features = _filter_noise_labels(labels, features)
    if filtered_features is None or len(set(filtered_labels.tolist())) < 2:
        return None
    if filtered_features.shape[0] <= len(set(filtered_labels.tolist())):
        return None
    return float(davies_bouldin_score(filtered_features, filtered_labels))


def compute_dunn_index(labels: np.ndarray, distances: np.ndarray) -> float | None:
    filtered_labels, filtered_distances = _filter_noise_labels(labels, distances)
    if filtered_distances is None or len(set(filtered_labels.tolist())) < 2:
        return None

    cluster_groups = _cluster_index_groups(filtered_labels)
    if len(cluster_groups) < 2:
        return None

    diameters = []
    for group in cluster_groups:
        if group.shape[0] < 2:
            diameters.append(0.0)
            continue
        diameters.append(float(np.max(filtered_distances[np.ix_(group, group)])))
    max_diameter = max(diameters)
    if max_diameter <= 0:
        return None

    min_intercluster = np.inf
    for left_index, left_group in enumerate(cluster_groups):
        for right_group in cluster_groups[left_index + 1 :]:
            distance_block = filtered_distances[np.ix_(left_group, right_group)]
            min_intercluster = min(min_intercluster, float(np.min(distance_block)))
    if not np.isfinite(min_intercluster):
        return None
    return float(min_intercluster / max_diameter)


def compute_separation_ratio(labels: np.ndarray, distances: np.ndarray) -> float | None:
    filtered_labels, filtered_distances = _filter_noise_labels(labels, distances)
    if filtered_distances is None or len(set(filtered_labels.tolist())) < 2:
        return None

    within_distances: list[np.ndarray] = []
    between_distances: list[np.ndarray] = []
    cluster_groups = _cluster_index_groups(filtered_labels)
    if len(cluster_groups) < 2:
        return None

    for left_index, left_group in enumerate(cluster_groups):
        if left_group.shape[0] >= 2:
            block = filtered_distances[np.ix_(left_group, left_group)]
            within_distances.append(block[np.triu_indices_from(block, k=1)])
        for right_group in cluster_groups[left_index + 1 :]:
            between_distances.append(filtered_distances[np.ix_(left_group, right_group)].ravel())

    within_arrays = [values for values in within_distances if values.size > 0]
    between_arrays = [values for values in between_distances if values.size > 0]
    if not within_arrays or not between_arrays:
        return None

    within_values = np.concatenate(within_arrays)
    between_values = np.concatenate(between_arrays)
    mean_within = float(np.mean(within_values))
    if mean_within <= 0:
        return None
    mean_between = float(np.mean(between_values))
    return mean_between / mean_within


def build_quality_metrics(
    labels: np.ndarray,
    distances: np.ndarray,
    features: np.ndarray | None,
) -> pd.DataFrame:
    non_noise = labels != -1
    assigned_fraction = float(np.mean(non_noise))
    noise_fraction = float(np.mean(labels == -1))
    cluster_ids = [cluster_id for cluster_id in sorted(set(labels.tolist())) if cluster_id != -1]
    cluster_sizes = np.asarray([np.sum(labels == cluster_id) for cluster_id in cluster_ids], dtype=float)
    largest_cluster_fraction = float(cluster_sizes.max() / len(labels)) if cluster_sizes.size else None
    cluster_size_cv = float(cluster_sizes.std() / cluster_sizes.mean()) if cluster_sizes.size and cluster_sizes.mean() > 0 else None

    rows = [
        {
            "metric_key": "silhouette",
            "metric": "Silhouette",
            "value": compute_silhouette(labels, features if features is not None else distances, metric="euclidean" if features is not None else "precomputed"),
            "optimize": "Higher",
            "description": "Average separation versus cohesion across assigned clusters.",
        },
        {
            "metric_key": "dunn_index",
            "metric": "Dunn index",
            "value": compute_dunn_index(labels, distances),
            "optimize": "Higher",
            "description": "Minimum inter-cluster separation divided by worst cluster diameter.",
        },
        {
            "metric_key": "separation_ratio",
            "metric": "Between/within ratio",
            "value": compute_separation_ratio(labels, distances),
            "optimize": "Higher",
            "description": "Mean between-cluster distance relative to mean within-cluster distance.",
        },
        {
            "metric_key": "calinski_harabasz",
            "metric": "Calinski-Harabasz",
            "value": compute_calinski_harabasz(labels, features) if features is not None else None,
            "optimize": "Higher",
            "description": "Variance-ratio score in feature space.",
        },
        {
            "metric_key": "davies_bouldin",
            "metric": "Davies-Bouldin",
            "value": compute_davies_bouldin(labels, features) if features is not None else None,
            "optimize": "Lower",
            "description": "Average similarity between each cluster and its nearest competing cluster.",
        },
        {
            "metric_key": "assigned_fraction",
            "metric": "Assigned fraction",
            "value": assigned_fraction,
            "optimize": "Higher",
            "description": "Share of models assigned to a non-noise cluster.",
        },
        {
            "metric_key": "noise_fraction",
            "metric": "Noise fraction",
            "value": noise_fraction,
            "optimize": "Lower",
            "description": "Share of models labeled as noise or outliers.",
        },
        {
            "metric_key": "largest_cluster_fraction",
            "metric": "Largest cluster share",
            "value": largest_cluster_fraction,
            "optimize": "Context",
            "description": "How much of the dataset is absorbed by the largest cluster.",
        },
        {
            "metric_key": "cluster_size_cv",
            "metric": "Cluster size CV",
            "value": cluster_size_cv,
            "optimize": "Lower",
            "description": "Coefficient of variation of cluster sizes; lower means more balanced clusters.",
        },
    ]
    return pd.DataFrame(rows)


def cluster_summary_table(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    profiles: np.ndarray,
) -> pd.DataFrame:
    rows = []
    overall_alpha = np.nanmean(profiles, axis=(1, 2))
    module_coverage = np.nanmean(np.isfinite(profiles), axis=(1, 2))

    with_labels = metadata.copy()
    with_labels["cluster"] = labels
    with_labels["overall_alpha"] = overall_alpha
    with_labels["module_coverage"] = module_coverage

    for cluster_id, cluster_frame in with_labels.groupby("cluster", sort=True):
        rows.append(
            {
                "cluster": cluster_id,
                "size": int(cluster_frame.shape[0]),
                "mean_layers": float(cluster_frame["num_layers"].mean()),
                "mean_alpha": float(cluster_frame["overall_alpha"].mean()),
                "mean_module_coverage": float(cluster_frame["module_coverage"].mean()),
                "example_model": cluster_frame.sort_values("overall_alpha").iloc[cluster_frame.shape[0] // 2]["display_name"],
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)
