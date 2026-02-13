"""Spike clustering and unit creation."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from speedsort.config import ClusteringMethod, SpikeSortingConfiguration, _HAS_SKLEARN
from speedsort.types import SpikeUnit

logger = logging.getLogger("speedsort")


def cluster_spikes(features: np.ndarray, config: SpikeSortingConfiguration) -> np.ndarray:
    """Cluster feature vectors into spike units. Returns integer label per spike."""
    if len(features) == 0:
        return np.array([])

    method = config.clustering_method

    # Auto select method based on data size
    if method == ClusteringMethod.AUTO:
        n_spikes = features.shape[0]
        if n_spikes < 1000:
            method = ClusteringMethod.KMEANS
        elif n_spikes < 10000:
            method = ClusteringMethod.GMM
        else:
            method = ClusteringMethod.DBSCAN

    if not _HAS_SKLEARN:
        return np.zeros(len(features), dtype=np.int64)

    if method == ClusteringMethod.KMEANS:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=config.max_clusters, random_state=42).fit_predict(features)
    elif method == ClusteringMethod.GMM:
        from sklearn.mixture import GaussianMixture
        labels = GaussianMixture(n_components=config.max_clusters, random_state=42).fit_predict(features)
    elif method == ClusteringMethod.DBSCAN:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=config.isolation_threshold, min_samples=config.min_cluster_size).fit_predict(features)
    elif method == ClusteringMethod.MEANSHIFT:
        from sklearn.cluster import MeanShift
        labels = MeanShift().fit_predict(features)
    elif method == ClusteringMethod.ISO_FOREST:
        from sklearn.ensemble import IsolationForest
        labels = IsolationForest(contamination=config.contamination_threshold, random_state=42).fit_predict(features)
    elif method == ClusteringMethod.AGGLO:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=config.max_clusters, linkage='ward').fit_predict(features)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    return labels


def create_units(
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    spike_channels: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> List[SpikeUnit]:
    """Group spikes by cluster label into SpikeUnit objects."""
    units = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise label

        unit_indices = np.where(labels == label)[0]

        unit = SpikeUnit(
            unit_id=label,
            waveforms=waveforms[unit_indices],
            timestamps=spike_times[unit_indices],
            channel_ids=spike_channels[unit_indices],
            features=features[unit_indices] if features is not None else None,
        )
        units.append(unit)

    return units
