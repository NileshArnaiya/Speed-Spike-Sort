"""Quality metrics: firing rate, presence ratio, ISI violations, SNR, isolation distance, silhouette score."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from speedsort.config import SpikeSortingConfiguration, _HAS_SKLEARN
from speedsort.types import SpikeUnit

logger = logging.getLogger("speedsort")


def compute_quality_metrics(
    units: List[SpikeUnit],
    features: np.ndarray,
    labels: np.ndarray,
    config: SpikeSortingConfiguration,
) -> Dict[str, Dict[str, float]]:
    """Compute real quality metrics for each unit.

    Metrics:
        firing_rate           Hz
        presence_ratio        fraction of 1-s bins with ≥1 spike
        isi_violations_ratio  fraction of spikes with ISI < 1.5 ms
        isi_violations_count  absolute count
        snr                   peak amplitude of mean waveform / baseline noise
        isolation_distance    Mahalanobis-based
        silhouette_score      per-unit (sklearn)
    """
    quality_metrics: Dict[str, Dict[str, float]] = {}

    if len(units) == 0:
        return quality_metrics

    all_times = np.concatenate([u.timestamps for u in units if len(u.timestamps) > 0])
    if len(all_times) == 0:
        return quality_metrics

    duration_sec = (all_times.max() - all_times.min()) / config.sampling_rate
    if duration_sec <= 0:
        duration_sec = 1.0

    refractory_samples = int(0.0015 * config.sampling_rate)

    # Per-unit silhouette scores
    per_unit_silhouette = _compute_silhouette_map(units, features, labels)

    for unit in units:
        unit_id = unit.unit_id
        m: Dict[str, float] = {}
        n_spikes = len(unit.timestamps)

        m['firing_rate'] = n_spikes / duration_sec if n_spikes > 0 else 0.0
        m['presence_ratio'] = _presence_ratio(unit, all_times, config)
        m['isi_violations_ratio'], m['isi_violations_count'] = _isi_violations(unit, refractory_samples)
        m['snr'] = _snr(unit)
        m['isolation_distance'] = _isolation_distance(unit_id, features, labels)
        m['silhouette_score'] = per_unit_silhouette.get(unit_id, 0.0)

        quality_metrics[unit_id] = m
        unit.quality_metrics = m

    return quality_metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _presence_ratio(unit: SpikeUnit, all_times: np.ndarray, config: SpikeSortingConfiguration) -> float:
    n_spikes = len(unit.timestamps)
    if n_spikes == 0:
        return 0.0
    bin_size_samples = int(config.sampling_rate)
    min_t, max_t = all_times.min(), all_times.max()
    n_bins = max(1, int(np.ceil((max_t - min_t) / bin_size_samples)))
    spike_bins = ((unit.timestamps - min_t) / bin_size_samples).astype(int)
    spike_bins = np.clip(spike_bins, 0, n_bins - 1)
    return len(np.unique(spike_bins)) / n_bins


def _isi_violations(unit: SpikeUnit, refractory_samples: int):
    n_spikes = len(unit.timestamps)
    if n_spikes <= 1:
        return 0.0, 0
    sorted_times = np.sort(unit.timestamps)
    isis = np.diff(sorted_times)
    n_violations = int(np.sum(isis < refractory_samples))
    return n_violations / n_spikes, n_violations


def _snr(unit: SpikeUnit) -> float:
    n_spikes = len(unit.timestamps)
    if n_spikes == 0 or unit.waveforms is None or len(unit.waveforms) == 0:
        return 0.0
    try:
        if unit.waveforms.ndim == 3:
            mean_waveform = np.mean(unit.waveforms, axis=0)
            peak_ch = np.argmax(np.max(np.abs(mean_waveform), axis=0))
            wf = mean_waveform[:, peak_ch]
        elif unit.waveforms.ndim == 2:
            wf = np.mean(unit.waveforms, axis=0)
        else:
            wf = unit.waveforms.flatten()

        peak_amplitude = np.max(np.abs(wf))
        n_baseline = max(1, len(wf) // 10)
        baseline = np.concatenate([wf[:n_baseline], wf[-n_baseline:]])
        noise_std = np.std(baseline)
        return float(peak_amplitude / noise_std) if noise_std > 0 else 0.0
    except Exception:
        return 0.0


def _isolation_distance(unit_id: int, features: np.ndarray, labels: np.ndarray) -> float:
    unit_indices = np.where(labels == unit_id)[0]
    other_indices = np.where((labels != unit_id) & (labels >= 0))[0]
    if len(unit_indices) <= 1 or len(other_indices) <= 1 or len(features) == 0:
        return 0.0
    try:
        unit_feats = features[unit_indices]
        other_feats = features[other_indices]
        cov = np.cov(unit_feats.T)
        if cov.ndim < 2:
            cov = np.array([[cov]])
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov)
        mean_unit = np.mean(unit_feats, axis=0)
        diff = other_feats - mean_unit
        mahal_dists = np.sum(diff @ cov_inv * diff, axis=1)
        mahal_dists_sorted = np.sort(mahal_dists)
        idx = min(len(unit_indices), len(mahal_dists_sorted)) - 1
        return float(mahal_dists_sorted[idx])
    except Exception:
        return 0.0


def _compute_silhouette_map(units: List[SpikeUnit], features: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    per_unit: Dict[int, float] = {}
    unique_ids = np.array([u.unit_id for u in units])
    if not (_HAS_SKLEARN and len(unique_ids) > 1 and len(features) > 0):
        return per_unit
    try:
        valid_mask = labels >= 0
        if valid_mask.sum() > 1 and len(np.unique(labels[valid_mask])) > 1:
            from sklearn.metrics import silhouette_samples
            sample_scores = silhouette_samples(features[valid_mask], labels[valid_mask])
            valid_labels = labels[valid_mask]
            for uid in unique_ids:
                uid_mask = valid_labels == uid
                if uid_mask.sum() > 0:
                    per_unit[uid] = float(np.mean(sample_scores[uid_mask]))
    except Exception as e:
        logger.warning(f"Could not compute silhouette scores: {e}")
    return per_unit
