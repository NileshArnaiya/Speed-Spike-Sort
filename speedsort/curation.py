"""Curation: auto-merge similar units and reject noise clusters."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from speedsort.config import SpikeSortingConfiguration
from speedsort.types import SpikeUnit

logger = logging.getLogger("speedsort")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_merge_units(
    units: List[SpikeUnit],
    config: SpikeSortingConfiguration,
) -> List[SpikeUnit]:
    """Merge unit pairs whose templates are highly similar AND whose
    cross-correlograms show a peak near zero lag.

    Algorithm:
        1. Compute mean waveform (template) for each unit on its peak channel.
        2. For every pair, compute cosine similarity of templates.
        3. If similarity > ``config.merge_template_threshold`` AND the pair has
           correlated spike trains, merge the smaller unit into the larger one.
        4. Repeat until no more merges are possible.

    Returns a new (potentially shorter) list of SpikeUnit objects.
    """
    if len(units) <= 1:
        return units

    merged = True
    current_units = list(units)

    while merged:
        merged = False
        templates = [_unit_template(u) for u in current_units]
        n = len(current_units)

        # Find best merge candidate
        best_sim = -1.0
        best_pair: Optional[Tuple[int, int]] = None

        for i in range(n):
            for j in range(i + 1, n):
                if templates[i] is None or templates[j] is None:
                    continue
                # Align lengths
                min_len = min(len(templates[i]), len(templates[j]))
                t_i = templates[i][:min_len]
                t_j = templates[j][:min_len]

                sim = _cosine_similarity(t_i, t_j)
                if sim > config.merge_template_threshold and sim > best_sim:
                    # Also check spike train correlation (optional fast check)
                    if _spike_trains_correlated(
                        current_units[i], current_units[j], config
                    ):
                        best_sim = sim
                        best_pair = (i, j)

        if best_pair is not None:
            i, j = best_pair
            logger.info(
                f"Merging unit {current_units[j].unit_id} into "
                f"unit {current_units[i].unit_id} (cosine={best_sim:.3f})"
            )
            current_units[i] = _merge_two_units(current_units[i], current_units[j])
            current_units.pop(j)
            merged = True

    logger.info(f"Auto-merge: {len(units)} → {len(current_units)} units")
    return current_units


def reject_noise_units(
    units: List[SpikeUnit],
    config: SpikeSortingConfiguration,
) -> List[SpikeUnit]:
    """Remove units that fail basic quality criteria.

    Rejection criteria (any one triggers removal):
        - ISI violation ratio > ``config.noise_isi_threshold`` (default 5 %)
        - SNR < ``config.noise_snr_threshold`` (default 1.5)
        - Firing rate outside ``config.noise_firing_rate_bounds`` (default 0.1–200 Hz)
    """
    kept: List[SpikeUnit] = []
    fr_lo, fr_hi = config.noise_firing_rate_bounds

    for unit in units:
        qm = unit.quality_metrics or {}
        isi_viol = qm.get("isi_violations_ratio", 0.0)
        snr = qm.get("snr", 999.0)
        fr = qm.get("firing_rate", 0.0)

        reasons: List[str] = []
        if isi_viol > config.noise_isi_threshold:
            reasons.append(f"ISI violations {isi_viol:.2%}")
        if snr < config.noise_snr_threshold:
            reasons.append(f"low SNR {snr:.2f}")
        if fr < fr_lo or fr > fr_hi:
            reasons.append(f"firing rate {fr:.2f} Hz")

        if reasons:
            logger.info(f"Rejecting unit {unit.unit_id}: {', '.join(reasons)}")
        else:
            kept.append(unit)

    logger.info(f"Noise rejection: {len(units)} → {len(kept)} units")
    return kept


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unit_template(unit: SpikeUnit) -> Optional[np.ndarray]:
    """Return the mean waveform on the peak channel, or None if unavailable."""
    if unit.waveforms is None or len(unit.waveforms) == 0:
        return None
    try:
        if unit.waveforms.ndim == 3:
            mean_wf = np.mean(unit.waveforms, axis=0)          # (n_samples, n_ch)
            peak_ch = np.argmax(np.max(np.abs(mean_wf), axis=0))
            return mean_wf[:, peak_ch]
        elif unit.waveforms.ndim == 2:
            return np.mean(unit.waveforms, axis=0)
        return unit.waveforms.flatten()
    except Exception:
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _spike_trains_correlated(
    u1: SpikeUnit,
    u2: SpikeUnit,
    config: SpikeSortingConfiguration,
    window_ms: float = 2.0,
) -> bool:
    """Return True if a significant fraction of spikes in u2 occur within
    ``window_ms`` of spikes in u1 (fast check for near-synchronous firing).
    """
    if config.sampling_rate is None or config.sampling_rate <= 0:
        return True  # Can't check, assume correlated
    window_samples = int(window_ms * config.sampling_rate / 1000.0)
    t1 = np.sort(u1.timestamps)
    t2 = np.sort(u2.timestamps)
    if len(t1) == 0 or len(t2) == 0:
        return False

    # For each spike in t2, check if there is a spike in t1 within window
    idx = np.searchsorted(t1, t2)
    n_close = 0
    for k, i in enumerate(idx):
        for candidate in [i - 1, i]:
            if 0 <= candidate < len(t1) and abs(int(t1[candidate]) - int(t2[k])) <= window_samples:
                n_close += 1
                break

    fraction = n_close / len(t2)
    return fraction > 0.3  # >30% of spikes are near-synchronous


def _merge_two_units(u_keep: SpikeUnit, u_merge: SpikeUnit) -> SpikeUnit:
    """Merge u_merge into u_keep, concatenating all spike data."""
    waveforms = np.concatenate([u_keep.waveforms, u_merge.waveforms], axis=0)
    timestamps = np.concatenate([u_keep.timestamps, u_merge.timestamps])
    channel_ids = np.concatenate([u_keep.channel_ids, u_merge.channel_ids])

    features = None
    if u_keep.features is not None and u_merge.features is not None:
        features = np.concatenate([u_keep.features, u_merge.features], axis=0)

    # Sort by timestamp
    order = np.argsort(timestamps)
    return SpikeUnit(
        unit_id=u_keep.unit_id,
        waveforms=waveforms[order],
        timestamps=timestamps[order],
        channel_ids=channel_ids[order],
        features=features[order] if features is not None else None,
        quality_metrics={},  # Will be recomputed
    )
