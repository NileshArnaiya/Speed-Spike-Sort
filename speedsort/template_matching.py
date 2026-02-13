"""Template matching / deconvolution: iterative template subtraction to resolve overlapping spikes."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from speedsort.config import SpikeSortingConfiguration
from speedsort.types import SpikeUnit

logger = logging.getLogger("speedsort")


def template_match_and_subtract(
    filtered_data: np.ndarray,
    units: List[SpikeUnit],
    config: SpikeSortingConfiguration,
    max_iterations: int = 2,
    residual_threshold_factor: float = 0.7,
) -> Tuple[List[SpikeUnit], np.ndarray]:
    """Iterative template matching / deconvolution.

    Algorithm (per iteration):
        1. Build a mean-waveform template for each unit (on all channels).
        2. Subtract each unit's template from the data at its spike times
           to produce a *residual*.
        3. Re-detect spikes in the residual (using a lower threshold).
        4. For each new spike, find the best-matching template (min L2 norm).
        5. Assign the new spike to that unit (if fit is good enough).
        6. The residual becomes the new data for the next iteration.

    This recovers overlapping spikes that the initial detection missed
    because two neurons fired near-simultaneously.

    Args:
        filtered_data: (samples, channels) filtered recording.
        units: initial units from clustering.
        config: pipeline configuration.
        max_iterations: number of deconvolution passes.
        residual_threshold_factor: multiply the normal detection threshold
            by this factor when re-detecting on the residual (lower = more
            sensitive).

    Returns:
        (updated_units, residual): units with any newly assigned spikes,
        and the final residual trace.
    """
    if len(units) == 0:
        return units, filtered_data

    # Ensure numpy — template matching uses pure numpy operations
    try:
        import torch as _torch
        if isinstance(filtered_data, _torch.Tensor):
            filtered_data = filtered_data.cpu().numpy()
    except ImportError:
        pass
    try:
        import cupy as _cp
        if isinstance(filtered_data, _cp.ndarray):
            filtered_data = _cp.asnumpy(filtered_data)
    except ImportError:
        pass

    if not isinstance(filtered_data, np.ndarray):
        filtered_data = np.asarray(filtered_data)

    if filtered_data.size == 0:
        return units, filtered_data

    window_start, window_end = config.waveform_extraction_window
    window_len = window_end - window_start
    n_channels = filtered_data.shape[1]
    n_samples = filtered_data.shape[0]

    residual = filtered_data.copy()

    # Precompute unit templates — (n_units, window_len, n_channels)
    templates: List[np.ndarray] = []
    for unit in units:
        if unit.waveforms is not None and len(unit.waveforms) > 0:
            if unit.waveforms.ndim == 3:
                templates.append(np.mean(unit.waveforms, axis=0))
            else:
                templates.append(np.zeros((window_len, n_channels)))
        else:
            templates.append(np.zeros((window_len, n_channels)))

    # Step 1: Subtract known spikes from residual
    for uid, unit in enumerate(units):
        template = templates[uid]
        for t in unit.timestamps:
            t_start = int(t) + window_start
            t_end = int(t) + window_end
            if 0 <= t_start and t_end <= n_samples:
                seg = residual[t_start:t_end, :]
                actual_len = seg.shape[0]
                if actual_len == template.shape[0]:
                    residual[t_start:t_end, :] -= template

    total_recovered = 0

    for iteration in range(max_iterations):
        # Step 2: Re-detect spikes in residual with lower threshold
        new_spikes = _detect_residual_spikes(
            residual, config, residual_threshold_factor
        )

        if len(new_spikes) == 0:
            logger.info(f"Template matching iter {iteration + 1}: no new spikes")
            break

        # Step 3: Match each new spike to the best template
        n_assigned = 0
        for spike_time, spike_ch in new_spikes:
            t_start = spike_time + window_start
            t_end = spike_time + window_end
            if t_start < 0 or t_end > n_samples:
                continue

            snippet = residual[t_start:t_end, :]
            if snippet.shape[0] != window_len:
                continue

            # Find best matching template
            best_uid = -1
            best_dist = np.inf
            for uid, template in enumerate(templates):
                if template.shape != snippet.shape:
                    continue
                dist = np.sum((snippet - template) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_uid = uid

            if best_uid < 0:
                continue

            # Accept if fit quality is reasonable:
            # distance should be < 50 % of template energy
            template_energy = np.sum(templates[best_uid] ** 2) + 1e-12
            if best_dist / template_energy < 1.0:
                # Assign spike to unit
                unit = units[best_uid]
                new_wf = snippet[np.newaxis, :, :]  # (1, window_len, n_ch)
                unit.waveforms = np.concatenate([unit.waveforms, new_wf], axis=0)
                unit.timestamps = np.append(unit.timestamps, spike_time)
                unit.channel_ids = np.append(unit.channel_ids, spike_ch)

                # Subtract from residual
                residual[t_start:t_end, :] -= templates[best_uid]
                n_assigned += 1

        total_recovered += n_assigned
        logger.info(
            f"Template matching iter {iteration + 1}: "
            f"detected {len(new_spikes)} candidates, assigned {n_assigned}"
        )

        if n_assigned == 0:
            break

    # Re-sort timestamps within each unit
    for unit in units:
        order = np.argsort(unit.timestamps)
        unit.timestamps = unit.timestamps[order]
        unit.waveforms = unit.waveforms[order]
        unit.channel_ids = unit.channel_ids[order]

    logger.info(f"Template matching recovered {total_recovered} overlapping spikes total")
    return units, residual


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_residual_spikes(
    residual: np.ndarray,
    config: SpikeSortingConfiguration,
    threshold_factor: float,
) -> List[Tuple[int, int]]:
    """Simple threshold detection on the residual with a reduced threshold.

    Returns list of (spike_time, channel) tuples.
    """
    spikes: List[Tuple[int, int]] = []
    threshold = config.detection_threshold * threshold_factor

    window_start, window_end = config.waveform_extraction_window
    left_margin = abs(window_start)
    right_margin = window_end
    min_spacing = max(30, int(0.001 * (config.sampling_rate or 30000)))

    for ch in range(residual.shape[1]):
        channel_data = residual[:, ch]
        noise_level = np.median(np.abs(channel_data)) / 0.6745
        thresh_val = threshold * noise_level

        crossings = np.where(channel_data < -thresh_val)[0]
        crossings = crossings[
            (crossings >= left_margin) & (crossings < len(channel_data) - right_margin)
        ]

        # Enforce minimum spacing
        if len(crossings) > 0:
            mask = np.concatenate(([True], np.diff(crossings) > min_spacing))
            crossings = crossings[mask]

        for t in crossings:
            spikes.append((int(t), ch))

    return spikes
