"""Spike detection: threshold, dynamic threshold, NEO, wavelet, neural net."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from speedsort.config import (
    DetectionMethod,
    SpikeSortingConfiguration,
    _HAS_CUPY,
    _HAS_TORCH,
)

logger = logging.getLogger("speedsort")

if _HAS_CUPY:
    import cupy as cp
if _HAS_TORCH:
    import torch


def detect_spikes(
    filtered_data: np.ndarray,
    config: SpikeSortingConfiguration,
    xp,
    device=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect spikes per channel. Returns (spike_times, spike_channels) arrays."""
    detection_method = config.detection_method
    threshold = config.detection_threshold

    spike_times_list = []
    spike_channels_list = []

    for ch in range(filtered_data.shape[1]):
        channel_data = filtered_data[:, ch]

        # Compute noise level
        if xp is np:
            noise_level = np.median(np.abs(channel_data)) / 0.6745
        elif _HAS_CUPY and xp is cp:
            noise_level = float(cp.median(cp.abs(channel_data))) / 0.6745
        elif _HAS_TORCH and xp is torch:
            noise_level = torch.median(torch.abs(channel_data)).item() / 0.6745
        else:
            noise_level = np.median(np.abs(np.array(channel_data))) / 0.6745

        threshold_value = threshold * noise_level

        # ----- Detect crossings based on method -----
        crossings = _detect_crossings(
            channel_data, detection_method, threshold, threshold_value,
            config, xp, device
        )

        # Ensure we're not too close to edges for waveform extraction
        alignment_window = config.alignment_window
        extraction_window = config.waveform_extraction_window
        left_margin = max(abs(alignment_window[0]), abs(extraction_window[0]))
        right_margin = max(alignment_window[1], extraction_window[1])

        valid_crossings = crossings[
            (crossings >= left_margin) & (crossings < len(channel_data) - right_margin)
        ]

        # Align spikes to their negative peak
        aligned_times = []
        for t in valid_crossings:
            spike_window = slice(t + alignment_window[0], t + alignment_window[1])
            if xp is np:
                window_data = channel_data[spike_window]
                peak_offset = np.argmin(window_data)
            elif _HAS_CUPY and xp is cp:
                window_data = channel_data[spike_window]
                peak_offset = cp.argmin(window_data).get()
            elif _HAS_TORCH and xp is torch:
                window_data = channel_data[spike_window]
                peak_offset = torch.argmin(window_data).item()
            else:
                window_data = np.array(channel_data[spike_window])
                peak_offset = np.argmin(window_data)

            aligned_time = t + alignment_window[0] + peak_offset
            if aligned_time >= left_margin and aligned_time < len(channel_data) - right_margin:
                aligned_times.append(aligned_time)

        if aligned_times:
            spike_times_list.extend(aligned_times)
            spike_channels_list.extend([ch] * len(aligned_times))

    # Convert to arrays and sort by time
    if spike_times_list:
        spike_times = np.array(spike_times_list)
        spike_channels = np.array(spike_channels_list)
        sort_idx = np.argsort(spike_times)
        spike_times = spike_times[sort_idx]
        spike_channels = spike_channels[sort_idx]
    else:
        spike_times = np.array([], dtype=np.int64)
        spike_channels = np.array([], dtype=np.int64)

    return spike_times, spike_channels


# ---------------------------------------------------------------------------
# Per-method crossing detection (internal)
# ---------------------------------------------------------------------------

def _detect_crossings(channel_data, method, threshold, threshold_value, config, xp, device):
    """Return an array of crossing indices for one channel."""

    if method == DetectionMethod.THRESHOLD:
        return _threshold_crossings(channel_data, threshold_value, xp)

    elif method == DetectionMethod.THRESHOLD_DYNAMIC:
        return _dynamic_threshold_crossings(channel_data, threshold, config, xp)

    elif method == DetectionMethod.NEO:
        return _neo_crossings(channel_data, threshold, config, xp, device)

    elif method == DetectionMethod.WAVELET:
        return _wavelet_crossings(channel_data, threshold, threshold_value, config, xp)

    elif method == DetectionMethod.NEURAL_NET:
        return _neural_net_crossings(channel_data, threshold_value, config, xp, device)

    else:
        return _threshold_crossings(channel_data, threshold_value, xp)


def _threshold_crossings(channel_data, threshold_value, xp):
    if xp is np:
        return np.where(channel_data < -threshold_value)[0]
    elif _HAS_CUPY and xp is cp:
        return cp.where(channel_data < -threshold_value)[0].get()
    elif _HAS_TORCH and xp is torch:
        return torch.where(channel_data < -threshold_value)[0].cpu().numpy()
    return np.where(np.array(channel_data) < -threshold_value)[0]


def _dynamic_threshold_crossings(channel_data, threshold, config, xp):
    window_size = int(1.0 * config.sampling_rate)

    if xp is np:
        thresholds = np.zeros_like(channel_data)
        for i in range(0, len(channel_data), window_size):
            window_end = min(i + window_size, len(channel_data))
            window_data = channel_data[i:window_end]
            window_noise = np.median(np.abs(window_data)) / 0.6745
            thresholds[i:window_end] = threshold * window_noise
        crossings = np.where(channel_data < -thresholds)[0]

    elif _HAS_CUPY and xp is cp:
        thresholds = cp.zeros_like(channel_data)
        for i in range(0, len(channel_data), window_size):
            window_end = min(i + window_size, len(channel_data))
            window_data = channel_data[i:window_end]
            window_noise = float(cp.median(cp.abs(window_data))) / 0.6745
            thresholds[i:window_end] = threshold * window_noise
        crossings = cp.where(channel_data < -thresholds)[0].get()

    elif _HAS_TORCH and xp is torch:
        thresholds = torch.zeros_like(channel_data)
        for i in range(0, len(channel_data), window_size):
            window_end = min(i + window_size, len(channel_data))
            window_data = channel_data[i:window_end]
            window_noise = torch.median(torch.abs(window_data)).item() / 0.6745
            thresholds[i:window_end] = threshold * window_noise
        crossings = torch.where(channel_data < -thresholds)[0].cpu().numpy()
    else:
        return np.array([], dtype=np.int64)

    # Remove duplicates (minimum spacing between spikes)
    min_spacing = int(0.001 * config.sampling_rate)
    if len(crossings) > 0:
        mask = np.diff(crossings) > min_spacing
        mask = np.concatenate(([True], mask))
        crossings = crossings[mask]

    return crossings


def _neo_crossings(channel_data, threshold, config, xp, device):
    """Nonlinear Energy Operator: y[n] = x[n]^2 - x[n-1]*x[n+1]."""
    if xp is np:
        energy = np.zeros_like(channel_data)
        energy[1:-1] = channel_data[1:-1]**2 - channel_data[0:-2] * channel_data[2:]
        kernel_size = int(0.0005 * config.sampling_rate)
        if kernel_size % 2 == 0:
            kernel_size += 1
        energy = np.convolve(energy, np.ones(kernel_size)/kernel_size, mode='same')
        neo_threshold = threshold * np.median(np.abs(energy)) / 0.6745
        crossings = np.where(energy > neo_threshold)[0]

    elif _HAS_CUPY and xp is cp:
        energy = cp.zeros_like(channel_data)
        energy[1:-1] = channel_data[1:-1]**2 - channel_data[0:-2] * channel_data[2:]
        energy_np = cp.asnumpy(energy)
        kernel_size = int(0.0005 * config.sampling_rate)
        if kernel_size % 2 == 0:
            kernel_size += 1
        energy_np = np.convolve(energy_np, np.ones(kernel_size)/kernel_size, mode='same')
        energy = cp.array(energy_np)
        neo_threshold = threshold * float(cp.median(cp.abs(energy))) / 0.6745
        crossings = cp.where(energy > neo_threshold)[0].get()

    elif _HAS_TORCH and xp is torch:
        energy = torch.zeros_like(channel_data)
        energy[1:-1] = channel_data[1:-1]**2 - channel_data[0:-2] * channel_data[2:]
        import torch.nn.functional as F
        kernel_size = int(0.0005 * config.sampling_rate)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        energy = energy.view(1, 1, -1)
        energy = F.conv1d(energy, kernel, padding=kernel_size//2)
        energy = energy.view(-1)
        neo_threshold = threshold * torch.median(torch.abs(energy)).item() / 0.6745
        crossings = torch.where(energy > neo_threshold)[0].cpu().numpy()
    else:
        return np.array([], dtype=np.int64)

    # Remove duplicates
    min_spacing = int(0.001 * config.sampling_rate)
    if len(crossings) > 0:
        mask = np.diff(np.concatenate(([0], crossings))) > min_spacing
        crossings = crossings[mask]

    return crossings


def _wavelet_crossings(channel_data, threshold, threshold_value, config, xp):
    """Wavelet-based detection."""
    try:
        import pywt

        if xp is np:
            coeffs = pywt.wavedec(channel_data, 'sym7', level=4)
        elif _HAS_CUPY and xp is cp:
            coeffs = pywt.wavedec(cp.asnumpy(channel_data), 'sym7', level=4)
        else:
            coeffs = pywt.wavedec(channel_data.cpu().numpy(), 'sym7', level=4)

        detail_coeffs = coeffs[1]
        wavelet_threshold = threshold * np.median(np.abs(detail_coeffs)) / 0.6745
        significant_indices = np.where(np.abs(detail_coeffs) > wavelet_threshold)[0]

        scale_factor = len(channel_data) / len(detail_coeffs)
        crossings = np.unique(np.floor(significant_indices * scale_factor).astype(int))
        crossings = crossings[crossings < len(channel_data)]

        min_spacing = int(0.001 * config.sampling_rate)
        if len(crossings) > 0:
            mask = np.diff(np.concatenate(([0], crossings))) > min_spacing
            crossings = crossings[mask]

        return crossings

    except ImportError:
        logger.warning("PyWavelets not available, falling back to threshold detection")
        return _threshold_crossings(channel_data, threshold_value, xp)


def _neural_net_crossings(channel_data, threshold_value, config, xp, device):
    """Neural network-based detection — requires PyTorch."""
    if not _HAS_TORCH:
        logger.warning("PyTorch not available, falling back to threshold detection")
        return _threshold_crossings(channel_data, threshold_value, xp)

    try:
        model_path = Path("models/spike_detector.pt")
        if not model_path.exists():
            logger.warning("Neural network model not found, falling back to threshold detection")
            return _threshold_crossings(channel_data, threshold_value, xp)

        model = torch.load(model_path)
        model.eval()

        if not isinstance(channel_data, torch.Tensor):
            channel_data_tensor = torch.tensor(channel_data.astype(np.float32) if isinstance(channel_data, np.ndarray) else np.array(channel_data, dtype=np.float32), device=device)
        else:
            channel_data_tensor = channel_data

        window_size = 64
        stride = 32
        windows = []
        for i in range(0, len(channel_data_tensor) - window_size, stride):
            windows.append(channel_data_tensor[i:i+window_size])

        if windows:
            windows = torch.stack(windows)
            windows = (windows - windows.mean(dim=1, keepdim=True)) / (windows.std(dim=1, keepdim=True) + 1e-8)

            batch_size = 1024
            predictions = []
            with torch.no_grad():
                for i in range(0, len(windows), batch_size):
                    batch = windows[i:i+batch_size]
                    preds = model(batch)
                    predictions.append(preds)

            predictions = torch.cat(predictions)
            spike_windows = torch.where(predictions > 0.5)[0].cpu().numpy()
            return stride * spike_windows + window_size // 2
        return np.array([], dtype=np.int64)

    except Exception as e:
        logger.error(f"Error in neural network spike detection: {e}")
        logger.warning("Falling back to threshold detection")
        return _threshold_crossings(channel_data, threshold_value, xp)
