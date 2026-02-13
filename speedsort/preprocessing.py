"""Preprocessing: bandpass/notch filtering, CAR, whitening, bad channel detection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import signal

from speedsort.config import (
    SpikeSortingConfiguration,
    _HAS_CUPY,
    _HAS_TORCH,
)

logger = logging.getLogger("speedsort")

# Conditional imports
if _HAS_CUPY:
    import cupy as cp
if _HAS_TORCH:
    import torch


# ---------------------------------------------------------------------------
# Bandpass + notch
# ---------------------------------------------------------------------------

def filter_data(data: np.ndarray, config: SpikeSortingConfiguration, xp, device=None) -> np.ndarray:
    """Apply bandpass and optional notch filters. Input/output shape: (samples, channels)."""
    # Convert to proper array type for backend
    if isinstance(xp, type(np)) and not isinstance(data, np.ndarray):
        data = np.array(data)
    elif _HAS_CUPY and xp is cp and not isinstance(data, cp.ndarray):
        data = cp.array(data)
    elif _HAS_TORCH and xp is torch and not isinstance(data, torch.Tensor):
        data = torch.tensor(data, device=device, dtype=torch.float32)

    nyquist = config.sampling_rate / 2
    low_cut = config.filter_low / nyquist
    high_cut = config.filter_high / nyquist

    # NumPy backend --------------------------------------------------------
    if xp is np:
        filtered_data = np.zeros_like(data)
        b, a = signal.butter(config.filter_order, [low_cut, high_cut], btype=config.filter_type)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = signal.filtfilt(b, a, data[:, ch])

        if config.notch_filter:
            for freq in [50, 60]:
                notch_low = (freq - 2) / nyquist
                notch_high = (freq + 2) / nyquist
                if notch_high < 1:
                    b_notch, a_notch = signal.butter(config.filter_order, [notch_low, notch_high], btype='bandstop')
                    for ch in range(data.shape[1]):
                        filtered_data[:, ch] = signal.filtfilt(b_notch, a_notch, filtered_data[:, ch])

    # PyTorch backend ------------------------------------------------------
    elif _HAS_TORCH and xp is torch:
        import torch.nn.functional as F

        kernel_size = 2 * config.filter_order + 1
        t = torch.linspace(-config.filter_order, config.filter_order, kernel_size, device=device)

        low_freq = config.filter_low
        high_freq = config.filter_high

        low_kernel = 2 * low_cut * torch.sinc(2 * low_freq * t)
        high_kernel = 2 * high_cut * torch.sinc(2 * high_freq * t)
        bandpass_kernel = high_kernel - low_kernel

        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(kernel_size, device=device) / (kernel_size - 1))
        bandpass_kernel = bandpass_kernel * window
        bandpass_kernel = bandpass_kernel / torch.sum(bandpass_kernel)
        bandpass_kernel = bandpass_kernel.view(1, 1, -1)

        data_input = data.permute(1, 0).unsqueeze(1)  # (channels, 1, samples)
        filtered_data = F.conv1d(data_input, bandpass_kernel, padding=config.filter_order)
        filtered_data = filtered_data.squeeze(1).permute(1, 0)  # Back to (samples, channels)

    # CuPy backend ---------------------------------------------------------
    elif _HAS_CUPY and xp is cp:
        filtered_data = cp.zeros_like(data)
        b, a = signal.butter(config.filter_order, [low_cut, high_cut], btype=config.filter_type)
        for ch in range(data.shape[1]):
            channel_data = cp.asnumpy(data[:, ch])
            filtered_channel = signal.filtfilt(b, a, channel_data)
            filtered_data[:, ch] = cp.array(filtered_channel)

        if config.notch_filter:
            for freq in [50, 60]:
                notch_low = (freq - 2) / nyquist
                notch_high = (freq + 2) / nyquist
                if notch_high < 1:
                    b_notch, a_notch = signal.butter(config.filter_order, [notch_low, notch_high], btype='bandstop')
                    for ch in range(data.shape[1]):
                        channel_data = cp.asnumpy(filtered_data[:, ch])
                        filtered_channel = signal.filtfilt(b_notch, a_notch, channel_data)
                        filtered_data[:, ch] = cp.array(filtered_channel)

    else:
        logger.warning(f"Unrecognized backend {type(xp)}, falling back to NumPy")
        return filter_data(data, config, np, device)

    return filtered_data


# ---------------------------------------------------------------------------
# Common Average / Median Reference
# ---------------------------------------------------------------------------

def apply_common_reference(data: np.ndarray, config: SpikeSortingConfiguration, xp, device=None) -> np.ndarray:
    """Subtract common average (or median) reference across channels per time sample."""
    if isinstance(data, np.ndarray):
        if config.common_reference_type == 'median':
            ref = np.median(data, axis=1, keepdims=True)
        else:
            ref = np.mean(data, axis=1, keepdims=True)
        return data - ref
    elif _HAS_CUPY and xp is cp:
        data_np = cp.asnumpy(data)
        ref = np.median(data_np, axis=1, keepdims=True) if config.common_reference_type == 'median' else np.mean(data_np, axis=1, keepdims=True)
        return cp.array(data_np - ref)
    elif _HAS_TORCH and xp is torch:
        if config.common_reference_type == 'median':
            ref = torch.median(data, dim=1, keepdim=True).values
        else:
            ref = torch.mean(data, dim=1, keepdim=True)
        return data - ref
    return data


# ---------------------------------------------------------------------------
# Bad channel detection & interpolation
# ---------------------------------------------------------------------------

def detect_and_fix_bad_channels(data: np.ndarray, config: SpikeSortingConfiguration, xp, device=None) -> np.ndarray:
    """Detect dead/noisy channels via variance analysis and interpolate from neighbors.

    Bad channels are identified using MAD-based z-score on log-variance.
    Dead channels (near-zero variance) are also flagged.
    Bad channels are replaced by the mean of their nearest non-bad neighbors.
    """
    if isinstance(data, np.ndarray):
        data_np = data
    elif _HAS_CUPY and xp is cp:
        data_np = cp.asnumpy(data)
    elif _HAS_TORCH and xp is torch:
        data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)

    n_channels = data_np.shape[1]
    if n_channels < 3:
        return data  # Need at least 3 channels for meaningful detection

    # Compute variance per channel
    channel_vars = np.var(data_np, axis=0)

    # MAD-based detection: channels whose log-variance deviates from median
    log_vars = np.log1p(channel_vars)
    median_lv = np.median(log_vars)
    mad_lv = np.median(np.abs(log_vars - median_lv))
    if mad_lv < 1e-10:
        mad_lv = 1e-10

    z_scores = np.abs(log_vars - median_lv) / mad_lv
    bad_mask = z_scores > config.bad_channel_std_threshold

    # Also flag dead channels (near-zero variance)
    dead_mask = channel_vars < np.median(channel_vars) * 0.01
    bad_mask = bad_mask | dead_mask

    bad_indices = np.where(bad_mask)[0]
    if len(bad_indices) == 0:
        logger.info("No bad channels detected")
        return data

    if len(bad_indices) >= n_channels - 1:
        logger.warning(f"Too many bad channels detected ({len(bad_indices)}/{n_channels}), skipping correction")
        return data

    logger.info(f"Bad channels detected: {bad_indices.tolist()} ({len(bad_indices)}/{n_channels})")

    # Interpolate: replace each bad channel with mean of nearest good neighbors
    good_indices = np.where(~bad_mask)[0]
    result = data_np.copy()

    for bad_ch in bad_indices:
        distances = np.abs(good_indices - bad_ch)
        n_neighbors = min(2, len(good_indices))
        nearest = good_indices[np.argsort(distances)[:n_neighbors]]
        result[:, bad_ch] = np.mean(data_np[:, nearest], axis=1)

    if _HAS_CUPY and xp is cp:
        return cp.array(result)
    elif _HAS_TORCH and xp is torch:
        return torch.tensor(result, device=device, dtype=torch.float32)
    return result


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------

def apply_whitening(data: np.ndarray, config: SpikeSortingConfiguration, xp, device=None) -> np.ndarray:
    """Apply ZCA whitening: decorrelate channels and equalize noise variance."""
    if _HAS_CUPY and xp is cp:
        data_np = cp.asnumpy(data).astype(np.float64)
    elif _HAS_TORCH and xp is torch:
        data_np = data.cpu().numpy().astype(np.float64)
    else:
        data_np = data.astype(np.float64)

    # Estimate covariance from random chunks (cap at 100k samples for speed)
    n_samples = data_np.shape[0]
    if n_samples > 100000:
        idx = np.random.choice(n_samples, 100000, replace=False)
        sample = data_np[idx]
    else:
        sample = data_np

    cov = np.cov(sample.T)
    if cov.ndim < 2:
        return data  # Single channel, nothing to whiten

    # ZCA whitening: W = cov^(-1/2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Regularize: clamp small eigenvalues to 1% of the largest
    reg_floor = max(np.max(eigvals) * 0.01, 1e-6)
    eigvals = np.maximum(eigvals, reg_floor)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T

    whitened = data_np @ W.T

    # Guard against numerical issues
    if np.any(~np.isfinite(whitened)):
        logger.warning("Whitening produced non-finite values, skipping whitening step")
        return data

    if _HAS_CUPY and xp is cp:
        return cp.array(whitened.astype(np.float32))
    elif _HAS_TORCH and xp is torch:
        return torch.tensor(whitened, device=device, dtype=torch.float32)
    return whitened.astype(np.float32)
