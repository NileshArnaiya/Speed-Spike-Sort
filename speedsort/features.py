"""Feature extraction: waveform cutting, PCA, t-SNE, UMAP, wavelet features."""

from __future__ import annotations

import logging
from typing import Union

import numpy as np

from speedsort.config import (
    DimensionalityReduction,
    SpikeSortingConfiguration,
    _HAS_CUPY,
    _HAS_SKLEARN,
    _HAS_TORCH,
)

logger = logging.getLogger("speedsort")

if _HAS_CUPY:
    import cupy as cp
if _HAS_TORCH:
    import torch
if _HAS_SKLEARN:
    import sklearn.decomposition
    import sklearn.preprocessing


# ---------------------------------------------------------------------------
# Waveform extraction
# ---------------------------------------------------------------------------

def extract_waveforms(
    filtered_data: np.ndarray,
    spike_times: np.ndarray,
    spike_channels: np.ndarray,
    config: SpikeSortingConfiguration,
    xp,
    device=None,
) -> np.ndarray:
    """Cut waveform snippets around each spike. Returns array (n_spikes, n_samples, n_channels)."""
    if len(spike_times) == 0:
        return np.array([])

    window_start, window_end = config.waveform_extraction_window
    window_length = window_end - window_start
    n_channels = filtered_data.shape[1]

    # Initialize waveform array
    if xp is np:
        waveforms = np.zeros((len(spike_times), window_length, n_channels))
    elif _HAS_CUPY and xp is cp:
        waveforms = cp.zeros((len(spike_times), window_length, n_channels))
    elif _HAS_TORCH and xp is torch:
        waveforms = torch.zeros((len(spike_times), window_length, n_channels), device=device)
    else:
        waveforms = np.zeros((len(spike_times), window_length, n_channels))

    for i, (spike_time, spike_channel) in enumerate(zip(spike_times, spike_channels)):
        t_start = spike_time + window_start
        t_end = spike_time + window_end
        if t_start >= 0 and t_end <= filtered_data.shape[0]:
            waveforms[i, :, :] = filtered_data[t_start:t_end, :]

    # Convert to NumPy for consistent output
    if _HAS_CUPY and xp is cp:
        waveforms = cp.asnumpy(waveforms)
    elif _HAS_TORCH and xp is torch:
        waveforms = waveforms.cpu().numpy()

    return waveforms


# ---------------------------------------------------------------------------
# Feature computation (dimensionality reduction)
# ---------------------------------------------------------------------------

def compute_features(
    waveforms: np.ndarray,
    config: SpikeSortingConfiguration,
    xp,
    device=None,
) -> np.ndarray:
    """Reduce waveforms to feature vectors via PCA, t-SNE, UMAP, or wavelets. Returns (n_spikes, n_features)."""
    if len(waveforms) == 0:
        return np.array([])

    n_spikes = waveforms.shape[0]

    # Convert to backend format
    if _HAS_CUPY and xp is cp and not isinstance(waveforms, cp.ndarray):
        waveforms_xp = cp.array(waveforms)
    elif _HAS_TORCH and xp is torch and not isinstance(waveforms, torch.Tensor):
        waveforms_xp = torch.tensor(waveforms, device=device)
    else:
        waveforms_xp = waveforms

    dim_reduction = config.dim_reduction

    # Auto selection based on data size
    if dim_reduction == DimensionalityReduction.AUTO:
        if n_spikes < 1000:
            dim_reduction = DimensionalityReduction.PCA
        elif n_spikes < 10000:
            dim_reduction = DimensionalityReduction.PCA if _HAS_SKLEARN else DimensionalityReduction.WAVELET
        else:
            dim_reduction = DimensionalityReduction.WAVELET

    # Reshape: (n_spikes, n_samples * n_channels)
    n_samples = waveforms_xp.shape[1]
    n_channels = waveforms_xp.shape[2]
    if xp is np or (_HAS_CUPY and xp is cp):
        X = waveforms_xp.reshape(n_spikes, n_samples * n_channels)
    elif _HAS_TORCH and xp is torch:
        X = waveforms_xp.reshape(n_spikes, n_samples * n_channels)
    else:
        X = np.array(waveforms_xp).reshape(n_spikes, n_samples * n_channels)

    # ----- Apply selected reduction -----
    if dim_reduction == DimensionalityReduction.PCA:
        features = _pca_features(X, config, xp)
    elif dim_reduction == DimensionalityReduction.TSNE:
        features = _tsne_features(X, config, xp)
    elif dim_reduction == DimensionalityReduction.UMAP:
        features = _umap_features(X, config, xp)
    elif dim_reduction == DimensionalityReduction.WAVELET:
        features = _compute_wavelet_features(waveforms_xp, config, xp)
    else:
        features = _raw_subsampled_features(X, config, xp)

    if not isinstance(features, np.ndarray):
        features = np.array(features)
    return features


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(X, xp):
    if xp is not np:
        if _HAS_CUPY and xp is cp:
            return X.get()
        elif _HAS_TORCH and xp is torch:
            return X.cpu().numpy()
    return X


def _pca_features(X, config, xp):
    if not _HAS_SKLEARN:
        logger.warning("scikit-learn not available, using wavelet features instead")
        return _compute_wavelet_features(X.reshape(X.shape[0], -1, 1), config, xp)

    X_np = _to_numpy(X, xp)
    if config.feature_normalize:
        X_np = sklearn.preprocessing.scale(X_np)
    pca = sklearn.decomposition.PCA(n_components=min(config.n_components, X_np.shape[1], X_np.shape[0]))
    features = pca.fit_transform(X_np)
    logger.info(f"PCA captured {sum(pca.explained_variance_ratio_):.2f} of variance with {config.n_components} components")
    return features


def _tsne_features(X, config, xp):
    if not _HAS_SKLEARN:
        logger.warning("scikit-learn not available, using wavelet features instead")
        return _compute_wavelet_features(X.reshape(X.shape[0], -1, 1), config, xp)

    X_np = _to_numpy(X, xp)
    if config.feature_normalize:
        X_np = sklearn.preprocessing.scale(X_np)
    if X_np.shape[1] > 50:
        pca = sklearn.decomposition.PCA(n_components=min(50, X_np.shape[1], X_np.shape[0]))
        X_np = pca.fit_transform(X_np)
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=min(config.n_components, X_np.shape[1]), random_state=42)
        return tsne.fit_transform(X_np)
    except Exception as e:
        logger.error(f"Error in t-SNE: {e}, falling back to PCA")
        return _pca_features(X, config, xp)


def _umap_features(X, config, xp):
    try:
        import umap
        X_np = _to_numpy(X, xp)
        if config.feature_normalize and _HAS_SKLEARN:
            X_np = sklearn.preprocessing.scale(X_np)
        reducer = umap.UMAP(n_components=min(config.n_components, X_np.shape[1]))
        return reducer.fit_transform(X_np)
    except ImportError:
        logger.warning("UMAP not available, using PCA instead")
        return _pca_features(X, config, xp)


def _raw_subsampled_features(X, config, xp):
    X_np = _to_numpy(X, xp)
    step = max(1, X_np.shape[1] // config.n_components)
    features = X_np[:, ::step]
    return features[:, :config.n_components]


# ---------------------------------------------------------------------------
# Wavelet features
# ---------------------------------------------------------------------------

def _compute_wavelet_features(
    waveforms: Union[np.ndarray, "cp.ndarray", "torch.Tensor"],
    config: SpikeSortingConfiguration,
    xp,
) -> np.ndarray:
    """Extract wavelet decomposition coefficients as features. Fallback: simple peak/width/energy features."""
    try:
        import pywt

        waveforms_np = _to_numpy(waveforms, xp)
        n_spikes = waveforms_np.shape[0]
        n_samples = waveforms_np.shape[1]
        n_channels = waveforms_np.shape[2]

        wavelet = 'sym5'
        level = 3
        features_list = []
        batch_size = min(5000, n_spikes)

        for i in range(0, n_spikes, batch_size):
            batch_end = min(i + batch_size, n_spikes)
            batch_features = []

            for j in range(i, batch_end):
                spike_wave = waveforms_np[j].reshape(n_samples, n_channels)
                peak_channel = np.argmax(np.max(np.abs(spike_wave), axis=0))
                peak_waveform = spike_wave[:, peak_channel]
                coeffs = pywt.wavedec(peak_waveform, wavelet, level=level)
                feature_vector = np.concatenate([coef for coef in coeffs])

                if len(feature_vector) > config.n_components:
                    step = max(1, len(feature_vector) // config.n_components)
                    feature_vector = feature_vector[::step][:config.n_components]

                batch_features.append(feature_vector)

            features_list.append(np.vstack(batch_features))

        features = np.vstack(features_list)

        if features.shape[1] > config.n_components:
            features = features[:, :config.n_components]
        elif features.shape[1] < config.n_components:
            padding = np.zeros((features.shape[0], config.n_components - features.shape[1]))
            features = np.hstack([features, padding])

        return features

    except ImportError:
        logger.warning("PyWavelets not available, using simple features instead")
        return _simple_waveform_features(waveforms, config, xp)


def _simple_waveform_features(waveforms, config, xp):
    """Fallback feature extractor using peak amplitude, width, energy."""
    waveforms_np = _to_numpy(waveforms, xp)
    n_spikes = waveforms_np.shape[0]
    n_samples = waveforms_np.shape[1]
    n_channels = waveforms_np.shape[2]

    features = np.zeros((n_spikes, config.n_components))

    for i in range(n_spikes):
        spike_wave = waveforms_np[i]
        peak_channel = np.argmax(np.max(np.abs(spike_wave), axis=0))
        peak_waveform = spike_wave[:, peak_channel]

        features[i, 0] = np.min(peak_waveform)
        features[i, 1] = np.argmin(peak_waveform)
        features[i, 2] = np.max(peak_waveform)
        features[i, 3] = np.argmax(peak_waveform)

        peak_val = np.min(peak_waveform)
        half_amp = peak_val / 2
        above_half = peak_waveform <= half_amp
        if np.any(above_half):
            cross_indices = np.where(above_half)[0]
            if len(cross_indices) >= 2:
                features[i, 4] = cross_indices[-1] - cross_indices[0]

        features[i, 5] = np.sum(peak_waveform**2)

        remaining = config.n_components - 6
        if remaining > 0:
            step = max(1, len(peak_waveform) // remaining)
            samples = peak_waveform[::step][:remaining]
            features[i, 6:6+len(samples)] = samples

    return features
