"""Enums and configuration dataclass for the spike sorting pipeline."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger("speedsort")

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
_HAS_TORCH = False
_HAS_CUPY = False
_HAS_GPU = False
_HAS_SKLEARN = False
_HAS_MNE = False

try:
    import torch
    _HAS_TORCH = True
    _HAS_GPU = torch.cuda.is_available()
    if _HAS_GPU:
        logger.info(f"GPU acceleration available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("PyTorch available but no GPU detected")
except ImportError:
    pass

try:
    import cupy as cp
    _HAS_CUPY = True
    logger.info(f"CuPy available with CUDA {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    logger.info("CuPy not available - using NumPy for array operations")

try:
    import sklearn
    _HAS_SKLEARN = True
except ImportError:
    logger.warning("scikit-learn not available - some features will be limited")

try:
    import mne
    _HAS_MNE = True
except ImportError:
    logger.info("MNE-Python not found - some data importers may be limited")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataFormat(str, Enum):
    """Supported electrophysiology data formats."""
    NUMPY = 'numpy'
    NEO = 'neo'
    SIG = 'sig'
    OPEN_EPHYS = 'open_ephys'
    MEA = 'mea'
    NWB = 'nwb'
    MDA = 'mda'
    BINARY = 'binary'
    AUTO = 'auto'


class ProcessingBackend(str, Enum):
    """Enumeration of possible processing backends."""
    CPU = 'cpu'
    NUMPY = 'numpy'
    TORCH_CPU = 'torch_cpu'
    TORCH_GPU = 'torch_gpu'
    CUPY = 'cupy'

    @classmethod
    def get_optimal(cls) -> "ProcessingBackend":
        """Determine the optimal backend based on available hardware/libraries."""
        if _HAS_GPU and _HAS_TORCH:
            return cls.TORCH_GPU
        if _HAS_CUPY:
            return cls.CUPY
        if _HAS_TORCH:
            return cls.TORCH_CPU
        return cls.NUMPY


class DetectionMethod(str, Enum):
    """Spike detection method options."""
    THRESHOLD = 'threshold'
    THRESHOLD_DYNAMIC = 'threshold_dynamic'
    TEMPLATE = 'template'
    NEO = 'neo'
    WAVELET = 'wavelet'
    NEURAL_NET = 'neural_net'


class ClusteringMethod(str, Enum):
    """Clustering method options."""
    KMEANS = 'kmeans'
    GMM = 'gmm'
    HDBSCAN = 'hdbscan'
    MEANSHIFT = 'meanshift'
    DBSCAN = 'dbscan'
    ISO_FOREST = 'isoforest'
    AGGLO = 'agglomerative'
    AUTO = 'auto'


class DimensionalityReduction(str, Enum):
    """Dimensionality reduction method options."""
    PCA = 'pca'
    TSNE = 'tsne'
    UMAP = 'umap'
    WAVELET = 'wavelet'
    NONE = 'none'
    AUTO = 'auto'


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpikeSortingConfiguration:
    """Complete configuration for spike sorting process."""
    # Input settings
    data_format: DataFormat = DataFormat.AUTO
    sampling_rate: Optional[float] = None  # Hz, will be auto-detected if possible
    n_channels: Optional[int] = None       # Will be auto-detected

    # Processing settings
    backend: ProcessingBackend = ProcessingBackend.get_optimal()
    n_jobs: int = max(1, mp.cpu_count() - 1)  # Default to all but one core
    chunk_size: Optional[int] = None           # Auto-determined if None
    use_gpu: bool = _HAS_GPU
    temp_dir: Optional[str] = None            # For temporary files

    # Filtering settings
    filter_type: str = 'bandpass'
    filter_low: float = 300.0                 # Hz
    filter_high: float = 6000.0               # Hz
    filter_order: int = 3                     # Filter order
    notch_filter: bool = True                 # Apply notch filter at 50/60 Hz

    # Detection settings
    detection_method: DetectionMethod = DetectionMethod.THRESHOLD_DYNAMIC
    detection_threshold: float = 4.5          # In terms of MAD
    alignment_window: Tuple[int, int] = (-10, 30)  # Samples around threshold crossing

    # Feature extraction settings
    waveform_extraction_window: Tuple[int, int] = (-40, 41)  # Samples around spike
    dim_reduction: DimensionalityReduction = DimensionalityReduction.AUTO
    n_components: int = 10                    # For dimensionality reduction
    feature_normalize: bool = True            # Normalize features

    # Clustering settings
    clustering_method: ClusteringMethod = ClusteringMethod.AUTO
    max_clusters: int = 50                    # Maximum number of clusters to consider
    min_cluster_size: int = 30                # Minimum spikes per cluster

    # Preprocessing — applied after bandpass filtering (requires ≥4 channels)
    detect_bad_channels: bool = True          # Detect and interpolate dead/noisy channels
    bad_channel_std_threshold: float = 5.0    # Channels outside this many MADs from median variance are bad
    common_reference: bool = True             # Apply common average/median reference
    common_reference_type: str = 'median'     # 'median' or 'mean'
    whiten: bool = True                       # Apply whitening transform
    min_channels_for_preprocessing: int = 4   # Skip CAR/whitening/bad-ch if fewer channels

    # Quality metrics and validation
    compute_quality_metrics: bool = True      # Calculate quality metrics
    isolation_threshold: float = 0.9          # Minimum isolation score to accept
    contamination_threshold: float = 0.1      # Maximum contamination to accept

    # P3: Curation — auto-merge + noise rejection (after quality metrics)
    auto_merge: bool = True                   # Merge units with similar templates
    merge_template_threshold: float = 0.92    # Cosine similarity above which to merge
    noise_rejection: bool = True              # Remove noise/artefact units
    noise_isi_threshold: float = 0.05         # Max ISI violation ratio (5 %)
    noise_snr_threshold: float = 1.5          # Min SNR to keep a unit
    noise_firing_rate_bounds: Tuple[float, float] = (0.1, 200.0)  # Hz

    # P3: Template matching / deconvolution
    template_matching: bool = True            # Iterative template subtraction
    template_matching_iterations: int = 2     # Max deconvolution passes
    template_residual_threshold: float = 0.7  # Threshold factor for residual detection

    # P3: Multi-channel spatial features
    use_spatial_features: bool = True         # Append spatial spread features
    n_neighbor_channels: int = 3              # Neighbors each side for spatial features

    # Output settings
    save_filtered_data: bool = False          # Save filtered data
    save_waveforms: bool = True               # Save extracted waveforms
    save_features: bool = True                # Save computed features
    output_format: str = 'numpy'              # Output format (numpy, csv, etc.)
