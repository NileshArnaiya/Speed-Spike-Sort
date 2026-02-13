#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spike sorting pipeline for extracellular electrophysiology recordings.

Takes a raw neural recording (numpy array, .npy, .nwb, .mda, or binary file)
and produces sorted spike units through these steps:

    1. Bandpass + notch filtering (default 300-6000 Hz)
    2. Spike detection (threshold, adaptive threshold, NEO, wavelet, or neural net)
    3. Waveform extraction around each detected spike
    4. Dimensionality reduction (PCA, t-SNE, UMAP, or wavelet coefficients)
    5. Clustering into units (K-means, GMM, DBSCAN, mean shift, etc.)
    6. Quality metrics (isolation distance, contamination rate)

Uses GPU (PyTorch/CuPy) automatically when available, otherwise falls back to NumPy.
"""

import os
import time
import warnings
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import json

# Core scientific libraries
import numpy as np
import scipy as sp
from scipy import signal, stats
import pandas as pd

# Optional dependencies - will be imported only when needed
_HAS_TORCH = False
_HAS_CUPY = False
_HAS_GPU = False
_HAS_SKLEARN = False
_HAS_MNE = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("speedsort")

# ------------------------------------------------------------------------------
# Data structures and enums
# ------------------------------------------------------------------------------

class DataFormat(Enum):
    """Supported electrophysiology data formats."""
    NUMPY = 'numpy'                   # Raw NumPy array
    NEO = 'neo'                       # Neo format
    SIG = 'sig'                       # SIG files
    OPEN_EPHYS = 'open_ephys'         # Open Ephys format
    MEA = 'mea'                       # Multi-electrode array format
    NWB = 'nwb'                       # Neurodata Without Borders
    MDA = 'mda'                       # MountainSort MDA format
    BINARY = 'binary'                 # Raw binary
    AUTO = 'auto'                     # Auto-detect format

# Check for available hardware acceleration
try:
    import torch
    _HAS_TORCH = True
    _HAS_GPU = torch.cuda.is_available()
    if _HAS_GPU:
        logger.info(f"GPU acceleration available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("PyTorch available but no GPU detected")
except ImportError:
    logger.info("PyTorch not available - CPU processing only")

try:
    import cupy as cp
    _HAS_CUPY = True
    logger.info("CuPy available for GPU-accelerated array operations")
except ImportError:
    _HAS_CUPY = False
    logger.info("CuPy not available - using NumPy for array operations")

try:
    import sklearn
    from sklearn import decomposition, cluster, preprocessing, metrics
    _HAS_SKLEARN = True
except ImportError:
    logger.warning("scikit-learn not found - advanced clustering features limited")

try:
    import mne
    _HAS_MNE = True
except ImportError:
    logger.info("MNE-Python not found - some data importers may be limited")

# ------------------------------------------------------------------------------
# Data structures and enums
# ------------------------------------------------------------------------------

class ProcessingBackend(Enum):
    """Enumeration of possible processing backends."""
    CPU = 'cpu'
    NUMPY = 'numpy'
    TORCH_CPU = 'torch_cpu'
    TORCH_GPU = 'torch_gpu'
    CUPY = 'cupy'
    
    @classmethod
    def get_optimal(cls) -> 'ProcessingBackend':
        """Determine the optimal backend based on available hardware/libraries."""
        if _HAS_GPU and _HAS_CUPY:
            return cls.CUPY
        elif _HAS_GPU and _HAS_TORCH:
            return cls.TORCH_GPU
        elif _HAS_TORCH:
            return cls.TORCH_CPU
        else:
            return cls.NUMPY


class DetectionMethod(Enum):
    """Spike detection method options."""
    THRESHOLD = 'threshold'           # Simple amplitude threshold
    THRESHOLD_DYNAMIC = 'threshold_dynamic'  # Adaptive threshold
    TEMPLATE = 'template'             # Template matching
    NEO = 'neo'                       # NEO operator
    WAVELET = 'wavelet'               # Wavelet-based detection
    NEURAL_NET = 'neural_net'         # Neural network-based detection


class ClusteringMethod(Enum):
    """Clustering method options."""
    KMEANS = 'kmeans'                 # K-means clustering
    GMM = 'gmm'                       # Gaussian Mixture Model
    HDBSCAN = 'hdbscan'               # HDBSCAN (density-based)
    MEANSHIFT = 'meanshift'           # Mean shift clustering
    DBSCAN = 'dbscan'                 # DBSCAN
    ISO_FOREST = 'isoforest'          # Isolation Forest (for outlier detection)
    AGGLO = 'agglomerative'           # Agglomerative clustering
    AUTO = 'auto'                     # Automatically select method
    
    
class DimensionalityReduction(Enum):
    """Dimensionality reduction method options."""
    PCA = 'pca'                       # Principal Component Analysis
    TSNE = 'tsne'                     # t-SNE
    UMAP = 'umap'                     # UMAP
    WAVELET = 'wavelet'               # Wavelet transform coefficients
    NONE = 'none'                     # No dimensionality reduction
    AUTO = 'auto'                     # Automatically select method


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
    
    # Output settings
    save_filtered_data: bool = False          # Save filtered data
    save_waveforms: bool = True               # Save extracted waveforms
    save_features: bool = True                # Save computed features
    output_format: str = 'numpy'              # Output format (numpy, csv, etc.)


@dataclass
class SpikeUnit:
    """Container for a detected spike unit after clustering."""
    unit_id: int
    waveforms: np.ndarray                    # Shape: (n_spikes, n_samples, n_channels)
    timestamps: np.ndarray                   # Spike timestamps (samples)
    channel_ids: np.ndarray                  # Associated channels
    features: Optional[np.ndarray] = None    # Extracted features
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SpikeSortingResults:
    """Complete results from a spike sorting run."""
    units: List[SpikeUnit]
    config: SpikeSortingConfiguration
    execution_time: float
    sampling_rate: float
    data_info: Dict[str, Any]
    quality_metrics: Dict[str, Dict[str, float]]
    
    def save(self, filename: str) -> None:
        """Save results to disk."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename: str) -> 'SpikeSortingResults':
        """Load results from disk."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert spike times to pandas DataFrame."""
        data = []
        for unit in self.units:
            for spike_idx, timestamp in enumerate(unit.timestamps):
                data.append({
                    'unit_id': unit.unit_id,
                    'timestamp': timestamp,
                    'time_seconds': timestamp / self.sampling_rate
                })
        return pd.DataFrame(data)
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the results."""
        return {
            'total_units': len(self.units),
            'total_spikes': sum(len(unit.timestamps) for unit in self.units),
            'units_per_channel': self._count_units_per_channel(),
            'mean_firing_rates': self._calc_mean_firing_rates(),
            'execution_time': self.execution_time
        }
    
    def _count_units_per_channel(self) -> Dict[int, int]:
        """Count number of units per channel."""
        channel_counts = {}
        for unit in self.units:
            if len(unit.channel_ids) > 0:
                primary_channel = unit.channel_ids[0]
                channel_counts[primary_channel] = channel_counts.get(primary_channel, 0) + 1
        return channel_counts
    
    def _calc_mean_firing_rates(self) -> Dict[int, float]:
        """Calculate mean firing rate for each unit."""
        rates = {}
        if not self.units:
            return rates
        max_time = max(max(unit.timestamps) for unit in self.units if len(unit.timestamps) > 0)
        duration_seconds = max_time / self.sampling_rate
        for unit in self.units:
            rates[unit.unit_id] = len(unit.timestamps) / duration_seconds if duration_seconds > 0 else 0
        return rates
    
    # ------------------------------------------------------------------
    # Pynapple / NWB export
    # ------------------------------------------------------------------
    
    def to_pynapple(self):
        """Convert sorted units to a pynapple TsGroup for downstream analysis.
        
        Returns:
            nap.TsGroup with one Ts per unit, timestamps in seconds.
        
        Raises:
            ImportError: if pynapple is not installed.
        """
        import pynapple as nap
        spike_dict = {}
        for i, unit in enumerate(self.units):
            times_sec = unit.timestamps.astype(np.float64) / self.sampling_rate
            spike_dict[i] = nap.Ts(t=times_sec)
        return nap.TsGroup(spike_dict)
    
    def to_nwb(self, filepath: str, session_description: str = "SpeedSort output") -> None:
        """Write sorted units to an NWB file with spike times and quality metrics.
        
        Args:
            filepath: output .nwb file path.
            session_description: description stored in the NWB file.
        
        Raises:
            ImportError: if pynwb is not installed.
        """
        from pynwb import NWBFile, NWBHDF5IO
        from datetime import datetime
        from dateutil.tz import tzlocal
        
        nwbfile = NWBFile(
            session_description=session_description,
            identifier=f"speedsort_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            session_start_time=datetime.now(tzlocal()),
        )
        nwbfile.add_unit_column(name='snr', description='Signal-to-noise ratio')
        nwbfile.add_unit_column(name='firing_rate', description='Firing rate (Hz)')
        nwbfile.add_unit_column(name='isi_violations_ratio', description='Fraction of ISI violations')
        nwbfile.add_unit_column(name='isolation_distance', description='Mahalanobis isolation distance')
        
        for unit in self.units:
            spike_times_sec = unit.timestamps.astype(np.float64) / self.sampling_rate
            qm = unit.quality_metrics or {}
            nwbfile.add_unit(
                spike_times=spike_times_sec,
                snr=qm.get('snr', 0.0),
                firing_rate=qm.get('firing_rate', 0.0),
                isi_violations_ratio=qm.get('isi_violations_ratio', 0.0),
                isolation_distance=qm.get('isolation_distance', 0.0),
            )
        
        with NWBHDF5IO(filepath, 'w') as io:
            io.write(nwbfile)
        logger.info(f"NWB file saved to {filepath}")
    
    # ------------------------------------------------------------------
    # Post-processing: correlograms and ISI histograms
    # ------------------------------------------------------------------
    
    def _get_unit_by_id(self, unit_id: int) -> SpikeUnit:
        """Look up a unit by its ID. Raises ValueError if not found."""
        for unit in self.units:
            if unit.unit_id == unit_id:
                return unit
        raise ValueError(f"Unit {unit_id} not found")
    
    def compute_acg(self, unit_id: int, bin_size_ms: float = 0.5, window_ms: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute auto-correlogram for a unit.
        
        Args:
            unit_id: which unit.
            bin_size_ms: histogram bin width in ms.
            window_ms: half-window size in ms.
        
        Returns:
            (counts, bin_edges_ms) — counts has the zero-lag bin removed.
        """
        unit = self._get_unit_by_id(unit_id)
        times_ms = unit.timestamps.astype(np.float64) / self.sampling_rate * 1000.0
        times_ms = np.sort(times_ms)
        
        n_bins = int(window_ms / bin_size_ms)
        bin_edges = np.linspace(-window_ms, window_ms, 2 * n_bins + 1)
        counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
        
        for i in range(len(times_ms)):
            diffs = times_ms[i+1:] - times_ms[i]
            valid = diffs[diffs <= window_ms]
            if len(valid) == 0:
                continue
            # positive lags
            idx = np.searchsorted(bin_edges, valid) - 1
            idx = idx[(idx >= 0) & (idx < len(counts))]
            for j in idx:
                counts[j] += 1
            # mirror for negative lags
            neg = -valid
            idx_neg = np.searchsorted(bin_edges, neg) - 1
            idx_neg = idx_neg[(idx_neg >= 0) & (idx_neg < len(counts))]
            for j in idx_neg:
                counts[j] += 1
        
        # Remove zero-lag bin
        zero_bin = n_bins
        counts[zero_bin] = 0
        
        return counts, bin_edges
    
    def compute_ccg(self, unit_id_a: int, unit_id_b: int, bin_size_ms: float = 0.5, window_ms: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-correlogram between two units.
        
        Returns:
            (counts, bin_edges_ms)
        """
        unit_a = self._get_unit_by_id(unit_id_a)
        unit_b = self._get_unit_by_id(unit_id_b)
        times_a = np.sort(unit_a.timestamps.astype(np.float64) / self.sampling_rate * 1000.0)
        times_b = np.sort(unit_b.timestamps.astype(np.float64) / self.sampling_rate * 1000.0)
        
        n_bins = int(window_ms / bin_size_ms)
        bin_edges = np.linspace(-window_ms, window_ms, 2 * n_bins + 1)
        counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
        
        j_start = 0
        for i in range(len(times_a)):
            while j_start < len(times_b) and times_b[j_start] < times_a[i] - window_ms:
                j_start += 1
            j = j_start
            while j < len(times_b) and times_b[j] <= times_a[i] + window_ms:
                diff = times_b[j] - times_a[i]
                bin_idx = int((diff + window_ms) / bin_size_ms)
                if 0 <= bin_idx < len(counts):
                    counts[bin_idx] += 1
                j += 1
        
        return counts, bin_edges
    
    def compute_isi_histogram(self, unit_id: int, bin_size_ms: float = 0.5, max_isi_ms: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inter-spike-interval histogram for a unit.
        
        Returns:
            (counts, bin_edges_ms)
        """
        unit = self._get_unit_by_id(unit_id)
        times_ms = np.sort(unit.timestamps.astype(np.float64) / self.sampling_rate * 1000.0)
        isis = np.diff(times_ms)
        
        bin_edges = np.arange(0, max_isi_ms + bin_size_ms, bin_size_ms)
        counts, _ = np.histogram(isis, bins=bin_edges)
        return counts, bin_edges
    
    def plot_unit_summary(self, unit_id: int, save_path: Optional[str] = None) -> None:
        """Plot 3-panel summary for a unit: mean waveform, auto-correlogram, ISI histogram.
        
        Args:
            unit_id: which unit to plot.
            save_path: if provided, saves figure to this path instead of showing.
        """
        import matplotlib.pyplot as plt
        
        unit = self._get_unit_by_id(unit_id)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'Unit {unit_id}', fontsize=14)
        
        # Panel 1: Mean waveform
        ax = axes[0]
        if unit.waveforms is not None and len(unit.waveforms) > 0:
            if unit.waveforms.ndim == 3:
                mean_wf = np.mean(unit.waveforms, axis=0)
                peak_ch = np.argmax(np.max(np.abs(mean_wf), axis=0))
                wf = mean_wf[:, peak_ch]
            elif unit.waveforms.ndim == 2:
                wf = np.mean(unit.waveforms, axis=0)
            else:
                wf = unit.waveforms.flatten()
            time_axis = np.arange(len(wf)) / self.sampling_rate * 1000
            ax.plot(time_axis, wf, 'k-', linewidth=1.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
        ax.set_title('Mean waveform')
        
        # Panel 2: Auto-correlogram
        ax = axes[1]
        acg_counts, acg_bins = self.compute_acg(unit_id)
        bin_centers = (acg_bins[:-1] + acg_bins[1:]) / 2
        ax.bar(bin_centers, acg_counts, width=acg_bins[1] - acg_bins[0], color='steelblue', edgecolor='none')
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Auto-correlogram')
        
        # Panel 3: ISI histogram
        ax = axes[2]
        isi_counts, isi_bins = self.compute_isi_histogram(unit_id)
        bin_centers = (isi_bins[:-1] + isi_bins[1:]) / 2
        ax.bar(bin_centers, isi_counts, width=isi_bins[1] - isi_bins[0], color='coral', edgecolor='none')
        # Mark refractory period
        ax.axvline(1.5, color='red', linestyle='--', alpha=0.7, label='1.5ms refractory')
        ax.set_xlabel('ISI (ms)')
        ax.set_ylabel('Count')
        ax.set_title('ISI histogram')
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Unit summary plot saved to {save_path}")
        else:
            plt.show()


# ------------------------------------------------------------------------------
# Core spike sorting functionality
# ------------------------------------------------------------------------------

class SpeedSort:
    """Runs the full spike sorting pipeline: load → filter → detect → extract → cluster → score."""
    
    def __init__(self, config: Optional[SpikeSortingConfiguration] = None):
        """Initialize with a SpikeSortingConfiguration (uses defaults if None)."""
        self.config = config or SpikeSortingConfiguration()
        self.results = None
        
        # Initialize backends based on configuration
        self._initialize_backends()
        
        # Track execution time
        self.start_time = None
        self.end_time = None
    
    def _initialize_backends(self) -> None:
        """Set up the array backend (NumPy, CuPy, or PyTorch) based on config."""
        # Initialize array module (numpy, torch, or cupy)
        if self.config.backend == ProcessingBackend.CUPY and _HAS_CUPY:
            self.xp = cp
        elif self.config.backend in (ProcessingBackend.TORCH_GPU, ProcessingBackend.TORCH_CPU) and _HAS_TORCH:
            self.xp = torch
            if self.config.backend == ProcessingBackend.TORCH_GPU and _HAS_GPU:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            # Fall back to NumPy if requested backend isn't available
            self.xp = np
            if self.config.backend not in (ProcessingBackend.CPU, ProcessingBackend.NUMPY):
                logger.warning(f"Requested backend {self.config.backend} not available, using NumPy instead")
            
        # Ensure the device attribute is set for all backends
        if not hasattr(self, 'device') and _HAS_TORCH:
            self.device = torch.device('cpu')
    
    def run(self, data: Union[str, np.ndarray, Path], sampling_rate: Optional[float] = None) -> SpikeSortingResults:
        """Run the full pipeline on a recording file or numpy array. Returns SpikeSortingResults."""
        # Start timing
        self.start_time = time.time()
        
        # Step 1: Load and preprocess the data
        raw_data, data_info = self._load_data(data, sampling_rate)
        
        # Update sampling rate from data_info if needed
        if sampling_rate is not None:
            self.config.sampling_rate = sampling_rate
        elif self.config.sampling_rate is None and 'sampling_rate' in data_info:
            self.config.sampling_rate = data_info['sampling_rate']
        else:
            raise ValueError("Sampling rate must be provided in config or as parameter")
        
        # Step 2: Filter the data
        filtered_data = self._filter_data(raw_data)
        
        n_ch = filtered_data.shape[1]
        has_enough_channels = n_ch >= self.config.min_channels_for_preprocessing
        
        # Step 2b: Bad channel detection & interpolation
        if self.config.detect_bad_channels and has_enough_channels:
            logger.info("Detecting bad channels...")
            filtered_data = self._detect_and_fix_bad_channels(filtered_data)
        
        # Step 2c: Common average reference
        if self.config.common_reference and has_enough_channels:
            logger.info(f"Applying common {self.config.common_reference_type} reference...")
            filtered_data = self._apply_common_reference(filtered_data)
        
        # Step 2d: Whitening
        if self.config.whiten and has_enough_channels:
            logger.info("Applying whitening transform...")
            filtered_data = self._apply_whitening(filtered_data)
        elif self.config.whiten and not has_enough_channels:
            logger.info(f"Skipping preprocessing (CAR/whiten/bad-ch): only {n_ch} channels, need ≥{self.config.min_channels_for_preprocessing}")
        
        # Step 3: Detect spikes
        logger.info("Detecting spikes...")
        spike_times, spike_channels = self._detect_spikes(filtered_data)
        logger.info(f"Detected {len(spike_times)} spikes")
        
        # Step 4: Extract waveforms around spike times
        logger.info("Extracting waveforms...")
        waveforms = self._extract_waveforms(filtered_data, spike_times, spike_channels)
        
        # Step 5: Compute features for spike waveforms
        logger.info("Computing features...")
        features = self._compute_features(waveforms)
        
        # Step 6: Cluster the spikes based on features
        logger.info("Clustering spikes...")
        labels = self._cluster_spikes(features)
        
        # Step 7: Create spike units from clustering results
        logger.info("Creating units...")
        units = self._create_units(waveforms, spike_times, spike_channels, features, labels)
        
        # Step 8: Calculate quality metrics
        if self.config.compute_quality_metrics:
            logger.info("Computing quality metrics...")
            quality_metrics = self._compute_quality_metrics(units, features, labels)
        else:
            quality_metrics = {}
        
        # Stop timing
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        # Step 9: Create the results object
        self.results = SpikeSortingResults(
            units=units,
            config=self.config,
            execution_time=execution_time,
            sampling_rate=self.config.sampling_rate,
            data_info=data_info,
            quality_metrics=quality_metrics
        )
        
        logger.info(f"Spike sorting completed in {execution_time:.2f} seconds")
        logger.info(f"Found {len(units)} units with {sum(len(unit.timestamps) for unit in units)} spikes")
        
        return self.results

    def _load_data(self, data: Union[str, np.ndarray, Path], sampling_rate: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load recording from file path or numpy array. Returns (samples×channels array, info dict)."""
        data_info = {}
        
        if isinstance(data, np.ndarray):
            if sampling_rate is None and self.config.sampling_rate is None:
                raise ValueError("Sampling rate must be provided when input is a NumPy array")
            
            data_array = data
            if sampling_rate:
                data_info['sampling_rate'] = sampling_rate
            elif self.config.sampling_rate:
                data_info['sampling_rate'] = self.config.sampling_rate
            
            # Infer number of channels
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)  # Convert to 2D: samples x 1 channel
            
            data_info['n_channels'] = data_array.shape[1]
            data_info['n_samples'] = data_array.shape[0]
            data_info['format'] = 'numpy'
            
            return data_array, data_info
        
        # Handle file paths
        if isinstance(data, str) or isinstance(data, Path):
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Auto-detect format if not specified
            data_format = self.config.data_format
            if data_format == DataFormat.AUTO:
                data_format = self._detect_data_format(data_path)
            
            # Load data based on format
            if data_format == DataFormat.NUMPY:
                data_array = np.load(data_path)
                data_info['format'] = 'numpy'
                
            elif data_format == DataFormat.BINARY:
                # Need to know dtype and shape for binary data
                if not self.config.n_channels:
                    raise ValueError("Number of channels must be specified for binary data")
                    
                dtype = np.float32  # Default dtype
                data_array = np.fromfile(data_path, dtype=dtype)
                data_array = data_array.reshape(-1, self.config.n_channels)
                data_info['format'] = 'binary'
            
            elif data_format == DataFormat.NEO and _HAS_MNE:
                # Use Neo to load various formats
                import neo
                reader = neo.get_io(str(data_path))
                block = reader.read_block()
                
                # Extract data from first segment
                if len(block.segments) > 0:
                    segment = block.segments[0]
                    if len(segment.analogsignals) > 0:
                        signal = segment.analogsignals[0]
                        data_array = signal.magnitude
                        data_info['sampling_rate'] = float(signal.sampling_rate)
                        data_info['format'] = 'neo'
                    else:
                        raise ValueError("No analog signals found in Neo file")
                else:
                    raise ValueError("No segments found in Neo file")
                
            elif data_format == DataFormat.OPEN_EPHYS:
                # Implement Open Ephys format loading
                if _HAS_MNE:
                    from mne.io import read_raw_openephys
                    raw = read_raw_openephys(data_path)
                    data_array = raw.get_data()
                    data_info['sampling_rate'] = raw.info['sfreq']
                    data_info['format'] = 'open_ephys'
                else:
                    raise ImportError("MNE-Python is required for Open Ephys format")
                
            elif data_format == DataFormat.NWB:
                # Implement NWB format loading
                try:
                    import pynwb
                    from pynwb import NWBHDF5IO
                    
                    io = NWBHDF5IO(str(data_path), 'r')
                    nwb_file = io.read()
                    
                    # Get the acquisition data
                    for key, data_obj in nwb_file.acquisition.items():
                        if isinstance(data_obj, pynwb.ecephys.ElectricalSeries):
                            data_array = data_obj.data[:]
                            data_info['sampling_rate'] = data_obj.rate
                            data_info['format'] = 'nwb'
                            break
                    else:
                        raise ValueError("No electrical series found in NWB file")
                    
                    io.close()
                except ImportError:
                    raise ImportError("pynwb is required for NWB format")
            
            elif data_format == DataFormat.MDA:
                # Implement MountainSort MDA format loading
                try:
                    data_array = self._read_mda(data_path)
                    data_info['format'] = 'mda'
                    # Need to get sampling rate from params file or config
                except Exception as e:
                    raise ValueError(f"Error loading MDA file: {e}")
            
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
            
            # Set dimensions in data_info
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            
            data_info['n_channels'] = data_array.shape[1]
            data_info['n_samples'] = data_array.shape[0]
            
            return data_array, data_info
        
        raise ValueError("Data must be a file path or numpy array")

    def _detect_data_format(self, filepath: Path) -> DataFormat:
        """Guess data format from file extension (.npy → NUMPY, .nwb → NWB, etc.)."""
        # Check by extension
        suffix = filepath.suffix.lower()
        
        if suffix == '.npy':
            return DataFormat.NUMPY
        elif suffix == '.dat' or suffix == '.bin':
            return DataFormat.BINARY
        elif suffix == '.nwb':
            return DataFormat.NWB
        elif suffix == '.mda':
            return DataFormat.MDA
        elif suffix in ('.mat', '.h5', '.hdf5'):
            # Need to check content for specific formats
            if _HAS_MNE:
                try:
                    import neo
                    neo.get_io(str(filepath))
                    return DataFormat.NEO
                except:
                    pass
        
        # Check for Open Ephys directory structure
        if filepath.is_dir() and any(filepath.glob('*.continuous')):
            return DataFormat.OPEN_EPHYS
        
        # Default to binary if we can't identify the format
        logger.warning(f"Could not detect format for {filepath}, assuming binary")
        return DataFormat.BINARY

    def _read_mda(self, filepath: Path) -> np.ndarray:
        """Read a MountainSort .mda file and return its data as a numpy array."""
        # Basic MDA format reader
        with open(filepath, 'rb') as f:
            # Read header
            header_size = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            num_dims = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            dims = np.fromfile(f, dtype=np.dtype('i4'), count=num_dims)
            
            # Read data
            data_type_code = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            
            # Map data type code to numpy dtype
            data_type = {
                -2: np.dtype('uint8'),
                -3: np.dtype('float32'),
                -4: np.dtype('int16'),
                -5: np.dtype('int32'),
                -6: np.dtype('uint16'),
                -7: np.dtype('double'),
                -8: np.dtype('uint32')
            }.get(data_type_code, np.dtype('float32'))
            
            # Read data
            data = np.fromfile(f, dtype=data_type)
            
            # Reshape data
            data = data.reshape(dims)
            
            return data

    def _filter_data(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass and optional notch filters. Input/output shape: (samples, channels)."""
        # Convert to proper array type for backend
        if isinstance(self.xp, type(np)) and not isinstance(data, np.ndarray):
            data = np.array(data)
        elif _HAS_CUPY and self.xp is cp and not isinstance(data, cp.ndarray):
            data = cp.array(data)
        elif self.xp is torch and not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=self.device, dtype=torch.float32)
        
        # Get parameters from config
        nyquist = self.config.sampling_rate / 2
        low_cut = self.config.filter_low / nyquist
        high_cut = self.config.filter_high / nyquist
        
        # For NumPy backend, use scipy
        if self.xp is np:
            filtered_data = np.zeros_like(data)
            
            # Apply bandpass filter to each channel
            b, a = signal.butter(self.config.filter_order, [low_cut, high_cut], btype=self.config.filter_type)
            
            for ch in range(data.shape[1]):
                filtered_data[:, ch] = signal.filtfilt(b, a, data[:, ch])
            
            # Apply notch filter if requested
            if self.config.notch_filter:
                for freq in [50, 60]:  # Common line noise frequencies
                    notch_low = (freq - 2) / nyquist
                    notch_high = (freq + 2) / nyquist
                    if notch_high < 1:  # Ensure filter is valid
                        b_notch, a_notch = signal.butter(self.config.filter_order, [notch_low, notch_high], btype='bandstop')
                        for ch in range(data.shape[1]):
                            filtered_data[:, ch] = signal.filtfilt(b_notch, a_notch, filtered_data[:, ch])
        
        # For PyTorch backend
        elif self.xp is torch:
            import torch.nn.functional as F
            
            # Create filter kernels
            kernel_size = 2 * self.config.filter_order + 1
            t = torch.linspace(-self.config.filter_order, self.config.filter_order, kernel_size, device=self.device)
            
            # Create bandpass filter
            low_freq = self.config.filter_low
            high_freq = self.config.filter_high
            
            # Create sinc filter
            low_kernel = 2 * low_cut * torch.sinc(2 * low_freq * t)
            high_kernel = 2 * high_cut * torch.sinc(2 * high_freq * t)
            bandpass_kernel = high_kernel - low_kernel
            
            # Apply Hamming window
            window = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(kernel_size, device=self.device) / (kernel_size - 1))
            bandpass_kernel = bandpass_kernel * window
            
            # Normalize kernel
            bandpass_kernel = bandpass_kernel / torch.sum(bandpass_kernel)
            
            # Reshape for conv1d
            bandpass_kernel = bandpass_kernel.view(1, 1, -1)
            
            # Apply filter
            data_input = data.permute(1, 0).unsqueeze(1)  # (channels, 1, samples)
            filtered_data = F.conv1d(data_input, bandpass_kernel, padding=self.config.filter_order)
            filtered_data = filtered_data.squeeze(1).permute(1, 0)  # Back to (samples, channels)
        
        # For CuPy backend
        elif _HAS_CUPY and self.xp is cp:
            filtered_data = cp.zeros_like(data)
            
            # Apply bandpass filter to each channel
            b, a = signal.butter(self.config.filter_order, [low_cut, high_cut], btype=self.config.filter_type)
            
            for ch in range(data.shape[1]):
                # Need to convert to NumPy, apply filter, then back to CuPy
                channel_data = cp.asnumpy(data[:, ch])
                filtered_channel = signal.filtfilt(b, a, channel_data)
                filtered_data[:, ch] = cp.array(filtered_channel)
            
            # Apply notch filter if requested
            if self.config.notch_filter:
                for freq in [50, 60]:  # Common line noise frequencies
                    notch_low = (freq - 2) / nyquist
                    notch_high = (freq + 2) / nyquist
                    if notch_high < 1:  # Ensure filter is valid
                        b_notch, a_notch = signal.butter(self.config.filter_order, [notch_low, notch_high], btype='bandstop')
                        for ch in range(data.shape[1]):
                            channel_data = cp.asnumpy(filtered_data[:, ch])
                            filtered_channel = signal.filtfilt(b_notch, a_notch, channel_data)
                            filtered_data[:, ch] = cp.array(filtered_channel)
        
        # Fall back to NumPy if backend not recognized
        else:
            logger.warning(f"Unrecognized backend {type(self.xp)}, falling back to NumPy")
            self.xp = np
            return self._filter_data(data)
        
        return filtered_data

    def _apply_common_reference(self, data: np.ndarray) -> np.ndarray:
        """Subtract common average (or median) reference across channels per time sample."""
        if isinstance(data, np.ndarray):
            if self.config.common_reference_type == 'median':
                ref = np.median(data, axis=1, keepdims=True)
            else:
                ref = np.mean(data, axis=1, keepdims=True)
            return data - ref
        elif _HAS_CUPY and self.xp is cp:
            data_np = cp.asnumpy(data)
            ref = np.median(data_np, axis=1, keepdims=True) if self.config.common_reference_type == 'median' else np.mean(data_np, axis=1, keepdims=True)
            return cp.array(data_np - ref)
        elif _HAS_TORCH and self.xp is torch:
            if self.config.common_reference_type == 'median':
                ref = torch.median(data, dim=1, keepdim=True).values
            else:
                ref = torch.mean(data, dim=1, keepdim=True)
            return data - ref
        return data

    def _detect_and_fix_bad_channels(self, data: np.ndarray) -> np.ndarray:
        """Detect dead/noisy channels via variance analysis and interpolate from neighbors.
        
        Bad channels are those whose variance is far outside the distribution of
        all channel variances (using MAD-based z-score). Dead channels (near-zero
        variance) and noisy channels (extremely high variance) are both caught.
        Bad channels are replaced by the mean of their nearest non-bad neighbors.
        """
        if isinstance(data, np.ndarray):
            data_np = data
        elif _HAS_CUPY and self.xp is cp:
            data_np = cp.asnumpy(data)
        elif _HAS_TORCH and self.xp is torch:
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
            mad_lv = 1e-10  # Avoid division by zero
        
        z_scores = np.abs(log_vars - median_lv) / mad_lv
        bad_mask = z_scores > self.config.bad_channel_std_threshold
        
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
            # Find nearest good neighbors
            distances = np.abs(good_indices - bad_ch)
            n_neighbors = min(2, len(good_indices))  # Use 2 nearest good channels
            nearest = good_indices[np.argsort(distances)[:n_neighbors]]
            result[:, bad_ch] = np.mean(data_np[:, nearest], axis=1)
        
        if _HAS_CUPY and self.xp is cp:
            return cp.array(result)
        elif _HAS_TORCH and self.xp is torch:
            return torch.tensor(result, device=self.device, dtype=torch.float32)
        return result

    def _apply_whitening(self, data: np.ndarray) -> np.ndarray:
        """Apply ZCA whitening: decorrelate channels and equalize noise variance."""
        # Work in numpy for covariance estimation
        if _HAS_CUPY and self.xp is cp:
            data_np = cp.asnumpy(data).astype(np.float64)
        elif _HAS_TORCH and self.xp is torch:
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
        
        # Apply whitening
        whitened = data_np @ W.T
        
        # Guard against numerical issues
        if np.any(~np.isfinite(whitened)):
            logger.warning("Whitening produced non-finite values, skipping whitening step")
            return data
        
        if _HAS_CUPY and self.xp is cp:
            return cp.array(whitened.astype(np.float32))
        elif _HAS_TORCH and self.xp is torch:
            return torch.tensor(whitened, device=self.device, dtype=torch.float32)
        return whitened.astype(np.float32)

    def _detect_spikes(self, filtered_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect spikes per channel. Returns (spike_times, spike_channels) arrays."""
        detection_method = self.config.detection_method
        threshold = self.config.detection_threshold
        
        # Initialize arrays for results
        spike_times_list = []
        spike_channels_list = []
        
        # Process each channel
        for ch in range(filtered_data.shape[1]):
            # Get channel data
            channel_data = filtered_data[:, ch]
            
            # Compute noise level
            if self.xp is np:
                noise_level = np.median(np.abs(channel_data)) / 0.6745  # MAD estimate
            elif _HAS_CUPY and self.xp is cp:
                noise_level = float(cp.median(cp.abs(channel_data))) / 0.6745
            elif self.xp is torch:
                noise_level = torch.median(torch.abs(channel_data)).item() / 0.6745
            
            # Compute threshold level
            threshold_value = threshold * noise_level
            
            # Detect spikes based on method
            if detection_method == DetectionMethod.THRESHOLD:
                # Simple threshold crossing
                if self.xp is np:
                    crossings = np.where(channel_data < -threshold_value)[0]
                elif _HAS_CUPY and self.xp is cp:
                    crossings = cp.where(channel_data < -threshold_value)[0].get()
                elif self.xp is torch:
                    crossings = torch.where(channel_data < -threshold_value)[0].cpu().numpy()
                else:
                    # Fall back to numpy
                    channel_data_np = np.array(channel_data)
                    crossings = np.where(channel_data_np < -threshold_value)[0]
            
            elif detection_method == DetectionMethod.THRESHOLD_DYNAMIC:
                # Adaptive threshold detection
                window_size = int(1.0 * self.config.sampling_rate)  # 1-second window
                
                if self.xp is np:
                    # Compute threshold per window
                    thresholds = np.zeros_like(channel_data)
                    for i in range(0, len(channel_data), window_size):
                        window_end = min(i + window_size, len(channel_data))
                        window_data = channel_data[i:window_end]
                        window_noise = np.median(np.abs(window_data)) / 0.6745
                        thresholds[i:window_end] = threshold * window_noise
                    
                    # Find crossings
                    crossings = np.where(channel_data < -thresholds)[0]
                
                elif _HAS_CUPY and self.xp is cp:
                    # Similar approach for CuPy
                    thresholds = cp.zeros_like(channel_data)
                    for i in range(0, len(channel_data), window_size):
                        window_end = min(i + window_size, len(channel_data))
                        window_data = channel_data[i:window_end]
                        window_noise = float(cp.median(cp.abs(window_data))) / 0.6745
                        thresholds[i:window_end] = threshold * window_noise
                    
                    crossings = cp.where(channel_data < -thresholds)[0].get()
                
                elif self.xp is torch:
                    # And for PyTorch
                    thresholds = torch.zeros_like(channel_data)
                    for i in range(0, len(channel_data), window_size):
                        window_end = min(i + window_size, len(channel_data))
                        window_data = channel_data[i:window_end]
                        window_noise = torch.median(torch.abs(window_data)).item() / 0.6745
                        thresholds[i:window_end] = threshold * window_noise
                    
                    crossings = torch.where(channel_data < -thresholds)[0].cpu().numpy()
                
                # Remove duplicates (ensure minimum spacing between spikes)
                min_spacing = int(0.001 * self.config.sampling_rate)  # 1ms default
                if len(crossings) > 0:
                    # Keep only first crossing in each cluster of crossings
                    mask = np.diff(crossings) > min_spacing
                    mask = np.concatenate(([True], mask))
                    crossings = crossings[mask]
            
            elif detection_method == DetectionMethod.NEO:
                # Nonlinear Energy Operator: y[n] = x[n]^2 - x[n-1]*x[n+1]
                if self.xp is np:
                    energy = np.zeros_like(channel_data)
                    energy[1:-1] = channel_data[1:-1]**2 - channel_data[0:-2] * channel_data[2:]
                    
                    # Smoothing
                    kernel_size = int(0.0005 * self.config.sampling_rate)  # 0.5ms
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # Make odd for symmetry
                    energy = np.convolve(energy, np.ones(kernel_size)/kernel_size, mode='same')
                    
                    # Find peaks in energy
                    neo_threshold = threshold * np.median(np.abs(energy)) / 0.6745
                    crossings = np.where(energy > neo_threshold)[0]
                
                elif _HAS_CUPY and self.xp is cp:
                    energy = cp.zeros_like(channel_data)
                    energy[1:-1] = channel_data[1:-1]**2 - channel_data[0:-2] * channel_data[2:]
                    
                    # Smoothing - convert to NumPy for convolution
                    energy_np = cp.asnumpy(energy)
                    kernel_size = int(0.0005 * self.config.sampling_rate)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    energy_np = np.convolve(energy_np, np.ones(kernel_size)/kernel_size, mode='same')
                    energy = cp.array(energy_np)
                    
                    neo_threshold = threshold * float(cp.median(cp.abs(energy))) / 0.6745
                    crossings = cp.where(energy > neo_threshold)[0].get()
                
                elif self.xp is torch:
                    energy = torch.zeros_like(channel_data)
                    energy[1:-1] = channel_data[1:-1]**2 - channel_data[0:-2] * channel_data[2:]
                    
                    # Smoothing using PyTorch's conv1d
                    import torch.nn.functional as F
                    kernel_size = int(0.0005 * self.config.sampling_rate)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
                    energy = energy.view(1, 1, -1)
                    energy = F.conv1d(energy, kernel, padding=kernel_size//2)
                    energy = energy.view(-1)
                    
                    neo_threshold = threshold * torch.median(torch.abs(energy)).item() / 0.6745
                    crossings = torch.where(energy > neo_threshold)[0].cpu().numpy()
                
                # Remove duplicates
                min_spacing = int(0.001 * self.config.sampling_rate)
                if len(crossings) > 0:
                    mask = np.diff(np.concatenate(([0], crossings))) > min_spacing
                    crossings = crossings[mask]
            
            elif detection_method == DetectionMethod.WAVELET:
                # Wavelet-based detection
                try:
                    import pywt
                    
                    # Decompose signal using wavelets
                    if self.xp is np:
                        coeffs = pywt.wavedec(channel_data, 'sym7', level=4)
                    elif _HAS_CUPY and self.xp is cp:
                        # Convert to NumPy for PyWavelets
                        np_channel_data = cp.asnumpy(channel_data)
                        coeffs = pywt.wavedec(np_channel_data, 'sym7', level=4)
                    else:
                        # Convert to NumPy for PyWavelets
                        np_channel_data = channel_data.cpu().numpy()
                        coeffs = pywt.wavedec(np_channel_data, 'sym7', level=4)
                    
                    # Focus on detail coefficients that capture spike frequencies
                    detail_coeffs = coeffs[1]  # First level of detail coefficients
                    
                    # Threshold the coefficients
                    wavelet_threshold = threshold * np.median(np.abs(detail_coeffs)) / 0.6745
                    significant_indices = np.where(np.abs(detail_coeffs) > wavelet_threshold)[0]
                    
                    # Map back to original signal domain
                    scale_factor = len(channel_data) / len(detail_coeffs)
                    crossings = np.unique(np.floor(significant_indices * scale_factor).astype(int))
                    crossings = crossings[crossings < len(channel_data)]
                    
                    # Remove duplicates
                    min_spacing = int(0.001 * self.config.sampling_rate)
                    if len(crossings) > 0:
                        mask = np.diff(np.concatenate(([0], crossings))) > min_spacing
                        crossings = crossings[mask]
                    
                except ImportError:
                    logger.warning("PyWavelets not available, falling back to threshold detection")
                    if self.xp is np:
                        crossings = np.where(channel_data < -threshold_value)[0]
                    elif _HAS_CUPY and self.xp is cp:
                        crossings = cp.where(channel_data < -threshold_value)[0].get()
                    elif self.xp is torch:
                        crossings = torch.where(channel_data < -threshold_value)[0].cpu().numpy()
            
            elif detection_method == DetectionMethod.NEURAL_NET:
                # Neural network-based detection - requires PyTorch
                if not _HAS_TORCH:
                    logger.warning("PyTorch not available, falling back to threshold detection")
                    if self.xp is np:
                        crossings = np.where(channel_data < -threshold_value)[0]
                    elif _HAS_CUPY and self.xp is cp:
                        crossings = cp.where(channel_data < -threshold_value)[0].get()
                else:
                    try:
                        # Check if model exists, otherwise use threshold
                        model_path = Path("models/spike_detector.pt")
                        if not model_path.exists():
                            logger.warning("Neural network model not found, falling back to threshold detection")
                            if self.xp is np:
                                crossings = np.where(channel_data < -threshold_value)[0]
                            elif _HAS_CUPY and self.xp is cp:
                                crossings = cp.where(channel_data < -threshold_value)[0].get()
                            elif self.xp is torch:
                                crossings = torch.where(channel_data < -threshold_value)[0].cpu().numpy()
                        else:
                            # Load model and run inference
                            model = torch.load(model_path)
                            model.eval()
                            
                            # Convert data to PyTorch tensor if needed
                            if not isinstance(channel_data, torch.Tensor):
                                channel_data_tensor = torch.tensor(channel_data.astype(np.float32), device=self.device)
                            else:
                                channel_data_tensor = channel_data
                            
                            # Window the data for the model
                            window_size = 64  # Typical model input size
                            stride = 32       # Overlap between windows
                            
                            windows = []
                            for i in range(0, len(channel_data_tensor) - window_size, stride):
                                windows.append(channel_data_tensor[i:i+window_size])
                            
                            if windows:
                                # Stack windows and normalize
                                windows = torch.stack(windows)
                                windows = (windows - windows.mean(dim=1, keepdim=True)) / (windows.std(dim=1, keepdim=True) + 1e-8)
                                
                                # Run in batches to avoid memory issues
                                batch_size = 1024
                                predictions = []
                                with torch.no_grad():
                                    for i in range(0, len(windows), batch_size):
                                        batch = windows[i:i+batch_size]
                                        preds = model(batch)
                                        predictions.append(preds)
                                
                                predictions = torch.cat(predictions)
                                
                                # Find windows with spikes
                                spike_windows = torch.where(predictions > 0.5)[0].cpu().numpy()
                                
                                # Convert window indices to sample indices
                                crossings = stride * spike_windows + window_size // 2
                            else:
                                crossings = np.array([])
                    except Exception as e:
                        logger.error(f"Error in neural network spike detection: {e}")
                        logger.warning("Falling back to threshold detection")
                        if self.xp is np:
                            crossings = np.where(channel_data < -threshold_value)[0]
                        elif _HAS_CUPY and self.xp is cp:
                            crossings = cp.where(channel_data < -threshold_value)[0].get()
                        elif self.xp is torch:
                            crossings = torch.where(channel_data < -threshold_value)[0].cpu().numpy()
            
            else:
                # Default to simple threshold
                if self.xp is np:
                    crossings = np.where(channel_data < -threshold_value)[0]
                elif _HAS_CUPY and self.xp is cp:
                    crossings = cp.where(channel_data < -threshold_value)[0].get()
                elif self.xp is torch:
                    crossings = torch.where(channel_data < -threshold_value)[0].cpu().numpy()
            
            # Ensure we're not too close to the edges for waveform extraction
            alignment_window = self.config.alignment_window
            extraction_window = self.config.waveform_extraction_window
            
            # Get the widest window needed
            left_margin = max(abs(alignment_window[0]), abs(extraction_window[0]))
            right_margin = max(alignment_window[1], extraction_window[1])
            
            valid_crossings = crossings[(crossings >= left_margin) & 
                                      (crossings < len(channel_data) - right_margin)]
            
            # Align spikes to their negative peak within a small window
            aligned_times = []
            
            for t in valid_crossings:
                # Extract window around initial detection
                spike_window = slice(t + alignment_window[0], t + alignment_window[1])

                if self.xp is np:
                    window_data = channel_data[spike_window]
                    peak_offset = np.argmin(window_data)
                elif _HAS_CUPY and self.xp is cp:
                    window_data = channel_data[spike_window]
                    peak_offset = cp.argmin(window_data).get()
                elif self.xp is torch:
                    window_data = channel_data[spike_window]
                    peak_offset = torch.argmin(window_data).item()
                
                # Calculate aligned spike time
                aligned_time = t + alignment_window[0] + peak_offset
                
                # Ensure we're not too close to edges after alignment
                if (aligned_time >= left_margin and 
                    aligned_time < len(channel_data) - right_margin):
                    aligned_times.append(aligned_time)
            
            # Append results for this channel
            if aligned_times:
                spike_times_list.extend(aligned_times)
                spike_channels_list.extend([ch] * len(aligned_times))
        
        # Convert to arrays
        if spike_times_list:
            spike_times = np.array(spike_times_list)
            spike_channels = np.array(spike_channels_list)
            
            # Sort by time
            sort_idx = np.argsort(spike_times)
            spike_times = spike_times[sort_idx]
            spike_channels = spike_channels[sort_idx]
        else:
            spike_times = np.array([], dtype=np.int64)
            spike_channels = np.array([], dtype=np.int64)
        
        return spike_times, spike_channels

    def _extract_waveforms(self, filtered_data: np.ndarray, spike_times: np.ndarray, 
                          spike_channels: np.ndarray) -> np.ndarray:
        """Cut waveform snippets around each spike. Returns array (n_spikes, n_samples, n_channels)."""
        if len(spike_times) == 0:
            return np.array([])
        
        # Get extraction window parameters
        window_start, window_end = self.config.waveform_extraction_window
        window_length = window_end - window_start
        n_channels = filtered_data.shape[1]
        
        # Initialize waveform array
        if self.xp is np:
            waveforms = np.zeros((len(spike_times), window_length, n_channels))
        elif _HAS_CUPY and self.xp is cp:
            waveforms = cp.zeros((len(spike_times), window_length, n_channels))
        elif self.xp is torch:
            waveforms = torch.zeros((len(spike_times), window_length, n_channels), device=self.device)
        else:
            # Fall back to NumPy
            waveforms = np.zeros((len(spike_times), window_length, n_channels))
        
        # Extract waveforms for each spike
        # We'll use the detected channel and neighbors for each spike
        for i, (spike_time, spike_channel) in enumerate(zip(spike_times, spike_channels)):
            # Extract time window for the spike
            t_start = spike_time + window_start
            t_end = spike_time + window_end
            
            # Ensure we're within data bounds (should already be checked, but just in case)
            if t_start >= 0 and t_end <= filtered_data.shape[0]:
                # Extract waveform for all channels
                spike_waveform = filtered_data[t_start:t_end, :]
                
                # Store in waveforms array
                if self.xp is np:
                    waveforms[i, :, :] = spike_waveform
                elif _HAS_CUPY and self.xp is cp:
                    waveforms[i, :, :] = spike_waveform
                elif self.xp is torch:
                    waveforms[i, :, :] = spike_waveform
                else:
                    # Fall back to NumPy
                    waveforms[i, :, :] = spike_waveform
        
        # Convert to NumPy for consistent output
        if _HAS_CUPY and self.xp is cp:
            waveforms = cp.asnumpy(waveforms)
        elif self.xp is torch:
            waveforms = waveforms.cpu().numpy()
        
        return waveforms

    def _compute_features(self, waveforms: np.ndarray) -> np.ndarray:
        """Reduce waveforms to feature vectors via PCA, t-SNE, UMAP, or wavelets. Returns (n_spikes, n_features)."""
        if len(waveforms) == 0:
            return np.array([])
        
        n_spikes = waveforms.shape[0]
        
        # Convert to appropriate backend format
        if _HAS_CUPY and self.xp is cp and not isinstance(waveforms, cp.ndarray):
            waveforms_xp = cp.array(waveforms)
        elif self.xp is torch and not isinstance(waveforms, torch.Tensor):
            waveforms_xp = torch.tensor(waveforms, device=self.device)
        else:
            waveforms_xp = waveforms
        
        # Select dimensionality reduction method
        dim_reduction = self.config.dim_reduction
        
        # Auto selection based on data size
        if dim_reduction == DimensionalityReduction.AUTO:
            if n_spikes < 1000:
                dim_reduction = DimensionalityReduction.PCA
            elif n_spikes < 10000:
                if _HAS_SKLEARN:
                    dim_reduction = DimensionalityReduction.PCA
                else:
                    dim_reduction = DimensionalityReduction.WAVELET
            else:
                dim_reduction = DimensionalityReduction.WAVELET
        
        # Reshape waveforms for feature extraction: (n_spikes, n_samples * n_channels)
        n_samples = waveforms_xp.shape[1]
        n_channels = waveforms_xp.shape[2]
        
        if self.xp is np or (_HAS_CUPY and self.xp is cp):
            X = waveforms_xp.reshape(n_spikes, n_samples * n_channels)
        elif self.xp is torch:
            X = waveforms_xp.reshape(n_spikes, n_samples * n_channels)
        else:
            # Fall back to NumPy
            waveforms_np = np.array(waveforms_xp)
            X = waveforms_np.reshape(n_spikes, n_samples * n_channels)
        
        # Apply dimensionality reduction
        if dim_reduction == DimensionalityReduction.PCA:
            if not _HAS_SKLEARN:
                logger.warning("scikit-learn not available, using wavelet features instead")
                features = self._compute_wavelet_features(waveforms_xp)
            else:
                # Convert to NumPy for sklearn
                if self.xp is not np:
                    X_np = X.get() if _HAS_CUPY and self.xp is cp else X.cpu().numpy()
                else:
                    X_np = X
                
                # Normalize features
                if self.config.feature_normalize:
                    X_np = sklearn.preprocessing.scale(X_np)
                
                # Apply PCA
                pca = sklearn.decomposition.PCA(n_components=min(self.config.n_components, X_np.shape[1], X_np.shape[0]))
                features = pca.fit_transform(X_np)
                
                # Log variance explained
                explained_var = sum(pca.explained_variance_ratio_)
                logger.info(f"PCA captured {explained_var:.2f} of variance with {self.config.n_components} components")
        
        elif dim_reduction == DimensionalityReduction.TSNE:
            if not _HAS_SKLEARN:
                logger.warning("scikit-learn not available, using wavelet features instead")
                features = self._compute_wavelet_features(waveforms_xp)
            else:
                # t-SNE is computationally expensive, so first reduce with PCA if many spikes
                if self.xp is not np:
                    X_np = X.get() if _HAS_CUPY and self.xp is cp else X.cpu().numpy()
                else:
                    X_np = X
                
                # Normalize features
                if self.config.feature_normalize:
                    X_np = sklearn.preprocessing.scale(X_np)
                
                # First reduce with PCA if high-dimensional
                if X_np.shape[1] > 50:
                    pca = sklearn.decomposition.PCA(n_components=min(50, X_np.shape[1], X_np.shape[0]))
                    X_np = pca.fit_transform(X_np)
                
                # Apply t-SNE
                try:
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=min(self.config.n_components, X_np.shape[1]), 
                                random_state=42)
                    features = tsne.fit_transform(X_np)
                except Exception as e:
                    logger.error(f"Error in t-SNE: {e}, falling back to PCA")
                    pca = sklearn.decomposition.PCA(n_components=min(self.config.n_components, X_np.shape[1], X_np.shape[0]))
                    features = pca.fit_transform(X_np)
        
        elif dim_reduction == DimensionalityReduction.UMAP:
            # UMAP requires separate installation
            try:
                import umap
                
                # Convert to NumPy for UMAP
                if self.xp is not np:
                    X_np = X.get() if _HAS_CUPY and self.xp is cp else X.cpu().numpy()
                else:
                    X_np = X
                
                # Normalize features
                if self.config.feature_normalize:
                    X_np = sklearn.preprocessing.scale(X_np)
                
                # Apply UMAP
                reducer = umap.UMAP(n_components=min(self.config.n_components, X_np.shape[1]))
                features = reducer.fit_transform(X_np)
            except ImportError:
                logger.warning("UMAP not available, using PCA instead")
                if not _HAS_SKLEARN:
                    features = self._compute_wavelet_features(waveforms_xp)
                else:
                    # Convert to NumPy for sklearn
                    if self.xp is not np:
                        X_np = X.get() if _HAS_CUPY and self.xp is cp else X.cpu().numpy()
                    else:
                        X_np = X
                    
                    # Normalize features
                    if self.config.feature_normalize:
                        X_np = sklearn.preprocessing.scale(X_np)
                    
                    # Apply PCA
                    pca = sklearn.decomposition.PCA(n_components=min(self.config.n_components, X_np.shape[1], X_np.shape[0]))
                    features = pca.fit_transform(X_np)
        
        elif dim_reduction == DimensionalityReduction.WAVELET:
            # Use wavelet coefficients as features
            features = self._compute_wavelet_features(waveforms_xp)
        
        else:  # No dimensionality reduction or unknown method
            # Use raw waveform data, but select a few key points to reduce dimensions
            if self.xp is not np:
                X_np = X.get() if _HAS_CUPY and self.xp is cp else X.cpu().numpy()
            else:
                X_np = X
            
            # Subsample waveform to reduce dimensions
            step = max(1, X_np.shape[1] // self.config.n_components)
            features = X_np[:, ::step]
            
            # Limit to requested number of components
            features = features[:, :self.config.n_components]
        
        # Ensure output is NumPy array for consistent interface
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        return features

    def _compute_wavelet_features(self, waveforms: Union[np.ndarray, 'cp.ndarray', 'torch.Tensor']) -> np.ndarray:
        """Extract wavelet decomposition coefficients as features. Fallback: simple peak/width/energy features."""
        try:
            import pywt
            
            # Convert to NumPy for PyWavelets
            if self.xp is not np:
                waveforms_np = waveforms.get() if _HAS_CUPY and self.xp is cp else waveforms.cpu().numpy()
            else:
                waveforms_np = waveforms
            
            n_spikes = waveforms_np.shape[0]
            n_samples = waveforms_np.shape[1]
            n_channels = waveforms_np.shape[2]
            
            # Reshape for convenience
            waveforms_flat = waveforms_np.reshape(n_spikes, -1)
            
            # Apply wavelet transform to extract coefficients
            wavelet = 'sym5'  # Symlet wavelet, good for spike shapes
            level = 3       # Decomposition level
            
            # Initialize feature array
            features_list = []
            
            # Process in batches for efficiency with large datasets
            batch_size = min(5000, n_spikes)
            for i in range(0, n_spikes, batch_size):
                batch_end = min(i + batch_size, n_spikes)
                batch_features = []
                
                for j in range(i, batch_end):
                    # Reshape waveform for this spike back to 2D
                    spike_wave = waveforms_np[j].reshape(n_samples, n_channels)
                    
                    # Get largest amplitude channel
                    peak_channel = np.argmax(np.max(np.abs(spike_wave), axis=0))
                    
                    # Extract waveform for peak channel
                    peak_waveform = spike_wave[:, peak_channel]
                    
                    # Apply wavelet transform
                    coeffs = pywt.wavedec(peak_waveform, wavelet, level=level)
                    
                    # Concatenate coefficients from different levels
                    feature_vector = np.concatenate([coef for coef in coeffs])
                    
                    # Limit number of features if needed
                    if len(feature_vector) > self.config.n_components:
                        # Either subsample or use PCA depending on implementation
                        step = max(1, len(feature_vector) // self.config.n_components)
                        feature_vector = feature_vector[::step][:self.config.n_components]
                    
                    batch_features.append(feature_vector)
                
                features_list.append(np.vstack(batch_features))
            
            features = np.vstack(features_list)
            
            # Ensure consistent feature dimension by padding or truncating
            if features.shape[1] > self.config.n_components:
                features = features[:, :self.config.n_components]
            elif features.shape[1] < self.config.n_components:
                # Pad with zeros if needed
                padding = np.zeros((features.shape[0], self.config.n_components - features.shape[1]))
                features = np.hstack([features, padding])
            
            return features
            
        except ImportError:
            logger.warning("PyWavelets not available, using simple features instead")
            
            # Convert to NumPy
            if self.xp is not np:
                waveforms_np = waveforms.get() if _HAS_CUPY and self.xp is cp else waveforms.cpu().numpy()
            else:
                waveforms_np = waveforms
            
            n_spikes = waveforms_np.shape[0]
            n_samples = waveforms_np.shape[1]
            n_channels = waveforms_np.shape[2]
            
            # Compute simple features: peak amplitude, width, etc.
            features = np.zeros((n_spikes, self.config.n_components))
            
            for i in range(n_spikes):
                spike_wave = waveforms_np[i]
                
                # Find peak channel
                peak_channel = np.argmax(np.max(np.abs(spike_wave), axis=0))
                peak_waveform = spike_wave[:, peak_channel]
                
                # Basic features
                features[i, 0] = np.min(peak_waveform)  # Peak amplitude
                features[i, 1] = np.argmin(peak_waveform)  # Time to peak
                features[i, 2] = np.max(peak_waveform)  # Max positive amplitude
                features[i, 3] = np.argmax(peak_waveform)  # Time to max positive
                
                # Compute width (half-amplitude duration)
                peak_val = np.min(peak_waveform)
                half_amp = peak_val / 2
                above_half = peak_waveform <= half_amp
                if np.any(above_half):
                    cross_indices = np.where(above_half)[0]
                    if len(cross_indices) >= 2:
                        width = cross_indices[-1] - cross_indices[0]
                        features[i, 4] = width
                
                # Energy
                features[i, 5] = np.sum(peak_waveform**2)
                
                # Fill remaining features with sample points from the waveform
                remaining_features = self.config.n_components - 6
                if remaining_features > 0:
                    step = max(1, len(peak_waveform) // remaining_features)
                    samples = peak_waveform[::step][:remaining_features]
                    features[i, 6:6+len(samples)] = samples
            
            return features

    def _cluster_spikes(self, features: np.ndarray) -> np.ndarray:
        """Cluster feature vectors into spike units. Returns integer label per spike."""
        if len(features) == 0:
            return np.array([])
        
        # Get clustering method from config
        method = self.config.clustering_method
        
        # Auto select method based on data size
        if method == ClusteringMethod.AUTO:
            n_spikes = features.shape[0]
            if n_spikes < 1000:
                method = ClusteringMethod.KMEANS
            elif n_spikes < 10000:
                method = ClusteringMethod.GMM
            else:
                method = ClusteringMethod.DBSCAN
        
        # Return noise cluster if scikit-learn not available
        if method == ClusteringMethod.AUTO and not _HAS_SKLEARN:
            return np.zeros(len(features), dtype=np.int64)

        # Perform clustering
        if method == ClusteringMethod.KMEANS:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.config.max_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
        elif method == ClusteringMethod.GMM:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=self.config.max_clusters, random_state=42)
            labels = gmm.fit_predict(features)
        elif method == ClusteringMethod.DBSCAN:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=self.config.isolation_threshold, min_samples=self.config.min_cluster_size)
            labels = dbscan.fit_predict(features)
        elif method == ClusteringMethod.KMEANS:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.config.max_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
        elif method == ClusteringMethod.MEANSHIFT:
            from sklearn.cluster import MeanShift
            meanshift = MeanShift()
            labels = meanshift.fit_predict(features)
        elif method == ClusteringMethod.DBSCAN:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=self.config.isolation_threshold, min_samples=self.config.min_cluster_size)
            labels = dbscan.fit_predict(features)
        elif method == ClusteringMethod.ISO_FOREST:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=self.config.contamination_threshold, random_state=42)
            labels = iso_forest.fit_predict(features)
        elif method == ClusteringMethod.AGGLO:
            from sklearn.cluster import AgglomerativeClustering
            agglo = AgglomerativeClustering(n_clusters=self.config.max_clusters, linkage='ward')
            labels = agglo.fit_predict(features)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        return labels

    def _create_units(self, waveforms: np.ndarray, spike_times: np.ndarray, spike_channels: np.ndarray, features: np.ndarray, labels: np.ndarray) -> List[SpikeUnit]:
        """Group spikes by cluster label into SpikeUnit objects."""
        units = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise label
            
            # Get indices of spikes in this unit
            unit_indices = np.where(labels == label)[0]
            
            # Extract waveforms, times, and channels for this unit
            unit_waveforms = waveforms[unit_indices]
            unit_times = spike_times[unit_indices]
            unit_channels = spike_channels[unit_indices]
            unit_features = features[unit_indices] if features is not None else None
            
            # Create SpikeUnit object
            unit = SpikeUnit(
                unit_id=label,
                waveforms=unit_waveforms,
                timestamps=unit_times,
                channel_ids=unit_channels,
                features=unit_features
            )
            
            units.append(unit)
        
        return units

    def _compute_quality_metrics(self, units: List[SpikeUnit], features: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute real quality metrics for each unit: firing rate, presence ratio, ISI violations, SNR, isolation distance, silhouette score."""
        quality_metrics = {}
        
        # Recording duration in seconds
        if len(units) == 0:
            return quality_metrics
        all_times = np.concatenate([u.timestamps for u in units if len(u.timestamps) > 0])
        if len(all_times) == 0:
            return quality_metrics
        duration_sec = (all_times.max() - all_times.min()) / self.config.sampling_rate
        if duration_sec <= 0:
            duration_sec = 1.0  # Avoid division by zero
        
        # Refractory period in samples (1.5 ms)
        refractory_samples = int(0.0015 * self.config.sampling_rate)
        
        # Compute per-unit silhouette scores if sklearn is available and we have enough labels
        per_unit_silhouette = {}
        unique_labels_clean = np.array([u.unit_id for u in units])
        if _HAS_SKLEARN and len(unique_labels_clean) > 1 and len(features) > 0:
            try:
                # Only use non-noise labels
                valid_mask = labels >= 0
                if valid_mask.sum() > 1 and len(np.unique(labels[valid_mask])) > 1:
                    from sklearn.metrics import silhouette_samples
                    sample_scores = silhouette_samples(features[valid_mask], labels[valid_mask])
                    valid_labels = labels[valid_mask]
                    for uid in unique_labels_clean:
                        uid_mask = valid_labels == uid
                        if uid_mask.sum() > 0:
                            per_unit_silhouette[uid] = float(np.mean(sample_scores[uid_mask]))
            except Exception as e:
                logger.warning(f"Could not compute silhouette scores: {e}")
        
        for unit in units:
            unit_id = unit.unit_id
            unit_metrics = {}
            n_spikes = len(unit.timestamps)
            
            # 1. Firing rate (Hz)
            unit_metrics['firing_rate'] = n_spikes / duration_sec if n_spikes > 0 else 0.0
            
            # 2. Presence ratio: fraction of 1-second bins containing at least 1 spike
            if n_spikes > 0:
                bin_size_samples = int(self.config.sampling_rate)  # 1 second
                min_t, max_t = all_times.min(), all_times.max()
                n_bins = max(1, int(np.ceil((max_t - min_t) / bin_size_samples)))
                spike_bins = ((unit.timestamps - min_t) / bin_size_samples).astype(int)
                spike_bins = np.clip(spike_bins, 0, n_bins - 1)
                occupied_bins = len(np.unique(spike_bins))
                unit_metrics['presence_ratio'] = occupied_bins / n_bins
            else:
                unit_metrics['presence_ratio'] = 0.0
            
            # 3. ISI violations ratio: fraction of spikes with ISI < refractory period
            if n_spikes > 1:
                sorted_times = np.sort(unit.timestamps)
                isis = np.diff(sorted_times)
                n_violations = np.sum(isis < refractory_samples)
                unit_metrics['isi_violations_ratio'] = n_violations / n_spikes
                unit_metrics['isi_violations_count'] = int(n_violations)
            else:
                unit_metrics['isi_violations_ratio'] = 0.0
                unit_metrics['isi_violations_count'] = 0
            
            # 4. SNR: peak amplitude of mean waveform / noise std
            if n_spikes > 0 and unit.waveforms is not None and len(unit.waveforms) > 0:
                try:
                    if unit.waveforms.ndim == 3:
                        # (n_spikes, n_samples, n_channels) -> mean over spikes, take peak channel
                        mean_waveform = np.mean(unit.waveforms, axis=0)  # (n_samples, n_channels)
                        # Find peak channel (largest absolute amplitude)
                        peak_ch = np.argmax(np.max(np.abs(mean_waveform), axis=0))
                        wf = mean_waveform[:, peak_ch]
                    elif unit.waveforms.ndim == 2:
                        wf = np.mean(unit.waveforms, axis=0)
                    else:
                        wf = unit.waveforms.flatten()
                    
                    peak_amplitude = np.max(np.abs(wf))
                    # Noise estimate from first/last 10% of waveform (baseline)
                    n_baseline = max(1, len(wf) // 10)
                    baseline = np.concatenate([wf[:n_baseline], wf[-n_baseline:]])
                    noise_std = np.std(baseline)
                    unit_metrics['snr'] = float(peak_amplitude / noise_std) if noise_std > 0 else 0.0
                except Exception:
                    unit_metrics['snr'] = 0.0
            else:
                unit_metrics['snr'] = 0.0
            
            # 5. Isolation distance (Mahalanobis-based)
            unit_indices = np.where(labels == unit_id)[0]
            other_indices = np.where((labels != unit_id) & (labels >= 0))[0]
            if len(unit_indices) > 1 and len(other_indices) > 1 and len(features) > 0:
                try:
                    unit_feats = features[unit_indices]
                    other_feats = features[other_indices]
                    # Covariance of the unit cluster
                    cov = np.cov(unit_feats.T)
                    if cov.ndim < 2:
                        cov = np.array([[cov]])
                    cov += np.eye(cov.shape[0]) * 1e-6  # Regularize
                    cov_inv = np.linalg.inv(cov)
                    mean_unit = np.mean(unit_feats, axis=0)
                    # Mahalanobis distances of other spikes to unit centroid
                    diff = other_feats - mean_unit
                    mahal_dists = np.sum(diff @ cov_inv * diff, axis=1)
                    mahal_dists_sorted = np.sort(mahal_dists)
                    # Isolation distance = Mahalanobis distance at the n_unit-th nearest other spike
                    idx = min(len(unit_indices), len(mahal_dists_sorted)) - 1
                    unit_metrics['isolation_distance'] = float(mahal_dists_sorted[idx])
                except Exception:
                    unit_metrics['isolation_distance'] = 0.0
            else:
                unit_metrics['isolation_distance'] = 0.0
            
            # 6. Silhouette score (per-unit)
            unit_metrics['silhouette_score'] = per_unit_silhouette.get(unit_id, 0.0)
            
            quality_metrics[unit_id] = unit_metrics
            
            # Also store in the unit object
            unit.quality_metrics = unit_metrics
        
        return quality_metrics