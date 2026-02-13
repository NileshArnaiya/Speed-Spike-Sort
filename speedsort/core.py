"""SpeedSort — thin pipeline orchestrator that calls the individual modules."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from speedsort.config import (
    ProcessingBackend,
    SpikeSortingConfiguration,
    _HAS_CUPY,
    _HAS_GPU,
    _HAS_TORCH,
)
from speedsort.types import SpikeSortingResults

# Import pipeline steps
from speedsort.io import load_data
from speedsort.preprocessing import (
    apply_common_reference,
    apply_whitening,
    detect_and_fix_bad_channels,
    filter_data,
)
from speedsort.detection import detect_spikes
from speedsort.features import compute_features, compute_spatial_features, extract_waveforms
from speedsort.clustering import cluster_spikes, create_units
from speedsort.quality import compute_quality_metrics
from speedsort.curation import auto_merge_units, reject_noise_units
from speedsort.template_matching import template_match_and_subtract

if _HAS_CUPY:
    import cupy as cp
if _HAS_TORCH:
    import torch

logger = logging.getLogger("speedsort")


class SpeedSort:
    """Runs the full spike sorting pipeline:
    load → filter → detect → extract → features (+spatial) → cluster
    → template match → quality metrics → auto-merge → noise rejection.
    """

    def __init__(self, config: Optional[SpikeSortingConfiguration] = None):
        """Initialize with a SpikeSortingConfiguration (uses defaults if None)."""
        self.config = config or SpikeSortingConfiguration()
        self.results: Optional[SpikeSortingResults] = None
        self._initialize_backends()

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------

    def _initialize_backends(self) -> None:
        """Set up the array backend (NumPy, CuPy, or PyTorch) based on config."""
        if self.config.backend == ProcessingBackend.CUPY and _HAS_CUPY:
            self.xp = cp
        elif self.config.backend in (ProcessingBackend.TORCH_GPU, ProcessingBackend.TORCH_CPU) and _HAS_TORCH:
            self.xp = torch
            if self.config.backend == ProcessingBackend.TORCH_GPU and _HAS_GPU:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.xp = np
            if self.config.backend not in (ProcessingBackend.CPU, ProcessingBackend.NUMPY):
                logger.warning(f"Requested backend {self.config.backend} not available, using NumPy instead")

        if not hasattr(self, 'device') and _HAS_TORCH:
            self.device = torch.device('cpu')
        elif not hasattr(self, 'device'):
            self.device = None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        data: Union[str, np.ndarray, Path],
        sampling_rate: Optional[float] = None,
    ) -> SpikeSortingResults:
        """Run the full pipeline on a recording file or numpy array. Returns SpikeSortingResults."""
        start = time.time()

        # Step 1: Load data
        raw_data, data_info = load_data(data, self.config, sampling_rate)

        # Resolve sampling rate
        if sampling_rate is not None:
            self.config.sampling_rate = sampling_rate
        elif self.config.sampling_rate is None and 'sampling_rate' in data_info:
            self.config.sampling_rate = data_info['sampling_rate']
        else:
            raise ValueError("Sampling rate must be provided in config or as parameter")

        # Step 2: Filter
        filtered = filter_data(raw_data, self.config, self.xp, self.device)

        n_ch = filtered.shape[1]
        enough_ch = n_ch >= self.config.min_channels_for_preprocessing

        # Step 2b: Bad channel detection
        if self.config.detect_bad_channels and enough_ch:
            logger.info("Detecting bad channels...")
            filtered = detect_and_fix_bad_channels(filtered, self.config, self.xp, self.device)

        # Step 2c: Common average reference
        if self.config.common_reference and enough_ch:
            logger.info(f"Applying common {self.config.common_reference_type} reference...")
            filtered = apply_common_reference(filtered, self.config, self.xp, self.device)

        # Step 2d: Whitening
        if self.config.whiten and enough_ch:
            logger.info("Applying whitening transform...")
            filtered = apply_whitening(filtered, self.config, self.xp, self.device)
        elif self.config.whiten and not enough_ch:
            logger.info(
                f"Skipping preprocessing (CAR/whiten/bad-ch): only {n_ch} channels, "
                f"need ≥{self.config.min_channels_for_preprocessing}"
            )

        # Step 3: Detect spikes
        logger.info("Detecting spikes...")
        spike_times, spike_channels = detect_spikes(filtered, self.config, self.xp, self.device)
        logger.info(f"Detected {len(spike_times)} spikes")

        # Step 4: Extract waveforms
        logger.info("Extracting waveforms...")
        waveforms = extract_waveforms(filtered, spike_times, spike_channels, self.config, self.xp, self.device)

        # Step 5: Compute features
        logger.info("Computing features...")
        features = compute_features(waveforms, self.config, self.xp, self.device)

        # Step 5b: Append spatial features (P3)
        if self.config.use_spatial_features and len(waveforms) > 0 and waveforms.ndim == 3:
            logger.info("Computing spatial features...")
            spatial = compute_spatial_features(
                waveforms, spike_channels, self.config.n_neighbor_channels
            )
            if len(features) > 0 and len(spatial) == len(features):
                features = np.hstack([features, spatial])

        # Step 6: Cluster
        logger.info("Clustering spikes...")
        labels = cluster_spikes(features, self.config)

        # Step 7: Create units
        logger.info("Creating units...")
        units = create_units(waveforms, spike_times, spike_channels, features, labels)

        # Step 7b: Template matching / deconvolution (P3)
        if self.config.template_matching and len(units) > 0:
            logger.info("Running template matching...")
            units, _ = template_match_and_subtract(
                filtered, units, self.config,
                max_iterations=self.config.template_matching_iterations,
                residual_threshold_factor=self.config.template_residual_threshold,
            )

        # Step 8: Quality metrics
        if self.config.compute_quality_metrics:
            logger.info("Computing quality metrics...")
            qm = compute_quality_metrics(units, features, labels, self.config)
        else:
            qm = {}

        # Step 8b: Auto-merge similar units (P3)
        if self.config.auto_merge and len(units) > 1:
            logger.info("Auto-merging similar units...")
            units = auto_merge_units(units, self.config)

        # Step 8c: Noise rejection (P3)
        if self.config.noise_rejection and len(units) > 0:
            logger.info("Rejecting noise units...")
            units = reject_noise_units(units, self.config)

        # Recompute quality metrics if curation changed the units
        if (self.config.auto_merge or self.config.noise_rejection) and self.config.compute_quality_metrics:
            if len(units) > 0:
                # Rebuild labels from remaining units
                all_feats = []
                all_labels = []
                for unit in units:
                    if unit.features is not None and len(unit.features) > 0:
                        all_feats.append(unit.features)
                        all_labels.extend([unit.unit_id] * len(unit.features))
                if all_feats:
                    features = np.vstack(all_feats)
                    labels = np.array(all_labels)
                qm = compute_quality_metrics(units, features, labels, self.config)

        execution_time = time.time() - start

        # Step 9: Package results
        self.results = SpikeSortingResults(
            units=units,
            config=self.config,
            execution_time=execution_time,
            sampling_rate=self.config.sampling_rate,
            data_info=data_info,
            quality_metrics=qm,
        )

        logger.info(f"Spike sorting completed in {execution_time:.2f} seconds")
        logger.info(f"Found {len(units)} units with {sum(len(u.timestamps) for u in units)} spikes")

        return self.results
