"""Data containers: SpikeUnit and SpikeSortingResults."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from speedsort.config import SpikeSortingConfiguration

logger = logging.getLogger("speedsort")


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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str) -> None:
        """Save results to disk."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "SpikeSortingResults":
        """Load results from disk."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

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
        channel_counts: Dict[int, int] = {}
        for unit in self.units:
            if len(unit.channel_ids) > 0:
                primary_channel = unit.channel_ids[0]
                channel_counts[primary_channel] = channel_counts.get(primary_channel, 0) + 1
        return channel_counts

    def _calc_mean_firing_rates(self) -> Dict[int, float]:
        """Calculate mean firing rate for each unit."""
        rates: Dict[int, float] = {}
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
