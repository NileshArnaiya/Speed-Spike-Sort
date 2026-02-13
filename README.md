# SpeedSort

A fast, modular Python spike sorting pipeline for extracellular electrophysiology.

## Pipeline

```
Raw recording
  │
  ├─ 1. Bandpass + notch filter (300–6000 Hz)
  ├─ 2. Bad channel detection & interpolation
  ├─ 3. Common average/median reference (CAR)
  ├─ 4. ZCA whitening
  │
  ├─ 5. Spike detection (adaptive threshold, NEO, wavelet, or neural net)
  ├─ 6. Waveform extraction
  ├─ 7. Feature reduction (PCA / t-SNE / UMAP / wavelet) + spatial features
  ├─ 8. Clustering (K-means, GMM, DBSCAN, mean shift, agglomerative)
  │
  ├─ 9.  Template matching & deconvolution (recovers overlapping spikes)
  ├─ 10. Quality metrics (SNR, ISI violations, firing rate, isolation distance, …)
  ├─ 11. Auto-merge similar units
  └─ 12. Noise rejection
       │
       ▼
  SpikeSortingResults → .pkl, NWB, Pynapple, DataFrame
```

**Input:** `.npy`, `.nwb`, `.mda`, or raw binary files (auto-detected).  
**Output:** Sorted units with waveforms, timestamps, features, and quality metrics.  
GPU acceleration via PyTorch/CuPy is used automatically when available.

## Installation

```bash
git clone https://github.com/NileshArnaiya/Speed-Spike-Sort.git
cd Speed-Spike-Sort
pip install -r requirements.txt
```

Core: `numpy`, `scipy`, `pandas`, `scikit-learn`, `pynwb`.  
Optional: `torch`, `cupy` (GPU), `pynapple` (analysis), `matplotlib` (plotting).

## Quick Start

### CLI

```bash
python spike-sort-run.py --data recording.npy --sampling-rate 30000
```

### Python API

```python
from speedsort import SpeedSort, SpikeSortingConfiguration

config = SpikeSortingConfiguration(sampling_rate=30000)
sorter = SpeedSort(config)
results = sorter.run("recording.npy")

print(results.summary())
results.save("results.pkl")
```

The old import path still works for backward compatibility:

```python
from spikesort import SpeedSort, SpikeSortingConfiguration  # also works
```

## Preprocessing

Preprocessing is applied after bandpass filtering when ≥ 4 channels are present.

```python
config = SpikeSortingConfiguration(
    sampling_rate=30000,
    # Bad channel detection
    detect_bad_channels=True,           # detect dead/noisy channels via variance
    bad_channel_std_threshold=5.0,      # MAD-based z-score threshold

    # Common reference
    common_reference=True,              # subtract median across channels
    common_reference_type='median',     # 'median' or 'mean'

    # Whitening
    whiten=True,                        # ZCA whitening to decorrelate channels
)
```

## Spike Detection

Six detection methods are available:

```python
from speedsort import SpikeSortingConfiguration

config = SpikeSortingConfiguration(
    detection_method='threshold_dynamic',  # adaptive threshold per time window
    # Also: 'threshold', 'neo', 'wavelet', 'neural_net', 'template'
    detection_threshold=4.5,               # in MADs
)
```

## Clustering

```python
config = SpikeSortingConfiguration(
    clustering_method='auto',  # auto-selects based on spike count
    # Also: 'kmeans', 'gmm', 'dbscan', 'meanshift', 'agglomerative', 'isoforest'
    max_clusters=50,
    min_cluster_size=30,
)
```

## Spatial Features

Multi-channel spatial features are appended alongside PCA/wavelet features to improve cluster separation:

```python
config = SpikeSortingConfiguration(
    use_spatial_features=True,   # append peak channel, spatial spread,
                                 # amplitude ratio, center-of-mass
    n_neighbor_channels=3,       # channels each side for spatial analysis
)
```

## Template Matching / Deconvolution

After initial clustering, SpeedSort performs iterative template subtraction to recover overlapping spikes:

```python
config = SpikeSortingConfiguration(
    template_matching=True,              # enable deconvolution
    template_matching_iterations=2,      # subtraction passes
    template_residual_threshold=0.7,     # lower threshold for residual detection
)
```

## Auto-Merge & Noise Rejection

Units are automatically curated after quality metrics are computed:

```python
config = SpikeSortingConfiguration(
    # Auto-merge: merge units with very similar templates
    auto_merge=True,
    merge_template_threshold=0.92,       # cosine similarity threshold

    # Noise rejection: remove bad units
    noise_rejection=True,
    noise_isi_threshold=0.05,            # max 5% ISI violations
    noise_snr_threshold=1.5,             # min SNR
    noise_firing_rate_bounds=(0.1, 200), # valid firing rate range (Hz)
)
```

## Quality Metrics

Six real quality metrics are computed for every unit:

| Metric | What it measures |
|--------|-----------------|
| `firing_rate` | Spikes per second |
| `presence_ratio` | Fraction of 1 s bins with ≥ 1 spike |
| `isi_violations_ratio` | Fraction of ISI < 1.5 ms (refractory) |
| `snr` | Peak amplitude / baseline noise |
| `isolation_distance` | Mahalanobis separation from other clusters |
| `silhouette_score` | Cluster cohesion vs. separation |

## Export & Analysis

```python
# Pandas DataFrame
df = results.to_dataframe()

# Pynapple TsGroup (requires pynapple)
tsgroup = results.to_pynapple()

# NWB file (requires pynwb)
results.to_nwb("sorted_output.nwb")

# Auto-correlogram, cross-correlogram, ISI histogram
acg_counts, acg_bins = results.compute_acg(unit_id=0)
ccg_counts, ccg_bins = results.compute_ccg(unit_id_a=0, unit_id_b=1)
isi_counts, isi_bins = results.compute_isi_histogram(unit_id=0)

# 3-panel unit summary plot (waveform, ACG, ISI)
results.plot_unit_summary(unit_id=0, save_path="unit_0.png")
```

## Download Test Data

```bash
python download_dandi_sample.py --dandiset-id 000006 --max-samples 60000
```

Downloads an NWB file from the [DANDI Archive](https://dandiarchive.org/) and extracts a `.npy` snippet for testing.

## Package Structure

```
speedsort/                 # Modular package
├── __init__.py            # Public API re-exports
├── config.py              # Enums + SpikeSortingConfiguration
├── types.py               # SpikeUnit, SpikeSortingResults
├── io.py                  # Data loading (numpy, NWB, NEO, MDA, binary)
├── preprocessing.py       # Filter, CAR, whitening, bad channel detection
├── detection.py           # Spike detection methods
├── features.py            # Waveform extraction + dimensionality reduction + spatial
├── clustering.py          # Clustering methods + unit creation
├── quality.py             # Quality metrics
├── curation.py            # Auto-merge + noise rejection
├── template_matching.py   # Iterative template subtraction / deconvolution
└── core.py                # SpeedSort pipeline orchestrator

spikesort.py               # Legacy monolith (backward compatible)
spike-sort-run.py          # CLI entry point
download_dandi_sample.py   # DANDI data downloader
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | — | Path to input data file (required) |
| `--sampling-rate` | 30000 | Sampling rate in Hz |
| `--data-format` | auto | `numpy`, `binary`, `neo`, `nwb`, `mda`, or `auto` |
| `--use-gpu` | off | Use GPU acceleration if available |
| `--detection-method` | threshold_dynamic | `threshold`, `threshold_dynamic`, `neo`, `wavelet` |
| `--detection-threshold` | 4.5 | Threshold in MADs |
| `--max-clusters` | 50 | Maximum clusters to detect |
| `--min-cluster-size` | 30 | Minimum spikes per cluster |
| `--output` | spike_sorting_results.pkl | Output file path |

## License

GNU General Public License. See [LICENSE](LICENSE).
