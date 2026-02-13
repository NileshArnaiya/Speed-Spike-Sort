# SpeedSort

A Python spike sorting pipeline for extracellular electrophysiology recordings.

## What It Does

SpeedSort takes a raw neural recording and produces sorted spike units. The pipeline:

1. **Filter** — Bandpass + notch filter to isolate spike-band activity (default 300–6000 Hz)
2. **Detect** — Find spikes via amplitude thresholding (static, adaptive, NEO, wavelet, or neural net)
3. **Extract** — Cut waveform snippets around each detected spike
4. **Reduce** — Dimensionality reduction on waveforms (PCA, t-SNE, UMAP, or wavelet coefficients)
5. **Cluster** — Group spikes into units (K-means, GMM, DBSCAN, mean shift, agglomerative, etc.)
6. **Score** — Compute isolation distance and contamination metrics per unit

**Input:** `.npy`, `.nwb`, `.mda`, or raw binary files (auto-detected from extension)
**Output:** A `.pkl` file containing sorted units with waveforms, timestamps, features, and quality metrics

GPU acceleration via PyTorch/CuPy is used automatically when available.

## Installation

Requires Python 3.6+.

```bash
git clone https://github.com/NileshArnaiya/Speed-Spike-Sort.git
cd Speed-Spike-Sort
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `pynwb`.
Optional (GPU): `torch`, `cupy`.

## Usage

### Basic

```bash
python spike-sort-run.py --data recording.npy --sampling-rate 30000
```

### Full Options

```bash
python spike-sort-run.py \
  --data recording.nwb \
  --sampling-rate 30000 \
  --data-format nwb \
  --use-gpu \
  --detection-method threshold_dynamic \
  --detection-threshold 4.5 \
  --filter-low 300 \
  --filter-high 6000 \
  --max-clusters 10 \
  --min-cluster-size 30 \
  --output results.pkl
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | — | Path to input data file (required) |
| `--sampling-rate` | 30000 | Sampling rate in Hz |
| `--data-format` | auto | `numpy`, `binary`, `neo`, `nwb`, `mda`, or `auto` |
| `--use-gpu` | off | Use GPU acceleration if available |
| `--jobs` | CPU count - 1 | Number of parallel jobs |
| `--filter-low` | 300 | Bandpass low cutoff (Hz) |
| `--filter-high` | 6000 | Bandpass high cutoff (Hz) |
| `--detection-method` | threshold_dynamic | `threshold`, `threshold_dynamic`, `neo`, `wavelet` |
| `--detection-threshold` | 4.5 | Threshold in MADs |
| `--max-clusters` | 50 | Maximum clusters to detect |
| `--min-cluster-size` | 30 | Minimum spikes per cluster |
| `--output` | spike_sorting_results.pkl | Output file path |

### Download Test Data from DANDI

```bash
python download_dandi_sample.py --dandiset-id 000006 --max-samples 60000
```

This downloads an NWB file from the [DANDI Archive](https://dandiarchive.org/) and extracts a `.npy` snippet you can use for testing.

### Python API

```python
from spikesort import SpeedSort, SpikeSortingConfiguration

config = SpikeSortingConfiguration(sampling_rate=30000, max_clusters=10)
sorter = SpeedSort(config)
results = sorter.run("recording.npy")

print(results.summary())
results.save("results.pkl")
```

## Project Structure

```
spikesort.py              # Core library: data loading, filtering, detection, clustering
spike-sort-run.py         # CLI entry point
download_dandi_sample.py  # Download NWB test data from DANDI
synthetic_spikes.npy      # Bundled synthetic test data
```

## License

GNU General Public License. See [LICENSE](LICENSE).
