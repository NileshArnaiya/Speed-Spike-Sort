# SpeedSort: High-Performance Neural Spike Sorting Framework

## Overview

SpeedSort is a high-performance framework for neural spike sorting, designed to handle various electrophysiology data formats with minimal configuration requirements. It supports GPU acceleration, adaptive dimensionality reduction, and integrated quality metrics for automated unit verification.

## Features

- Universal data format handling with automatic detection
- GPU-accelerated preprocessing and clustering when available
- Adaptive dimensionality reduction and feature selection
- Parallelized processing for multi-channel recordings
- Integrated quality metrics with automated unit verification
- Minimal configuration with intelligent defaults

## Installation

To run this project, you need to have Python 3.6 or higher installed. You can clone the repository and install the required dependencies using pip.

1. Clone the repository:
   ```bash
   git clone https://github.com/NileshArnaiya/Speed-Spike-Sort.git
   cd speed-spike-sort
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Make sure to install `pynwb`, `numpy`, `scipy`, `pandas`, and `scikit-learn`. If you want to use GPU acceleration, install `torch` and `cupy` as well.

## Usage

To run the spike sorting process, you can use the provided `spike-sort-run.py` script. Here's how to execute it:

```bash
python feuapp/spike-sort-run.py --data <path_to_your_data_file> --sampling-rate <sampling_rate> --data-format <data_format>
```

### Command Line Arguments

- `--data`: Path to the input data file (required).
- `--sampling-rate`: Sampling rate in Hz (default: 30000).
- `--data-format`: Data format (options: `numpy`, `binary`, `neo`, `nwb`, `mda`, `auto`, default: `auto`).
- `--use-gpu`: Use GPU acceleration if available (optional).
- `--jobs`: Number of parallel jobs (default: number of CPU cores - 1).
- `--filter-low`: Low cutoff frequency in Hz (default: 300).
- `--filter-high`: High cutoff frequency in Hz (default: 6000).
- `--detection-method`: Spike detection method (default: `threshold_dynamic`).
- `--max-clusters`: Maximum number of clusters to detect (default: 50).
- `--min-cluster-size`: Minimum number of spikes per cluster (default: 30).
- `--output`: Path to save results (default: `spike_sorting_results.pkl`).

### Example

```bash
python spike-sort-run.py --data test_data/random_numpy_data.npy --sampling-rate 30000 --max-clusters 5 --data-format nwb
```

## Contributing

We welcome contributions to SpeedSort! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push your changes to your fork:
   ```bash
   git push origin feature/my-feature
   ```
5. Create a pull request describing your changes.

## License

This project is licensed under the GNU Public License. See the LICENSE file for details.

## Acknowledgments

- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [PyTorch](https://pytorch.org/)
- [CuPy](https://cupy.chainer.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [MNE-Python](https://mne.tools/stable/index.html)
