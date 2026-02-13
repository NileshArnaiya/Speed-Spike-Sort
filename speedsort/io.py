"""Data loading from file paths and numpy arrays."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from speedsort.config import DataFormat, SpikeSortingConfiguration, _HAS_MNE

logger = logging.getLogger("speedsort")


def load_data(
    data: Union[str, np.ndarray, Path],
    config: SpikeSortingConfiguration,
    sampling_rate: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load recording from file path or numpy array.

    Returns:
        (samples × channels array, info dict)
    """
    data_info: Dict[str, Any] = {}

    if isinstance(data, np.ndarray):
        if sampling_rate is None and config.sampling_rate is None:
            raise ValueError("Sampling rate must be provided when input is a NumPy array")

        data_array = data
        if sampling_rate:
            data_info['sampling_rate'] = sampling_rate
        elif config.sampling_rate:
            data_info['sampling_rate'] = config.sampling_rate

        # Infer number of channels
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)  # Convert to 2D: samples x 1 channel

        data_info['n_channels'] = data_array.shape[1]
        data_info['n_samples'] = data_array.shape[0]
        data_info['format'] = 'numpy'

        return data_array, data_info

    # Handle file paths
    if isinstance(data, (str, Path)):
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Auto-detect format if not specified
        data_format = config.data_format
        if data_format == DataFormat.AUTO:
            data_format = detect_data_format(data_path)

        # Load data based on format
        if data_format == DataFormat.NUMPY:
            data_array = np.load(data_path)
            data_info['format'] = 'numpy'

        elif data_format == DataFormat.BINARY:
            if not config.n_channels:
                raise ValueError("Number of channels must be specified for binary data")
            dtype = np.float32
            data_array = np.fromfile(data_path, dtype=dtype)
            data_array = data_array.reshape(-1, config.n_channels)
            data_info['format'] = 'binary'

        elif data_format == DataFormat.NEO and _HAS_MNE:
            import neo
            reader = neo.get_io(str(data_path))
            block = reader.read_block()
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
            if _HAS_MNE:
                from mne.io import read_raw_openephys
                raw = read_raw_openephys(data_path)
                data_array = raw.get_data()
                data_info['sampling_rate'] = raw.info['sfreq']
                data_info['format'] = 'open_ephys'
            else:
                raise ImportError("MNE-Python is required for Open Ephys format")

        elif data_format == DataFormat.NWB:
            try:
                import pynwb
                from pynwb import NWBHDF5IO

                io = NWBHDF5IO(str(data_path), 'r')
                nwb_file = io.read()

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
            try:
                data_array = read_mda(data_path)
                data_info['format'] = 'mda'
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


def detect_data_format(filepath: Path) -> DataFormat:
    """Guess data format from file extension (.npy → NUMPY, .nwb → NWB, etc.)."""
    suffix = filepath.suffix.lower()

    if suffix == '.npy':
        return DataFormat.NUMPY
    elif suffix in ('.dat', '.bin'):
        return DataFormat.BINARY
    elif suffix == '.nwb':
        return DataFormat.NWB
    elif suffix == '.mda':
        return DataFormat.MDA
    elif suffix in ('.mat', '.h5', '.hdf5'):
        if _HAS_MNE:
            try:
                import neo
                neo.get_io(str(filepath))
                return DataFormat.NEO
            except Exception:
                pass

    # Check for Open Ephys directory structure
    if filepath.is_dir() and any(filepath.glob('*.continuous')):
        return DataFormat.OPEN_EPHYS

    logger.warning(f"Could not detect format for {filepath}, assuming binary")
    return DataFormat.BINARY


def read_mda(filepath: Path) -> np.ndarray:
    """Read a MountainSort .mda file and return its data as a numpy array."""
    with open(filepath, 'rb') as f:
        header_size = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
        num_dims = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
        dims = np.fromfile(f, dtype=np.dtype('i4'), count=num_dims)
        data_type_code = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]

        data_type = {
            -2: np.dtype('uint8'),
            -3: np.dtype('float32'),
            -4: np.dtype('int16'),
            -5: np.dtype('int32'),
            -6: np.dtype('uint16'),
            -7: np.dtype('double'),
            -8: np.dtype('uint32')
        }.get(data_type_code, np.dtype('float32'))

        data = np.fromfile(f, dtype=data_type)
        data = data.reshape(dims)
        return data
