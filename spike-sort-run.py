#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from bro import SpeedSort, SpikeSortingConfiguration, DataFormat, DetectionMethod, ProcessingBackend
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run neural spike sorting with SpeedSort.')
    
    # Input data arguments
    parser.add_argument('--data', type=str, help='Path to the input data file')
    parser.add_argument('--sampling-rate', type=float, default=30000.0, 
                        help='Sampling rate in Hz (default: 30000)')
    parser.add_argument('--data-format', type=str, choices=['numpy', 'binary', 'neo', 'nwb', 'mda', 'auto'], 
                        default='auto', help='Data format (default: auto)')
    
    # Processing settings
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--jobs', type=int, default=0, 
                        help='Number of parallel jobs (default: number of CPU cores - 1)')
    
    # Filter settings
    parser.add_argument('--filter-low', type=float, default=300.0, 
                        help='Low cutoff frequency in Hz (default: 300)')
    parser.add_argument('--filter-high', type=float, default=6000.0, 
                        help='High cutoff frequency in Hz (default: 6000)')
    parser.add_argument('--no-notch-filter', action='store_true', 
                        help='Disable notch filter for line noise')
    
    # Detection settings
    parser.add_argument('--detection-method', type=str, 
                        choices=['threshold', 'threshold_dynamic', 'neo', 'wavelet'], 
                        default='threshold_dynamic', help='Spike detection method (default: threshold_dynamic)')
    parser.add_argument('--detection-threshold', type=float, default=4.5, 
                        help='Detection threshold in MADs (default: 4.5)')
    
    # Clustering settings
    parser.add_argument('--max-clusters', type=int, default=50,
                        help='Maximum number of clusters to detect (default: 50)')
    parser.add_argument('--min-cluster-size', type=int, default=30,
                        help='Minimum number of spikes per cluster (default: 30)')
    
    # Output settings
    parser.add_argument('--output', type=str, default='spike_sorting_results.pkl', 
                        help='Path to save results (default: spike_sorting_results.pkl)')
    parser.add_argument('--no-save-waveforms', action='store_true', 
                        help='Do not save waveforms in results')
    parser.add_argument('--no-quality-metrics', action='store_true', 
                        help='Do not compute quality metrics')
    
    args = parser.parse_args()
    
    # Validate data path if provided
    if args.data and not os.path.exists(args.data):
        parser.error(f"Data file not found: {args.data}")
    
    return args


def create_config_from_args(args):
    """Create SpikeSortingConfiguration from command line arguments."""
    # Map string choices to enum values
    data_format_map = {
        'numpy': DataFormat.NUMPY,
        'binary': DataFormat.BINARY,
        'neo': DataFormat.NEO,
        'nwb': DataFormat.NWB,
        'mda': DataFormat.MDA,
        'auto': DataFormat.AUTO,
    }
    
    detection_method_map = {
        'threshold': DetectionMethod.THRESHOLD,
        'threshold_dynamic': DetectionMethod.THRESHOLD_DYNAMIC,
        'neo': DetectionMethod.NEO,
        'wavelet': DetectionMethod.WAVELET,
    }
    
    # Determine number of jobs
    n_jobs = args.jobs if args.jobs > 0 else max(1, os.cpu_count() - 1)
    
    # Create configuration
    config = SpikeSortingConfiguration(
        # Basic settings
        sampling_rate=args.sampling_rate,
        data_format=data_format_map.get(args.data_format, DataFormat.AUTO),
        
        # Processing settings
        n_jobs=n_jobs,
        use_gpu=args.use_gpu,
        
        # Filter settings
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        notch_filter=not args.no_notch_filter,
        
        # Detection settings
        detection_method=detection_method_map.get(args.detection_method, DetectionMethod.THRESHOLD_DYNAMIC),
        detection_threshold=args.detection_threshold,
        
        # Clustering settings
        max_clusters=args.max_clusters,
        min_cluster_size=args.min_cluster_size,
        
        # Feature and clustering settings
        compute_quality_metrics=not args.no_quality_metrics,
        save_waveforms=not args.no_save_waveforms,
    )
    
    return config


def plot_spike_times(spike_times, output_path):
    """
    Plot spike times and save the figure.
    
    Args:
        spike_times: Array of spike timestamps.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.eventplot(spike_times, color='black', linewidths=0.5)
    plt.title('Spike Times')
    plt.xlabel('Time (samples)')
    plt.ylabel('Spike Events')
    plt.xlim(0, max(spike_times) + 1000)  # Adjust x-axis limit as needed
    plt.grid(True)
    
    # Save the plot
    plot_path = output_path.replace('.pkl', '_spike_times.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Spike times plot saved to {plot_path}")


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Initialize SpeedSort
    sorter = SpeedSort(config)
    
    # Load data from command line argument or use default path
    data_path = args.data if args.data else Path("./test_data/random_numpy_data.npy")
    data_path = Path(data_path)
    
    try:
        # Load data based on format
        if data_path.exists():
            print(f"Loading data from {data_path}...")
            
            # Handle different data formats
            if args.data_format == 'nwb':
                try:
                    print("Loading NWB file...")
                    import pynwb
                    from pynwb import NWBHDF5IO
                    
                    io = NWBHDF5IO(str(data_path), 'r')
                    nwb_file = io.read()
                    
                    # Get electrical series
                    found_data = False
                    for electrical_series in nwb_file.acquisition.values():
                        if isinstance(electrical_series, pynwb.ecephys.ElectricalSeries):
                            data = electrical_series.data[:]
                            if args.sampling_rate is None:
                                args.sampling_rate = float(electrical_series.rate)
                            print(f"Found electrical series with shape {data.shape}")
                            print(f"Sampling rate: {args.sampling_rate} Hz")
                            found_data = True
                            break
                    
                    if not found_data:
                        raise ValueError("No electrical series found in NWB file")
                    
                except ImportError:
                    print("Error: pynwb library required for NWB format")
                    print("Please install it with: pip install pynwb")
                    sys.exit(1)
            elif data_path.suffix == '.p' or data_path.suffix == '.pkl' or data_path.suffix == '.pickle':
                import pickle
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded pickled data with type {type(data)}")
                
                # Convert list to numpy array if necessary
                if isinstance(data, list):
                    try:
                        data = np.array(data)
                        print(f"Converted list to numpy array with shape {data.shape}")
                    except:
                        raise ValueError("Could not convert list data to numpy array")
                
                if isinstance(data, np.ndarray):
                    print(f"Array shape: {data.shape}")
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}, must be convertible to numpy array")
            else:
                # Default to numpy loading for npy files and others
                data = np.load(data_path)
                print(f"Loaded data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Run spike sorting
        print("Running spike sorting...")
        results = sorter.run(data, sampling_rate=args.sampling_rate)
        
        # Print results summary
        summary = results.summary()
        print("\nSpike Sorting Results Summary:")
        print(f"Total units found: {summary['total_units']}")
        print(f"Total spikes detected: {summary['total_spikes']}")
        print(f"Execution time: {summary['execution_time']:.2f} seconds")
        
        # Print info about each unit
        if summary['total_units'] > 0:
            print("\nUnit Information:")
            for unit in results.units:
                print(f"Unit {unit.unit_id}: {len(unit.timestamps)} spikes, "
                      f"mean rate: {summary['mean_firing_rates'][unit.unit_id]:.2f} Hz")
                
                if unit.quality_metrics:
                    print("  Quality metrics:")
                    for metric, value in unit.quality_metrics.items():
                        print(f"    {metric}: {value:.3f}")
        
        # Save results
        output_path = args.output
        print(f"\nSaving results to {output_path}")
        results.save(output_path)
        print("Done!")
        
        # Call the plotting function only if there are detected units
        if summary['total_units'] > 0:
            all_spike_times = np.concatenate([unit.timestamps for unit in results.units])  # Collect all spike times
            plot_spike_times(all_spike_times, output_path)
        else:
            print("No spike units detected. Skipping plot.")
        
    except Exception as e:
        print(f"Error during spike sorting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()