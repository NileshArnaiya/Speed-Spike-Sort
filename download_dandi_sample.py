#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download NWB sample files from the DANDI Archive and extract numpy arrays for testing the spike sorter."""

import os
import sys
import argparse
from pathlib import Path
from dandi.dandiapi import DandiAPIClient
import pynwb
from pynwb import NWBHDF5IO
import numpy as np
import h5py

# Create directory if it doesn't exist
DOWNLOAD_DIR = Path("dandi_test_data")
DOWNLOAD_DIR.mkdir(exist_ok=True)

def download_dandi_file(dandiset_id="000006", file_path=None):
    """
    Download a sample NWB file from DANDI archive.
    
    Args:
        dandiset_id: The DANDI dataset ID
        file_path: Specific file to download. If None, a suitable NWB file is downloaded.
    
    Returns:
        Path to the downloaded file
    """
    print(f"Connecting to DANDI API to download a sample file from dandiset {dandiset_id}...")
    
    with DandiAPIClient() as client:
        # Get dandiset
        dandiset = client.get_dandiset(dandiset_id)
        print(f"Found dandiset: {dandiset.identifier}")
        
        # Get assets (files) in the dandiset
        print("Fetching assets list (this may take a while for large dandisets)...")
        assets = list(dandiset.get_assets())
        print(f"Found {len(assets)} assets in the dandiset")
        
        # Find NWB files
        nwb_assets = [asset for asset in assets if asset.path.endswith(".nwb")]
        print(f"Found {len(nwb_assets)} NWB files")
        
        if not nwb_assets:
            print(f"No NWB files found in dandiset {dandiset_id}")
            sys.exit(1)
        
        # Select specific file or a suitable NWB file
        if file_path:
            asset_to_download = next((a for a in nwb_assets if file_path in a.path), None)
            if not asset_to_download:
                print(f"Could not find file matching {file_path}")
                sys.exit(1)
        else:
            # For dandiset 000006, choose a good test file
            # These are known good files with electrophysiology data
            suggested_files = [
                "sub-anm369962_ses-20170310.nwb",  # Try this one first
                "sub-anm369963_ses-20170309.nwb"   # Backup option
            ]
            
            for suggested in suggested_files:
                asset_to_download = next((a for a in nwb_assets if suggested in a.path), None)
                if asset_to_download:
                    print(f"Selected known good test file: {suggested}")
                    break
            
            # If no suggested file found
            if not asset_to_download:
                # Look for small files
                for asset in nwb_assets:
                    if hasattr(asset, 'blob') and hasattr(asset.blob, 'size') and asset.blob.size < 50_000_000:  # Less than 50MB
                        asset_to_download = asset
                        break
                else:
                    # If no suitable file found, take the first one
                    asset_to_download = nwb_assets[0]
        
        # Output file path
        output_path = DOWNLOAD_DIR / f"dandi_{dandiset_id}_{Path(asset_to_download.path).name}"
        
        # Download the file if it doesn't exist
        if not output_path.exists():
            print(f"Downloading {asset_to_download.path}...")
            if hasattr(asset_to_download, 'blob') and hasattr(asset_to_download.blob, 'size'):
                size_mb = asset_to_download.blob.size / 1_000_000
                print(f"File size: {size_mb:.2f} MB")
            
            # Use the download method provided by the asset
            try:
                asset_to_download.download(output_path)
                print(f"Downloaded to {output_path}")
            except Exception as e:
                print(f"Error with primary download method: {e}")
                print("Trying alternative download method...")
                try:
                    with open(output_path, 'wb') as f:
                        client.download_asset_file(asset_to_download, f)
                    print(f"Downloaded to {output_path}")
                except Exception as e:
                    print(f"Error downloading file: {e}")
                    sys.exit(1)
        else:
            print(f"File already exists at {output_path}")
        
        return output_path

def analyze_nwb_file(file_path):
    """
    Analyze the NWB file to find electrophysiology data.
    
    Args:
        file_path: Path to the NWB file
    
    Returns:
        Basic information about the file
    """
    print(f"\nAnalyzing NWB file: {file_path}")
    
    # First, try to use low-level h5py to directly inspect file structure
    # This is helpful for datasets with unusual structures
    try:
        print("Examining file structure with h5py...")
        with h5py.File(file_path, 'r') as f:
            # Check for common electrophysiology data paths in DANDI 000006
            ephys_paths = []
            
            # Search for common electrophysiology data paths
            for path in [
                '/acquisition/ElectricalSeries',
                '/acquisition/e-series',
                '/acquisition/ElectricalSeries/data',
                '/processing/ecephys',
                '/processing/ecephys/LFP',
                '/processing/ecephys/LFP/electrical_series',
                '/processing/ecephys/LFP/electrical_series/data',
                '/intervals/trials',
                '/intervals/trials/ElectricalSeries',
                '/stimulus/presentation',
                '/stimulus/presentation/ElectricalSeries',
                '/analysis/spike_times'
            ]:
                if path in f:
                    print(f"Found potential data path: {path}")
                    if isinstance(f[path], h5py.Dataset) and len(f[path].shape) > 0:
                        print(f"  Shape: {f[path].shape}, Type: {f[path].dtype}")
                        ephys_paths.append(path)
            
            # If we found direct paths, get the first viable one
            if ephys_paths:
                for path in ephys_paths:
                    if path in f and isinstance(f[path], h5py.Dataset) and len(f[path].shape) > 0:
                        if 'rate' in f[path].attrs:
                            rate = f[path].attrs['rate']
                        elif 'starting_time' in f[path].attrs and 'rate' in f[path].attrs:
                            rate = f[path].attrs['rate']
                        else:
                            rate = 30000  # Default rate
                        
                        return {
                            'path': file_path,
                            'data_path': path,
                            'h5py_direct': True,
                            'shape': f[path].shape,
                            'sampling_rate': rate
                        }
            
            # If we didn't find direct paths, look for any dataset with likely shapes
            print("Searching for any dataset with suitable shape...")
            found_data = []
            
            def explore_group(group, path=''):
                for key in group.keys():
                    item_path = f"{path}/{key}" if path else key
                    item = group[key]
                    
                    if isinstance(item, h5py.Dataset):
                        if len(item.shape) >= 1 and item.shape[0] > 1000:  # Time series typically have many samples
                            found_data.append((item_path, item.shape, item.dtype))
                    elif isinstance(item, h5py.Group):
                        explore_group(item, item_path)
            
            explore_group(f)
            
            if found_data:
                print("Found datasets with suitable shapes:")
                for path, shape, dtype in found_data:
                    print(f"  {path}: Shape {shape}, Type {dtype}")
                
                # Choose the first dataset with substantial data
                chosen_path, chosen_shape, _ = found_data[0]
                print(f"Using dataset: {chosen_path}")
                
                return {
                    'path': file_path,
                    'data_path': chosen_path,
                    'h5py_direct': True,
                    'shape': chosen_shape,
                    'sampling_rate': 30000  # Default rate
                }
    
    except Exception as e:
        print(f"Error in h5py inspection: {e}")
    
    # Fall back to pynwb if h5py direct inspection fails
    try:
        print("\nFalling back to pynwb inspection...")
        with NWBHDF5IO(file_path, 'r') as io:
            nwb_file = io.read()
            
            # Print basic info
            try:
                if hasattr(nwb_file, 'nwb_version'):
                    print(f"NWB version: {nwb_file.nwb_version}")
                elif hasattr(nwb_file, 'container_source') and hasattr(nwb_file.container_source, 'version'):
                    print(f"NWB version: {nwb_file.container_source.version}")
                else:
                    print("NWB version: Unknown")
                
                if hasattr(nwb_file, 'session_id'):
                    print(f"Session ID: {nwb_file.session_id}")
                
                if hasattr(nwb_file, 'session_description'):
                    print(f"Session description: {nwb_file.session_description}")
            except Exception as e:
                print(f"Error getting basic info: {e}")
            
            # Look for electrophysiology data in acquisition
            for section_name, section in [
                ('acquisition', nwb_file.acquisition if hasattr(nwb_file, 'acquisition') else {}),
                ('processing', nwb_file.processing if hasattr(nwb_file, 'processing') else {})
            ]:
                print(f"\n{section_name.capitalize()} data:")
                if section:
                    for name, item in section.items():
                        if hasattr(item, 'data') and hasattr(item.data, 'shape'):
                            print(f"  - {name}: {type(item).__name__}")
                            print(f"    Shape: {item.data.shape}, Type: {item.data.dtype}")
                            
                            rate = 30000  # Default rate
                            if hasattr(item, 'rate'):
                                rate = item.rate
                                print(f"    Sampling rate: {rate} Hz")
                            
                            return {
                                'path': file_path,
                                'data_key': name,
                                'h5py_direct': False,
                                'shape': item.data.shape,
                                'sampling_rate': rate
                            }
                
                print(f"  No suitable data found in {section_name}")
            
            print("\nNo suitable electrophysiology data found in this file")
            return None
            
    except Exception as e:
        print(f"Error analyzing NWB file with pynwb: {e}")
        return None

def extract_data_for_testing(nwb_info, output_dir=DOWNLOAD_DIR, max_samples=60000):
    """
    Extract data from the NWB file for testing with the spike sorting code.
    
    Args:
        nwb_info: Information about the NWB file
        output_dir: Directory to save the extracted data
        max_samples: Maximum number of samples to extract
    
    Returns:
        Path to the extracted data
    """
    if not nwb_info:
        return None
    
    try:
        file_path = nwb_info['path']
        
        # Check if we're using h5py direct access
        if nwb_info.get('h5py_direct', False):
            data_path = nwb_info['data_path']
            
            print(f"\nExtracting data from {file_path} using h5py direct access")
            print(f"Data path: {data_path}")
            
            with h5py.File(file_path, 'r') as f:
                # Extract a subset of the data
                data_shape = nwb_info['shape']
                samples_to_extract = min(max_samples, data_shape[0])
                
                try:
                    print(f"Extracting {samples_to_extract} samples...")
                    data = f[data_path][:samples_to_extract]
                except (MemoryError, ValueError) as me:
                    print(f"Warning: {me}. Trying to extract fewer samples...")
                    samples_to_extract = min(10000, data_shape[0])
                    data = f[data_path][:samples_to_extract]
                
                # If data is 1D, reshape to 2D for spike sorting
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                
                # Save as numpy array
                output_path = output_dir / f"ecephys_data_{samples_to_extract}_{data.shape[1] if len(data.shape) > 1 else 1}.npy"
                np.save(output_path, data)
                
                print(f"Extracted {samples_to_extract} samples, shape: {data.shape}")
                print(f"Saved to {output_path}")
                
                # Create a metadata file
                meta_path = output_dir / f"ecephys_data_{samples_to_extract}_{data.shape[1] if len(data.shape) > 1 else 1}_metadata.npz"
                np.savez(meta_path,
                         sampling_rate=nwb_info['sampling_rate'],
                         original_shape=data_shape,
                         extracted_samples=samples_to_extract)
                
                print(f"Metadata saved to {meta_path}")
                
                return output_path
        else:
            # Regular pynwb access
            data_key = nwb_info['data_key']
            
            print(f"\nExtracting data from {file_path} using pynwb")
            print(f"Data key: {data_key}")
            
            with NWBHDF5IO(file_path, 'r') as io:
                nwb_file = io.read()
                
                # Navigate to the data
                if '/' in data_key:
                    module_name, data_name = data_key.split('/')
                    data_obj = nwb_file.processing[module_name].data_interfaces[data_name]
                else:
                    data_obj = nwb_file.acquisition[data_key]
                
                # Extract a subset of the data
                data_shape = data_obj.data.shape
                samples_to_extract = min(max_samples, data_shape[0])
                
                # Extract data
                try:
                    print(f"Extracting {samples_to_extract} samples...")
                    data = data_obj.data[:samples_to_extract]
                except (MemoryError, ValueError) as me:
                    print(f"Warning: {me}. Trying to extract fewer samples...")
                    samples_to_extract = min(10000, data_shape[0])
                    data = data_obj.data[:samples_to_extract]
                
                # If data is 1D, reshape to 2D for spike sorting
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                
                # Save as numpy array
                output_path = output_dir / f"ecephys_data_{samples_to_extract}_{data.shape[1]}.npy"
                np.save(output_path, data)
                
                print(f"Extracted {samples_to_extract} samples for {data.shape[1]} channels")
                print(f"Saved to {output_path}")
                
                # Create a metadata file
                meta_path = output_dir / f"ecephys_data_{samples_to_extract}_{data.shape[1]}_metadata.npz"
                np.savez(meta_path,
                         sampling_rate=nwb_info['sampling_rate'],
                         original_shape=data_shape,
                         extracted_samples=samples_to_extract)
                
                print(f"Metadata saved to {meta_path}")
                
                return output_path
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download and process a sample NWB file from DANDI archive')
    parser.add_argument('--dandiset-id', type=str, default="000006", 
                        help='DANDI dataset ID (default: 000006)')
    parser.add_argument('--file-path', type=str, default=None,
                        help='Specific file to download (optional)')
    parser.add_argument('--max-samples', type=int, default=60000,
                        help='Maximum number of samples to extract (default: 60000)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force download even if file exists')
    
    args = parser.parse_args()
    
    # Download the file
    nwb_file_path = download_dandi_file(args.dandiset_id, args.file_path)
    
    # Analyze the file
    nwb_info = analyze_nwb_file(nwb_file_path)
    
    # Extract data for testing
    if nwb_info:
        extract_data_for_testing(nwb_info, max_samples=args.max_samples)
    else:
        print("\nNo suitable electrophysiology data found to extract.")
        print("Try specifying a different file with --file-path or using a different dandiset with --dandiset-id")

if __name__ == "__main__":
    main()