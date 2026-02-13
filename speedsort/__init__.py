"""
SpeedSort — fast spike sorting for extracellular electrophysiology.

Public API:
    SpeedSort              pipeline class
    SpikeSortingConfiguration   all tunable parameters
    SpikeSortingResults    output container with export/analysis methods
    SpikeUnit              single neuron cluster
"""

from speedsort.config import (
    DataFormat,
    ProcessingBackend,
    DetectionMethod,
    ClusteringMethod,
    DimensionalityReduction,
    SpikeSortingConfiguration,
)
from speedsort.types import SpikeUnit, SpikeSortingResults
from speedsort.core import SpeedSort
from speedsort.curation import auto_merge_units, reject_noise_units
from speedsort.template_matching import template_match_and_subtract
from speedsort.features import compute_spatial_features

__all__ = [
    "SpeedSort",
    "SpikeSortingConfiguration",
    "SpikeSortingResults",
    "SpikeUnit",
    "DataFormat",
    "ProcessingBackend",
    "DetectionMethod",
    "ClusteringMethod",
    "DimensionalityReduction",
    "auto_merge_units",
    "reject_noise_units",
    "template_match_and_subtract",
    "compute_spatial_features",
]

