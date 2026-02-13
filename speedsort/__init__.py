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
]
