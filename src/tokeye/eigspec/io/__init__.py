"""
Input/Output module for eigspec package.

This module provides comprehensive data I/O functionality for scientific data formats:
- Data loading from multiple formats (HDF5, NetCDF, MATLAB, ASCII, CSV)
- Data export and saving with format validation
- Data structure definitions for time series and analysis results
- Format detection and compatibility checking
- Data validation and processing utilities

Based on the MATLAB eigspec toolbox I/O functions:
- fetch_mpi_arrays_ptdata64.m - MPI array data loading
- bundle_shot_data.m - Shot data bundling and organization
- savebdotimage.m - B-dot image saving and export
- eigspec_mmain.m - Main I/O workflow coordination
- collect_rep_data.m - Analysis result data collection
- Various file format handlers throughout the MATLAB codebase
"""

from .data_structures import (
    ShotData,
    ProcessedData, 
    AnalysisResult,
    TimeSeriesData,
    MirnovData,
    ConfigData
)

from .loaders import (
    load_shot_data,
    load_time_series,
    load_mirnov_data,
    load_config
)

from .exporters import (
    save_shot_data,
    save_analysis_results,
    save_time_series,
    export_ascii,
    export_prototypes,
    save_config
)

from .formats import (
    detect_format,
    supported_formats,
    validate_format_support
)

from .utils import (
    validate_data,
    merge_datasets,
    filter_bad_channels,
    bundle_shot_data
)

# Optional dependencies with graceful fallback
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    import netCDF4
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

try:
    import scipy.io
    HAS_SCIPY_IO = True
except ImportError:
    HAS_SCIPY_IO = False

__all__ = [
    # Data structures
    'ShotData',
    'ProcessedData', 
    'AnalysisResult',
    'TimeSeriesData',
    'MirnovData', 
    'ConfigData',
    
    # Loading functions
    'load_shot_data',
    'load_time_series',
    'load_mirnov_data',
    'load_config',
    
    # Export functions
    'save_shot_data',
    'save_analysis_results', 
    'save_time_series',
    'export_ascii',
    'export_prototypes',
    'save_config',
    
    # Format utilities
    'detect_format',
    'supported_formats',
    'validate_format_support',
    
    # Data utilities
    'validate_data',
    'merge_datasets',
    'filter_bad_channels',
    'bundle_shot_data',
    
    # Feature flags
    'HAS_HDF5',
    'HAS_NETCDF', 
    'HAS_SCIPY_IO',
] 