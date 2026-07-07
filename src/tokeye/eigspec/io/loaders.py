"""
Data loading functions for eigspec.

This module provides functions to load various data formats commonly used
in plasma physics and spectral analysis, including time series data,
shot data, configuration files, and analysis results.
"""

from typing import Optional, Dict, List, Any, Union, Tuple
import numpy as np
import numpy.typing as npt
from pathlib import Path
import warnings
import json
import pickle

from .data_structures import (
    ShotData, 
    ProcessedData,
    AnalysisResult, 
    TimeSeriesData,
    MirnovData,
    ConfigData
)

# Optional imports with graceful fallback
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
    import scipy.io as sio
    HAS_SCIPY_IO = True
except ImportError:
    HAS_SCIPY_IO = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_shot_data(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs
) -> ShotData:
    """Load plasma shot data from file.
    
    Args:
        file_path: Path to data file
        file_format: File format ('hdf5', 'netcdf', 'matlab', 'ascii', None for auto-detect)
        **kwargs: Additional format-specific parameters
        
    Returns:
        ShotData object with loaded data
        
    Raises:
        ValueError: If file format is not supported or file cannot be read
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_format is None:
        file_format = _detect_file_format(file_path)
    
    if file_format == "hdf5":
        return _load_shot_data_hdf5(file_path, **kwargs)
    elif file_format == "netcdf":
        return _load_shot_data_netcdf(file_path, **kwargs)
    elif file_format == "matlab":
        return _load_shot_data_matlab(file_path, **kwargs)
    elif file_format == "ascii":
        return _load_shot_data_ascii(file_path, **kwargs)
    elif file_format == "binary":
        return _load_shot_data_binary(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_time_series(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs
) -> TimeSeriesData:
    """Load time series data from file.
    
    Args:
        file_path: Path to data file
        file_format: File format (None for auto-detect)
        **kwargs: Additional format-specific parameters
        
    Returns:
        TimeSeriesData object with loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_format is None:
        file_format = _detect_file_format(file_path)
    
    if file_format == "ascii":
        return _load_time_series_ascii(file_path, **kwargs)
    elif file_format == "csv":
        return _load_time_series_csv(file_path, **kwargs)
    elif file_format == "hdf5":
        return _load_time_series_hdf5(file_path, **kwargs)
    elif file_format == "matlab":
        return _load_time_series_matlab(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_mirnov_data(
    file_path: Union[str, Path],
    shot_number: int,
    probe_type: str = "mixed",
    **kwargs
) -> MirnovData:
    """Load Mirnov probe data from file.
    
    Args:
        file_path: Path to data file
        shot_number: Plasma shot number
        probe_type: Type of probe array ('poloidal', 'toroidal', 'mixed')
        **kwargs: Additional parameters
        
    Returns:
        MirnovData object with loaded probe data
    """
    # Load as time series first
    ts_data = load_time_series(file_path, **kwargs)
    
    # Convert to MirnovData
    return MirnovData(
        time=ts_data.time,
        data=ts_data.data,
        channels=ts_data.channels,
        sample_rate=ts_data.sample_rate,
        units=ts_data.units,
        coordinates=ts_data.coordinates,
        metadata=ts_data.metadata,
        shot_number=shot_number,
        probe_type=probe_type
    )


def load_config(
    file_path: Union[str, Path],
    file_format: Optional[str] = None
) -> ConfigData:
    """Load configuration data from file.
    
    Args:
        file_path: Path to configuration file
        file_format: File format ('json', 'yaml', 'matlab', None for auto-detect)
        
    Returns:
        ConfigData object with configuration
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    if file_format is None:
        file_format = _detect_config_format(file_path)
    
    if file_format == "json":
        return _load_config_json(file_path)
    elif file_format == "matlab":
        return _load_config_matlab(file_path)
    elif file_format == "pickle":
        return _load_config_pickle(file_path)
    else:
        raise ValueError(f"Unsupported config format: {file_format}")


def load_analysis_results(
    file_path: Union[str, Path],
    file_format: Optional[str] = None
) -> AnalysisResult:
    """Load analysis results from file.
    
    Args:
        file_path: Path to results file
        file_format: File format (None for auto-detect)
        
    Returns:
        AnalysisResult object with loaded results
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    if file_format is None:
        file_format = _detect_file_format(file_path)
    
    if file_format == "hdf5":
        return _load_results_hdf5(file_path)
    elif file_format == "matlab":
        return _load_results_matlab(file_path)
    elif file_format == "pickle":
        return _load_results_pickle(file_path)
    else:
        raise ValueError(f"Unsupported results format: {file_format}")


# Format detection utilities
def _detect_file_format(file_path: Path) -> str:
    """Detect file format from extension and content."""
    suffix = file_path.suffix.lower()
    
    if suffix in [".h5", ".hdf5"]:
        return "hdf5"
    elif suffix in [".nc", ".netcdf"]:
        return "netcdf"
    elif suffix in [".mat"]:
        return "matlab"
    elif suffix in [".txt", ".dat", ".asc"]:
        return "ascii"
    elif suffix in [".csv"]:
        return "csv"
    elif suffix in [".bin", ".raw"]:
        return "binary"
    elif suffix in [".pkl", ".pickle"]:
        return "pickle"
    else:
        # Try to detect from content
        return _detect_from_content(file_path)


def _detect_config_format(file_path: Path) -> str:
    """Detect configuration file format."""
    suffix = file_path.suffix.lower()
    
    if suffix in [".json"]:
        return "json"
    elif suffix in [".mat"]:
        return "matlab"
    elif suffix in [".pkl", ".pickle"]:
        return "pickle"
    else:
        return "json"  # Default


def _detect_from_content(file_path: Path) -> str:
    """Detect format from file content."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        # Check for HDF5 signature
        if header.startswith(b'\x89HDF\r\n\x1a\n'):
            return "hdf5"
        
        # Check for NetCDF signature
        if header.startswith(b'CDF\x01') or header.startswith(b'CDF\x02'):
            return "netcdf"
        
        # Default to ASCII
        return "ascii"
    except:
        return "ascii"


# Format-specific loaders
def _load_shot_data_hdf5(file_path: Path, **kwargs) -> ShotData:
    """Load shot data from HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")
    
    with h5py.File(file_path, 'r') as f:
        shot_number = int(f.attrs.get('shot_number', 0))
        time_range = tuple(f.attrs.get('time_range', (0.0, 1.0)))
        
        # Load channel data
        channels = {}
        coordinates = {}
        channel_names = []
        
        if 'channels' in f:
            for ch_name in f['channels'].keys():
                ch_data = f['channels'][ch_name][...]
                channels[ch_name] = ch_data
                channel_names.append(ch_name)
                
                # Load coordinates if available
                if 'coordinates' in f['channels'][ch_name].attrs:
                    coords = f['channels'][ch_name].attrs['coordinates']
                    coordinates[ch_name] = tuple(coords)
        
        # Load metadata
        acquisition_info = dict(f.attrs) if f.attrs else {}
        
    return ShotData(
        shot_number=shot_number,
        time_range=time_range,
        channels=channels,
        coordinates=coordinates,
        channel_names=channel_names,
        acquisition_info=acquisition_info
    )


def _load_shot_data_matlab(file_path: Path, **kwargs) -> ShotData:
    """Load shot data from MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    data = sio.loadmat(str(file_path))
    
    # Extract shot information
    shot_number = int(data.get('shot_number', [0])[0])
    time_range = tuple(data.get('time_range', [0.0, 1.0]))
    
    # Load channel data
    channels = {}
    coordinates = {}
    channel_names = []
    
    if 'channels' in data:
        ch_data = data['channels']
        for i, ch_name in enumerate(data.get('channel_names', [])):
            if isinstance(ch_name, np.ndarray):
                ch_name = str(ch_name[0])
            channels[ch_name] = ch_data[:, i:i+2]  # time and data
            channel_names.append(ch_name)
    
    return ShotData(
        shot_number=shot_number,
        time_range=time_range,
        channels=channels,
        coordinates=coordinates,
        channel_names=channel_names
    )


def _load_shot_data_ascii(file_path: Path, **kwargs) -> ShotData:
    """Load shot data from ASCII file."""
    delimiter = kwargs.get('delimiter', None)
    skip_header = kwargs.get('skip_header', 0)
    
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_header)
    
    # Assume first column is time, rest are channels
    time_vec = data[:, 0]
    channels = {}
    channel_names = []
    
    for i in range(1, data.shape[1]):
        ch_name = f"channel_{i}"
        channels[ch_name] = np.column_stack([time_vec, data[:, i]])
        channel_names.append(ch_name)
    
    return ShotData(
        shot_number=kwargs.get('shot_number', 0),
        time_range=(float(time_vec[0]), float(time_vec[-1])),
        channels=channels,
        coordinates={},
        channel_names=channel_names
    )


def _load_shot_data_binary(file_path: Path, **kwargs) -> ShotData:
    """Load shot data from binary file."""
    dtype = kwargs.get('dtype', np.float64)
    shape = kwargs.get('shape', None)
    
    data = np.fromfile(file_path, dtype=dtype)
    
    if shape is not None:
        data = data.reshape(shape)
    
    # Assume first column is time
    time_vec = data[:, 0]
    channels = {}
    channel_names = []
    
    for i in range(1, data.shape[1]):
        ch_name = f"channel_{i}"
        channels[ch_name] = np.column_stack([time_vec, data[:, i]])
        channel_names.append(ch_name)
    
    return ShotData(
        shot_number=kwargs.get('shot_number', 0),
        time_range=(float(time_vec[0]), float(time_vec[-1])),
        channels=channels,
        coordinates={},
        channel_names=channel_names
    )


def _load_shot_data_netcdf(file_path: Path, **kwargs) -> ShotData:
    """Load shot data from NetCDF file."""
    if not HAS_NETCDF:
        raise ImportError("netCDF4 is required for NetCDF support")
    
    with netCDF4.Dataset(file_path, 'r') as nc:
        shot_number = int(nc.getncattr('shot_number') if 'shot_number' in nc.ncattrs() else 0)
        
        # Load variables
        channels = {}
        channel_names = []
        coordinates = {}
        
        for var_name in nc.variables:
            if var_name != 'time':
                var_data = nc.variables[var_name][...]
                time_data = nc.variables['time'][...]
                channels[var_name] = np.column_stack([time_data, var_data])
                channel_names.append(var_name)
        
        time_range = (float(time_data[0]), float(time_data[-1])) if 'time_data' in locals() else (0.0, 1.0)
    
    return ShotData(
        shot_number=shot_number,
        time_range=time_range,
        channels=channels,
        coordinates=coordinates,
        channel_names=channel_names
    )


def _load_time_series_ascii(file_path: Path, **kwargs) -> TimeSeriesData:
    """Load time series from ASCII file."""
    delimiter = kwargs.get('delimiter', None)
    skip_header = kwargs.get('skip_header', 0)
    channel_names = kwargs.get('channel_names', None)
    
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_header)
    
    # Assume first column is time
    time_vec = data[:, 0]
    data_matrix = data[:, 1:]
    
    if channel_names is None:
        channel_names = [f"channel_{i+1}" for i in range(data_matrix.shape[1])]
    
    sample_rate = float(1.0 / np.mean(np.diff(time_vec))) if len(time_vec) > 1 else 1.0
    
    return TimeSeriesData(
        time=time_vec,
        data=data_matrix,
        channels=channel_names,
        sample_rate=sample_rate,
        units=kwargs.get('units', 'unknown')
    )


def _load_time_series_csv(file_path: Path, **kwargs) -> TimeSeriesData:
    """Load time series from CSV file."""
    if not HAS_PANDAS:
        # Fallback to numpy
        return _load_time_series_ascii(file_path, delimiter=',', **kwargs)
    
    df = pd.read_csv(file_path, **kwargs)
    
    # Assume first column is time
    time_col = df.columns[0]
    time_vec = df[time_col].values
    
    channel_names = list(df.columns[1:])
    data_matrix = df[channel_names].values
    
    sample_rate = float(1.0 / np.mean(np.diff(time_vec))) if len(time_vec) > 1 else 1.0
    
    return TimeSeriesData(
        time=time_vec,
        data=data_matrix,
        channels=channel_names,
        sample_rate=sample_rate
    )


def _load_time_series_hdf5(file_path: Path, **kwargs) -> TimeSeriesData:
    """Load time series from HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")
    
    with h5py.File(file_path, 'r') as f:
        time_vec = f['time'][...]
        data_matrix = f['data'][...]
        
        channel_names = []
        if 'channel_names' in f:
            channel_names = [s.decode() if isinstance(s, bytes) else str(s) for s in f['channel_names'][...]]
        else:
            channel_names = [f"channel_{i+1}" for i in range(data_matrix.shape[1])]
        
        sample_rate = float(f.attrs.get('sample_rate', 1.0 / np.mean(np.diff(time_vec))))
        units = f.attrs.get('units', 'unknown')
        if isinstance(units, bytes):
            units = units.decode()
        
        coordinates = None
        if 'coordinates' in f:
            coordinates = f['coordinates'][...]
    
    return TimeSeriesData(
        time=time_vec,
        data=data_matrix,
        channels=channel_names,
        sample_rate=sample_rate,
        units=str(units),
        coordinates=coordinates
    )


def _load_time_series_matlab(file_path: Path, **kwargs) -> TimeSeriesData:
    """Load time series from MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    data = sio.loadmat(str(file_path))
    
    time_vec = data.get('T', data.get('time', data.get('t', np.array([])))).flatten()
    data_matrix = data.get('Y', data.get('data', data.get('Bdot', np.array([]))))
    
    if data_matrix.ndim == 1:
        data_matrix = data_matrix.reshape(-1, 1)
    
    # Get channel names
    channel_names = []
    if 'ynames' in data:
        ynames = data['ynames']
        if ynames.dtype.names:  # Structured array
            channel_names = [str(ynames[i][0][0]) for i in range(len(ynames))]
        else:
            channel_names = [str(name[0]) if hasattr(name, '__len__') and len(name) > 0 else f"channel_{i+1}" 
                           for i, name in enumerate(ynames.flatten())]
    else:
        channel_names = [f"channel_{i+1}" for i in range(data_matrix.shape[1])]
    
    sample_rate = float(1.0 / np.mean(np.diff(time_vec))) if len(time_vec) > 1 else 1.0
    
    coordinates = None
    if 'Yxy' in data:
        coordinates = data['Yxy']
    
    return TimeSeriesData(
        time=time_vec,
        data=data_matrix,
        channels=channel_names,
        sample_rate=sample_rate,
        coordinates=coordinates
    )


def _load_config_json(file_path: Path) -> ConfigData:
    """Load configuration from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return ConfigData(
        analysis_params=data.get('analysis_params', {}),
        processing_params=data.get('processing_params', {}),
        plot_params=data.get('plot_params', {}),
        io_params=data.get('io_params', {}),
        file_paths=data.get('file_paths', {}),
        user_settings=data.get('user_settings', {}),
        version=data.get('version', '1.0')
    )


def _load_config_matlab(file_path: Path) -> ConfigData:
    """Load configuration from MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    data = sio.loadmat(str(file_path))
    
    # Convert MATLAB structure to dictionary
    config_dict = {}
    for key, value in data.items():
        if not key.startswith('__'):
            if isinstance(value, np.ndarray) and value.dtype.names:
                # Structured array (MATLAB struct)
                config_dict[key] = _matlab_struct_to_dict(value)
            else:
                config_dict[key] = value
    
    return ConfigData(
        analysis_params=config_dict.get('analysis_params', {}),
        processing_params=config_dict.get('processing_params', {}),
        plot_params=config_dict.get('plot_params', {}),
        io_params=config_dict.get('io_params', {}),
        file_paths=config_dict.get('file_paths', {}),
        user_settings=config_dict.get('user_settings', {}),
        version=str(config_dict.get('version', '1.0'))
    )


def _load_config_pickle(file_path: Path) -> ConfigData:
    """Load configuration from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, ConfigData):
        return data
    elif isinstance(data, dict):
        return ConfigData(**data)
    else:
        raise ValueError("Invalid pickle file format for ConfigData")


def _load_results_hdf5(file_path: Path) -> AnalysisResult:
    """Load analysis results from HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")
    
    with h5py.File(file_path, 'r') as f:
        return AnalysisResult(
            analysis_type=f.attrs.get('analysis_type', 'unknown').decode() if isinstance(f.attrs.get('analysis_type'), bytes) else str(f.attrs.get('analysis_type', 'unknown')),
            shot_number=int(f.attrs.get('shot_number', 0)),
            time_blocks=list(f['time_blocks'][...]),
            frequencies=f['frequencies'][...],
            eigenvalues=f['eigenvalues'][...],
            mode_shapes=f['mode_shapes'][...],
            stability=f['stability'][...],
            parameters=dict(f.attrs) if f.attrs else {}
        )


def _load_results_matlab(file_path: Path) -> AnalysisResult:
    """Load analysis results from MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    data = sio.loadmat(str(file_path))
    
    return AnalysisResult(
        analysis_type=str(data.get('analysis_type', ['unknown'])[0]),
        shot_number=int(data.get('shot_number', [0])[0]),
        time_blocks=data.get('time_blocks', []).tolist(),
        frequencies=data.get('frequencies', np.array([])),
        eigenvalues=data.get('eigenvalues', np.array([])),
        mode_shapes=data.get('mode_shapes', np.array([])),
        stability=data.get('stability', np.array([])),
        parameters=_extract_matlab_params(data)
    )


def _load_results_pickle(file_path: Path) -> AnalysisResult:
    """Load analysis results from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, AnalysisResult):
        return data
    else:
        raise ValueError("Invalid pickle file format for AnalysisResult")


def _matlab_struct_to_dict(struct_array: np.ndarray) -> Dict[str, Any]:
    """Convert MATLAB struct array to Python dictionary."""
    if struct_array.size == 0:
        return {}
    
    result = {}
    struct = struct_array.flat[0]
    
    for field_name in struct_array.dtype.names:
        field_data = struct[field_name]
        if isinstance(field_data, np.ndarray):
            if field_data.dtype.names:  # Nested struct
                result[field_name] = _matlab_struct_to_dict(field_data)
            else:
                result[field_name] = field_data
        else:
            result[field_name] = field_data
    
    return result


def _extract_matlab_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters from MATLAB data structure."""
    params = {}
    
    # Look for common parameter fields
    param_fields = ['parameters', 'params', 'options', 'config']
    
    for field in param_fields:
        if field in data:
            field_data = data[field]
            if isinstance(field_data, np.ndarray) and field_data.dtype.names:
                params.update(_matlab_struct_to_dict(field_data))
            else:
                params[field] = field_data
    
    return params 