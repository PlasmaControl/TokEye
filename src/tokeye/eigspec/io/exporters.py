"""
Data export functions for eigspec.

This module provides functions to export/save data in various formats
commonly used in plasma physics and spectral analysis, including analysis
results, time series data, and configuration files.
"""

from typing import Optional, Dict, List, Any, Union
import numpy as np
import numpy.typing as npt
from pathlib import Path
import json
import pickle
import warnings

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


def save_shot_data(
    data: ShotData,
    file_path: Union[str, Path],
    file_format: str = "hdf5",
    **kwargs
) -> None:
    """Save shot data to file.
    
    Args:
        data: ShotData object to save
        file_path: Output file path
        file_format: File format ('hdf5', 'netcdf', 'matlab', 'pickle')
        **kwargs: Additional format-specific parameters
        
    Raises:
        ValueError: If file format is not supported
        ImportError: If required dependencies are missing
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format == "hdf5":
        _save_shot_data_hdf5(data, file_path, **kwargs)
    elif file_format == "netcdf":
        _save_shot_data_netcdf(data, file_path, **kwargs)
    elif file_format == "matlab":
        _save_shot_data_matlab(data, file_path, **kwargs)
    elif file_format == "pickle":
        _save_shot_data_pickle(data, file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def save_analysis_results(
    results: AnalysisResult,
    file_path: Union[str, Path],
    file_format: str = "hdf5",
    **kwargs
) -> None:
    """Save analysis results to file.
    
    Args:
        results: AnalysisResult object to save
        file_path: Output file path
        file_format: File format ('hdf5', 'matlab', 'pickle')
        **kwargs: Additional format-specific parameters
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format == "hdf5":
        _save_results_hdf5(results, file_path, **kwargs)
    elif file_format == "matlab":
        _save_results_matlab(results, file_path, **kwargs)
    elif file_format == "pickle":
        _save_results_pickle(results, file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def export_ascii(
    data: Union[TimeSeriesData, ProcessedData, npt.NDArray],
    file_path: Union[str, Path],
    header: Optional[str] = None,
    delimiter: str = " ",
    **kwargs
) -> None:
    """Export data to ASCII format.
    
    Args:
        data: Data to export (TimeSeriesData, ProcessedData, or numpy array)
        file_path: Output file path
        header: Optional header text
        delimiter: Column delimiter
        **kwargs: Additional parameters for numpy.savetxt
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, (TimeSeriesData, ProcessedData)):
        # Export time series or processed data
        if hasattr(data, 'time'):
            export_data = np.column_stack([data.time, data.data])
        else:
            export_data = data.data
        
        if header is None:
            if isinstance(data, TimeSeriesData) and hasattr(data, 'channels'):
                header = f"Time\t{delimiter.join(data.channels)}"
            elif isinstance(data, ProcessedData) and hasattr(data, 'channel_names'):
                header = f"Time\t{delimiter.join(data.channel_names)}"
    
    elif isinstance(data, np.ndarray):
        export_data = data
    else:
        raise ValueError("Unsupported data type for ASCII export")
    
    # Save using numpy.savetxt
    np.savetxt(
        file_path,
        export_data,
        delimiter=delimiter,
        header=header or "",
        comments='# ',
        **kwargs
    )


def export_prototypes(
    mode_shapes: npt.NDArray[np.complexfloating],
    coordinates: npt.NDArray[np.floating],
    file_path: Union[str, Path],
    threshold: float = 0.95,
    m_max: int = 8,
    n_max: int = 5,
    **kwargs
) -> None:
    """Export mode shape prototypes in ASCII format.
    
    Similar to MATLAB export_prototypes_mn1_ascii function.
    
    Args:
        mode_shapes: Complex mode shapes (channels x modes)
        coordinates: Channel coordinates
        file_path: Output file path
        threshold: Threshold for prototype selection
        m_max: Maximum poloidal mode number
        n_max: Maximum toroidal mode number
        **kwargs: Additional parameters
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_channels, num_modes = mode_shapes.shape
    
    # Process each mode shape
    export_data = []
    
    for mode_idx in range(num_modes):
        mode_shape = mode_shapes[:, mode_idx]
        
        # Apply threshold and processing
        amplitude = np.abs(mode_shape)
        phase = np.angle(mode_shape)
        
        # Select channels above threshold
        significant_channels = amplitude > threshold * np.max(amplitude)
        
        if np.any(significant_channels):
            for ch_idx in np.where(significant_channels)[0]:
                coord = coordinates[ch_idx] if coordinates is not None else [0, 0, 0]
                
                export_data.append([
                    mode_idx + 1,  # Mode number (1-indexed)
                    ch_idx + 1,    # Channel number (1-indexed)
                    coord[0],      # R coordinate
                    coord[1],      # Z coordinate
                    coord[2],      # Phi coordinate
                    amplitude[ch_idx],
                    phase[ch_idx]
                ])
    
    if export_data:
        export_array = np.array(export_data)
        header = "Mode Channel R Z Phi Amplitude Phase"
        
        np.savetxt(
            file_path,
            export_array,
            header=header,
            fmt=['%d', '%d', '%.6f', '%.6f', '%.6f', '%.6e', '%.6f'],
            delimiter=' ',
            comments='# '
        )
    else:
        # Create empty file
        with open(file_path, 'w') as f:
            f.write("# No prototypes found above threshold\n")


def save_config(
    config: ConfigData,
    file_path: Union[str, Path],
    file_format: str = "json",
    **kwargs
) -> None:
    """Save configuration data to file.
    
    Args:
        config: ConfigData object to save
        file_path: Output file path
        file_format: File format ('json', 'matlab', 'pickle')
        **kwargs: Additional format-specific parameters
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format == "json":
        _save_config_json(config, file_path, **kwargs)
    elif file_format == "matlab":
        _save_config_matlab(config, file_path, **kwargs)
    elif file_format == "pickle":
        _save_config_pickle(config, file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported config format: {file_format}")


def save_time_series(
    data: TimeSeriesData,
    file_path: Union[str, Path],
    file_format: str = "hdf5",
    **kwargs
) -> None:
    """Save time series data to file.
    
    Args:
        data: TimeSeriesData object to save
        file_path: Output file path
        file_format: File format ('hdf5', 'csv', 'ascii', 'matlab')
        **kwargs: Additional format-specific parameters
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format == "hdf5":
        _save_time_series_hdf5(data, file_path, **kwargs)
    elif file_format == "csv":
        _save_time_series_csv(data, file_path, **kwargs)
    elif file_format == "ascii":
        export_ascii(data, file_path, **kwargs)
    elif file_format == "matlab":
        _save_time_series_matlab(data, file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported time series format: {file_format}")


def save_bdot_image(
    time: npt.NDArray[np.floating],
    data: npt.NDArray[np.floating],
    coordinates: Optional[npt.NDArray[np.floating]],
    channel_names: List[str],
    file_path: Union[str, Path],
    file_format: str = "matlab"
) -> None:
    """Save B-dot data in organized format.
    
    Similar to MATLAB savebdotimage function.
    
    Args:
        time: Time vector
        data: Data matrix (time x channels)
        coordinates: Channel coordinates (channels x 3)
        channel_names: Channel names
        file_path: Output file path
        file_format: File format ('matlab', 'hdf5')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if coordinates is not None:
        # Sort channels by array type and position
        sorted_data, sorted_coords, sorted_names = _sort_array_channels(
            data, coordinates, channel_names
        )
    else:
        sorted_data = data
        sorted_coords = coordinates
        sorted_names = channel_names
    
    if file_format == "matlab":
        _save_bdot_matlab(time, sorted_data, sorted_coords, sorted_names, file_path)
    elif file_format == "hdf5":
        _save_bdot_hdf5(time, sorted_data, sorted_coords, sorted_names, file_path)
    else:
        raise ValueError(f"Unsupported format for B-dot data: {file_format}")


# Format-specific save functions
def _save_shot_data_hdf5(data: ShotData, file_path: Path, **kwargs) -> None:
    """Save shot data to HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")
    
    import h5py
    
    with h5py.File(file_path, 'w') as f:
        # Save metadata as attributes
        f.attrs['shot_number'] = data.shot_number
        f.attrs['time_range'] = data.time_range
        
        # Save channel data
        if data.channels:
            channels_group = f.create_group('channels')
            for ch_name, ch_data in data.channels.items():
                ch_dataset = channels_group.create_dataset(ch_name, data=ch_data)
                
                # Save coordinates as attributes
                if ch_name in data.coordinates:
                    ch_dataset.attrs['coordinates'] = data.coordinates[ch_name]
        
        # Save channel names
        if data.channel_names:
            f.create_dataset('channel_names', data=[s.encode() for s in data.channel_names])
        
        # Save acquisition info
        for key, value in data.acquisition_info.items():
            if isinstance(value, (int, float, str)):
                f.attrs[f'acq_{key}'] = value


def _save_shot_data_matlab(data: ShotData, file_path: Path, **kwargs) -> None:
    """Save shot data to MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    import scipy.io as sio
    
    # Prepare data for MATLAB format
    save_dict = {
        'shot_number': data.shot_number,
        'time_range': data.time_range,
        'channel_names': data.channel_names
    }
    
    # Concatenate channel data if available
    if data.channels:
        channel_data = []
        for ch_name in data.channel_names:
            if ch_name in data.channels:
                channel_data.append(data.channels[ch_name])
        
        if channel_data:
            save_dict['channels'] = np.array(channel_data)
    
    # Save coordinates
    if data.coordinates:
        coords_array = np.array([data.coordinates.get(ch, [0, 0, 0]) for ch in data.channel_names])
        save_dict['coordinates'] = coords_array
    
    sio.savemat(file_path, save_dict)


def _save_shot_data_netcdf(data: ShotData, file_path: Path, **kwargs) -> None:
    """Save shot data to NetCDF file."""
    if not HAS_NETCDF:
        raise ImportError("netCDF4 is required for NetCDF support")
    
    import netCDF4
    
    with netCDF4.Dataset(file_path, 'w') as nc:
        # Global attributes
        nc.setncattr('shot_number', data.shot_number)
        nc.setncattr('time_range_start', data.time_range[0])
        nc.setncattr('time_range_end', data.time_range[1])
        
        # Create dimensions
        if data.channels:
            # Determine dimensions from first channel
            first_channel = next(iter(data.channels.values()))
            time_dim = nc.createDimension('time', first_channel.shape[0])
            channel_dim = nc.createDimension('channels', len(data.channel_names))
            
            # Create time variable
            time_var = nc.createVariable('time', 'f8', ('time',))
            time_var[:] = first_channel[:, 0]
            time_var.units = 'seconds'
            
            # Create channel variables
            for i, ch_name in enumerate(data.channel_names):
                if ch_name in data.channels:
                    ch_var = nc.createVariable(f'channel_{i:03d}', 'f8', ('time',))
                    ch_var[:] = data.channels[ch_name][:, 1]
                    ch_var.channel_name = ch_name


def _save_shot_data_pickle(data: ShotData, file_path: Path, **kwargs) -> None:
    """Save shot data to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def _save_results_hdf5(results: AnalysisResult, file_path: Path, **kwargs) -> None:
    """Save analysis results to HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")
    
    import h5py
    
    with h5py.File(file_path, 'w') as f:
        # Save basic attributes
        f.attrs['analysis_type'] = results.analysis_type.encode()
        f.attrs['shot_number'] = results.shot_number
        f.attrs['timestamp'] = results.timestamp.isoformat().encode()
        
        # Save datasets
        f.create_dataset('time_blocks', data=results.time_blocks)
        f.create_dataset('frequencies', data=results.frequencies)
        f.create_dataset('eigenvalues', data=results.eigenvalues)
        f.create_dataset('mode_shapes', data=results.mode_shapes)
        f.create_dataset('stability', data=results.stability)
        
        # Save parameters as attributes
        for key, value in results.parameters.items():
            if isinstance(value, (int, float, str)):
                f.attrs[f'param_{key}'] = value


def _save_results_matlab(results: AnalysisResult, file_path: Path, **kwargs) -> None:
    """Save analysis results to MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    import scipy.io as sio
    
    save_dict = {
        'analysis_type': results.analysis_type,
        'shot_number': results.shot_number,
        'time_blocks': results.time_blocks,
        'frequencies': results.frequencies,
        'eigenvalues': results.eigenvalues,
        'mode_shapes': results.mode_shapes,
        'stability': results.stability,
        'parameters': results.parameters,
        'timestamp': results.timestamp.isoformat()
    }
    
    sio.savemat(file_path, save_dict)


def _save_results_pickle(results: AnalysisResult, file_path: Path, **kwargs) -> None:
    """Save analysis results to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def _save_config_json(config: ConfigData, file_path: Path, **kwargs) -> None:
    """Save configuration to JSON file."""
    config_dict = {
        'analysis_params': config.analysis_params,
        'processing_params': config.processing_params,
        'plot_params': config.plot_params,
        'io_params': config.io_params,
        'file_paths': config.file_paths,
        'user_settings': config.user_settings,
        'version': config.version
    }
    
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=_json_serializer)


def _save_config_matlab(config: ConfigData, file_path: Path, **kwargs) -> None:
    """Save configuration to MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    import scipy.io as sio
    
    save_dict = {
        'analysis_params': config.analysis_params,
        'processing_params': config.processing_params,
        'plot_params': config.plot_params,
        'io_params': config.io_params,
        'file_paths': config.file_paths,
        'user_settings': config.user_settings,
        'version': config.version
    }
    
    sio.savemat(file_path, save_dict)


def _save_config_pickle(config: ConfigData, file_path: Path, **kwargs) -> None:
    """Save configuration to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(config, f)


def _save_time_series_hdf5(data: TimeSeriesData, file_path: Path, **kwargs) -> None:
    """Save time series to HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")
    
    import h5py
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('time', data=data.time)
        f.create_dataset('data', data=data.data)
        f.create_dataset('channel_names', data=[s.encode() for s in data.channels])
        
        f.attrs['sample_rate'] = data.sample_rate
        f.attrs['units'] = data.units.encode()
        
        if data.coordinates is not None:
            f.create_dataset('coordinates', data=data.coordinates)
        
        # Save metadata
        for key, value in data.metadata.items():
            if isinstance(value, (int, float, str)):
                f.attrs[f'meta_{key}'] = value


def _save_time_series_csv(data: TimeSeriesData, file_path: Path, **kwargs) -> None:
    """Save time series to CSV file."""
    if HAS_PANDAS:
        import pandas as pd
        
        # Create DataFrame
        df_data = {'time': data.time}
        for i, ch_name in enumerate(data.channels):
            df_data[ch_name] = data.data[:, i]
        
        df = pd.DataFrame(df_data)
        df.to_csv(file_path, index=False, **kwargs)
    else:
        # Fallback to ASCII export
        export_ascii(data, file_path, delimiter=',', **kwargs)


def _save_time_series_matlab(data: TimeSeriesData, file_path: Path, **kwargs) -> None:
    """Save time series to MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    import scipy.io as sio
    
    save_dict = {
        'T': data.time,
        'Y': data.data,
        'ynames': data.channels,
        'sample_rate': data.sample_rate,
        'units': data.units
    }
    
    if data.coordinates is not None:
        save_dict['Yxy'] = data.coordinates
    
    sio.savemat(file_path, save_dict)


def _save_bdot_matlab(
    time: npt.NDArray[np.floating],
    data: npt.NDArray[np.floating],
    coordinates: Optional[npt.NDArray[np.floating]],
    channel_names: List[str],
    file_path: Path
) -> None:
    """Save B-dot data to MATLAB file."""
    if not HAS_SCIPY_IO:
        raise ImportError("scipy is required for MATLAB file support")
    
    import scipy.io as sio
    
    save_dict = {
        'T': time,
        'Y': data,
        'ynames': channel_names
    }
    
    if coordinates is not None:
        save_dict['Yxy'] = coordinates
    
    sio.savemat(file_path, save_dict)


def _save_bdot_hdf5(
    time: npt.NDArray[np.floating],
    data: npt.NDArray[np.floating],
    coordinates: Optional[npt.NDArray[np.floating]],
    channel_names: List[str],
    file_path: Path
) -> None:
    """Save B-dot data to HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")
    
    import h5py
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('time', data=time)
        f.create_dataset('data', data=data)
        f.create_dataset('channel_names', data=[s.encode() for s in channel_names])
        
        if coordinates is not None:
            f.create_dataset('coordinates', data=coordinates)


def _sort_array_channels(
    data: npt.NDArray[np.floating],
    coordinates: npt.NDArray[np.floating],
    channel_names: List[str]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], List[str]]:
    """Sort channels by array type (toroidal/poloidal) and position."""
    # Identify toroidal and poloidal arrays based on coordinates
    # This is a simplified version of the MATLAB logic
    eps_x = 2e-2
    eps_y = 1e-1
    pol_array_y = 5.6287
    tor_array_x = 0.0
    
    # Find indices for different array types
    tor_indices = np.where(np.abs(coordinates[:, 0] - tor_array_x) <= eps_x)[0]
    pol_indices = np.where(np.abs(coordinates[:, 1] - pol_array_y) <= eps_y)[0]
    
    # Sort toroidal array by y-coordinate
    if len(tor_indices) > 0:
        tor_sort_idx = np.argsort(coordinates[tor_indices, 1])
        tor_indices = tor_indices[tor_sort_idx]
    
    # Sort poloidal array by x-coordinate
    if len(pol_indices) > 0:
        pol_sort_idx = np.argsort(coordinates[pol_indices, 0])
        pol_indices = pol_indices[pol_sort_idx]
    
    # Remove overlap between arrays
    pol_indices = np.setdiff1d(pol_indices, tor_indices)
    
    # Get remaining indices
    all_indices = np.arange(len(channel_names))
    other_indices = np.setdiff1d(all_indices, np.concatenate([tor_indices, pol_indices]))
    
    # Combine in order: toroidal, poloidal, others
    new_order = np.concatenate([tor_indices, pol_indices, other_indices])
    
    # Reorder data, coordinates, and names
    sorted_data = data[:, new_order]
    sorted_coords = coordinates[new_order]
    sorted_names = [channel_names[i] for i in new_order]
    
    return sorted_data, sorted_coords, sorted_names


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy arrays and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.complexfloating):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable") 