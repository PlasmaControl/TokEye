"""
Data utilities for eigspec I/O operations.

This module provides utility functions for data validation, manipulation,
merging, filtering, and bundling operations commonly used in plasma physics
data processing workflows.
"""

from typing import Optional, Dict, List, Any, Union, Tuple
import numpy as np
import numpy.typing as npt
from pathlib import Path
import warnings

from .data_structures import (
    ShotData,
    ProcessedData,
    AnalysisResult,
    TimeSeriesData,
    MirnovData,
    ConfigData
)


def validate_data(
    data: Union[TimeSeriesData, ProcessedData, ShotData],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """Validate data consistency and integrity.
    
    Args:
        data: Data object to validate
        strict: Whether to apply strict validation rules
        
    Returns:
        Tuple of (is_valid, warning_messages)
        
    Raises:
        ValueError: If critical validation errors are found
    """
    warnings_list = []
    
    if isinstance(data, TimeSeriesData):
        warnings_list.extend(_validate_time_series(data, strict))
    elif isinstance(data, ProcessedData):
        warnings_list.extend(_validate_processed_data(data, strict))
    elif isinstance(data, ShotData):
        warnings_list.extend(_validate_shot_data(data, strict))
    else:
        raise ValueError(f"Unsupported data type for validation: {type(data)}")
    
    # Check for critical errors vs warnings
    critical_errors = [w for w in warnings_list if "ERROR:" in w]
    
    if critical_errors and strict:
        raise ValueError(f"Critical validation errors: {critical_errors}")
    
    return len(critical_errors) == 0, warnings_list


def merge_datasets(
    datasets: List[Union[TimeSeriesData, ProcessedData]],
    method: str = "concatenate",
    axis: str = "time"
) -> Union[TimeSeriesData, ProcessedData]:
    """Merge multiple datasets along specified axis.
    
    Args:
        datasets: List of datasets to merge
        method: Merge method ('concatenate', 'average', 'stack')
        axis: Axis along which to merge ('time', 'channels')
        
    Returns:
        Merged dataset
        
    Raises:
        ValueError: If datasets are incompatible for merging
    """
    if not datasets:
        raise ValueError("No datasets provided for merging")
    
    if len(datasets) == 1:
        return datasets[0]
    
    # Check compatibility
    _check_merge_compatibility(datasets, axis)
    
    if method == "concatenate":
        return _concatenate_datasets(datasets, axis)
    elif method == "average":
        return _average_datasets(datasets)
    elif method == "stack":
        return _stack_datasets(datasets, axis)
    else:
        raise ValueError(f"Unknown merge method: {method}")


def filter_bad_channels(
    data: Union[TimeSeriesData, ProcessedData, ShotData],
    bad_channels: List[str],
    in_place: bool = False
) -> Union[TimeSeriesData, ProcessedData, ShotData]:
    """Remove bad channels from data.
    
    Args:
        data: Data object to filter
        bad_channels: List of channel names to remove
        in_place: Whether to modify data in place
        
    Returns:
        Filtered data object
    """
    if not bad_channels:
        return data if in_place else _copy_data_object(data)
    
    if isinstance(data, TimeSeriesData):
        return _filter_time_series_channels(data, bad_channels, in_place)
    elif isinstance(data, ProcessedData):
        return _filter_processed_data_channels(data, bad_channels, in_place)
    elif isinstance(data, ShotData):
        return _filter_shot_data_channels(data, bad_channels, in_place)
    else:
        raise ValueError(f"Unsupported data type for channel filtering: {type(data)}")


def bundle_shot_data(
    time: npt.NDArray[np.floating],
    data: npt.NDArray[np.floating],
    coordinates: npt.NDArray[np.floating],
    channel_names: List[str],
    shot_number: int,
    bad_channels: Optional[List[str]] = None,
    interpolation_method: str = "linear",
    detrend_method: str = "linear"
) -> ProcessedData:
    """Bundle shot data into ProcessedData format.
    
    Similar to MATLAB bundle_shot_data function.
    
    Args:
        time: Time vector
        data: Data matrix (time x channels)
        coordinates: Channel coordinates
        channel_names: Channel names
        shot_number: Shot number
        bad_channels: Channels to exclude
        interpolation_method: Method used for interpolation
        detrend_method: Method used for detrending
        
    Returns:
        ProcessedData object with bundled data
    """
    if bad_channels is None:
        bad_channels = []
    
    # Validate input dimensions
    if len(time) != data.shape[0]:
        raise ValueError("Time vector length must match data rows")
    if len(channel_names) != data.shape[1]:
        raise ValueError("Number of channel names must match data columns")
    if len(coordinates) != len(channel_names):
        raise ValueError("Coordinates length must match number of channels")
    
    # Filter out bad channels
    good_indices = []
    good_channels = []
    filtered_bad = []
    
    for i, ch_name in enumerate(channel_names):
        if ch_name in bad_channels:
            print(f"*** removing: pointname \"{ch_name}\"; found at index #{i+1}.")
            filtered_bad.append(ch_name)
        else:
            good_indices.append(i)
            good_channels.append(ch_name)
    
    # Extract good channels
    if good_indices:
        filtered_data = data[:, good_indices]
        filtered_coords = coordinates[good_indices]
    else:
        raise ValueError("No good channels remaining after filtering")
    
    return ProcessedData(
        shot_number=shot_number,
        time=time,
        data=filtered_data,
        coordinates=filtered_coords,
        channel_names=good_channels,
        bad_channels=filtered_bad,
        processing_steps=["channel_filtering"],
        interpolation_method=interpolation_method,
        detrend_method=detrend_method
    )


def resample_time_series(
    data: TimeSeriesData,
    new_sample_rate: float,
    method: str = "linear"
) -> TimeSeriesData:
    """Resample time series data to new sampling rate.
    
    Args:
        data: Input time series data
        new_sample_rate: Target sampling rate (Hz)
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        Resampled time series data
    """
    # Create new time vector
    time_span = data.time[-1] - data.time[0]
    num_new_samples = int(time_span * new_sample_rate) + 1
    new_time = np.linspace(data.time[0], data.time[-1], num_new_samples)
    
    # Interpolate data
    new_data = np.zeros((len(new_time), data.data.shape[1]))
    
    for ch_idx in range(data.data.shape[1]):
        new_data[:, ch_idx] = np.interp(new_time, data.time, data.data[:, ch_idx])
    
    return TimeSeriesData(
        time=new_time,
        data=new_data,
        channels=data.channels.copy(),
        sample_rate=new_sample_rate,
        units=data.units,
        coordinates=data.coordinates.copy() if data.coordinates is not None else None,
        metadata=data.metadata.copy()
    )


def align_time_series(
    datasets: List[TimeSeriesData],
    method: str = "intersection"
) -> List[TimeSeriesData]:
    """Align multiple time series to common time base.
    
    Args:
        datasets: List of time series to align
        method: Alignment method ('intersection', 'union', 'first')
        
    Returns:
        List of aligned time series
    """
    if not datasets:
        return []
    
    if len(datasets) == 1:
        return datasets
    
    # Determine common time base
    if method == "intersection":
        # Use intersection of all time ranges
        start_time = max(data.time[0] for data in datasets)
        end_time = min(data.time[-1] for data in datasets)
    elif method == "union":
        # Use union of all time ranges
        start_time = min(data.time[0] for data in datasets)
        end_time = max(data.time[-1] for data in datasets)
    elif method == "first":
        # Use time base of first dataset
        start_time = datasets[0].time[0]
        end_time = datasets[0].time[-1]
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # Find common sampling rate (use minimum for safety)
    common_sample_rate = min(data.sample_rate for data in datasets)
    
    # Create common time vector
    time_span = end_time - start_time
    num_samples = int(time_span * common_sample_rate) + 1
    common_time = np.linspace(start_time, end_time, num_samples)
    
    # Interpolate all datasets to common time base
    aligned_datasets = []
    for data in datasets:
        new_data = np.zeros((len(common_time), data.data.shape[1]))
        
        for ch_idx in range(data.data.shape[1]):
            new_data[:, ch_idx] = np.interp(common_time, data.time, data.data[:, ch_idx])
        
        aligned_data = TimeSeriesData(
            time=common_time,
            data=new_data,
            channels=data.channels.copy(),
            sample_rate=common_sample_rate,
            units=data.units,
            coordinates=data.coordinates.copy() if data.coordinates is not None else None,
            metadata=data.metadata.copy()
        )
        aligned_datasets.append(aligned_data)
    
    return aligned_datasets


def compute_data_statistics(
    data: Union[TimeSeriesData, ProcessedData]
) -> Dict[str, Any]:
    """Compute statistical summary of data.
    
    Args:
        data: Data object to analyze
        
    Returns:
        Dictionary with statistical information
    """
    stats = {}
    
    # Basic statistics
    stats['mean'] = np.mean(data.data, axis=0)
    stats['std'] = np.std(data.data, axis=0)
    stats['min'] = np.min(data.data, axis=0)
    stats['max'] = np.max(data.data, axis=0)
    stats['median'] = np.median(data.data, axis=0)
    
    # Data quality metrics
    stats['num_samples'] = data.data.shape[0]
    stats['num_channels'] = data.data.shape[1]
    stats['nan_count'] = np.sum(np.isnan(data.data), axis=0)
    stats['inf_count'] = np.sum(np.isinf(data.data), axis=0)
    
    # Time statistics
    if hasattr(data, 'time'):
        stats['time_span'] = float(data.time[-1] - data.time[0])
        stats['sample_rate'] = float(1.0 / np.mean(np.diff(data.time)))
        stats['time_gaps'] = _detect_time_gaps(data.time)
    
    # Signal quality metrics
    stats['snr_estimate'] = _estimate_snr(data.data)
    stats['dynamic_range'] = stats['max'] - stats['min']
    
    return stats


def detect_outliers(
    data: npt.NDArray[np.floating],
    method: str = "iqr",
    threshold: float = 3.0
) -> npt.NDArray[np.bool_]:
    """Detect outliers in data.
    
    Args:
        data: Data array (samples x channels)
        method: Detection method ('iqr', 'zscore', 'mad')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if method == "iqr":
        return _detect_outliers_iqr(data, threshold)
    elif method == "zscore":
        return _detect_outliers_zscore(data, threshold)
    elif method == "mad":
        return _detect_outliers_mad(data, threshold)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def create_channel_map(
    channel_names: List[str],
    coordinates: Optional[npt.NDArray[np.floating]] = None
) -> Dict[str, Dict[str, Any]]:
    """Create a mapping of channel information.
    
    Args:
        channel_names: List of channel names
        coordinates: Optional channel coordinates
        
    Returns:
        Dictionary mapping channel names to information
    """
    channel_map = {}
    
    for i, ch_name in enumerate(channel_names):
        ch_info = {
            'index': i,
            'name': ch_name
        }
        
        if coordinates is not None and i < len(coordinates):
            ch_info['coordinates'] = {
                'R': float(coordinates[i, 0]) if coordinates.shape[1] > 0 else 0.0,
                'Z': float(coordinates[i, 1]) if coordinates.shape[1] > 1 else 0.0,
                'phi': float(coordinates[i, 2]) if coordinates.shape[1] > 2 else 0.0
            }
        
        channel_map[ch_name] = ch_info
    
    return channel_map


# Private helper functions
def _validate_time_series(data: TimeSeriesData, strict: bool) -> List[str]:
    """Validate TimeSeriesData object."""
    warnings_list = []
    
    # Check time vector
    if len(data.time) == 0:
        warnings_list.append("ERROR: Empty time vector")
    elif len(np.unique(data.time)) != len(data.time):
        warnings_list.append("WARNING: Non-unique time values detected")
    elif not np.all(np.diff(data.time) > 0):
        warnings_list.append("ERROR: Time vector is not monotonically increasing")
    
    # Check data matrix
    if data.data.size == 0:
        warnings_list.append("ERROR: Empty data matrix")
    elif np.any(np.isnan(data.data)):
        nan_count = np.sum(np.isnan(data.data))
        warnings_list.append(f"WARNING: {nan_count} NaN values in data")
    elif np.any(np.isinf(data.data)):
        inf_count = np.sum(np.isinf(data.data))
        warnings_list.append(f"WARNING: {inf_count} infinite values in data")
    
    # Check sampling rate consistency
    if len(data.time) > 1:
        actual_rate = 1.0 / np.mean(np.diff(data.time))
        rate_diff = abs(actual_rate - data.sample_rate) / data.sample_rate
        if rate_diff > 0.05:  # 5% tolerance
            warnings_list.append(f"WARNING: Sample rate mismatch - stated: {data.sample_rate:.2f}, actual: {actual_rate:.2f}")
    
    return warnings_list


def _validate_processed_data(data: ProcessedData, strict: bool) -> List[str]:
    """Validate ProcessedData object."""
    warnings_list = []
    
    # Inherit time series validations
    time_series_like = TimeSeriesData(
        time=data.time,
        data=data.data,
        channels=data.channel_names,
        sample_rate=1.0 / np.mean(np.diff(data.time)) if len(data.time) > 1 else 1.0
    )
    warnings_list.extend(_validate_time_series(time_series_like, strict))
    
    # Additional processed data checks
    if len(data.channel_names) != data.data.shape[1]:
        warnings_list.append("ERROR: Channel names count mismatch with data columns")
    
    if len(data.coordinates) != len(data.channel_names):
        warnings_list.append("ERROR: Coordinates count mismatch with channels")
    
    return warnings_list


def _validate_shot_data(data: ShotData, strict: bool) -> List[str]:
    """Validate ShotData object."""
    warnings_list = []
    
    # Check shot number
    if data.shot_number <= 0:
        warnings_list.append("WARNING: Invalid shot number")
    
    # Check time range
    if data.time_range[0] >= data.time_range[1]:
        warnings_list.append("ERROR: Invalid time range")
    
    # Check channel consistency
    if len(data.channel_names) != len(data.channels):
        warnings_list.append("ERROR: Channel names count mismatch with channel data")
    
    return warnings_list


def _check_merge_compatibility(
    datasets: List[Union[TimeSeriesData, ProcessedData]],
    axis: str
) -> None:
    """Check if datasets are compatible for merging."""
    if axis == "time":
        # Check that all datasets have same number of channels
        ref_channels = len(datasets[0].channels if hasattr(datasets[0], 'channels') else datasets[0].channel_names)
        for i, data in enumerate(datasets[1:], start=1):
            data_channels = len(data.channels if hasattr(data, 'channels') else data.channel_names)
            if data_channels != ref_channels:
                raise ValueError(f"Dataset {i} has {data_channels} channels, expected {ref_channels}")
    
    elif axis == "channels":
        # Check that all datasets have same time length
        ref_time_len = len(datasets[0].time)
        for i, data in enumerate(datasets[1:], start=1):
            if len(data.time) != ref_time_len:
                raise ValueError(f"Dataset {i} has {len(data.time)} time samples, expected {ref_time_len}")


def _concatenate_datasets(
    datasets: List[Union[TimeSeriesData, ProcessedData]],
    axis: str
) -> Union[TimeSeriesData, ProcessedData]:
    """Concatenate datasets along specified axis."""
    if axis == "time":
        # Concatenate along time axis
        merged_time = np.concatenate([data.time for data in datasets])
        merged_data = np.concatenate([data.data for data in datasets], axis=0)
        
        # Use first dataset as template
        template = datasets[0]
        if isinstance(template, TimeSeriesData):
            return TimeSeriesData(
                time=merged_time,
                data=merged_data,
                channels=template.channels.copy(),
                sample_rate=template.sample_rate,
                units=template.units,
                coordinates=template.coordinates.copy() if template.coordinates is not None else None,
                metadata=template.metadata.copy()
            )
        else:  # ProcessedData
            return ProcessedData(
                shot_number=template.shot_number,
                time=merged_time,
                data=merged_data,
                coordinates=template.coordinates.copy(),
                channel_names=template.channel_names.copy(),
                bad_channels=template.bad_channels.copy(),
                processing_steps=template.processing_steps.copy() + ["time_concatenation"],
                interpolation_method=template.interpolation_method,
                detrend_method=template.detrend_method
            )
    
    elif axis == "channels":
        # Concatenate along channels axis
        merged_data = np.concatenate([data.data for data in datasets], axis=1)
        
        # Merge channel information
        all_channels = []
        all_coords = []
        
        for data in datasets:
            if hasattr(data, 'channels'):
                all_channels.extend(data.channels)
            else:
                all_channels.extend(data.channel_names)
            
            if hasattr(data, 'coordinates') and data.coordinates is not None:
                all_coords.append(data.coordinates)
        
        merged_coords = np.concatenate(all_coords) if all_coords else None
        
        # Use first dataset as template
        template = datasets[0]
        if isinstance(template, TimeSeriesData):
            return TimeSeriesData(
                time=template.time.copy(),
                data=merged_data,
                channels=all_channels,
                sample_rate=template.sample_rate,
                units=template.units,
                coordinates=merged_coords,
                metadata=template.metadata.copy()
            )
        else:  # ProcessedData
            return ProcessedData(
                shot_number=template.shot_number,
                time=template.time.copy(),
                data=merged_data,
                coordinates=merged_coords,
                channel_names=all_channels,
                bad_channels=template.bad_channels.copy(),
                processing_steps=template.processing_steps.copy() + ["channel_concatenation"],
                interpolation_method=template.interpolation_method,
                detrend_method=template.detrend_method
            )
    else:
        raise ValueError(f"Unknown concatenation axis: {axis}")


def _average_datasets(
    datasets: List[Union[TimeSeriesData, ProcessedData]]
) -> Union[TimeSeriesData, ProcessedData]:
    """Average datasets (requires same time base)."""
    # First align all datasets to same time base
    aligned_datasets = align_time_series(datasets) if all(isinstance(d, TimeSeriesData) for d in datasets) else datasets
    
    # Average data
    data_arrays = [data.data for data in aligned_datasets]
    averaged_data = np.mean(data_arrays, axis=0)
    
    # Use first dataset as template
    template = aligned_datasets[0]
    if isinstance(template, TimeSeriesData):
        return TimeSeriesData(
            time=template.time.copy(),
            data=averaged_data,
            channels=template.channels.copy(),
            sample_rate=template.sample_rate,
            units=template.units,
            coordinates=template.coordinates.copy() if template.coordinates is not None else None,
            metadata=template.metadata.copy()
        )
    else:  # ProcessedData
        return ProcessedData(
            shot_number=template.shot_number,
            time=template.time.copy(),
            data=averaged_data,
            coordinates=template.coordinates.copy(),
            channel_names=template.channel_names.copy(),
            bad_channels=template.bad_channels.copy(),
            processing_steps=template.processing_steps.copy() + ["ensemble_averaging"],
            interpolation_method=template.interpolation_method,
            detrend_method=template.detrend_method
        )


def _stack_datasets(
    datasets: List[Union[TimeSeriesData, ProcessedData]],
    axis: str
) -> Union[TimeSeriesData, ProcessedData]:
    """Stack datasets (adds new dimension)."""
    # This is a placeholder - actual implementation would depend on specific requirements
    return _concatenate_datasets(datasets, axis)


def _copy_data_object(
    data: Union[TimeSeriesData, ProcessedData, ShotData]
) -> Union[TimeSeriesData, ProcessedData, ShotData]:
    """Create a copy of data object."""
    if isinstance(data, TimeSeriesData):
        return TimeSeriesData(
            time=data.time.copy(),
            data=data.data.copy(),
            channels=data.channels.copy(),
            sample_rate=data.sample_rate,
            units=data.units,
            coordinates=data.coordinates.copy() if data.coordinates is not None else None,
            metadata=data.metadata.copy()
        )
    elif isinstance(data, ProcessedData):
        return ProcessedData(
            shot_number=data.shot_number,
            time=data.time.copy(),
            data=data.data.copy(),
            coordinates=data.coordinates.copy(),
            channel_names=data.channel_names.copy(),
            bad_channels=data.bad_channels.copy(),
            processing_steps=data.processing_steps.copy(),
            interpolation_method=data.interpolation_method,
            detrend_method=data.detrend_method
        )
    else:  # ShotData
        return ShotData(
            shot_number=data.shot_number,
            time_range=data.time_range,
            channels={k: v.copy() for k, v in data.channels.items()},
            coordinates=data.coordinates.copy(),
            channel_names=data.channel_names.copy(),
            acquisition_info=data.acquisition_info.copy(),
            preprocessing=data.preprocessing.copy(),
            quality_flags=data.quality_flags.copy()
        )


def _filter_time_series_channels(
    data: TimeSeriesData,
    bad_channels: List[str],
    in_place: bool
) -> TimeSeriesData:
    """Filter channels from TimeSeriesData."""
    if not in_place:
        data = _copy_data_object(data)
    
    # Find indices of good channels
    good_indices = [i for i, ch in enumerate(data.channels) if ch not in bad_channels]
    
    if not good_indices:
        raise ValueError("No good channels remaining after filtering")
    
    # Filter data
    data.data = data.data[:, good_indices]
    data.channels = [data.channels[i] for i in good_indices]
    
    if data.coordinates is not None:
        data.coordinates = data.coordinates[good_indices]
    
    return data


def _filter_processed_data_channels(
    data: ProcessedData,
    bad_channels: List[str],
    in_place: bool
) -> ProcessedData:
    """Filter channels from ProcessedData."""
    if not in_place:
        data = _copy_data_object(data)
    
    # Find indices of good channels
    good_indices = [i for i, ch in enumerate(data.channel_names) if ch not in bad_channels]
    
    if not good_indices:
        raise ValueError("No good channels remaining after filtering")
    
    # Filter data
    data.data = data.data[:, good_indices]
    data.coordinates = data.coordinates[good_indices]
    data.channel_names = [data.channel_names[i] for i in good_indices]
    
    # Update bad channels list
    data.bad_channels.extend([ch for ch in bad_channels if ch not in data.bad_channels])
    data.processing_steps.append("channel_filtering")
    
    return data


def _filter_shot_data_channels(
    data: ShotData,
    bad_channels: List[str],
    in_place: bool
) -> ShotData:
    """Filter channels from ShotData."""
    if not in_place:
        data = _copy_data_object(data)
    
    # Remove bad channels
    for ch in bad_channels:
        if ch in data.channels:
            del data.channels[ch]
        if ch in data.coordinates:
            del data.coordinates[ch]
        if ch in data.channel_names:
            data.channel_names.remove(ch)
    
    return data


def _detect_time_gaps(time: npt.NDArray[np.floating], threshold: float = 2.0) -> List[int]:
    """Detect gaps in time series."""
    dt = np.diff(time)
    median_dt = np.median(dt)
    gap_indices = np.where(dt > threshold * median_dt)[0]
    return gap_indices.tolist()


def _estimate_snr(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Estimate signal-to-noise ratio for each channel."""
    # Simple SNR estimate using signal variance vs noise estimate
    signal_power = np.var(data, axis=0)
    
    # Estimate noise from high-frequency content (crude approximation)
    if data.shape[0] > 10:
        noise_estimate = np.var(np.diff(data, axis=0), axis=0) / 2.0
        snr = signal_power / (noise_estimate + 1e-12)  # Avoid division by zero
    else:
        snr = np.ones(data.shape[1])  # Fallback
    
    return snr


def _detect_outliers_iqr(data: npt.NDArray[np.floating], threshold: float) -> npt.NDArray[np.bool_]:
    """Detect outliers using IQR method."""
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


def _detect_outliers_zscore(data: npt.NDArray[np.floating], threshold: float) -> npt.NDArray[np.bool_]:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    return z_scores > threshold


def _detect_outliers_mad(data: npt.NDArray[np.floating], threshold: float) -> npt.NDArray[np.bool_]:
    """Detect outliers using Median Absolute Deviation method."""
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    
    # Modified Z-score using MAD
    modified_z_scores = 0.6745 * (data - median) / (mad + 1e-12)
    return np.abs(modified_z_scores) > threshold 