"""
Data structures for eigspec I/O operations.

This module defines the core data structures used for storing and managing
spectral analysis and modal identification data, including time series,
analysis results, and configuration information.
"""

from typing import Optional, Dict, List, Any, Union, Tuple
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TimeSeriesData:
    """Time series data container.
    
    Attributes:
        time: Time vector (seconds)
        data: Data matrix (time x channels)
        channels: Channel names/identifiers
        sample_rate: Sampling rate (Hz)
        units: Data units (e.g., 'Tesla', 'Gauss')
        coordinates: Physical coordinates for each channel
        metadata: Additional metadata dictionary
    """
    time: npt.NDArray[np.floating]
    data: npt.NDArray[np.floating]
    channels: List[str]
    sample_rate: float
    units: str = "Tesla"
    coordinates: Optional[npt.NDArray[np.floating]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate data consistency after initialization."""
        if len(self.time) != self.data.shape[0]:
            raise ValueError("Time vector length must match data rows")
        if len(self.channels) != self.data.shape[1]:
            raise ValueError("Number of channels must match data columns")
        if self.coordinates is not None and len(self.coordinates) != len(self.channels):
            raise ValueError("Coordinates length must match number of channels")


@dataclass 
class MirnovData(TimeSeriesData):
    """Mirnov probe data container (specialized time series).
    
    Attributes:
        shot_number: Plasma shot identifier
        probe_type: Type of probe ('poloidal', 'toroidal', 'mixed')
        array_geometry: Geometric arrangement information
        bad_channels: List of channels to exclude from analysis
        calibration: Calibration factors for each channel
    """
    shot_number: int = 0
    probe_type: str = "mixed"
    array_geometry: Optional[Dict[str, Any]] = None
    bad_channels: List[str] = field(default_factory=list)
    calibration: Optional[npt.NDArray[np.floating]] = None


@dataclass
class ShotData:
    """Plasma shot data container.
    
    Attributes:
        shot_number: Unique shot identifier
        time_range: Time range [start, end] in seconds
        channels: Raw time series data organized by channel
        coordinates: Physical coordinates (R, Z, phi) for each channel
        channel_names: Names/identifiers for each channel  
        acquisition_info: Data acquisition metadata
        preprocessing: Applied preprocessing steps
        quality_flags: Data quality indicators
    """
    shot_number: int
    time_range: Tuple[float, float]
    channels: Dict[str, npt.NDArray[np.floating]]
    coordinates: Dict[str, Tuple[float, float, float]]
    channel_names: List[str]
    acquisition_info: Dict[str, Any] = field(default_factory=dict)
    preprocessing: List[str] = field(default_factory=list)
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    
    def get_time_series(self, channel_subset: Optional[List[str]] = None) -> TimeSeriesData:
        """Extract time series data for specified channels.
        
        Args:
            channel_subset: List of channels to extract (None for all)
            
        Returns:
            TimeSeriesData object with requested channels
        """
        if channel_subset is None:
            channel_subset = self.channel_names
        
        # Filter channels that exist in the data
        valid_channels = [ch for ch in channel_subset if ch in self.channels]
        
        if not valid_channels:
            raise ValueError("No valid channels found")
        
        # Assume all channels have the same time base (first channel)
        time_vec = self.channels[valid_channels[0]][:, 0]
        data_matrix = np.column_stack([self.channels[ch][:, 1] for ch in valid_channels])
        
        # Extract coordinates if available
        coords = None
        if self.coordinates:
            coords = np.array([self.coordinates.get(ch, (0, 0, 0)) for ch in valid_channels])
        
        return TimeSeriesData(
            time=time_vec,
            data=data_matrix,
            channels=valid_channels,
            sample_rate=float(1.0 / np.mean(np.diff(time_vec))),
            coordinates=coords,
            metadata={"shot_number": self.shot_number}
        )


@dataclass
class ProcessedData:
    """Processed/bundled data container.
    
    Attributes:
        shot_number: Source shot identifier
        time: Common time vector
        data: Data matrix (time x channels) with bad channels removed
        coordinates: Physical coordinates for remaining channels
        channel_names: Names of remaining channels
        bad_channels: List of removed channel names
        processing_steps: Applied processing operations
        interpolation_method: Method used for time alignment
        detrend_method: Detrending method applied
        filter_info: Filtering information if applied
    """
    shot_number: int
    time: npt.NDArray[np.floating]
    data: npt.NDArray[np.floating]  
    coordinates: npt.NDArray[np.floating]
    channel_names: List[str]
    bad_channels: List[str] = field(default_factory=list)
    processing_steps: List[str] = field(default_factory=list)
    interpolation_method: str = "linear"
    detrend_method: str = "linear"
    filter_info: Optional[Dict[str, Any]] = None
    
    @property
    def num_channels(self) -> int:
        """Number of channels in processed data."""
        return len(self.channel_names)
    
    @property
    def num_samples(self) -> int:
        """Number of time samples."""
        return len(self.time)
    
    @property
    def time_range(self) -> Tuple[float, float]:
        """Time range [start, end] in seconds."""
        return (float(self.time[0]), float(self.time[-1]))


@dataclass
class AnalysisResult:
    """Analysis results container.
    
    Attributes:
        analysis_type: Type of analysis performed ('ssi', 'arpca', 'spectral')
        shot_number: Source shot identifier
        time_blocks: Time block information for analysis
        frequencies: Identified frequencies (Hz)
        eigenvalues: System eigenvalues (complex)
        mode_shapes: Spatial mode shapes (complex)
        stability: Stability indicators (damping ratios)
        parameters: Analysis parameters used
        quality_metrics: Quality assessment metrics
        cluster_info: Clustering analysis results if applicable
        timestamp: Analysis timestamp
    """
    analysis_type: str
    shot_number: int
    time_blocks: List[Tuple[float, float]]
    frequencies: npt.NDArray[np.floating]
    eigenvalues: npt.NDArray[np.complexfloating]
    mode_shapes: npt.NDArray[np.complexfloating]
    stability: npt.NDArray[np.floating]
    parameters: Dict[str, Any]
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    cluster_info: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def num_modes(self) -> int:
        """Number of identified modes."""
        return len(self.frequencies)
    
    @property 
    def stable_modes(self) -> npt.NDArray[np.bool_]:
        """Boolean array indicating stable modes (damping > 0)."""
        return self.stability > 0
    
    def get_mode_info(self, mode_idx: int) -> Dict[str, Any]:
        """Get information for a specific mode.
        
        Args:
            mode_idx: Mode index
            
        Returns:
            Dictionary with mode information
        """
        if mode_idx < 0 or mode_idx >= self.num_modes:
            raise IndexError(f"Mode index {mode_idx} out of range")
        
        return {
            "frequency": float(self.frequencies[mode_idx]),
            "eigenvalue": complex(self.eigenvalues[mode_idx]),
            "mode_shape": self.mode_shapes[:, mode_idx],
            "stability": float(self.stability[mode_idx]),
            "is_stable": bool(self.stable_modes[mode_idx])
        }


@dataclass
class ConfigData:
    """Configuration data container.
    
    Attributes:
        analysis_params: Analysis algorithm parameters
        processing_params: Data processing parameters  
        plot_params: Plotting and visualization parameters
        io_params: Input/output parameters
        file_paths: Important file paths
        user_settings: User-specific settings
        version: Configuration version
    """
    analysis_params: Dict[str, Any] = field(default_factory=dict)
    processing_params: Dict[str, Any] = field(default_factory=dict)
    plot_params: Dict[str, Any] = field(default_factory=dict)
    io_params: Dict[str, Any] = field(default_factory=dict)
    file_paths: Dict[str, str] = field(default_factory=dict)
    user_settings: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    def get_param(self, category: str, param_name: str, default: Any = None) -> Any:
        """Get a specific parameter value.
        
        Args:
            category: Parameter category ('analysis', 'processing', 'plot', 'io')
            param_name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        category_map = {
            "analysis": self.analysis_params,
            "processing": self.processing_params,
            "plot": self.plot_params,
            "io": self.io_params
        }
        
        if category not in category_map:
            raise ValueError(f"Unknown parameter category: {category}")
        
        return category_map[category].get(param_name, default)
    
    def set_param(self, category: str, param_name: str, value: Any) -> None:
        """Set a specific parameter value.
        
        Args:
            category: Parameter category
            param_name: Parameter name
            value: Parameter value
        """
        category_map = {
            "analysis": self.analysis_params,
            "processing": self.processing_params,
            "plot": self.plot_params,
            "io": self.io_params
        }
        
        if category not in category_map:
            raise ValueError(f"Unknown parameter category: {category}")
        
        category_map[category][param_name] = value


# Type aliases for common data types
DataMatrix = npt.NDArray[np.floating]
ComplexDataMatrix = npt.NDArray[np.complexfloating]
TimeVector = npt.NDArray[np.floating]
ChannelList = List[str]
CoordinateArray = npt.NDArray[np.floating] 