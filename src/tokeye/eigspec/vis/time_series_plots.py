"""
Time series visualization functions for eigspec package.

This module provides specialized plotting functions for time-domain analysis:
- Multi-channel time series plots with modal overlays
- Block-wise analysis timeline visualization
- RMS evolution and amplitude tracking plots
- Time-frequency analysis and trend visualization

Based on the MATLAB eigspec toolbox time series plotting functions:
- Time series plotting in view_pcaspec_results.m
- Block-wise timeline plots in eigspec_mmain.m
- RMS and amplitude evolution plots
- collect_rep_refvec_trace.m - Reference vector trace plotting
- Various time-domain plotting utilities
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass


@dataclass
class TimeSeriesPlotOptions:
    """Configuration options for time series plots.
    
    Attributes:
        figsize: Figure size (width, height) in inches
        dpi: Figure DPI for resolution
        fontsize: Base font size for text elements
        title_fontsize: Font size for plot titles
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend text
        grid: Whether to show grid lines
        colormap: Colormap name for multi-series plots
        marker_size: Size of scatter plot markers
        line_width: Width of plot lines
        alpha: Transparency level (0-1)
        save_format: Default format for saving figures
    """
    figsize: Tuple[float, float] = (14, 8)
    dpi: int = 100
    fontsize: int = 12
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid: bool = True
    colormap: str = 'tab10'
    marker_size: float = 4.0
    line_width: float = 1.5
    alpha: float = 0.8
    save_format: str = 'png'


def plot_time_traces(
    time: npt.NDArray[np.floating],
    data: npt.NDArray[np.floating],
    channel_labels: Optional[List[str]] = None,
    plot_type: str = 'line',
    normalize: bool = False,
    options: Optional[TimeSeriesPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot time series traces for multiple channels.
    
    Args:
        time: Time vector
        data: Data matrix (n_time x n_channels)
        channel_labels: Optional labels for channels
        plot_type: Type of plot ('line', 'scatter', 'both')
        normalize: Whether to normalize each trace
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = TimeSeriesPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    n_channels = data.shape[1]
    colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, n_channels))
    
    # Normalize data if requested
    if normalize:
        plot_data = data / np.max(np.abs(data), axis=0)
    else:
        plot_data = data
    
    # Plot each channel
    for i in range(n_channels):
        label = channel_labels[i] if channel_labels else f'Channel {i+1}'
        
        if plot_type == 'line':
            ax.plot(time, plot_data[:, i], color=colors[i], 
                   linewidth=options.line_width, alpha=options.alpha, label=label)
        elif plot_type == 'scatter':
            ax.scatter(time, plot_data[:, i], c=[colors[i]], 
                      s=options.marker_size**2, alpha=options.alpha, label=label)
        elif plot_type == 'both':
            ax.plot(time, plot_data[:, i], color=colors[i], 
                   linewidth=options.line_width, alpha=options.alpha)
            ax.scatter(time, plot_data[:, i], c=[colors[i]], 
                      s=options.marker_size**2, alpha=options.alpha, label=label)
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel('Amplitude', fontsize=options.label_fontsize)
    ax.set_title('Time Series Traces', fontsize=options.title_fontsize)
    
    if n_channels <= 10:
        ax.legend(fontsize=options.legend_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_prototype_traces(
    time_data: List[npt.NDArray[np.floating]],
    frequency_data: List[npt.NDArray[np.floating]], 
    rms_data: List[npt.NDArray[np.floating]],
    mode_labels: List[str],
    plot_frequency: bool = True,
    plot_rms: bool = True,
    filter_beta: Optional[float] = None,
    options: Optional[TimeSeriesPlotOptions] = None
) -> Figure:
    """Plot prototype trace data for multiple modes.
    
    Args:
        time_data: List of time vectors for each mode
        frequency_data: List of frequency data for each mode
        rms_data: List of RMS data for each mode
        mode_labels: Labels for each mode (e.g., 'm/n=1/1')
        plot_frequency: Whether to plot frequency traces
        plot_rms: Whether to plot RMS traces
        filter_beta: Optional filtering parameter
        options: Plot configuration options
        
    Returns:
        Figure object with subplots
    """
    if options is None:
        options = TimeSeriesPlotOptions()
    
    n_plots = sum([plot_frequency, plot_rms])
    fig, axes = plt.subplots(n_plots, 1, figsize=(options.figsize[0], 
                                                 options.figsize[1]*n_plots/2),
                           dpi=options.dpi)
    
    if n_plots == 1:
        axes = [axes]
    
    n_modes = len(time_data)
    colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, n_modes))
    
    plot_idx = 0
    
    if plot_frequency:
        ax = axes[plot_idx]
        for i in range(n_modes):
            ax.scatter(time_data[i] * 1e3, frequency_data[i] / 1e3,
                      c=[colors[i]], marker='o', s=options.marker_size**2,
                      alpha=options.alpha, label=mode_labels[i])
        
        ax.set_xlabel('Time [ms]', fontsize=options.label_fontsize)
        ax.set_ylabel('Frequency [kHz]', fontsize=options.label_fontsize)
        ax.set_title('Prototype Frequency Traces', fontsize=options.title_fontsize)
        ax.legend(fontsize=options.legend_fontsize)
        
        if options.grid:
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    if plot_rms:
        ax = axes[plot_idx]
        for i in range(n_modes):
            ax.scatter(time_data[i] * 1e3, rms_data[i],
                      c=[colors[i]], marker='o', s=options.marker_size**2,
                      alpha=options.alpha, label=mode_labels[i])
        
        ax.set_xlabel('Time [ms]', fontsize=options.label_fontsize)
        ax.set_ylabel('RMS [T/s]', fontsize=options.label_fontsize)
        
        title = 'Prototype RMS Traces'
        if filter_beta is not None:
            title += f' (filtered, β={filter_beta:.2f})'
        ax.set_title(title, fontsize=options.title_fontsize)
        
        ax.legend(fontsize=options.legend_fontsize)
        
        if options.grid:
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_filtered_traces(
    time: npt.NDArray[np.floating],
    original_data: npt.NDArray[np.floating],
    filtered_data: npt.NDArray[np.floating],
    filter_params: Dict[str, Any],
    channel_indices: Optional[List[int]] = None,
    options: Optional[TimeSeriesPlotOptions] = None
) -> Figure:
    """Plot comparison of original and filtered traces.
    
    Args:
        time: Time vector
        original_data: Original data matrix
        filtered_data: Filtered data matrix
        filter_params: Dictionary with filter parameters
        channel_indices: Optional indices of channels to plot
        options: Plot configuration options
        
    Returns:
        Figure object with subplots
    """
    if options is None:
        options = TimeSeriesPlotOptions()
    
    if channel_indices is None:
        channel_indices = list(range(min(4, original_data.shape[1])))
    
    n_channels = len(channel_indices)
    fig, axes = plt.subplots(n_channels, 1, figsize=(options.figsize[0], 
                                                    options.figsize[1]*n_channels/3),
                           dpi=options.dpi)
    
    if n_channels == 1:
        axes = [axes]
    
    for i, ch_idx in enumerate(channel_indices):
        ax = axes[i]
        
        ax.plot(time, original_data[:, ch_idx], 'b-', 
               linewidth=options.line_width, alpha=options.alpha,
               label='Original')
        ax.plot(time, filtered_data[:, ch_idx], 'r-', 
               linewidth=options.line_width, alpha=options.alpha,
               label='Filtered')
        
        ax.set_ylabel(f'Channel {ch_idx+1}', fontsize=options.label_fontsize)
        ax.legend(fontsize=options.legend_fontsize-2)
        
        if options.grid:
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time', fontsize=options.label_fontsize)
    
    # Create title with filter information
    filter_info = ', '.join([f'{k}={v}' for k, v in filter_params.items()])
    fig.suptitle(f'Original vs Filtered Data ({filter_info})', 
                fontsize=options.title_fontsize)
    
    plt.tight_layout()
    return fig


def plot_multi_channel_overlay(
    time: npt.NDArray[np.floating],
    data: npt.NDArray[np.floating],
    overlay_style: str = 'offset',
    channel_labels: Optional[List[str]] = None,
    offset_scale: float = 1.0,
    options: Optional[TimeSeriesPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot multiple channels with offset or overlay styling.
    
    Args:
        time: Time vector
        data: Data matrix (n_time x n_channels)
        overlay_style: Style of overlay ('offset', 'transparent', 'normalized')
        channel_labels: Optional labels for channels
        offset_scale: Scaling factor for offset style
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = TimeSeriesPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    n_channels = data.shape[1]
    colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, n_channels))
    
    if overlay_style == 'offset':
        # Calculate offset based on data range
        data_range = np.max(data) - np.min(data)
        offset = data_range * offset_scale
        
        for i in range(n_channels):
            label = channel_labels[i] if channel_labels else f'Ch {i+1}'
            offset_data = data[:, i] + i * offset
            ax.plot(time, offset_data, color=colors[i], 
                   linewidth=options.line_width, alpha=options.alpha, label=label)
        
        # Set custom y-tick labels
        if channel_labels:
            y_positions = [i * offset for i in range(n_channels)]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(channel_labels)
        
    elif overlay_style == 'transparent':
        for i in range(n_channels):
            label = channel_labels[i] if channel_labels else f'Ch {i+1}'
            ax.plot(time, data[:, i], color=colors[i], 
                   linewidth=options.line_width, alpha=options.alpha*0.7, label=label)
    
    elif overlay_style == 'normalized':
        # Normalize each channel to [0, 1] range
        normalized_data = np.zeros_like(data)
        for i in range(n_channels):
            ch_data = data[:, i]
            normalized_data[:, i] = (ch_data - np.min(ch_data)) / (np.max(ch_data) - np.min(ch_data))
        
        for i in range(n_channels):
            label = channel_labels[i] if channel_labels else f'Ch {i+1}'
            ax.plot(time, normalized_data[:, i], color=colors[i], 
                   linewidth=options.line_width, alpha=options.alpha, label=label)
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel('Amplitude', fontsize=options.label_fontsize)
    ax.set_title(f'Multi-Channel Overlay ({overlay_style})', fontsize=options.title_fontsize)
    
    if n_channels <= 15:
        ax.legend(fontsize=options.legend_fontsize, bbox_to_anchor=(1.05, 1), 
                 loc='upper left')
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def save_time_series_plot(
    fig: Figure, 
    filename: str, 
    options: Optional[TimeSeriesPlotOptions] = None
) -> None:
    """Save a time series plot to file.
    
    Args:
        fig: Figure to save
        filename: Output filename
        options: Plot configuration options
    """
    if options is None:
        options = TimeSeriesPlotOptions()
    
    if not filename.endswith(f'.{options.save_format}'):
        filename += f'.{options.save_format}'
    
    fig.savefig(filename, format=options.save_format, dpi=options.dpi, 
               bbox_inches='tight')
    print(f"Saved time series plot to: {filename}") 