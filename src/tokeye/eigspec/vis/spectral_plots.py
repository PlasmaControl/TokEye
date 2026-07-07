"""
Spectral plotting functions for eigspec package.

This module provides specialized plotting functions for frequency-domain analysis:
- Power spectral density plots with modal overlays
- Frequency-time spectrograms and waterfall plots
- Coherence and phase plots for multi-channel analysis
- Eigenvalue and stability plots

Based on the MATLAB eigspec toolbox spectral plotting functions:
- Various plotting routines in view_pcaspec_results.m
- Spectral analysis plots in eigspec_mmain.m
- FFT and frequency domain plotting utilities
- Coherence plotting in oddevenupdate_emc.m
"""

from typing import Optional, List, Tuple, Union, Literal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass, field


@dataclass
class SpectralPlotOptions:
    """Configuration options for spectral analysis plots.
    
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
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    fontsize: int = 12
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid: bool = True
    colormap: str = 'viridis'
    marker_size: float = 4.0
    line_width: float = 1.5
    alpha: float = 0.8
    save_format: str = 'png'


def setup_spectral_figure(
    nrows: int = 1, 
    ncols: int = 1, 
    options: Optional[SpectralPlotOptions] = None
) -> Tuple[Figure, Union[Axes, npt.NDArray]]:
    """Set up a figure for spectral analysis plots.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        options: Plot configuration options
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = SpectralPlotOptions()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=options.figsize, dpi=options.dpi)
    
    # Set default font sizes
    plt.rcParams.update({
        'font.size': options.fontsize,
        'axes.titlesize': options.title_fontsize,
        'axes.labelsize': options.label_fontsize,
        'legend.fontsize': options.legend_fontsize,
        'lines.linewidth': options.line_width,
        'lines.markersize': options.marker_size,
    })
    
    return fig, axes


def setup_single_spectral_figure(
    options: Optional[SpectralPlotOptions] = None
) -> Tuple[Figure, Axes]:
    """Set up a figure with a single axes for spectral analysis plots.
    
    Args:
        options: Plot configuration options
        
    Returns:
        Figure and single axes objects
    """
    if options is None:
        options = SpectralPlotOptions()
    
    fig, ax = plt.subplots(1, 1, figsize=options.figsize, dpi=options.dpi)
    
    # Set default font sizes
    plt.rcParams.update({
        'font.size': options.fontsize,
        'axes.titlesize': options.title_fontsize,
        'axes.labelsize': options.label_fontsize,
        'legend.fontsize': options.legend_fontsize,
        'lines.linewidth': options.line_width,
        'lines.markersize': options.marker_size,
    })
    
    return fig, ax


def plot_eigenvalue_evolution(
    time: npt.NDArray[np.floating],
    eigenvalues: npt.NDArray[np.floating],
    n_retained: Optional[int] = None,
    log_scale: bool = True,
    highlight_time: Optional[float] = None,
    options: Optional[SpectralPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot eigenvalue evolution over time.
    
    Args:
        time: Time vector
        eigenvalues: Eigenvalue matrix (time x eigenvalue_index)
        n_retained: Number of retained eigenvalues to highlight
        log_scale: Whether to use logarithmic scale for eigenvalues
        highlight_time: Time point to highlight with vertical line
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = SpectralPlotOptions()
    
    if ax is None:
        fig, ax = setup_single_spectral_figure(options=options)
    else:
        fig = ax.figure
    
    # Convert to log scale if requested
    if log_scale:
        plot_eigenvalues = np.log10(eigenvalues + 1e-16)  # Add small value to avoid log(0)
        ylabel = r'$\log_{10}(\lambda)$'
    else:
        plot_eigenvalues = eigenvalues
        ylabel = r'$\lambda$'
    
    # Plot eigenvalues
    n_eigs = eigenvalues.shape[1]
    colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, n_eigs))
    
    for i in range(n_eigs):
        if n_retained is not None and i < n_retained:
            ax.plot(time, plot_eigenvalues[:, i], color=colors[i], 
                   alpha=options.alpha, linewidth=options.line_width)
        else:
            ax.plot(time, plot_eigenvalues[:, i], 'k-', 
                   alpha=options.alpha * 0.5, linewidth=options.line_width * 0.7)
    
    # Highlight specific time if provided
    if highlight_time is not None:
        ax.axvline(highlight_time, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=options.label_fontsize)
    ax.set_title('Eigenvalue Evolution', fontsize=options.title_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_frequency_time(
    time: npt.NDArray[np.floating],
    frequencies: npt.NDArray[np.floating],
    amplitudes: Optional[npt.NDArray[np.floating]] = None,
    labels: Optional[List[str]] = None,
    sampling_frequency: Optional[float] = None,
    options: Optional[SpectralPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot frequency vs time scatter plot.
    
    Args:
        time: Time vector
        frequencies: Frequency values
        amplitudes: Optional amplitude values for color coding
        labels: Optional labels for different mode types
        sampling_frequency: Sampling frequency for normalization
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = SpectralPlotOptions()
    
    if ax is None:
        fig, ax = setup_single_spectral_figure(options=options)
    else:
        fig = ax.figure
    
    # Convert frequencies to appropriate units
    if sampling_frequency is not None and sampling_frequency > 0:
        plot_frequencies = frequencies / (1e3)  # Convert to kHz
        freq_label = 'Frequency [kHz]'
    else:
        plot_frequencies = frequencies
        freq_label = 'Frequency [rad/sample]'
    
    # Plot based on whether we have labels or amplitudes
    if labels is not None:
        unique_labels = list(set(labels))
        colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            if label == 'nolabel':
                ax.scatter(time[mask], plot_frequencies[mask], 
                         c='black', s=options.marker_size**2, alpha=options.alpha,
                         label=label)
            else:
                ax.scatter(time[mask], plot_frequencies[mask], 
                         c=[colors[i]], s=options.marker_size**2, alpha=options.alpha,
                         label=label)
        ax.legend(fontsize=options.legend_fontsize)
        
    elif amplitudes is not None:
        scatter = ax.scatter(time, plot_frequencies, c=amplitudes, 
                           s=options.marker_size**2, alpha=options.alpha,
                           cmap=options.colormap)
        plt.colorbar(scatter, ax=ax, label='Amplitude')
        
    else:
        ax.scatter(time, plot_frequencies, s=options.marker_size**2, 
                  alpha=options.alpha, color='blue')
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel(freq_label, fontsize=options.label_fontsize)
    ax.set_title('Frequency vs Time', fontsize=options.title_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_phase_time(
    time: npt.NDArray[np.floating],
    eigenvalues: npt.NDArray[np.complex128],
    mode_indices: Optional[List[int]] = None,
    options: Optional[SpectralPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot phase angles of eigenvalues vs time.
    
    Args:
        time: Time vector
        eigenvalues: Complex eigenvalues
        mode_indices: Indices of modes to plot
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = SpectralPlotOptions()
    
    if ax is None:
        fig, ax = setup_single_spectral_figure(options=options)
    else:
        fig = ax.figure
    
    # Extract phase angles
    phases = np.angle(eigenvalues)
    
    if mode_indices is not None:
        for i in mode_indices:
            ax.scatter(time, phases[:, i], s=options.marker_size**2, 
                      alpha=options.alpha, label=f'Mode {i+1}')
        ax.legend(fontsize=options.legend_fontsize)
    else:
        # Plot all modes as black dots
        for i in range(phases.shape[1]):
            ax.scatter(time, phases[:, i], c='black', s=options.marker_size**2, 
                      alpha=options.alpha)
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel(r'Phase [rad]', fontsize=options.label_fontsize)
    ax.set_title('Phase Evolution', fontsize=options.title_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_rms_time(
    time: npt.NDArray[np.floating],
    rms_values: npt.NDArray[np.floating],
    frequencies: Optional[npt.NDArray[np.floating]] = None,
    plot_type: Literal["rms_dot", "rms_b"] = "rms_dot",
    labels: Optional[List[str]] = None,
    options: Optional[SpectralPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot RMS values vs time.
    
    Args:
        time: Time vector
        rms_values: RMS amplitude values
        frequencies: Optional frequency values for RMS(B) calculation
        plot_type: Type of RMS plot ("rms_dot" or "rms_b")
        labels: Optional labels for different modes
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = SpectralPlotOptions()
    
    if ax is None:
        fig, ax = setup_single_spectral_figure(options=options)
    else:
        fig = ax.figure
    
    # Prepare data based on plot type
    if plot_type == "rms_b" and frequencies is not None:
        plot_values = (1e3) * rms_values / (2 * np.pi * frequencies)
        ylabel = 'RMS (B) [mT]'
        title = 'RMS Magnetic Field vs Time'
    else:
        plot_values = rms_values
        ylabel = 'RMS (Ḃ) [T/s]'
        title = 'RMS Magnetic Field Derivative vs Time'
    
    # Plot with or without labels
    if labels is not None:
        unique_labels = list(set(labels))
        colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            if label == 'nolabel':
                ax.scatter(time[mask], plot_values[mask], 
                         c='black', s=options.marker_size**2, alpha=options.alpha,
                         label=label)
            else:
                ax.scatter(time[mask], plot_values[mask], 
                         c=[colors[i]], s=options.marker_size**2, alpha=options.alpha,
                         label=label)
        ax.legend(fontsize=options.legend_fontsize)
    else:
        ax.scatter(time, plot_values, s=options.marker_size**2, 
                  alpha=options.alpha, color='blue')
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=options.label_fontsize)
    ax.set_title(title, fontsize=options.title_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_spectral_summary(
    time: npt.NDArray[np.floating],
    eigenvalues: npt.NDArray[np.floating],
    frequencies: npt.NDArray[np.floating],
    rms_values: npt.NDArray[np.floating],
    n_retained: Optional[int] = None,
    sampling_frequency: Optional[float] = None,
    options: Optional[SpectralPlotOptions] = None
) -> Figure:
    """Create a summary plot of spectral analysis results.
    
    Args:
        time: Time vector
        eigenvalues: Eigenvalue matrix
        frequencies: Frequency values
        rms_values: RMS values
        n_retained: Number of retained eigenvalues
        sampling_frequency: Sampling frequency
        options: Plot configuration options
        
    Returns:
        Figure object with subplots
    """
    if options is None:
        options = SpectralPlotOptions()
    
    fig, axes = setup_spectral_figure(nrows=2, ncols=2, options=options)
    
    # Eigenvalue evolution
    plot_eigenvalue_evolution(time, eigenvalues, n_retained=n_retained, 
                            ax=axes[0, 0], options=options)
    
    # Frequency vs time
    plot_frequency_time(time, frequencies, sampling_frequency=sampling_frequency,
                       ax=axes[0, 1], options=options)
    
    # Phase evolution (if eigenvalues are complex)
    if np.iscomplexobj(eigenvalues):
        plot_phase_time(time, eigenvalues.astype(np.complex128), ax=axes[1, 0], 
                       options=options)
    else:
        axes[1, 0].text(0.5, 0.5, 'No phase data\n(real eigenvalues)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Phase Evolution')
    
    # RMS vs time
    plot_rms_time(time, rms_values, ax=axes[1, 1], options=options)
    
    plt.tight_layout()
    return fig


def save_spectral_plot(
    fig: Figure, 
    filename: str, 
    options: Optional[SpectralPlotOptions] = None
) -> None:
    """Save a spectral plot to file.
    
    Args:
        fig: Figure to save
        filename: Output filename
        options: Plot configuration options
    """
    if options is None:
        options = SpectralPlotOptions()
    
    if not filename.endswith(f'.{options.save_format}'):
        filename += f'.{options.save_format}'
    
    fig.savefig(filename, format=options.save_format, dpi=options.dpi, 
               bbox_inches='tight')
    print(f"Saved plot to: {filename}")


# =============================================================================
# Main view_pcaspec visualization function (MATLAB equivalent)
# =============================================================================

def view_pcaspec(
    analysis_result,
    sensor_coordinates: Optional[npt.NDArray[np.floating]] = None,
    time_point: Optional[float] = None,
    options: Optional[SpectralPlotOptions] = None
) -> List[Figure]:
    """
    Main visualization function for spectral analysis results.
    
    Python equivalent of MATLAB view_pcaspec.m that provides comprehensive
    visualization of spectral analysis results including:
    - Eigenvalue evolution over time
    - Frequency-time spectrograms  
    - Modal shape visualization
    - PCA component analysis
    
    Args:
        analysis_result: Analysis result structure from eigspec functions
        sensor_coordinates: Optional sensor coordinate matrix (N_sensors, 2)
        time_point: Optional specific time point to visualize
        options: Plot configuration options
        
    Returns:
        List of Figure objects created
    """
    if options is None:
        options = SpectralPlotOptions()
    
    figures = []
    
    # Extract time vector and block results
    if hasattr(analysis_result, 'L') and analysis_result.L:
        # Block-based analysis results
        TL = np.array([block.centre_t for block in analysis_result.L])
        NBlock = len(TL)
        
        # Plot eigenvalue evolution if available
        if hasattr(analysis_result.L[0], 'D'):
            M = len(analysis_result.L[0].D)
            r = analysis_result.L[0].mrep.m0.shape.shape[0] if hasattr(analysis_result.L[0].mrep.m0, 'shape') else 10
            
            DD = np.zeros((NBlock, M))
            for jj, block in enumerate(analysis_result.L):
                DD[jj, :] = block.D[:M]
            
            # Create eigenvalue evolution plot
            fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
            
            for mode_idx in range(min(r, M)):
                ax.semilogy(TL, DD[:, mode_idx], 'o-', 
                          markersize=options.marker_size,
                          linewidth=options.line_width,
                          label=f'Mode {mode_idx+1}')
            
            ax.set_xlabel('Time', fontsize=options.label_fontsize)
            ax.set_ylabel('Eigenvalue Magnitude', fontsize=options.label_fontsize)
            ax.set_title('PCA Eigenvalue Evolution', fontsize=options.title_fontsize)
            ax.grid(options.grid)
            ax.legend(fontsize=options.legend_fontsize)
            
            figures.append(fig)
        
        # Plot frequency evolution if available
        if hasattr(analysis_result.L[0].mrep, 'm0') and hasattr(analysis_result.L[0].mrep.m0, 'lambda'):
            n_modes_max = max(len(block.mrep.imode) for block in analysis_result.L if hasattr(block.mrep, 'imode'))
            
            if n_modes_max > 0:
                freq_data = np.full((NBlock, n_modes_max), np.nan)
                
                for jj, block in enumerate(analysis_result.L):
                    if hasattr(block.mrep, 'imode') and hasattr(block.mrep.m0, 'lambda'):
                        # 'lambda' is a keyword: attribute access needs getattr
                        eigenvalues = getattr(block.mrep.m0, 'lambda')
                        for idx, mode_idx in enumerate(block.mrep.imode):
                            if idx < n_modes_max and mode_idx < len(eigenvalues):
                                eigenval = eigenvalues[mode_idx]
                                # Convert complex eigenvalue to frequency
                                if np.iscomplexobj(eigenval):
                                    freq_data[jj, idx] = np.abs(np.angle(eigenval))
                                else:
                                    freq_data[jj, idx] = np.abs(eigenval)
                
                # Create frequency-time plot
                fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
                
                for mode_idx in range(n_modes_max):
                    valid_mask = ~np.isnan(freq_data[:, mode_idx])
                    if np.any(valid_mask):
                        ax.plot(TL[valid_mask], freq_data[valid_mask, mode_idx], 'o-',
                               markersize=options.marker_size,
                               linewidth=options.line_width,
                               label=f'Mode {mode_idx+1}')
                
                ax.set_xlabel('Time', fontsize=options.label_fontsize)
                ax.set_ylabel('Frequency (rad/sample)', fontsize=options.label_fontsize)
                ax.set_title('Modal Frequency Evolution', fontsize=options.title_fontsize)
                ax.grid(options.grid)
                ax.legend(fontsize=options.legend_fontsize)
                
                figures.append(fig)
    
    # Single time point visualization
    if time_point is not None and hasattr(analysis_result, 'L'):
        if time_point < TL.min() or time_point > TL.max():
            print(f"Warning: time point {time_point} out of range [{TL.min():.3f}, {TL.max():.3f}]")
            return figures
        
        # Find closest time block
        ll = np.argmin(np.abs(TL - time_point))
        Lll = analysis_result.L[ll]
        
        # Plot mode shapes if sensor coordinates provided
        if sensor_coordinates is not None and hasattr(Lll.mrep, 'imode'):
            for mm, mode_idx in enumerate(Lll.mrep.imode):
                if hasattr(Lll.mrep.m0, 'shape') and mode_idx < len(Lll.mrep.m0.shape):
                    fig, axes = plt.subplots(1, 2, figsize=(2*options.figsize[0]/3, options.figsize[1]), 
                                           dpi=options.dpi)
                    
                    mode_shape = Lll.mrep.m0.shape[mode_idx]
                    freq = getattr(Lll.mrep.m0, 'lambda')[mode_idx] if hasattr(Lll.mrep.m0, 'lambda') else 0
                    
                    # Normalize mode shape
                    mode_shape = mode_shape / np.sqrt(np.vdot(mode_shape, mode_shape))
                    
                    # Real part
                    axes[0].scatter(sensor_coordinates[:, 0], sensor_coordinates[:, 1], 
                                  c=np.real(mode_shape), s=options.marker_size*20,
                                  cmap=options.colormap, alpha=options.alpha)
                    axes[0].set_title(f'Mode {mm+1} - Real Part', fontsize=options.title_fontsize)
                    axes[0].set_xlabel('X Coordinate', fontsize=options.label_fontsize)
                    axes[0].set_ylabel('Y Coordinate', fontsize=options.label_fontsize)
                    axes[0].grid(options.grid)
                    
                    # Imaginary part
                    axes[1].scatter(sensor_coordinates[:, 0], sensor_coordinates[:, 1],
                                  c=np.imag(mode_shape), s=options.marker_size*20, 
                                  cmap=options.colormap, alpha=options.alpha)
                    axes[1].set_title(f'Mode {mm+1} - Imaginary Part', fontsize=options.title_fontsize)
                    axes[1].set_xlabel('X Coordinate', fontsize=options.label_fontsize)
                    axes[1].set_ylabel('Y Coordinate', fontsize=options.label_fontsize)
                    axes[1].grid(options.grid)
                    
                    fig.suptitle(f'Mode Shape at t={time_point:.3f}s, f={np.abs(freq):.3f}', 
                               fontsize=options.title_fontsize)
                    plt.tight_layout()
                    
                    figures.append(fig)
    
    # Summary statistics plot
    if hasattr(analysis_result, 'L') and analysis_result.L:
        fig, axes = plt.subplots(2, 2, figsize=options.figsize, dpi=options.dpi)
        axes = axes.flatten()
        
        # Number of modes per time block
        n_modes_per_block = [len(block.mrep.imode) if hasattr(block.mrep, 'imode') else 0 
                           for block in analysis_result.L]
        
        axes[0].plot(TL, n_modes_per_block, 'o-', 
                    markersize=options.marker_size, linewidth=options.line_width)
        axes[0].set_xlabel('Time', fontsize=options.label_fontsize)
        axes[0].set_ylabel('Number of Modes', fontsize=options.label_fontsize)
        axes[0].set_title('Mode Count Evolution', fontsize=options.title_fontsize)
        axes[0].grid(options.grid)
        
        # Processing time per block if available
        if hasattr(analysis_result.L[0], 'block_processing_time'):
            proc_times = [block.block_processing_time for block in analysis_result.L]
            axes[1].plot(TL, proc_times, 'o-',
                        markersize=options.marker_size, linewidth=options.line_width)
            axes[1].set_xlabel('Time', fontsize=options.label_fontsize)
            axes[1].set_ylabel('Processing Time (s)', fontsize=options.label_fontsize)
            axes[1].set_title('Block Processing Time', fontsize=options.title_fontsize)
            axes[1].grid(options.grid)
        else:
            axes[1].text(0.5, 0.5, 'No timing data\navailable', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Block Processing Time', fontsize=options.title_fontsize)
        
        # Mode frequency distribution
        all_freqs = []
        for block in analysis_result.L:
            if hasattr(block.mrep, 'm0') and hasattr(block.mrep.m0, 'lambda'):
                # 'lambda' is a keyword: attribute access needs getattr
                eigenvalues = getattr(block.mrep.m0, 'lambda')
                for mode_idx in block.mrep.imode:
                    if mode_idx < len(eigenvalues):
                        eigenval = eigenvalues[mode_idx]
                        freq = np.abs(np.angle(eigenval)) if np.iscomplexobj(eigenval) else np.abs(eigenval)
                        all_freqs.append(freq)
        
        if all_freqs:
            axes[2].hist(all_freqs, bins=20, alpha=options.alpha, edgecolor='black')
            axes[2].set_xlabel('Frequency (rad/sample)', fontsize=options.label_fontsize)
            axes[2].set_ylabel('Count', fontsize=options.label_fontsize)
            axes[2].set_title('Mode Frequency Distribution', fontsize=options.title_fontsize)
            axes[2].grid(options.grid)
        else:
            axes[2].text(0.5, 0.5, 'No frequency data\navailable', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Mode Frequency Distribution', fontsize=options.title_fontsize)
        
        # Hide unused subplot
        axes[3].axis('off')
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures


def view_pcaspec_results(
    analysis_result,
    sensor_coordinates: npt.NDArray[np.floating],
    threshold: float = 0.1,
    options: Optional[SpectralPlotOptions] = None
) -> List[Figure]:
    """
    Results visualization with minimal classification.
    
    Python equivalent of MATLAB view_pcaspec_results.m for displaying
    analysis results with basic mode classification.
    
    Args:
        analysis_result: Analysis result structure
        sensor_coordinates: Sensor coordinate matrix (3, 2) for toroidal geometry
        threshold: Classification threshold
        options: Plot configuration options
        
    Returns:
        List of Figure objects created
    """
    if sensor_coordinates.shape != (3, 2):
        raise ValueError("sensor_coordinates must be (3, 2) for toroidal geometry")
    
    if options is None:
        options = SpectralPlotOptions()
    
    figures = []
    
    # Extract results similar to MATLAB version
    if hasattr(analysis_result, 'L') and analysis_result.L:
        # Create frequency evolution plot
        fig = view_pcaspec(analysis_result, sensor_coordinates, options=options)
        figures.extend(fig)
    
    return figures 