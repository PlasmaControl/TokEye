"""
Modal analysis visualization functions for eigspec package.

This module provides specialized plotting functions for modal analysis results:
- Mode shape visualization with spatial coordinates
- MAC (Modal Assurance Criterion) matrix plots
- Frequency-damping stability diagrams
- Modal parameter evolution and validation plots

Based on the MATLAB eigspec toolbox modal visualization functions:
- view_pcaspec_cluster_pattern.m - Modal pattern visualization
- mnfit_ptref.m - Point-reference mode shape fitting and display
- mnfit_clus_medoid.m - Cluster-based modal analysis plots
- Various modal plotting utilities in view_pcaspec_results.m
- Gaussian Process Regression plots in gp2dp.m
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass
from scipy.interpolate import griddata


@dataclass
class ModalPlotOptions:
    """Configuration options for modal analysis plots.
    
    Attributes:
        figsize: Figure size (width, height) in inches
        dpi: Figure DPI for resolution
        fontsize: Base font size for text elements
        title_fontsize: Font size for plot titles
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend text
        grid: Whether to show grid lines
        colormap: Colormap name for mode shape plots
        contour_levels: Number of contour levels
        marker_size: Size of scatter plot markers
        line_width: Width of plot lines
        alpha: Transparency level (0-1)
        save_format: Default format for saving figures
    """
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 100
    fontsize: int = 12
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid: bool = True
    colormap: str = 'RdBu'
    contour_levels: int = 20
    marker_size: float = 6.0
    line_width: float = 1.5
    alpha: float = 0.8
    save_format: str = 'png'


def plot_array_geometry(
    coordinates: npt.NDArray[np.floating],
    sensor_labels: Optional[List[str]] = None,
    highlight_subsets: Optional[Dict[str, List[int]]] = None,
    options: Optional[ModalPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot sensor array geometry.
    
    Args:
        coordinates: Sensor coordinates (n_sensors x 2) - [theta, phi] or [x, y]
        sensor_labels: Optional labels for each sensor
        highlight_subsets: Dictionary of subset names to sensor indices
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ModalPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    # Plot all sensors
    ax.scatter(coordinates[:, 0], coordinates[:, 1], 
              c='black', marker='x', s=options.marker_size**2, 
              alpha=options.alpha, label='All sensors')
    
    # Highlight subsets if provided
    if highlight_subsets is not None:
        colors = colormaps.get_cmap('tab10')(np.linspace(0, 1, len(highlight_subsets)))
        for i, (subset_name, indices) in enumerate(highlight_subsets.items()):
            ax.scatter(coordinates[indices, 0], coordinates[indices, 1],
                      c=[colors[i]], s=options.marker_size**2,
                      alpha=options.alpha, label=subset_name)
    
    # Add sensor labels if provided
    if sensor_labels is not None:
        for i, label in enumerate(sensor_labels):
            ax.annotate(label, (coordinates[i, 0], coordinates[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=options.fontsize-2)
    
    ax.set_xlabel('θ [rad]' if np.max(coordinates[:, 0]) <= 2*np.pi else 'X', 
                 fontsize=options.label_fontsize)
    ax.set_ylabel('φ [rad]' if np.max(coordinates[:, 1]) <= 2*np.pi else 'Y', 
                 fontsize=options.label_fontsize)
    ax.set_title('Sensor Array Geometry', fontsize=options.title_fontsize)
    
    if highlight_subsets is not None:
        ax.legend(fontsize=options.legend_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_shape_vectors(
    coordinates: npt.NDArray[np.floating],
    shape_vector: npt.NDArray[np.complex128],
    mode_info: Optional[Dict[str, Any]] = None,
    show_phase: bool = True,
    normalize: bool = True,
    options: Optional[ModalPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot mode shape vectors as scatter plot.
    
    Args:
        coordinates: Sensor coordinates (n_sensors x 2)
        shape_vector: Complex mode shape vector
        mode_info: Optional dictionary with mode information (frequency, etc.)
        show_phase: Whether to show phase information
        normalize: Whether to normalize the shape vector
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ModalPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    # Normalize if requested
    if normalize:
        shape_vector = shape_vector / np.sqrt(np.sum(np.abs(shape_vector)**2))
    
    # Extract real and imaginary parts
    real_part = np.real(shape_vector)
    imag_part = np.imag(shape_vector)
    
    # Plot real and imaginary parts
    ax.scatter(coordinates[:, 0], real_part, 
              c='blue', marker='o', s=options.marker_size**2,
              alpha=options.alpha, label='Real part')
    ax.scatter(coordinates[:, 1], imag_part, 
              c='red', marker='s', s=options.marker_size**2,
              alpha=options.alpha, label='Imaginary part')
    
    # Add phase information if requested
    if show_phase:
        phases = np.angle(shape_vector)
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                           c=phases, cmap='hsv', s=options.marker_size**2,
                           alpha=options.alpha, marker='^')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Phase [rad]', fontsize=options.label_fontsize)
    
    # Create title with mode information
    title = 'Mode Shape Vector'
    if mode_info is not None:
        if 'frequency' in mode_info:
            title += f" (f = {mode_info['frequency']:.2f})"
        if 'mode_number' in mode_info:
            title += f" - Mode {mode_info['mode_number']}"
    
    ax.set_xlabel('Sensor Coordinate', fontsize=options.label_fontsize)
    ax.set_ylabel('Amplitude', fontsize=options.label_fontsize)
    ax.set_title(title, fontsize=options.title_fontsize)
    ax.legend(fontsize=options.legend_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_mode_shape_2d(
    coordinates: npt.NDArray[np.floating],
    shape_vector: npt.NDArray[np.complex128],
    grid_size: Tuple[int, int] = (100, 75),
    component: str = 'magnitude',
    mode_info: Optional[Dict[str, Any]] = None,
    options: Optional[ModalPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot 2D mode shape using interpolation and contours.
    
    Args:
        coordinates: Sensor coordinates (n_sensors x 2) - [theta, phi]
        shape_vector: Complex mode shape vector
        grid_size: Size of interpolation grid (ntheta, nphi)
        component: Component to plot ('magnitude', 'real', 'imag', 'phase')
        mode_info: Optional dictionary with mode information
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ModalPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    # Create interpolation grid
    theta_min, theta_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    phi_min, phi_max = coordinates[:, 1].min(), coordinates[:, 1].max()
    
    theta_grid = np.linspace(theta_min, theta_max, grid_size[0])
    phi_grid = np.linspace(phi_min, phi_max, grid_size[1])
    THETA, PHI = np.meshgrid(theta_grid, phi_grid)
    
    # Select component to plot
    if component == 'magnitude':
        values = np.abs(shape_vector)
        cmap = options.colormap
        label = 'Magnitude'
    elif component == 'real':
        values = np.real(shape_vector)
        cmap = 'RdBu'
        label = 'Real Part'
    elif component == 'imag':
        values = np.imag(shape_vector)
        cmap = 'RdBu'
        label = 'Imaginary Part'
    elif component == 'phase':
        values = np.angle(shape_vector)
        cmap = 'hsv'
        label = 'Phase [rad]'
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # Interpolate to grid
    grid_values = griddata(coordinates, values, (THETA, PHI), method='cubic')
    
    # Create contour plot
    contour = ax.contourf(THETA, PHI, grid_values, levels=options.contour_levels, 
                         cmap=cmap, alpha=options.alpha)
    
    # Add contour lines
    ax.contour(THETA, PHI, grid_values, levels=options.contour_levels, 
              colors='black', alpha=0.3, linewidths=0.5)
    
    # Add sensor positions
    ax.scatter(coordinates[:, 0], coordinates[:, 1], 
              c='black', marker='x', s=options.marker_size**2, alpha=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(label, fontsize=options.label_fontsize)
    
    # Create title
    title = f'2D Mode Shape - {label}'
    if mode_info is not None:
        if 'frequency' in mode_info:
            title += f" (f = {mode_info['frequency']:.2f})"
        if 'mode_number' in mode_info:
            title += f" - Mode {mode_info['mode_number']}"
    
    ax.set_xlabel('θ [rad]', fontsize=options.label_fontsize)
    ax.set_ylabel('φ [rad]', fontsize=options.label_fontsize)
    ax.set_title(title, fontsize=options.title_fontsize)
    
    return fig, ax


def plot_mode_shape_polar(
    coordinates: npt.NDArray[np.floating],
    shape_vector: npt.NDArray[np.complex128],
    subset_indices: Optional[List[int]] = None,
    smooth: bool = True,
    mode_info: Optional[Dict[str, Any]] = None,
    options: Optional[ModalPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot mode shape on a polar/toroidal array.
    
    Args:
        coordinates: Sensor coordinates (n_sensors x 2) - [theta, phi]
        shape_vector: Complex mode shape vector
        subset_indices: Indices of sensors to use for polar plot
        smooth: Whether to apply smoothing interpolation
        mode_info: Optional dictionary with mode information
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ModalPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    # Select subset if provided
    if subset_indices is not None:
        plot_coords = coordinates[subset_indices]
        plot_shape = shape_vector[subset_indices]
    else:
        plot_coords = coordinates
        plot_shape = shape_vector
    
    # Extract angles (assuming second coordinate is the angular one)
    angles = plot_coords[:, 1]
    
    # Sort by angle for proper plotting
    sort_indices = np.argsort(angles)
    angles = angles[sort_indices]
    plot_shape = plot_shape[sort_indices]
    
    # Create smooth interpolation if requested
    if smooth:
        from scipy.interpolate import interp1d
        phi_smooth = np.linspace(0, 2*np.pi, 128)
        
        # Handle periodicity
        extended_angles = np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi])
        extended_real = np.concatenate([np.real(plot_shape), np.real(plot_shape), 
                                      np.real(plot_shape)])
        extended_imag = np.concatenate([np.imag(plot_shape), np.imag(plot_shape), 
                                      np.imag(plot_shape)])
        
        interp_real = interp1d(extended_angles, extended_real, kind='cubic')
        interp_imag = interp1d(extended_angles, extended_imag, kind='cubic')
        
        smooth_real = interp_real(phi_smooth)
        smooth_imag = interp_imag(phi_smooth)
        
        ax.plot(phi_smooth, smooth_real, 'b-', linewidth=options.line_width, 
               alpha=options.alpha, label='Real part')
        ax.plot(phi_smooth, smooth_imag, 'r-', linewidth=options.line_width, 
               alpha=options.alpha, label='Imaginary part')
    
    # Plot original data points
    ax.scatter(angles, np.real(plot_shape), c='blue', marker='o', 
              s=options.marker_size**2, alpha=options.alpha, 
              label='Real (data)' if smooth else 'Real part')
    ax.scatter(angles, np.imag(plot_shape), c='red', marker='s', 
              s=options.marker_size**2, alpha=options.alpha,
              label='Imag (data)' if smooth else 'Imaginary part')
    
    # Create title
    title = 'Polar Mode Shape'
    if mode_info is not None:
        if 'frequency' in mode_info:
            title += f" (f = {mode_info['frequency']:.2f})"
        if 'mode_number' in mode_info:
            title += f" - Mode {mode_info['mode_number']}"
    
    ax.set_xlabel('φ [rad]', fontsize=options.label_fontsize)
    ax.set_ylabel('Amplitude', fontsize=options.label_fontsize)
    ax.set_title(title, fontsize=options.title_fontsize)
    ax.legend(fontsize=options.legend_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_mode_shapes(
    coordinates: npt.NDArray[np.floating],
    shape_vectors: npt.NDArray[np.complex128],
    mode_info: Optional[List[Dict[str, Any]]] = None,
    plot_types: List[str] = ['2d', 'polar'],
    options: Optional[ModalPlotOptions] = None
) -> Figure:
    """Create a comprehensive plot of multiple mode shapes.
    
    Args:
        coordinates: Sensor coordinates (n_sensors x 2)
        shape_vectors: Complex mode shape vectors (n_sensors x n_modes)
        mode_info: List of dictionaries with mode information
        plot_types: Types of plots to create ('2d', 'polar', 'vectors')
        options: Plot configuration options
        
    Returns:
        Figure object with subplots
    """
    if options is None:
        options = ModalPlotOptions()
    
    n_modes = shape_vectors.shape[1]
    n_plots = len(plot_types)
    
    fig, axes = plt.subplots(n_modes, n_plots, figsize=(options.figsize[0]*n_plots, 
                                                       options.figsize[1]*n_modes),
                           dpi=options.dpi)
    
    # Handle case of single mode or single plot type
    if n_modes == 1:
        axes = axes.reshape(1, -1)
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_modes):
        shape_vector = shape_vectors[:, i]
        info = mode_info[i] if mode_info is not None else None
        
        for j, plot_type in enumerate(plot_types):
            if plot_type == '2d':
                plot_mode_shape_2d(coordinates, shape_vector, mode_info=info,
                                  options=options, ax=axes[i, j])
            elif plot_type == 'polar':
                plot_mode_shape_polar(coordinates, shape_vector, mode_info=info,
                                     options=options, ax=axes[i, j])
            elif plot_type == 'vectors':
                plot_shape_vectors(coordinates, shape_vector, mode_info=info,
                                  options=options, ax=axes[i, j])
    
    plt.tight_layout()
    return fig


def save_modal_plot(
    fig: Figure, 
    filename: str, 
    options: Optional[ModalPlotOptions] = None
) -> None:
    """Save a modal plot to file.
    
    Args:
        fig: Figure to save
        filename: Output filename
        options: Plot configuration options
    """
    if options is None:
        options = ModalPlotOptions()
    
    if not filename.endswith(f'.{options.save_format}'):
        filename += f'.{options.save_format}'
    
    fig.savefig(filename, format=options.save_format, dpi=options.dpi, 
               bbox_inches='tight')
    print(f"Saved modal plot to: {filename}") 