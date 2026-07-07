"""
Clustering visualization functions for eigspec package.

This module provides specialized plotting functions for clustering analysis results:
- Cluster scatter plots with frequency-time coloring
- MAC-based similarity and distance visualization
- Medoid and prototype shape plotting
- Cluster validation and quality metrics visualization

Based on the MATLAB eigspec toolbox clustering visualization functions:
- view_pcaspec_cluster.m - Main cluster visualization and analysis
- view_pcaspec_unsupervised.m - Unsupervised clustering plots
- view_pcaspec_cluster_nearness.m - Cluster similarity visualization
- view_pcaspec_cluster_pattern.m - Pattern estimation and plotting
- redraw_clustering() - Cluster result plotting utilities
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


@dataclass
class ClusterPlotOptions:
    """Configuration options for clustering analysis plots.
    
    Attributes:
        figsize: Figure size (width, height) in inches
        dpi: Figure DPI for resolution
        fontsize: Base font size for text elements
        title_fontsize: Font size for plot titles
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend text
        grid: Whether to show grid lines
        colormap: Colormap name for cluster colors
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
    colormap: str = 'tab10'
    marker_size: float = 6.0
    line_width: float = 1.5
    alpha: float = 0.8
    save_format: str = 'png'


def plot_clustering_results(
    time: npt.NDArray[np.floating],
    frequencies: npt.NDArray[np.floating],
    cluster_labels: npt.NDArray[np.integer],
    medoid_indices: Optional[npt.NDArray[np.integer]] = None,
    amplitudes: Optional[npt.NDArray[np.floating]] = None,
    options: Optional[ClusterPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot clustering results in time-frequency space.
    
    Args:
        time: Time vector
        frequencies: Frequency values
        cluster_labels: Cluster assignment for each point
        medoid_indices: Indices of cluster medoids
        amplitudes: Optional amplitude values for marker sizing
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ClusterPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Create colors for clusters
    if n_clusters <= 10:
        colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, n_clusters))
    else:
        colors = colormaps.get_cmap('hsv')(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        
        # Determine marker size
        if amplitudes is not None:
            sizes = options.marker_size**2 * (1 + amplitudes[mask])
        else:
            sizes = options.marker_size**2
        
        ax.scatter(time[mask], frequencies[mask], 
                  c=[colors[i]], s=sizes, alpha=options.alpha,
                  label=f'Cluster {cluster_id}')
    
    # Highlight medoids if provided
    if medoid_indices is not None:
        ax.scatter(time[medoid_indices], frequencies[medoid_indices],
                  c='black', marker='x', s=(options.marker_size*2)**2,
                  alpha=1.0, linewidths=3, label='Medoids')
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel('Frequency', fontsize=options.label_fontsize)
    ax.set_title('Clustering Results', fontsize=options.title_fontsize)
    
    # Add legend with reasonable number of entries
    if n_clusters <= 20:
        ax.legend(fontsize=options.legend_fontsize, bbox_to_anchor=(1.05, 1), 
                 loc='upper left')
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_cluster_similarity_matrix(
    similarity_matrix: npt.NDArray[np.floating],
    cluster_labels: Optional[npt.NDArray[np.integer]] = None,
    method: str = 'similarity',
    options: Optional[ClusterPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot similarity or distance matrix as heatmap.
    
    Args:
        similarity_matrix: Similarity or distance matrix
        cluster_labels: Optional cluster labels for ordering
        method: Type of matrix ('similarity' or 'distance')
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ClusterPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    # Order by clusters if labels provided
    if cluster_labels is not None:
        order = np.argsort(cluster_labels)
        ordered_matrix = similarity_matrix[order][:, order]
        ordered_labels = cluster_labels[order]
    else:
        ordered_matrix = similarity_matrix
        ordered_labels = None
    
    # Choose colormap based on method
    if method == 'similarity':
        cmap = 'Blues'
        label = 'Similarity'
    else:
        cmap = 'Reds'
        label = 'Distance'
    
    # Create heatmap
    im = ax.imshow(ordered_matrix, cmap=cmap, alpha=options.alpha,
                   aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, fontsize=options.label_fontsize)
    
    # Add cluster boundaries if labels provided
    if ordered_labels is not None:
        # Find cluster boundaries
        boundaries = []
        current_cluster = ordered_labels[0]
        for i, label in enumerate(ordered_labels[1:], 1):
            if label != current_cluster:
                boundaries.append(i - 0.5)
                current_cluster = label
        
        # Draw boundary lines
        for boundary in boundaries:
            ax.axhline(boundary, color='red', linewidth=2, alpha=0.7)
            ax.axvline(boundary, color='red', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Data Point Index', fontsize=options.label_fontsize)
    ax.set_ylabel('Data Point Index', fontsize=options.label_fontsize)
    ax.set_title(f'{label} Matrix', fontsize=options.title_fontsize)
    
    return fig, ax


def plot_cluster_medoids(
    shape_vectors: npt.NDArray[np.complex128],
    cluster_labels: npt.NDArray[np.integer],
    medoid_indices: npt.NDArray[np.integer],
    coordinates: Optional[npt.NDArray[np.floating]] = None,
    options: Optional[ClusterPlotOptions] = None
) -> Figure:
    """Plot cluster medoids and their shape vectors.
    
    Args:
        shape_vectors: Complex shape vectors (n_channels x n_vectors)
        cluster_labels: Cluster assignment for each vector
        medoid_indices: Indices of cluster medoids
        coordinates: Optional sensor coordinates for spatial plotting
        options: Plot configuration options
        
    Returns:
        Figure object with subplots
    """
    if options is None:
        options = ClusterPlotOptions()
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    n_channels = shape_vectors.shape[0]
    
    # Determine subplot layout
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(options.figsize[0]*n_cols/3, 
                                                     options.figsize[1]*n_rows/2),
                           dpi=options.dpi)
    
    # Handle single subplot case
    if n_clusters == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    colors = colormaps.get_cmap(options.colormap)(np.linspace(0, 1, n_clusters))
    
    for i, cluster_id in enumerate(unique_clusters):
        if i >= len(axes):
            break
            
        medoid_idx = medoid_indices[i]
        medoid_vector = shape_vectors[:, medoid_idx]
        
        # Normalize medoid
        medoid_vector = medoid_vector / np.sqrt(np.sum(np.abs(medoid_vector)**2))
        
        if coordinates is not None:
            # Spatial plot
            axes[i].scatter(coordinates[:, 0], np.real(medoid_vector),
                          c='blue', marker='o', s=options.marker_size**2,
                          alpha=options.alpha, label='Real')
            axes[i].scatter(coordinates[:, 1], np.imag(medoid_vector),
                          c='red', marker='s', s=options.marker_size**2,
                          alpha=options.alpha, label='Imaginary')
            axes[i].set_xlabel('Coordinate')
        else:
            # Channel index plot
            channels = np.arange(n_channels)
            axes[i].plot(channels, np.real(medoid_vector), 'b-o',
                        linewidth=options.line_width, markersize=options.marker_size,
                        alpha=options.alpha, label='Real')
            axes[i].plot(channels, np.imag(medoid_vector), 'r-s',
                        linewidth=options.line_width, markersize=options.marker_size,
                        alpha=options.alpha, label='Imaginary')
            axes[i].set_xlabel('Channel Index')
        
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'Cluster {cluster_id} Medoid', fontsize=options.title_fontsize)
        axes[i].legend(fontsize=options.legend_fontsize-2)
        
        if options.grid:
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_mac_similarity(
    mac_matrix: npt.NDArray[np.floating],
    threshold: Optional[float] = None,
    cluster_labels: Optional[npt.NDArray[np.integer]] = None,
    options: Optional[ClusterPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot MAC (Modal Assurance Criterion) similarity matrix.
    
    Args:
        mac_matrix: MAC similarity matrix
        threshold: Optional threshold for highlighting
        cluster_labels: Optional cluster labels for ordering
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ClusterPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    # Order by clusters if labels provided
    if cluster_labels is not None:
        order = np.argsort(cluster_labels)
        ordered_matrix = mac_matrix[order][:, order]
    else:
        ordered_matrix = mac_matrix
    
    # Create heatmap
    im = ax.imshow(ordered_matrix, cmap='viridis', alpha=options.alpha,
                   aspect='auto', origin='lower', vmin=0, vmax=1)
    
    # Add threshold contour if provided
    if threshold is not None:
        contour = ax.contour(ordered_matrix, levels=[threshold], colors='red',
                           linewidths=2, alpha=0.8)
        ax.clabel(contour, inline=True, fontsize=options.fontsize-2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAC Value', fontsize=options.label_fontsize)
    
    ax.set_xlabel('Mode Index', fontsize=options.label_fontsize)
    ax.set_ylabel('Mode Index', fontsize=options.label_fontsize)
    title = 'MAC Similarity Matrix'
    if threshold is not None:
        title += f' (threshold = {threshold:.2f})'
    ax.set_title(title, fontsize=options.title_fontsize)
    
    return fig, ax


def plot_soft_clustering(
    time: npt.NDArray[np.floating],
    frequencies: npt.NDArray[np.floating],
    membership_matrix: npt.NDArray[np.floating],
    method: str = 'interpolated',
    options: Optional[ClusterPlotOptions] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot soft clustering results with membership probabilities.
    
    Args:
        time: Time vector
        frequencies: Frequency values
        membership_matrix: Membership probabilities (n_points x n_clusters)
        method: Soft clustering visualization method
        options: Plot configuration options
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if options is None:
        options = ClusterPlotOptions()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    else:
        fig = ax.figure
    
    n_clusters = membership_matrix.shape[1]
    
    if method == 'interpolated':
        # Use membership probabilities as colors
        for i in range(n_clusters):
            membership = membership_matrix[:, i]
            scatter = ax.scatter(time, frequencies, c=membership, 
                               cmap='viridis', s=options.marker_size**2,
                               alpha=options.alpha, vmin=0, vmax=1)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Membership Probability', fontsize=options.label_fontsize)
        
    elif method == 'nearest_neighbor':
        # Use k-nearest neighbor style visualization
        from scipy.spatial.distance import cdist
        
        # Create grid for interpolation
        time_grid = np.linspace(time.min(), time.max(), 50)
        freq_grid = np.linspace(frequencies.min(), frequencies.max(), 50)
        TIME, FREQ = np.meshgrid(time_grid, freq_grid)
        
        # Find nearest neighbors and interpolate membership
        points = np.column_stack([time, frequencies])
        grid_points = np.column_stack([TIME.ravel(), FREQ.ravel()])
        
        distances = cdist(grid_points, points)
        nearest_indices = np.argmin(distances, axis=1)
        
        for i in range(n_clusters):
            grid_membership = membership_matrix[nearest_indices, i].reshape(TIME.shape)
            contour = ax.contourf(TIME, FREQ, grid_membership, levels=20,
                                cmap='viridis', alpha=options.alpha/2)
        
        # Overlay original points
        max_membership_cluster = np.argmax(membership_matrix, axis=1)
        colors = colormaps.get_cmap(options.colormap)(
            np.linspace(0, 1, n_clusters))[max_membership_cluster]
        
        ax.scatter(time, frequencies, c=colors, s=options.marker_size**2,
                  alpha=options.alpha, edgecolors='black', linewidths=0.5)
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Membership Probability', fontsize=options.label_fontsize)
    
    ax.set_xlabel('Time', fontsize=options.label_fontsize)
    ax.set_ylabel('Frequency', fontsize=options.label_fontsize)
    ax.set_title(f'Soft Clustering ({method})', fontsize=options.title_fontsize)
    
    if options.grid:
        ax.grid(True, alpha=0.3)
    
    return fig, ax


def save_clustering_plot(
    fig: Figure, 
    filename: str, 
    options: Optional[ClusterPlotOptions] = None
) -> None:
    """Save a clustering plot to file.
    
    Args:
        fig: Figure to save
        filename: Output filename
        options: Plot configuration options
    """
    if options is None:
        options = ClusterPlotOptions()
    
    if not filename.endswith(f'.{options.save_format}'):
        filename += f'.{options.save_format}'
    
    fig.savefig(filename, format=options.save_format, dpi=options.dpi, 
               bbox_inches='tight')
    print(f"Saved clustering plot to: {filename}") 