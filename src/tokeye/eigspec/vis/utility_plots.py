"""
Utility plotting functions for eigspec package.

This module provides general-purpose plotting utilities and diagnostic plots:
- Analysis diagnostics and validation plots
- Parameter convergence and optimization plots
- Error analysis and statistical visualization
- General-purpose scientific plotting utilities

Based on the MATLAB eigspec toolbox utility plotting functions:
- Diagnostic plots throughout the MATLAB codebase
- GCV and cross-validation plots in gcv1dp.m and df1dp.m
- Optimization and convergence plots
- Statistical analysis and error plotting utilities
- General plotting utilities used across the toolbox
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_quantiles(
    data: npt.NDArray[np.floating],
    quantiles: Optional[npt.NDArray[np.floating]] = None,
    column_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Create a quantile plot showing statistical distribution for each column.
    
    This function is equivalent to the MATLAB qplot function, showing
    percentile ranges for each data column.
    
    Args:
        data: Data matrix (n_samples x n_features)
        quantiles: Quantile levels to plot (default: [0.01, 0.1, 0.33, 0.5, 0.67, 0.9, 0.99])
        column_labels: Labels for each column
        figsize: Figure size
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if quantiles is None:
        quantiles = np.array([0.01, 0.1, 1/3, 0.5, 2/3, 0.9, 0.99])
    
    if len(quantiles) != 7:
        raise ValueError("quantiles must have exactly 7 elements")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    n_samples, n_features = data.shape
    
    if column_labels is None:
        column_labels = [f'#{i+1}' for i in range(n_features)]
    elif len(column_labels) != n_features:
        raise ValueError("Number of labels must match number of columns")
    
    # Sort quantiles to ensure proper ordering
    quantiles = np.sort(quantiles)
    
    # Calculate quantile values for each column
    for i in range(n_features):
        col_data = np.sort(data[:, i])
        quantile_indices = np.round(quantiles * (n_samples - 1)).astype(int)
        quantile_values = col_data[quantile_indices]
        
        y_pos = i + 1
        
        # Plot different ranges with different line styles and colors
        # Extreme range (1-99%)
        ax.plot([quantile_values[0], quantile_values[6]], [y_pos, y_pos], 
               'g-', linewidth=2, alpha=0.7)
        
        # Outer range (10-90%)
        ax.plot([quantile_values[1], quantile_values[5]], [y_pos, y_pos], 
               'b-', linewidth=3, alpha=0.8)
        
        # Inner range (33-67%)
        ax.plot([quantile_values[2], quantile_values[4]], [y_pos, y_pos], 
               'r-', linewidth=4, alpha=0.9)
        
        # Median line
        ax.plot([quantile_values[3], quantile_values[3]], [y_pos-0.4, y_pos+0.4], 
               'k-', linewidth=2)
    
    # Set y-axis properties
    ax.set_ylim(0, n_features + 1)
    ax.set_yticks(range(1, n_features + 1))
    ax.set_yticklabels(column_labels)
    
    ax.set_xlabel('Value')
    ax.set_title('Quantile Plot')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', linewidth=2, alpha=0.7, label='1-99%'),
        Line2D([0], [0], color='b', linewidth=3, alpha=0.8, label='10-90%'),
        Line2D([0], [0], color='r', linewidth=4, alpha=0.9, label='33-67%'),
        Line2D([0], [0], color='k', linewidth=2, label='Median')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig, ax


def plot_statistical_summary(
    data: npt.NDArray[np.floating],
    labels: Optional[List[str]] = None,
    stats: List[str] = ['mean', 'std', 'min', 'max'],
    figsize: Tuple[float, float] = (12, 8)
) -> Figure:
    """Create a comprehensive statistical summary plot.
    
    Args:
        data: Data matrix (n_samples x n_features)
        labels: Labels for each feature
        stats: List of statistics to compute ('mean', 'std', 'min', 'max', 'median')
        figsize: Figure size
        
    Returns:
        Figure object with subplots
    """
    n_features = data.shape[1]
    n_stats = len(stats)
    
    if labels is None:
        labels = [f'Feature {i+1}' for i in range(n_features)]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    x_pos = np.arange(n_features)
    
    # Calculate statistics
    stat_values = {}
    for stat in stats:
        if stat == 'mean':
            stat_values[stat] = np.mean(data, axis=0)
        elif stat == 'std':
            stat_values[stat] = np.std(data, axis=0)
        elif stat == 'min':
            stat_values[stat] = np.min(data, axis=0)
        elif stat == 'max':
            stat_values[stat] = np.max(data, axis=0)
        elif stat == 'median':
            stat_values[stat] = np.median(data, axis=0)
    
    # Plot each statistic
    for i, stat in enumerate(stats[:4]):  # Limit to 4 subplots
        if i >= len(axes):
            break
            
        axes[i].bar(x_pos, stat_values[stat], alpha=0.7, 
                   color=plt.cm.tab10(i))
        axes[i].set_title(f'{stat.capitalize()}')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(stats), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def create_colormap(
    colors: List[str], 
    name: str = 'custom',
    n_segments: int = 256
) -> mcolors.LinearSegmentedColormap:
    """Create a custom colormap from a list of colors.
    
    Args:
        colors: List of color names or hex codes
        name: Name for the colormap
        n_segments: Number of segments in the colormap
        
    Returns:
        Custom colormap object
    """
    if len(colors) < 2:
        raise ValueError("At least 2 colors are required")
    
    # Convert colors to RGB if needed
    rgb_colors = []
    for color in colors:
        if isinstance(color, str):
            rgb_colors.append(mcolors.to_rgb(color))
        else:
            rgb_colors.append(color)
    
    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(name, rgb_colors, N=n_segments)
    return cmap


def setup_figure_style(
    style: str = 'default',
    font_scale: float = 1.0,
    grid: bool = True
) -> None:
    """Set up matplotlib figure style and parameters.
    
    Args:
        style: Style name ('default', 'publication', 'presentation')
        font_scale: Scale factor for font sizes
        grid: Whether to show grids by default
    """
    if style == 'publication':
        # Publication-ready style
        plt.rcParams.update({
            'font.size': 10 * font_scale,
            'axes.titlesize': 12 * font_scale,
            'axes.labelsize': 10 * font_scale,
            'xtick.labelsize': 9 * font_scale,
            'ytick.labelsize': 9 * font_scale,
            'legend.fontsize': 9 * font_scale,
            'figure.titlesize': 14 * font_scale,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'axes.grid': grid,
            'grid.alpha': 0.3,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
        })
    
    elif style == 'presentation':
        # Presentation style with larger fonts
        plt.rcParams.update({
            'font.size': 14 * font_scale,
            'axes.titlesize': 18 * font_scale,
            'axes.labelsize': 16 * font_scale,
            'xtick.labelsize': 12 * font_scale,
            'ytick.labelsize': 12 * font_scale,
            'legend.fontsize': 14 * font_scale,
            'figure.titlesize': 20 * font_scale,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'axes.grid': grid,
            'grid.alpha': 0.3,
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
        })
    
    else:  # default
        # Default matplotlib style with minor adjustments
        plt.rcParams.update({
            'font.size': 12 * font_scale,
            'axes.titlesize': 14 * font_scale,
            'axes.labelsize': 12 * font_scale,
            'xtick.labelsize': 11 * font_scale,
            'ytick.labelsize': 11 * font_scale,
            'legend.fontsize': 11 * font_scale,
            'figure.titlesize': 16 * font_scale,
            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'axes.grid': grid,
            'grid.alpha': 0.3,
            'figure.dpi': 100,
            'savefig.dpi': 200,
            'savefig.bbox': 'tight',
        })


def create_comparison_plot(
    data_dict: Dict[str, npt.NDArray[np.floating]],
    x_values: Optional[npt.NDArray[np.floating]] = None,
    plot_type: str = 'line',
    title: str = 'Comparison Plot',
    xlabel: str = 'X',
    ylabel: str = 'Y',
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Create a comparison plot for multiple datasets.
    
    Args:
        data_dict: Dictionary of {label: data_array} pairs
        x_values: X-axis values (uses indices if None)
        plot_type: Type of plot ('line', 'scatter', 'bar')
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        ax: Existing axes to plot on
        
    Returns:
        Figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    n_datasets = len(data_dict)
    colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    
    for i, (label, data) in enumerate(data_dict.items()):
        if x_values is None:
            x = np.arange(len(data))
        else:
            x = x_values[:len(data)]
        
        if plot_type == 'line':
            ax.plot(x, data, color=colors[i], linewidth=2, alpha=0.8, label=label)
        elif plot_type == 'scatter':
            ax.scatter(x, data, color=colors[i], s=40, alpha=0.8, label=label)
        elif plot_type == 'bar':
            width = 0.8 / n_datasets
            offset = (i - n_datasets/2 + 0.5) * width
            ax.bar(x + offset, data, width=width, color=colors[i], 
                  alpha=0.8, label=label)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def save_all_figures(
    figures: List[Figure],
    base_filename: str,
    formats: List[str] = ['png'],
    dpi: int = 200,
    close_after_save: bool = True
) -> None:
    """Save multiple figures with sequential numbering.
    
    Args:
        figures: List of figure objects to save
        base_filename: Base filename (will add numbers and extensions)
        formats: List of formats to save ('png', 'pdf', 'svg', etc.)
        dpi: DPI for raster formats
        close_after_save: Whether to close figures after saving
    """
    for i, fig in enumerate(figures):
        for fmt in formats:
            filename = f"{base_filename}_{i+1:02d}.{fmt}"
            fig.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        if close_after_save:
            plt.close(fig) 