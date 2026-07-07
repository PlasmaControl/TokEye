"""
Visualization module for eigspec package.

This module provides plotting and visualization functions for spectral analysis results:
- Spectral plots for frequency-domain analysis
- Time series plotting with modal overlays
- Clustering visualization and pattern recognition plots
- Modal shape and pattern visualization
- Utility plots for analysis diagnostics

Based on the MATLAB eigspec toolbox visualization functions:
- view_pcaspec_cluster.m - Cluster visualization and analysis
- view_pcaspec_results.m - Main results plotting
- view_pcaspec_prototypes.m - Prototype-based visualization
- view_pcaspec_cluster_pattern.m - Pattern estimation and plotting
- view_pcaspec_cluster_nearness.m - Cluster similarity visualization
- Various plotting utilities throughout the MATLAB toolbox
"""

# Use conditional imports to handle missing modules gracefully
try:
    from .spectral_plots import (
        SpectralPlotOptions,
        plot_eigenvalue_evolution,
        plot_frequency_time,
        plot_phase_time,
        plot_rms_time,
        plot_spectral_summary,
        view_pcaspec,
        view_pcaspec_results,
    )
    _SPECTRAL_AVAILABLE = True
except ImportError:
    _SPECTRAL_AVAILABLE = False

try:
    from .modal_plots import (
        ModalPlotOptions,
        plot_mode_shapes,
        plot_mode_shape_2d,
        plot_mode_shape_polar,
        plot_shape_vectors,
        plot_array_geometry,
    )
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False

try:
    from .clustering_plots import (
        ClusterPlotOptions,
        plot_clustering_results,
        plot_cluster_similarity_matrix,
        plot_cluster_medoids,
        plot_mac_similarity,
        plot_soft_clustering,
    )
    _CLUSTERING_AVAILABLE = True
except ImportError:
    _CLUSTERING_AVAILABLE = False

try:
    from .time_series_plots import (
        TimeSeriesPlotOptions,
        plot_time_traces,
        plot_prototype_traces,
        plot_filtered_traces,
        plot_multi_channel_overlay,
    )
    _TIME_SERIES_AVAILABLE = True
except ImportError:
    _TIME_SERIES_AVAILABLE = False

try:
    from .utility_plots import (
        plot_quantiles,
        plot_statistical_summary,
        create_colormap,
        setup_figure_style,
    )
    _UTILITY_AVAILABLE = True
except ImportError:
    _UTILITY_AVAILABLE = False

# Build __all__ list dynamically based on available imports
__all__ = []

if _SPECTRAL_AVAILABLE:
    __all__.extend([
        'SpectralPlotOptions',
        'plot_eigenvalue_evolution',
        'plot_frequency_time',
        'plot_phase_time', 
        'plot_rms_time',
        'plot_spectral_summary',
        'view_pcaspec',
        'view_pcaspec_results',
    ])

if _MODAL_AVAILABLE:
    __all__.extend([
        'ModalPlotOptions',
        'plot_mode_shapes',
        'plot_mode_shape_2d',
        'plot_mode_shape_polar',
        'plot_shape_vectors',
        'plot_array_geometry',
    ])

if _CLUSTERING_AVAILABLE:
    __all__.extend([
        'ClusterPlotOptions',
        'plot_clustering_results',
        'plot_cluster_similarity_matrix',
        'plot_cluster_medoids',
        'plot_mac_similarity',
        'plot_soft_clustering',
    ])

if _TIME_SERIES_AVAILABLE:
    __all__.extend([
        'TimeSeriesPlotOptions',
        'plot_time_traces',
        'plot_prototype_traces',
        'plot_filtered_traces',
        'plot_multi_channel_overlay',
    ])

if _UTILITY_AVAILABLE:
    __all__.extend([
        'plot_quantiles',
        'plot_statistical_summary',
        'create_colormap',
        'setup_figure_style',
    ]) 