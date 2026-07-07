"""
eigspec - Python port of eigspec MATLAB toolbox for spectral analysis and modal identification

This package provides tools for:
- Stochastic Subspace Identification (SSI) 
- AR/PCA time-series modeling
- Random projection spectral analysis
- Modal analysis and shape estimation
- Block-based processing for large datasets
- Clustering analysis for modal pattern recognition
- Comprehensive visualization and data I/O

Python port of the MATLAB eigspec toolbox developed for plasma physics applications.
Main MATLAB entry points correspond to:
- eigspec_mmain.m - Main analysis workflow
- rndspecx.m - Random projection spectral analysis
- view_pcaspec_*.m - Visualization and results analysis
- clus_*.m - Clustering analysis functions
- ssi*.m - Subspace identification algorithms
"""

__version__ = "0.1.0"

# Main analysis functions
from .analysis.random_projection import RandomProjectionSpectralAnalysisResult, random_projection_spectral_analysis

# Core modal analysis utilities  
from .utils.modal_analysis import ModalList, ModalShortlist, ShapeEstimates, extract_modal_parameters, order_mac, shapes_from_freq, complex_vector_scalar_fit, shape2mn

# Block processing functions
from .utils.block_processing import (
    BlockAnalysisResult, 
    RandomProjectionResult, 
    random_projection_block_analysis
)

# System identification algorithms
from .utils.subspace_identification import (
    covariance_driven_ssi,
    canonical_correlation_ssi, 
    ssi1ca,
    ssicca,
    SubspaceIdentificationResult,
    StateSpaceModel
)

# AR/PCA modeling
from .utils.autoregressive_pca import arpca, ARPCAResult, ARPCAModel

# Signal processing utilities
from .utils.signal_processing import (
    FFTSpectralResult,
    CoherenceResult,
    AssessmentResult,
    CorrelationAssessmentResult,
    compute_zero_mean_spectrum,
    aggregate_fft,
    filter_signal,
    create_window,
    fftspec,
    fftspec1,
    fftspecwin,
    zmfftspec,
    yintegrate,
    ydecimate,
    yresample,
    yaddgauss,
    coherence_filter,
    ar_assessment,
    correlation_assessment,
    arassess,
    corrassess,
    fdmspec1,
    kdftspec,
    yfilt,
    yinterpolate,
    qplot_data
)

# MATLAB utility functions
from .utils.matlab_utilities import (
    kfoldcov,
    logdet,
    srteig,
    zpdftmatrix,
    fdm1dk,
    weighted_rms
)

# FDM analysis functions
from .utils.fdm_analysis import (
    fdm1d,
    rndspec, 
    pcaspecx
)

# Data extraction utilities
from .utils.data_extraction import (
    extract_ptrefs,
    collect_rep_data,
    PointReference,
    PointReferences
)

# Clustering analysis utilities
from .utils.clustering import (
    DistanceMetric,
    ClusteringResult,
    distance_matrix,
    similarity_matrix,
    kmeans_clustering,
    spectral_clustering,
    medoid_clustering,
    mac_value,
    trim_cluster_mac,
    clus_similarity_matrix,
    clus_distance_matrix,
    spclus_spectral,
    spclus_knn_similarity_matrix,
    clus_krnn_enhance,
)

# Import visualization module if available
try:
    from . import vis
    _VIS_AVAILABLE = True
except ImportError:
    _VIS_AVAILABLE = False

# Import I/O module if available
try:
    from . import io
    _IO_AVAILABLE = True
except ImportError:
    _IO_AVAILABLE = False

__all__ = [
    # Main analysis entry points
    'random_projection_spectral_analysis',
    'RandomProjectionSpectralAnalysisResult',
    
    # Modal analysis data structures
    'ModalList',
    'ModalShortlist', 
    'ShapeEstimates',
    
    # Block processing
    'BlockAnalysisResult',
    'RandomProjectionResult',
    'random_projection_block_analysis',
    
    # Core analysis functions
    'extract_modal_parameters',
'order_mac',
'shapes_from_freq',
'complex_vector_scalar_fit',
'shape2mn',
'modal_mac_matrix',
'sort_modes_by_frequency',
'normalize_mode_shapes',
'mode_shape_scaling_factor',
'modal_correlation_coefficient',
'extract_modal_parameters',
    
    # System identification
    'covariance_driven_ssi',
    'canonical_correlation_ssi',
    'ssi1ca',
    'ssicca',
    'SubspaceIdentificationResult',
    'StateSpaceModel',
    
    # AR/PCA modeling  
    'arpca',
    'ARPCAResult',
    'ARPCAModel',
    
    # Signal processing
    'FFTSpectralResult',
    'CoherenceResult',
    'AssessmentResult',
    'CorrelationAssessmentResult',
    'compute_zero_mean_spectrum',
    'aggregate_fft',
    'filter_signal',
    'create_window',
    'fftspec',
    'coherence_filter',
    'ar_assessment',
    'correlation_assessment',
    
    # MATLAB utilities
    'kfoldcov',
    'logdet', 
    'srteig',
    'zpdftmatrix',
    'fdm1dk',
    'weighted_rms',
    
    # FDM analysis
    'fdm1d',
    'rndspec',
    'pcaspecx',
    
    # Data extraction
    'extract_ptrefs',
    'collect_rep_data',
    'PointReference', 
    'PointReferences',
    
    # Clustering analysis
    'DistanceMetric',
    'ClusteringResult',
    'distance_matrix',
    'similarity_matrix',
    'kmeans_clustering',
    'spectral_clustering',
    'medoid_clustering',
    'mac_value',
    'trim_cluster_mac',
] 

# Add visualization to exports if available
if _VIS_AVAILABLE:
    __all__.append('vis')

# Add I/O to exports if available  
if _IO_AVAILABLE:
    __all__.append('io') 