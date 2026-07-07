"""
Modal analysis functions for eigspec package.

This module provides modal analysis functions for system identification and mode shape analysis:
- Modal frequency and shape vector extraction
- Order-MAC modal matching and validation
- Shape vector estimation from frequency analysis

Based on the MATLAB eigspec toolbox modal analysis functions:
- ac2modelist.m - Extract modal frequencies and shapes from state-space models
- order_mac() in rndspecx.m - Order-MAC modal matching procedure
- shapes_from_fshortlistx() in rndspecx.m - Shape estimation from frequency shortlist
- collect_rep_data.m - Harvest modal features from analysis reports
- mnfit_ptref.m - Point-reference mode shape fitting
- mnfit_clus_medoid.m - Cluster-based mode shape fitting
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union, TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
import numpy.typing as npt

# Import will be done with TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from ..analysis.random_projection import RandomProjectionSpectralAnalysisResult

if TYPE_CHECKING:
    from .clustering import ClusteringResult

@dataclass
class ModalList:
    """
    Container for modal frequencies and shape vectors.
    
    Attributes:
        eigenvalues: Complex eigenvalues representing modal frequencies
        shape: Complex modal shape vectors, shape (n_outputs, n_modes)
    """
    eigenvalues: NDArray[np.complexfloating]
    shape: NDArray[np.complexfloating]
    
    # Add property aliases for backward compatibility with tests
    @property
    def lambda_vals(self) -> NDArray[np.complexfloating]:
        """Alias for eigenvalues for test compatibility."""
        return self.eigenvalues

@dataclass
class ModalShortlist:
    """
    Container for modal analysis shortlist results.
    
    Attributes:
        input_modes: Original modal list
        mac: Modal Assurance Criterion matrix, shape (n_modes1, n_modes2)
        distance: Distance matrix, shape (n_modes1, n_modes2)
        mode_indices: Selected mode indices
    """
    input_modes: ModalList
    mac: NDArray[np.floating]
    distance: NDArray[np.floating]
    mode_indices: List[int]
    
    # Add property aliases for backward compatibility with tests
    @property
    def imode(self) -> List[int]:
        """Alias for mode_indices for test compatibility."""
        return self.mode_indices
    
    @property
    def m0(self) -> ModalList:
        """Alias for input_modes for test compatibility."""
        return self.input_modes
    
    @property
    def dst(self) -> NDArray[np.floating]:
        """Alias for distance matrix for test compatibility."""
        return self.distance

@dataclass
class ShapeEstimates:
    """
    Container for estimated shape vectors.
    
    Attributes:
        shape_estimates: Estimated shape matrix, shape (n_outputs, n_modes) - complex
        shape_estimates_rms: RMS values for each shape, shape (n_modes,)
    """
    shape_estimates: NDArray[np.complexfloating]
    shape_estimates_rms: NDArray[np.floating]
    
    # Add property aliases for backward compatibility with tests
    @property
    def dhat(self) -> NDArray[np.complexfloating]:
        """Alias for shape_estimates for test compatibility."""
        return self.shape_estimates
    
    @property 
    def drms(self) -> NDArray[np.floating]:
        """Alias for shape_estimates_rms for test compatibility."""
        return self.shape_estimates_rms

@dataclass
class ModalFittingResult:
    """Result from modal fitting analysis.
    
    Attributes:
        mode_numbers: Array of (m,n) mode number pairs
        mac_values: MAC values for each (m,n) pair
        best_fit_indices: Indices sorted by decreasing MAC value
        reference_shape: Reference shape vector used for fitting
        spatial_coordinates: Spatial coordinate arrays (Yxy)
        query_point: Query [time, frequency] point
        closest_point: Actual [time, frequency] of closest match
        distance: Distance between query and closest point
    """
    mode_numbers: npt.NDArray[np.int32]  # (N, 2) array of (m,n) pairs
    mac_values: npt.NDArray[np.floating]
    best_fit_indices: npt.NDArray[np.int32]
    reference_shape: npt.NDArray[np.complexfloating]
    spatial_coordinates: npt.NDArray[np.floating]
    query_point: npt.NDArray[np.floating]  # [time_ms, freq_kHz]
    closest_point: npt.NDArray[np.floating]  # [time_ms, freq_kHz]
    distance: float


@dataclass
class PrototypeExtraction:
    """Container for extracted prototype modes.
    
    Attributes:
        modes: List of individual mode information
        spatial_coordinates: Spatial coordinate arrays (Yxy)
        sampling_time: Sampling time in seconds
        block_settings: Block analysis parameters [bss]
        projection_settings: Random projection parameters [rfpn]
        thresholds: Analysis thresholds
    """
    modes: List[Dict[str, Union[float, npt.NDArray]]]
    spatial_coordinates: npt.NDArray[np.floating]
    sampling_time: float
    block_settings: npt.NDArray[np.int32]
    projection_settings: npt.NDArray[np.floating] 
    thresholds: npt.NDArray[np.floating]


def extract_modal_parameters(
    state_matrix: NDArray[np.number], 
    output_matrix: NDArray[np.number], 
    transformation_matrix: Optional[NDArray[np.number]] = None,
    imag_threshold: float = 1e-14
) -> ModalList:
    """
    Returns modal frequencies and shape-vectors for the matrix pair (A,C).
    
    Args:
        state_matrix: State transition matrix A, shape (n_states, n_states)
        output_matrix: Output matrix C, shape (n_outputs, n_states)
        transformation_matrix: Optional transformation matrix U, shape (n_outputs, n_outputs)
        imag_threshold: Threshold for considering imaginary part zero
        
    Returns:
        ModalList containing eigenvalues and shape vectors
        
    Example:
        >>> a = np.array([[0, -1], [1, 0]])  # Rotation matrix
        >>> c = np.eye(2)
        >>> modes = ac2modelist(a, c)
        >>> np.allclose(modes.lambda_vals, [1j, -1j])
        True
    """
    if state_matrix.ndim != 2 or state_matrix.shape[0] != state_matrix.shape[1]:
        raise ValueError(f"A must be square, got shape {state_matrix.shape}")
    if output_matrix.ndim != 2 or output_matrix.shape[1] != state_matrix.shape[0]:
        raise ValueError(f"C must have shape (n_outputs, n_states), got {output_matrix.shape}")
    if transformation_matrix is not None and (transformation_matrix.ndim != 2 or transformation_matrix.shape[1] != output_matrix.shape[0]):
        raise ValueError(f"U must have shape (n_outputs, n_outputs), got {transformation_matrix.shape}")
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eig(state_matrix)
    
    # Select modes with non-negative imaginary part
    positive_imaginary_mask = np.imag(eigenvals) >= 0
    rr = np.where(positive_imaginary_mask)[0]
    
    # Extract relevant eigenvalues and shapes, ensure complex type
    eigenvalues = eigenvals[rr].astype(complex)
    shape = output_matrix[:, rr].astype(complex)
    
    # Handle real modes
    real_modes = np.abs(np.imag(eigenvalues)) <= imag_threshold
    eigenvalues[real_modes] = np.real(eigenvalues[real_modes])
    shape[:, real_modes] = np.real(shape[:, real_modes])
            
    return ModalList(eigenvalues=eigenvalues, shape=shape)

def order_mac(
    output_matrix_1: NDArray[np.number],
    state_matrix_1: NDArray[np.number],
    output_matrix_2: NDArray[np.number],
    state_matrix_2: NDArray[np.number],
    thresh: Tuple[float, float]
) -> ModalShortlist:
    """
    Return a modal shortlist using the order-MAC procedure.
    
    Args:
        output_matrix_1: First output matrix C1, shape (n_outputs, n_states1)
        state_matrix_1: First state matrix A1, shape (n_states1, n_states1)
        output_matrix_2: Second output matrix C2, shape (n_outputs, n_states2)
        state_matrix_2: Second state matrix A2, shape (n_states2, n_states2)
        thresh: (MAC threshold, DST threshold) for mode matching
        
    Returns:
        ModalShortlist containing analysis results
        
    Example:
        >>> a1 = np.array([[0, -1], [1, 0]])
        >>> a2 = np.array([[0, -2], [2, 0]])
        >>> c1 = c2 = np.eye(2)
        >>> result = order_mac(c1, a1, c2, a2, (0.9, 0.9))
    """
    mac_thresh, dst_thresh = thresh
    
    # Create lists of modal shape vectors
    input_modes_1 = extract_modal_parameters(state_matrix_1, output_matrix_1)
    input_modes_2 = extract_modal_parameters(state_matrix_2, output_matrix_2)
    
    n_modes = len(input_modes_1.eigenvalues)
    
    # Compute MAC values using vectorized operations
    # Mode shapes should be (n_outputs, n_modes) - use shape directly
    mode_shapes_1 = input_modes_1.shape  # Shape (n_outputs, n_modes1)  
    mode_shapes_2 = input_modes_2.shape  # Shape (n_outputs, n_modes2)
    
    # MAC = |v'w|^2 / (|v|^2 * |w|^2) 
    # Cross-correlation matrix: (n_modes1, n_modes2)
    mac_values = mode_shapes_1.conj().T @ mode_shapes_2  # (n_modes1, n_modes2)
    
    # Compute norms for normalization
    mode_shapes_1_norms = np.sum(np.abs(mode_shapes_1)**2, axis=0)  # (n_modes1,)
    mode_shapes_2_norms = np.sum(np.abs(mode_shapes_2)**2, axis=0)  # (n_modes2,)
    
    # Broadcast norms for division: (n_modes1, 1) * (1, n_modes2)
    norm_matrix = mode_shapes_1_norms[:, np.newaxis] * mode_shapes_2_norms[np.newaxis, :]
    
    # Handle zero norms
    norm_matrix = np.where(norm_matrix == 0, 1, norm_matrix)
    mac = np.abs(mac_values)**2 / norm_matrix
    
    # Compute distance metric
    eigenvalues_1 = input_modes_1.eigenvalues[:, np.newaxis]  # Shape (n_modes1, 1)
    eigenvalues_2 = input_modes_2.eigenvalues[np.newaxis, :]  # Shape (1, n_modes2)
    distance = np.maximum(1 - np.abs(eigenvalues_1 - eigenvalues_2) / np.abs(eigenvalues_1), 0)
    
    # Find matching modes
    max_mac_values = np.max(mac, axis=1)
    i_max_mac_values = np.argmax(mac, axis=1)
    max_distance_values = np.max(distance, axis=1)
    i_max_distance_values = np.argmax(distance, axis=1)
    
    # Select modes meeting criteria
    mode_indices = [i for i in range(n_modes)
            if (max_mac_values[i] >= mac_thresh and 
                max_distance_values[i] >= dst_thresh and
                i_max_mac_values[i] == i_max_distance_values[i] and
                np.imag(input_modes_1.eigenvalues[i]) > 0)]
            
    return ModalShortlist(input_modes=input_modes_1, mac=mac, distance=distance, mode_indices=mode_indices)

def shapes_from_freq(
    signal: NDArray[np.number],
    modal_shortlist: ModalShortlist
) -> Optional[ShapeEstimates]:
    """
    Estimate shape vectors based on the frequency-shortlist.
    
    Args:
        signal: Block of data, shape (n_samples, n_outputs)
        modal_shortlist: Modal shortlist from order_mac
        
    Returns:
        ShapeEstimates containing estimated shapes and RMS values,
        or None if no modes were found
        
    Example:
        >>> t = np.linspace(0, 10, 1000)
        >>> y = np.sin(2*np.pi*t)[:, np.newaxis]
        >>> modal_shortlist = ...  # Modal shortlist with one mode
        >>> shapes = shapes_from_freq(y, modal_shortlist)
    """
    if not modal_shortlist.mode_indices:
        return None
        
    n_samples, n_outputs = signal.shape
    n_modes = len(modal_shortlist.mode_indices)
    
    # Build time basis matrix efficiently
    time_basis_matrix = np.arange(n_samples)[:, np.newaxis]  # Shape (n, 1)
    mode_angles = np.angle(modal_shortlist.input_modes.eigenvalues[modal_shortlist.mode_indices]).astype(complex)  # Shape (n_modes,)
    
    # Create time basis for all modes at once
    # Shape: (n, 2*n_modes)
    time_basis_matrix = np.column_stack([
        np.cos(time_basis_matrix * mode_angles.reshape(1, -1)),
        np.sin(time_basis_matrix * mode_angles.reshape(1, -1))
    ]).reshape(n_samples, -1)
    
    # Estimate shapes using least squares
    coeffs = np.linalg.lstsq(time_basis_matrix, signal, rcond=None)[0].T
    
    # Convert cos/sin coefficients to complex mode shapes
    # coeffs shape: (n_outputs, 2*n_modes) -> (n_outputs, n_modes, 2)
    coeffs_reshaped = coeffs.reshape(n_outputs, n_modes, 2)
    
    # Combine cos and sin coefficients into complex form: cos_coeff + j*sin_coeff
    shape_estimates = coeffs_reshaped[:, :, 0] + 1j * coeffs_reshaped[:, :, 1]
    
    # Compute RMS values efficiently
    shape_estimates_rms = np.sqrt(np.sum(coeffs_reshaped**2, axis=(0, 2)) / n_outputs)
        
    return ShapeEstimates(shape_estimates=shape_estimates, shape_estimates_rms=shape_estimates_rms)


def extract_prototypes(
    analysis_result: "RandomProjectionSpectralAnalysisResult",
    query_points: npt.NDArray[np.floating],
    spatial_coordinates: npt.NDArray[np.floating],
    warning_distance: float = 5.0
) -> PrototypeExtraction:
    """Extract prototype mode shapes from spectral analysis results.
    
    Python equivalent of extract_ptrefs.m
    
    Finds the reference points from analysis results by locating the closest 
    matches to provided time-frequency query points.
    
    Args:
        analysis_result: Random projection spectral analysis results
        query_points: Query points as (N, 2) array of [time_ms, freq_kHz]
        spatial_coordinates: Spatial coordinate arrays (typically Yxy)
        warning_distance: Warn if distance to query point exceeds this (euclidean in ms,kHz)
        
    Returns:
        PrototypeExtraction containing extracted mode information
        
    Raises:
        ValueError: If inputs are malformed or analysis_result is invalid
    """
    if not hasattr(analysis_result, 'block_results') or len(analysis_result.block_results) == 0:
        raise ValueError("Invalid analysis_result: no block results available")
    
    query_points = np.asarray(query_points)
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)
    
    if query_points.shape[1] != 2:
        raise ValueError("query_points must have shape (N, 2) for [time_ms, freq_kHz]")
    
    num_refs = query_points.shape[0]
    
    # Extract time-frequency data from all blocks
    tf_data = []
    shape_vectors = []
    rms_values = []
    
    # Get sampling rate from analysis result
    sampling_rate = 1.0  # Default fallback
    if (hasattr(analysis_result, 'block_results') and len(analysis_result.block_results) > 0 and
        hasattr(analysis_result.block_results[0], 'time_step') and 
        analysis_result.block_results[0].time_step > 0):
        sampling_rate = 1.0 / analysis_result.block_results[0].time_step
    
    for block_result in analysis_result.block_results:
        centre_time_ms = block_result.centre_time * 1000  # Convert to ms
        
        # Get modal analysis results from the block - it's a ModalShortlist
        modal_shortlist = block_result.reduced_dimension_matrix
        if modal_shortlist is not None and hasattr(modal_shortlist, 'input_modes'):
            # Access eigenvalues and mode shapes from the ModalShortlist
            modal_list = getattr(modal_shortlist, 'input_modes', None)
            if modal_list is not None and hasattr(modal_list, 'eigenvalues') and len(modal_list.eigenvalues) > 0:
                for idx, eigenval in enumerate(modal_list.eigenvalues):
                    # Convert eigenvalue to frequency and damping
                    real_part = np.real(eigenval)
                    imag_part = np.imag(eigenval)
                    
                    # Frequency from imaginary part - include sampling rate for discrete-time eigenvalues
                    if imag_part > 0:  # Only positive frequencies
                        freq_hz = imag_part * sampling_rate / (2 * np.pi)
                        freq_khz = freq_hz / 1e3  # Convert to kHz
                        
                        # Damping ratio from real part
                        damping = -real_part / abs(eigenval) if abs(eigenval) > 0 else 0.0
                        
                        tf_data.append([centre_time_ms, damping, freq_khz])
                        
                        # Get shape vector if available
                        modal_shape = getattr(modal_list, 'shape', None)
                        if (modal_shape is not None and modal_shape.size > 0 and
                            idx < modal_shape.shape[1]):
                            shape_vectors.append(modal_shape[:, idx])
                            
                            # RMS from shape estimates if available
                            shape_estimates = block_result.reduced_dimension_array
                            if (shape_estimates is not None and 
                                hasattr(shape_estimates, 'shape_estimates_rms')):
                                shape_rms = getattr(shape_estimates, 'shape_estimates_rms', None)
                                if shape_rms is not None and idx < len(shape_rms):
                                    rms_values.append(shape_rms[idx])
                                else:
                                    rms_values.append(np.linalg.norm(modal_shape[:, idx]))
                            else:
                                rms_values.append(np.linalg.norm(modal_shape[:, idx]))
                        else:
                            # Create dummy shape vector
                            shape_vectors.append(np.zeros(len(spatial_coordinates), dtype=complex))
                            rms_values.append(0.0)
    
    if len(tf_data) == 0:
        raise ValueError("No mode data found in analysis results")
    
    tf_data = np.array(tf_data)
    shape_vectors = np.column_stack(shape_vectors) if shape_vectors else np.array([]).reshape(len(spatial_coordinates), 0)
    rms_values = np.array(rms_values)
    
    modes = []
    
    for i in range(num_refs):
        query_time, query_freq = query_points[i]
        
        # Find closest point in (time, frequency) space
        distances_sq = (
            (tf_data[:, 0] - query_time)**2 + 
            (tf_data[:, 2] - query_freq)**2
        )
        
        closest_idx = np.argmin(distances_sq)
        distance = np.sqrt(distances_sq[closest_idx])
        
        closest_time = tf_data[closest_idx, 0]
        closest_freq = tf_data[closest_idx, 2]
        
        print(f"Query {i+1}: closest to [time,freq]=[{query_time:.6f},{query_freq:.6f}] "
              f"found at index {closest_idx+1}/{len(tf_data)} "
              f"[time,freq]=[{closest_time:.6f},{closest_freq:.6f}] (ms,kHz)")
        
        if distance > warning_distance:
            print(f"(Warning: distance to query point is large, d={distance:.3f})")
        
        mode_info = {
            'query_tifr': np.array([query_time, query_freq]),
            'shapevector': shape_vectors[:, closest_idx].copy() if shape_vectors.size > 0 else np.array([]),
            'rms': float(rms_values[closest_idx]) if len(rms_values) > closest_idx else 0.0,
            'frequency': float(tf_data[closest_idx, 2] * 1e3),  # Convert back to Hz
            'radius': float(tf_data[closest_idx, 1]),
            'centre_time': float(tf_data[closest_idx, 0] / 1e3),  # Convert to seconds
            'closest_point': np.array([closest_time, closest_freq]),
            'distance': float(distance)
        }
        
        modes.append(mode_info)
    
    return PrototypeExtraction(
        modes=modes,
        spatial_coordinates=spatial_coordinates,
        sampling_time=analysis_result.block_results[0].time_step if analysis_result.block_results else 1.0,
        block_settings=np.array(analysis_result.block_parameters, dtype=np.int32),
        projection_settings=np.array(analysis_result.reduced_dimension, dtype=np.floating),
        thresholds=np.array(analysis_result.threshold_parameters, dtype=np.floating)
    )


def modal_fitting_ptref(
    analysis_result: "RandomProjectionSpectralAnalysisResult",
    query_point: npt.NDArray[np.floating],
    spatial_coordinates: npt.NDArray[np.floating],
    m_max: int = 8,
    n_max: int = 5,
    num_display: int = 10
) -> ModalFittingResult:
    """Fit modal harmonics to reference prototype from spectral analysis.
    
    Python equivalent of mnfit_ptref.m
    
    Finds the closest reference point to the query [time, frequency] and 
    computes MAC values for a range of (m,n) mode numbers, displaying
    the best-fitting modes.
    
    Args:
        analysis_result: Random projection spectral analysis results  
        query_point: Query point as [time_ms, freq_kHz] 
        spatial_coordinates: Spatial coordinates (N, 2) array, typically [Y, X] or [R, Phi]
        m_max: Maximum m mode number (toroidal/azimuthal)
        n_max: Maximum n mode number (poloidal/radial)
        num_display: Number of best fits to display/return
        
    Returns:
        ModalFittingResult with mode fitting analysis
        
    Raises:
        ValueError: If inputs are malformed
    """
    query_point = np.asarray(query_point)
    if query_point.shape != (2,):
        raise ValueError("query_point must be [time_ms, freq_kHz]")
    
    # Extract prototype at query point
    prototype = extract_prototypes(
        analysis_result, query_point.reshape(1, 2), spatial_coordinates
    )
    
    if len(prototype.modes) == 0:
        raise ValueError("No prototype modes extracted")
    
    mode_data = prototype.modes[0]
    reference_shape = mode_data['shapevector']
    closest_point = mode_data['closest_point']
    distance = mode_data['distance']
    
    # Ensure reference_shape is a proper complex numpy array
    reference_shape = np.asarray(reference_shape, dtype=np.complexfloating)
    # Ensure closest_point is a proper array 
    closest_point = np.asarray(closest_point, dtype=np.floating)
    
    print(f"Reference mode for fitting:")
    print(f"  Query: [time,freq]=[{query_point[0]:.6f},{query_point[1]:.6f}] (ms,kHz)")
    print(f"  Found: [time,freq]=[{closest_point[0]:.6f},{closest_point[1]:.6f}] (ms,kHz)")
    print(f"  Distance: {distance:.6f}")
    
    # Generate (m,n) mode number grid
    m_vec = np.arange(-abs(m_max), abs(m_max) + 1)
    n_vec = np.arange(-abs(n_max), abs(n_max) + 1)
    m_grid, n_grid = np.meshgrid(m_vec, n_vec, indexing='ij')
    mode_numbers = np.column_stack([m_grid.ravel(), n_grid.ravel()])
    
    # Compute MAC values for all (m,n) combinations
    mac_values = _compute_modal_mac_values(
        reference_shape, mode_numbers, spatial_coordinates
    )
    
    # Sort by decreasing MAC value
    sorted_indices = np.argsort(mac_values)[::-1]
    
    print(f"\nBest (m,n) mode fits:")
    for i in range(min(num_display, len(sorted_indices))):
        idx = sorted_indices[i]
        m, n = mode_numbers[idx]
        mac_val = mac_values[idx]
        print(f"  m,n={m:2d},{n:2d}; MAC={mac_val:.6f}")
    
    return ModalFittingResult(
        mode_numbers=mode_numbers,
        mac_values=mac_values,
        best_fit_indices=sorted_indices.astype(np.int32),
        reference_shape=reference_shape,
        spatial_coordinates=spatial_coordinates,
        query_point=np.asarray(query_point, dtype=np.floating),
        closest_point=closest_point,
        distance=float(distance)
    )


def modal_fitting_cluster_medoid(
    data_matrix: npt.NDArray[np.complexfloating],
    cluster_result: 'ClusteringResult',
    cluster_index: int,
    spatial_coordinates: npt.NDArray[np.floating],
    m_max: int = 8,
    n_max: int = 5,
    num_display: int = 10
) -> ModalFittingResult:
    """Fit modal harmonics to cluster medoid shape vector.
    
    Python equivalent of mnfit_clus_medoid.m
    
    Takes a cluster medoid representative and computes MAC values for 
    a range of (m,n) mode numbers.
    
    Args:
        data_matrix: Data matrix where each column is a feature vector (M, N)
        cluster_result: ClusteringResult containing labels and medoid indices
        cluster_index: Index of cluster to analyze (1-based as in MATLAB)
        spatial_coordinates: Spatial coordinates (M, 2) array
        m_max: Maximum m mode number  
        n_max: Maximum n mode number
        num_display: Number of best fits to display/return
        
    Returns:
        ModalFittingResult with mode fitting analysis
        
    Raises:
        ValueError: If cluster_index is invalid or inputs malformed
    """
    # Import here to avoid circular imports
    from .clustering import ClusteringResult
    
    if not isinstance(cluster_result, ClusteringResult):
        raise ValueError("cluster_result must be a ClusteringResult instance")
    
    if cluster_result.medoid_indices is None:
        raise ValueError("cluster_result must contain medoid_indices")
    
    # Get number of clusters
    num_clusters = len(cluster_result.medoid_indices)
    
    if cluster_index < 1 or cluster_index > num_clusters:
        raise ValueError(f"cluster_index {cluster_index} out of range [1, {num_clusters}]")
    
    # Get medoid shape (convert from 1-based to 0-based indexing)
    medoid_idx = cluster_result.medoid_indices[cluster_index - 1]
    reference_shape = data_matrix[:, medoid_idx]
    
    print(f"Analyzing cluster #{cluster_index} medoid shape vector")
    print(f"  Medoid index: {medoid_idx}")
    
    # Generate (m,n) mode number grid  
    m_vec = np.arange(-abs(m_max), abs(m_max) + 1)
    n_vec = np.arange(-abs(n_max), abs(n_max) + 1)
    m_grid, n_grid = np.meshgrid(m_vec, n_vec, indexing='ij')
    mode_numbers = np.column_stack([m_grid.ravel(), n_grid.ravel()])
    
    # Compute MAC values for all (m,n) combinations
    mac_values = _compute_modal_mac_values(
        reference_shape, mode_numbers, spatial_coordinates
    )
    
    # Sort by decreasing MAC value
    sorted_indices = np.argsort(mac_values)[::-1]
    
    print(f"\nBest (m,n) mode fits for cluster #{cluster_index} medoid:")
    for i in range(min(num_display, len(sorted_indices))):
        idx = sorted_indices[i]
        m, n = mode_numbers[idx]
        mac_val = mac_values[idx]
        print(f"  m,n={m:2d},{n:2d}; MAC={mac_val:.6f}")
    
    return ModalFittingResult(
        mode_numbers=mode_numbers,
        mac_values=mac_values,
        best_fit_indices=sorted_indices.astype(np.int32),
        reference_shape=reference_shape,
        spatial_coordinates=spatial_coordinates,
        query_point=np.array([0.0, 0.0], dtype=np.floating),  # Not applicable for cluster medoids
        closest_point=np.array([0.0, 0.0], dtype=np.floating),  # Not applicable
        distance=0.0  # Not applicable
    )


def _compute_modal_mac_values(
    reference_shape: npt.NDArray[np.complexfloating],
    mode_numbers: npt.NDArray[np.int32],
    spatial_coordinates: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute MAC values between reference shape and (m,n) modal harmonics.
    
    Python equivalent of mns2mac function in mnfit_ptref.m
    
    Args:
        reference_shape: Complex reference shape vector (M,)
        mode_numbers: Array of (m,n) pairs (N, 2)
        spatial_coordinates: Spatial coordinates (M, 2), typically [Y, X] 
        
    Returns:
        MAC values for each (m,n) pair (N,)
    """
    M = len(reference_shape)
    if M != spatial_coordinates.shape[0]:
        raise ValueError("Dimension mismatch between reference_shape and spatial_coordinates")
    
    Y = spatial_coordinates[:, 0]  # First coordinate (e.g., Y or R)
    X = spatial_coordinates[:, 1]  # Second coordinate (e.g., X or Phi)
    
    num_modes = mode_numbers.shape[0]
    mac_values = np.zeros(num_modes)
    
    for i in range(num_modes):
        m, n = mode_numbers[i]
        
        # Compute spatial harmonic: exp(i * (m*X + n*Y))
        # Split into cos and sin components for real computation
        kx = m * X + n * Y
        harmonic = np.cos(kx) + 1j * np.sin(kx)
        
        # Compute MAC value
        mac_values[i] = _mac_value(reference_shape, harmonic)
    
    return mac_values


def _mac_value(v: npt.NDArray[np.complexfloating], w: npt.NDArray[np.complexfloating]) -> float:
    """Compute Modal Assurance Criterion between two complex vectors.
    
    Python equivalent of macvalue function in mnfit_ptref.m
    
    Args:
        v: First complex vector
        w: Second complex vector
        
    Returns:
        MAC value (0 to 1)
    """
    numerator = (v.conj().T @ w) * (w.conj().T @ v)
    denominator = (v.conj().T @ v) * (w.conj().T @ w)
    
    if np.abs(denominator) == 0:
        return 0.0
    
    return float(np.real(numerator / denominator))


def complex_vector_scalar_fit(
    a: npt.NDArray[np.complexfloating], 
    b: npt.NDArray[np.complexfloating]
) -> complex:
    """Find scalar complex number c such that a ≈ c*b in least-squares sense.
    
    Python equivalent of complex_vector_scalar_fit.m
    
    Solves the complex least-squares problem to find the optimal complex scalar
    that best fits one complex vector to another via scaling.
    
    Args:
        a: Target complex vector (result of c*b)
        b: Reference complex vector (to be scaled)
        
    Returns:
        Complex scalar c that minimizes ||a - c*b||²
        
    Raises:
        ValueError: If vectors have different lengths
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    
    if len(a) != len(b):
        raise ValueError("Vectors a and b must have same length")
    
    if len(a) == 0:
        return 0.0 + 0.0j
    
    # Solve for c in a ≈ c*b using least squares: c = <a,b> / <b,b>
    b_conj_b = np.real(np.conj(b) @ b)
    if b_conj_b > 0:
        c = (np.conj(b) @ a) / b_conj_b
    else:
        c = 0.0 + 0.0j
    
    return complex(c)


def modal_mac_matrix(
    mode_shapes: npt.NDArray[np.complexfloating]
) -> npt.NDArray[np.floating]:
    """Compute MAC matrix between all pairs of mode shapes.
    
    Args:
        mode_shapes: Mode shape matrix (n_dofs, n_modes)
        
    Returns:
        MAC matrix (n_modes, n_modes) with MAC values between all mode pairs
    """
    n_modes = mode_shapes.shape[1]
    mac_matrix = np.zeros((n_modes, n_modes))
    
    for i in range(n_modes):
        for j in range(n_modes):
            mac_matrix[i, j] = _mac_value(mode_shapes[:, i], mode_shapes[:, j])
    
    return mac_matrix


def sort_modes_by_frequency(
    frequencies: npt.NDArray[np.floating],
    mode_shapes: Optional[npt.NDArray[np.complexfloating]] = None,
    damping_ratios: Optional[npt.NDArray[np.floating]] = None
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.floating], 
           Optional[npt.NDArray[np.complexfloating]], Optional[npt.NDArray[np.floating]]]:
    """Sort modes by increasing frequency.
    
    Args:
        frequencies: Natural frequencies (Hz)
        mode_shapes: Mode shapes matrix (n_dofs, n_modes), optional
        damping_ratios: Damping ratios, optional
        
    Returns:
        Tuple of (sort_indices, sorted_frequencies, sorted_mode_shapes, sorted_damping)
    """
    sort_indices = np.argsort(frequencies).astype(np.int32)
    sorted_frequencies = frequencies[sort_indices]
    
    sorted_mode_shapes = None
    if mode_shapes is not None:
        sorted_mode_shapes = mode_shapes[:, sort_indices]
    
    sorted_damping = None
    if damping_ratios is not None:
        sorted_damping = damping_ratios[sort_indices]
    
    return sort_indices, sorted_frequencies, sorted_mode_shapes, sorted_damping


def normalize_mode_shapes(
    mode_shapes: npt.NDArray[np.complexfloating],
    method: Literal["unity_modal_mass", "max_displacement", "euclidean"] = "max_displacement"
) -> npt.NDArray[np.complexfloating]:
    """Normalize mode shapes using different criteria.
    
    Args:
        mode_shapes: Mode shapes matrix (n_dofs, n_modes)
        method: Normalization method
            - "unity_modal_mass": Normalize to unit modal mass (requires mass matrix)
            - "max_displacement": Normalize to unit maximum displacement
            - "euclidean": Normalize to unit Euclidean norm
            
    Returns:
        Normalized mode shapes matrix
    """
    normalized_shapes = mode_shapes.copy()
    n_modes = mode_shapes.shape[1]
    
    for mode_idx in range(n_modes):
        mode_shape = mode_shapes[:, mode_idx]
        
        if method == "max_displacement":
            # Normalize by maximum absolute displacement
            max_disp = np.max(np.abs(mode_shape))
            if max_disp > 0:
                normalized_shapes[:, mode_idx] = mode_shape / max_disp
                
        elif method == "euclidean":
            # Normalize to unit Euclidean norm
            norm = np.linalg.norm(mode_shape)
            if norm > 0:
                normalized_shapes[:, mode_idx] = mode_shape / norm
                
        elif method == "unity_modal_mass":
            # This would require mass matrix - placeholder for now
            raise NotImplementedError("Unity modal mass normalization requires mass matrix")
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_shapes


def mode_shape_scaling_factor(
    reference_shape: npt.NDArray[np.complexfloating],
    test_shape: npt.NDArray[np.complexfloating]
) -> complex:
    """Compute optimal complex scaling factor between two mode shapes.
    
    Finds the complex scalar that best matches test_shape to reference_shape.
    
    Args:
        reference_shape: Reference mode shape vector
        test_shape: Test mode shape vector to be scaled
        
    Returns:
        Complex scaling factor such that test_shape ≈ scaling_factor * reference_shape
    """
    return complex_vector_scalar_fit(test_shape, reference_shape)


def modal_correlation_coefficient(
    mode1: npt.NDArray[np.complexfloating],
    mode2: npt.NDArray[np.complexfloating]
) -> float:
    """Compute modal correlation coefficient between two mode shapes.
    
    This computes the absolute value of the normalized complex inner product,
    which is appropriate for complex mode shapes and should give 0 for 
    orthogonal modes and 1 for identical (up to scaling) modes.
    
    Args:
        mode1: First mode shape vector
        mode2: Second mode shape vector
        
    Returns:
        Correlation coefficient (0 to 1)
    """
    if len(mode1) != len(mode2):
        raise ValueError("Mode shapes must have same length")
    
    # Compute normalized complex inner product
    numerator = np.abs(np.vdot(mode1, mode2))
    denominator = np.linalg.norm(mode1) * np.linalg.norm(mode2)
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def complex_vector_scalar_fit(
    a: npt.NDArray[np.complexfloating],
    b: npt.NDArray[np.complexfloating]
) -> np.complexfloating:
    """
    Find scalar complex number c such that a = c*b in least-squares sense.
    
    Python port of MATLAB complex_vector_scalar_fit.m that finds the optimal
    complex scalar to fit one complex vector to another.
    
    Args:
        a: Target complex vector
        b: Reference complex vector
        
    Returns:
        Complex scalar c such that a ≈ c*b
    """
    a = a.ravel()
    b = b.ravel()
    
    if len(a) != len(b):
        raise ValueError("Vectors a and b must have same length")
    
    # Set up least squares system: [Re(b) -Im(b); Im(b) Re(b)] * [Re(c); Im(c)] = [Re(a); Im(a)]
    M = np.block([
        [b.real.reshape(-1, 1), -b.imag.reshape(-1, 1)],
        [b.imag.reshape(-1, 1), b.real.reshape(-1, 1)]
    ])
    
    Z = np.concatenate([a.real, a.imag])
    
    # Solve least squares
    ab = np.linalg.lstsq(M, Z, rcond=None)[0]
    c = ab[0] + 1j * ab[1]
    
    return c


def shape2mn(
    shape_vector: npt.NDArray[np.complexfloating],
    mn_candidates: npt.NDArray[np.int32],
    sensor_coordinates: npt.NDArray[np.floating]
) -> Tuple[int, float]:
    """
    Find best-fit (m,n) harmonic for a complex shape vector.
    
    Python port of MATLAB shape2mn.m that determines the best (m,n) mode
    numbers for a given complex shape vector by comparing with trial harmonics
    using MAC (Modal Assurance Criterion).
    
    Args:
        shape_vector: Complex shape vector, shape (M,)
        mn_candidates: Trial (m,n) pairs, shape (N_trials, 2)
        sensor_coordinates: Sensor angular positions, shape (M, 2)
        
    Returns:
        J: Index of best-fit (m,n) pair (0-based)
        E: MAC value of best fit
    """
    M = len(shape_vector)
    
    if M != sensor_coordinates.shape[0]:
        raise ValueError("Shape vector and sensor coordinates dimension mismatch")
    
    N_trials = mn_candidates.shape[0]
    mac_values = np.zeros(N_trials)
    
    for jj in range(N_trials):
        m, n = mn_candidates[jj]
        
        # Compute trial harmonic: exp(i*(m*theta + n*phi))
        kx = m * sensor_coordinates[:, 0] + n * sensor_coordinates[:, 1]
        trial_shape = np.cos(kx) + 1j * np.sin(kx)
        
        # Compute MAC value
        mac_values[jj] = _mac_value_util(shape_vector, trial_shape)
    
    # Find best match
    J = np.argmax(mac_values)
    E = mac_values[J]
    
    return int(J), float(E)


def _mac_value_util(v: npt.NDArray[np.complexfloating], w: npt.NDArray[np.complexfloating]) -> float:
    """Compute MAC (Modal Assurance Criterion) value between two vectors."""
    numerator = np.abs(np.vdot(v, w))**2
    denominator = np.real(np.vdot(v, v) * np.vdot(w, w))
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def filter_modal_parameters(
    modal_list: ModalList,
    frequency_range: Optional[Tuple[float, float]] = None,
    damping_threshold: Optional[float] = None
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
    """Filter modal parameters from ModalList within specified criteria.
    
    Args:
        modal_list: ModalList containing identified modes
        frequency_range: Optional (min_freq, max_freq) filter in Hz
        damping_threshold: Optional maximum damping ratio filter
        
    Returns:
        Tuple of (frequencies, damping_ratios, mode_shapes)
    """
    if len(modal_list.eigenvalues) == 0:
        return np.array([]), np.array([]), np.array([]).reshape(0, 0)
    
    # Extract frequencies and damping from eigenvalues
    frequencies = []
    damping_ratios = []
    valid_indices = []
    
    for i, eigenval in enumerate(modal_list.eigenvalues):
        # For complex eigenvalue s = σ + jω, frequency = |ω|/(2π), damping = -σ/|s|
        real_part = np.real(eigenval)
        imag_part = np.imag(eigenval)
        
        frequency = abs(imag_part) / (2 * np.pi)
        if abs(eigenval) > 0:
            damping_ratio = -real_part / abs(eigenval)
        else:
            damping_ratio = 0.0
        
        # Apply filters
        if frequency_range is not None:
            min_freq, max_freq = frequency_range
            if not (min_freq <= frequency <= max_freq):
                continue
        
        if damping_threshold is not None:
            if damping_ratio > damping_threshold:
                continue
        
        frequencies.append(frequency)
        damping_ratios.append(damping_ratio)
        valid_indices.append(i)
    
    if not frequencies:
        return np.array([]), np.array([]), np.array([]).reshape(0, 0)
    
    # Extract corresponding mode shapes
    if modal_list.shape.size > 0:
        filtered_shapes = modal_list.shape[:, valid_indices]
    else:
        filtered_shapes = np.array([]).reshape(0, len(valid_indices))
    
    return (
        np.array(frequencies),
        np.array(damping_ratios),
        filtered_shapes
    )