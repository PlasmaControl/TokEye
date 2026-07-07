"""
Clustering analysis module for eigspec.

This module provides clustering algorithms and pattern recognition tools specifically 
designed for modal analysis applications, including:
- Distance/similarity metrics (Euclidean, Cosine, MAC)
- Clustering algorithms (K-means, Spectral clustering, Medoids)
- Graph construction methods (Full, kNN, mutual kNN, epsilon neighborhoods)
- Specialized modal analysis clustering utilities

Based on the MATLAB eigspec toolbox clustering functions.
"""

from typing import Union, Optional, Literal, cast, Dict
import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass
import warnings


@dataclass
class ClusteringResult:
    """Result from clustering analysis.
    
    Attributes:
        labels: Cluster labels for each data point
        centroids: Cluster centroids (for k-means)
        medoid_indices: Indices of cluster medoids
        cost: Final clustering cost/objective function value
        eigenvalues: Eigenvalues from spectral clustering (if applicable)
        iterations: Number of iterations to convergence
    """
    labels: npt.NDArray[np.int32]
    centroids: Optional[npt.NDArray[np.floating]] = None
    medoid_indices: Optional[npt.NDArray[np.int32]] = None
    cost: Optional[float] = None
    eigenvalues: Optional[npt.NDArray[np.floating]] = None
    iterations: Optional[int] = None


class DistanceMetric:
    """Distance and similarity metrics for clustering analysis."""
    
    @staticmethod
    def euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> float:
        """Euclidean distance between two vectors."""
        return float(np.linalg.norm(x - y))
    
    @staticmethod
    def cosine_distance(x: npt.NDArray, y: npt.NDArray) -> float:
        """Cosine distance between two real vectors."""
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError("Cosine distance not compatible with complex data")
        
        x_flat = x.ravel()
        y_flat = y.ravel()
        
        norm_x = np.linalg.norm(x_flat)
        norm_y = np.linalg.norm(y_flat)
        
        if norm_x == 0 or norm_y == 0:
            return 1.0
        
        cosine_sim = np.dot(x_flat, y_flat) / (norm_x * norm_y)
        return float(1.0 - cosine_sim)
    
    @staticmethod
    def mac_distance(x: npt.NDArray, y: npt.NDArray) -> float:
        """Modal Assurance Criterion (MAC) distance."""
        numerator = np.abs(np.vdot(x, y)) ** 2
        denominator = (np.vdot(x, x) * np.vdot(y, y))
        
        if denominator == 0:
            return 1.0
        
        mac_value = np.real(numerator / denominator)
        return float(1.0 - mac_value)
    
    @staticmethod
    def euclidean_similarity(x: npt.NDArray, y: npt.NDArray, sigma: float) -> float:
        """Gaussian similarity based on Euclidean distance."""
        dist_sq = np.linalg.norm(x - y) ** 2
        return float(np.exp(-dist_sq / (2 * sigma**2)))
    
    @staticmethod
    def cosine_similarity(x: npt.NDArray, y: npt.NDArray) -> float:
        """Cosine similarity between two real vectors."""
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError("Cosine similarity not compatible with complex data")
        
        x_flat = x.ravel()
        y_flat = y.ravel()
        
        norm_x = np.linalg.norm(x_flat)
        norm_y = np.linalg.norm(y_flat)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return float(np.dot(x_flat, y_flat) / (norm_x * norm_y))
    
    @staticmethod
    def mac_similarity(x: npt.NDArray, y: npt.NDArray) -> float:
        """Modal Assurance Criterion (MAC) similarity."""
        numerator = np.abs(np.vdot(x, y)) ** 2
        denominator = (np.vdot(x, x) * np.vdot(y, y))
        
        if denominator == 0:
            return 0.0
        
        return float(np.real(numerator / denominator))


def distance_matrix(
    X: npt.NDArray[np.floating], 
    metric: Literal["euclidean", "cosine", "mac"] = "euclidean"
) -> npt.NDArray[np.floating]:
    """Compute symmetric distance matrix between data points.
    
    Args:
        X: Data matrix where each column is a feature vector
        metric: Distance metric to use
        
    Returns:
        Symmetric distance matrix
    """
    n = X.shape[1]
    
    if metric == "euclidean":
        # Use scipy's optimized pdist for Euclidean distance
        distances = pdist(X.T, metric='euclidean')
        return cast(npt.NDArray[np.floating], squareform(distances))
    
    elif metric == "mac":
        # Vectorized MAC distance computation
        # Normalize columns for MAC computation
        norms = np.sqrt(np.sum(X.conj() * X, axis=0, keepdims=True))
        norms[norms == 0] = 1  # Avoid division by zero
        X_norm = X / norms
        
        # MAC similarity matrix
        similarity = np.abs(X_norm.conj().T @ X_norm) ** 2
        
        # Convert to distance and ensure zero diagonal
        distance = 1 - similarity
        np.fill_diagonal(distance, 0)
        return cast(npt.NDArray[np.floating], distance)
    
    elif metric == "cosine":
        if np.iscomplexobj(X):
            raise ValueError("Cosine distance not compatible with complex data")
        
        # Use scipy's cosine distance
        distances = pdist(X.T, metric='cosine')
        return cast(npt.NDArray[np.floating], squareform(distances))
    
    else:
        raise ValueError(f"Distance metric '{metric}' not recognized")


def similarity_matrix(
    X: npt.NDArray[np.floating],
    distance_method: Literal["euclidean", "cosine", "mac"] = "euclidean",
    graph_method: Literal["full", "knn", "mknn", "epsilon"] = "full",
    **kwargs
) -> npt.NDArray[np.floating]:
    """Compute similarity matrix for spectral clustering.
    
    Args:
        X: Data matrix where each column is a feature vector
        distance_method: Distance/similarity method
        graph_method: Graph construction method
        **kwargs: Additional parameters:
            - sigma: For euclidean similarity (required)
            - k: For knn/mknn methods (required)
            - epsilon: For epsilon method (required)
    
    Returns:
        Symmetric similarity matrix
    """
    d, n = X.shape
    
    print(f"Processing {n} features (dimension {d}, {'complex' if np.iscomplexobj(X) else 'real'})")
    
    # Validate parameters
    if distance_method == "euclidean" and "sigma" not in kwargs:
        raise ValueError("sigma > 0 required for euclidean similarity")
    if graph_method in ["knn", "mknn"] and "k" not in kwargs:
        raise ValueError("k > 0 required for kNN methods")
    if graph_method == "epsilon" and "epsilon" not in kwargs:
        raise ValueError("epsilon > 0 required for epsilon method")
    
    if graph_method == "full":
        if distance_method == "mac":
            # Optimized vectorized MAC similarity
            print("Using optimized full/MAC computation")
            norms = np.sqrt(np.sum(X.conj() * X, axis=0, keepdims=True))
            norms[norms == 0] = 1
            X_norm = X / norms
            S = np.abs(X_norm.conj().T @ X_norm) ** 2
            np.fill_diagonal(S, 0.0)
            return cast(npt.NDArray[np.floating], S)
        
        elif distance_method == "euclidean":
            sigma = kwargs["sigma"]
            # Vectorized Euclidean similarity
            distances = pdist(X.T, metric='euclidean')
            dist_matrix = squareform(distances)
            S = np.exp(-dist_matrix**2 / (2 * sigma**2))
            np.fill_diagonal(S, 0.0)
            return cast(npt.NDArray[np.floating], S)
        
        elif distance_method == "cosine":
            if np.iscomplexobj(X):
                raise ValueError("Cosine similarity not compatible with complex data")
            # Use cosine similarity (1 - cosine distance)
            distances = pdist(X.T, metric='cosine')
            dist_matrix = squareform(distances)
            S = 1 - dist_matrix
            np.fill_diagonal(S, 0.0)
            return cast(npt.NDArray[np.floating], S)
    
    elif graph_method in ["knn", "mknn"]:
        k = kwargs["k"]
        # Remove k from kwargs to avoid conflicts
        knn_kwargs = {key: value for key, value in kwargs.items() if key != "k"}
        return _knn_similarity_matrix(X, distance_method, k, mutual=(graph_method == "mknn"), **knn_kwargs)
    
    elif graph_method == "epsilon":
        epsilon = kwargs["epsilon"]
        # First compute full similarity matrix
        full_S = similarity_matrix(X, distance_method, "full", **kwargs)
        # Apply epsilon threshold
        threshold = 1 - epsilon
        S = np.where(full_S >= threshold, full_S, 0)
        return cast(npt.NDArray[np.floating], S)
    
    else:
        raise ValueError(f"Graph method '{graph_method}' not recognized")


def _knn_similarity_matrix(
    X: npt.NDArray[np.floating], 
    distance_method: Literal["euclidean", "cosine", "mac"],
    k: int, 
    mutual: bool = False,
    **kwargs
) -> npt.NDArray[np.floating]:
    """Construct k-nearest neighbors similarity matrix."""
    n = X.shape[1]
    
    # First compute full similarity matrix efficiently
    full_S = similarity_matrix(X, distance_method, "full", **kwargs)
    
    # Find k-nearest neighbors for each point
    # Sort similarities in descending order to get top k
    sorted_indices = np.argsort(-full_S, axis=1)
    knn_indices = sorted_indices[:, :k]
    
    # Construct sparse similarity matrix
    S = np.zeros((n, n))
    
    for i in range(n):
        neighbors_i = knn_indices[i]
        for j in neighbors_i:
            if i != j:
                if mutual:
                    # Mutual kNN: check if i is also in j's neighborhood
                    neighbors_j = knn_indices[j]
                    if i in neighbors_j:
                        S[i, j] = full_S[i, j]
                        S[j, i] = full_S[i, j]
                else:
                    # Regular kNN: set both directions to ensure symmetry
                    S[i, j] = full_S[i, j]
                    S[j, i] = full_S[i, j]
    
    return cast(npt.NDArray[np.floating], S)


def kmeans_clustering(
    X: npt.NDArray[np.floating], 
    k: int, 
    n_trials: int = 10,
    max_iter: int = 300,
    random_state: Optional[int] = None
) -> ClusteringResult:
    """K-means clustering with multiple random initializations.
    
    Args:
        X: Data matrix where each column is a feature vector
        k: Number of clusters
        n_trials: Number of random initialization trials
        max_iter: Maximum iterations per trial
        random_state: Random seed for reproducibility
        
    Returns:
        ClusteringResult with best clustering from all trials
    """
    if k < 2:
        raise ValueError("Number of clusters should be at least 2")
    if n_trials < 1:
        raise ValueError("At least 1 trial is required")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    d, n = X.shape
    best_cost = float('inf')
    best_result: Optional[ClusteringResult] = None
    
    for trial in range(n_trials):
        # Random initialization from actual data points
        initial_indices = np.random.choice(n, size=k, replace=False)
        centroids = X[:, initial_indices].copy()
        
        result = _kmeans_single_trial(X, centroids, max_iter)
        
        if result.cost is not None and result.cost < best_cost:
            best_cost = result.cost
            best_result = result
    
    if best_result is None:
        raise RuntimeError("K-means failed to converge in all trials")
    
    return best_result


def _kmeans_single_trial(
    X: npt.NDArray[np.floating], 
    initial_centroids: npt.NDArray[np.floating],
    max_iter: int
) -> ClusteringResult:
    """Single trial of k-means clustering."""
    d, n = X.shape
    k = initial_centroids.shape[1]
    
    centroids = initial_centroids.copy()
    labels = np.zeros(n, dtype=np.int32)
    
    old_cost = float('inf')
    total_cost = 0.0
    iteration = 0
    
    for iteration in range(max_iter):
        # Vectorized distance computation and assignment
        distances = np.zeros((n, k))
        for j in range(k):
            diff = X - centroids[:, j:j+1]
            distances[:, j] = np.sum(np.real(diff.conj() * diff), axis=0)
        
        labels = np.argmin(distances, axis=1).astype(np.int32)
        
        # Update centroids and compute cost
        total_cost = 0.0
        for j in range(k):
            cluster_mask = labels == j
            if np.any(cluster_mask):
                cluster_points = X[:, cluster_mask]
                centroids[:, j] = np.mean(cluster_points, axis=1)
                
                # Compute within-cluster sum of squares
                diff = cluster_points - centroids[:, j:j+1]
                total_cost += np.sum(np.real(diff.conj() * diff))
        
        # Check convergence
        if total_cost >= old_cost:
            break
        old_cost = total_cost
    
    return ClusteringResult(
        labels=labels,
        centroids=centroids,
        cost=total_cost,
        iterations=iteration + 1
    )


def spectral_clustering(
    similarity_matrix: npt.NDArray[np.floating],
    k: Union[int, list],
    method: Literal["standard", "shi", "ng"] = "standard",
    n_trials: int = 10,
    auto_select: bool = False
) -> ClusteringResult:
    """Spectral clustering using various Laplacian methods.
    
    Args:
        similarity_matrix: Symmetric similarity/adjacency matrix
        k: Number of clusters (or list of k values, or 0 for auto-selection)
        method: Laplacian method ('standard', 'shi', 'ng')
        n_trials: Number of k-means trials in final stage
        auto_select: Whether to auto-select k using eigengap heuristic
        
    Returns:
        ClusteringResult with spectral clustering results
    """
    if isinstance(k, (list, np.ndarray)):
        if len(k) > 1:
            raise NotImplementedError("Multiple k values not yet implemented")
        k = k[0] if len(k) == 1 else 0
    
    if k == 0:
        auto_select = True
    
    n = similarity_matrix.shape[0]
    if similarity_matrix.shape[1] != n:
        raise ValueError("Similarity matrix must be square")
    
    # Ensure zero diagonal
    W = similarity_matrix.copy()
    if np.linalg.norm(np.diag(W)) != 0:
        warnings.warn("Similarity matrix has nonzero diagonal (ignored)")
        np.fill_diagonal(W, 0)
    
    # Compute degree matrix
    degrees = np.sum(W, axis=1)
    if np.any(degrees <= 0):
        raise ValueError("All degree matrix entries must be positive")
    
    # Solve eigenvalue problem based on method
    if method == "standard":
        # Standard Laplacian: L = D - W
        D = np.diag(degrees)
        L = D - W
        eigenvalues, eigenvectors = eigh(L)
        
    elif method in ["shi", "ng"]:
        # Both use normalized Laplacian: I - D^(-1/2) W D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        eigenvalues, eigenvectors = eigh(L_norm)
        
    else:
        raise ValueError(f"Laplacian method '{method}' not recognized")
    
    # Auto-select k using eigengap heuristic
    if auto_select:
        k = _eigengap_heuristic(eigenvalues)
        if k < 2:
            raise ValueError("Auto-selected k < 2, clustering not possible")
    
    if k < 2:
        raise ValueError("Number of clusters should be at least 2")
    
    # Prepare features for k-means
    if method == "standard":
        features = eigenvectors[:, :k]
    elif method == "shi":
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        features = D_inv_sqrt @ eigenvectors[:, :k]
    elif method == "ng":
        # Row-normalize the eigenvectors
        features = eigenvectors[:, :k]
        row_norms = np.linalg.norm(features, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        features = features / row_norms
    
    # Final k-means clustering
    kmeans_result = kmeans_clustering(features.T, k, n_trials)
    
    return ClusteringResult(
        labels=kmeans_result.labels,
        centroids=kmeans_result.centroids,
        cost=kmeans_result.cost,
        eigenvalues=eigenvalues,
        iterations=kmeans_result.iterations
    )


def _eigengap_heuristic(eigenvalues: npt.NDArray[np.floating]) -> int:
    """Auto-select number of clusters using eigengap heuristic."""
    n_eval = min(len(eigenvalues) // 2, len(eigenvalues))
    if n_eval < 2:
        return 2
    
    gaps = np.diff(eigenvalues[:n_eval])
    k = int(np.argmax(gaps)) + 1
    return max(k, 2)


def medoid_clustering(
    X: npt.NDArray[np.floating],
    labels: npt.NDArray[np.int32], 
    metric: Literal["euclidean", "cosine", "mac"] = "euclidean"
) -> npt.NDArray[np.int32]:
    """Find cluster medoids given clustering labels.
    
    Args:
        X: Data matrix where each column is a feature vector
        labels: Cluster labels for each data point
        metric: Distance metric for medoid computation
        
    Returns:
        Array of medoid indices for each cluster
    """
    unique_labels = np.unique(labels[labels > 0])
    num_clusters = len(unique_labels)
    medoid_indices = np.zeros(num_clusters, dtype=np.int32)
    
    for i, cluster_id in enumerate(unique_labels):
        cluster_points = np.where(labels == cluster_id)[0]
        if len(cluster_points) == 0:
            raise ValueError(f"Empty cluster #{cluster_id}")
        
        medoid_idx = _find_medoid(X[:, cluster_points], metric)
        medoid_indices[i] = cluster_points[medoid_idx]
    
    return medoid_indices


def _find_medoid(X: npt.NDArray[np.floating], metric: Literal["euclidean", "cosine", "mac"]) -> int:
    """Find medoid (point with smallest average distance to all others)."""
    n = X.shape[1]
    D = distance_matrix(X, metric)
    
    # Find point with minimum sum of distances
    costs = np.sum(D, axis=1)
    return int(np.argmin(costs))


def mac_value(v: npt.NDArray, w: npt.NDArray) -> float:
    """Compute Modal Assurance Criterion (MAC) value between two vectors."""
    return DistanceMetric.mac_similarity(v, w)


def trim_cluster_mac(
    X: npt.NDArray[np.floating],
    labels: npt.NDArray[np.int32],
    medoid_indices: npt.NDArray[np.int32],
    threshold: float
) -> npt.NDArray[np.int32]:
    """Remove vectors from clusters if MAC with medoid is below threshold.
    
    Args:
        X: Data matrix where each column is a feature vector
        labels: Current cluster labels
        medoid_indices: Indices of cluster medoids
        threshold: MAC threshold (0 < threshold < 1)
        
    Returns:
        Updated cluster labels (demoted points get label 0)
    """
    if threshold <= 0 or threshold >= 1:
        raise ValueError("Threshold must satisfy 0 < threshold < 1")
    
    updated_labels = labels.copy()
    total_removed = 0
    num_clusters = len(medoid_indices)
    
    for cluster_id in range(1, num_clusters + 1):
        cluster_points = np.where(labels == cluster_id)[0]
        if len(cluster_points) == 0:
            continue
        
        medoid_idx = medoid_indices[cluster_id - 1]
        medoid_shape = X[:, medoid_idx]
        
        # Vectorized MAC computation for all points in cluster
        cluster_shapes = X[:, cluster_points]
        # Compute MAC for each column
        numerators = np.abs(medoid_shape.conj().T @ cluster_shapes) ** 2
        medoid_norm_sq = np.real(medoid_shape.conj().T @ medoid_shape)
        cluster_norms_sq = np.sum(cluster_shapes.conj() * cluster_shapes, axis=0)
        denominators = medoid_norm_sq * cluster_norms_sq
        
        # Avoid division by zero
        mac_values = np.where(denominators > 0, numerators / denominators, 0.0)
        mac_values = np.real(mac_values)
        
        # Find points below threshold
        below_threshold = mac_values < threshold
        removed_points = cluster_points[below_threshold]
        updated_labels[removed_points] = 0
        
        removed_count = len(removed_points)
        print(f"#{removed_count} vectors removed from cluster #{cluster_id}")
        total_removed += removed_count
    
    print(f"Total of #{total_removed} vectors removed from clusters")
    return updated_labels


# =============================================================================
# Missing MATLAB Clustering Functions
# =============================================================================

def clus_similarity_matrix(
    X: npt.NDArray,
    distance_method: str = 'euclidean',
    graph_method: str = 'full',
    method_args: Optional[Dict] = None
) -> npt.NDArray[np.floating]:
    """
    Calculate symmetric similarity matrix for clustering.
    
    This function calculates similarity matrices from feature data using various
    distance and graph construction methods, following the MATLAB clus_similarity_matrix.m.
    
    Args:
        X: Feature matrix where each column is a feature vector, shape (d, n)
        distance_method: Distance method ('euclidean', 'cosine', 'mac')
        graph_method: Graph construction method ('full', 'knn', 'mknn', 'epsilon')
        method_args: Additional arguments for graph methods (e.g., {'k': 10} for knn)
        
    Returns:
        Symmetric similarity matrix, shape (n, n)
        
    Example:
        >>> X = np.random.randn(5, 100)  # 100 features of dimension 5
        >>> S = clus_similarity_matrix(X, 'euclidean', 'knn', {'k': 10})
        >>> print(f"Similarity matrix shape: {S.shape}")
    """
    if method_args is None:
        method_args = {}
    
    d, n = X.shape
    data_is_complex = np.iscomplexobj(X)
    
    # Compute distance matrix
    if distance_method == 'euclidean':
        if data_is_complex:
            # For complex data, use magnitude
            X_real = np.abs(X)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    dist = np.linalg.norm(X_real[:, i] - X_real[:, j])
                    distances[i, j] = distances[j, i] = dist
        else:
            # Standard Euclidean distance
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    dist = np.linalg.norm(X[:, i] - X[:, j])
                    distances[i, j] = distances[j, i] = dist
                    
    elif distance_method == 'cosine':
        if data_is_complex:
            raise NotImplementedError("Cosine similarity not implemented for complex data")
        # Cosine distance = 1 - cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(X.T)
        distances = 1 - similarities
        
    elif distance_method == 'mac':
        # Modal Assurance Criterion (1 - MAC)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if data_is_complex:
                    numerator = np.abs(X[:, i].conj().T @ X[:, j])**2
                    denominator = np.real((X[:, i].conj().T @ X[:, i]) * 
                                        (X[:, j].conj().T @ X[:, j]))
                else:
                    numerator = (X[:, i].T @ X[:, j])**2
                    denominator = (X[:, i].T @ X[:, i]) * (X[:, j].T @ X[:, j])
                
                mac_value = numerator / denominator if denominator > 0 else 0
                distances[i, j] = distances[j, i] = 1 - mac_value
    else:
        raise ValueError(f"Unknown distance method: {distance_method}")
    
    # Convert distances to similarities
    if distance_method == 'euclidean':
        # Use Gaussian kernel
        sigma = method_args.get('sigma', np.std(distances[distances > 0]))
        similarities = np.exp(-distances**2 / (2 * sigma**2))
    else:
        similarities = 1 - distances
    
    # Apply graph construction method
    if graph_method == 'full':
        # Use full similarity matrix
        S = similarities
        
    elif graph_method == 'knn':
        # k-nearest neighbors
        k = method_args.get('k', 10)
        S = np.zeros_like(similarities)
        
        for i in range(n):
            # Find k nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i, :])[:k+1]
            neighbor_indices = neighbor_indices[neighbor_indices != i][:k]
            S[i, neighbor_indices] = similarities[i, neighbor_indices]
            
    elif graph_method == 'mknn':
        # Mutual k-nearest neighbors
        k = method_args.get('k', 10)
        knn_graph = np.zeros_like(similarities)
        
        # First build knn graph
        for i in range(n):
            neighbor_indices = np.argsort(distances[i, :])[:k+1]
            neighbor_indices = neighbor_indices[neighbor_indices != i][:k]
            knn_graph[i, neighbor_indices] = 1
        
        # Make it mutual (symmetric)
        mutual_graph = knn_graph * knn_graph.T
        S = similarities * mutual_graph
        
    elif graph_method == 'epsilon':
        # Epsilon neighborhood
        epsilon = method_args.get('epsilon', np.median(distances[distances > 0]))
        S = similarities * (distances <= epsilon)
        
    else:
        raise ValueError(f"Unknown graph method: {graph_method}")
    
    # Ensure symmetry
    S = (S + S.T) / 2
    
    return S


def clus_distance_matrix(
    X: npt.NDArray,
    distance_method: str = 'euclidean'
) -> npt.NDArray[np.floating]:
    """
    Compute pairwise distance matrix between feature vectors.
    
    Args:
        X: Feature matrix where each column is a feature vector, shape (d, n)
        distance_method: Distance method ('euclidean', 'manhattan', 'mac')
        
    Returns:
        Distance matrix, shape (n, n)
    """
    d, n = X.shape
    distances = np.zeros((n, n))
    
    if distance_method == 'euclidean':
        for i in range(n):
            for j in range(i, n):
                dist = np.linalg.norm(X[:, i] - X[:, j])
                distances[i, j] = distances[j, i] = dist
                
    elif distance_method == 'manhattan':
        for i in range(n):
            for j in range(i, n):
                dist = np.sum(np.abs(X[:, i] - X[:, j]))
                distances[i, j] = distances[j, i] = dist
                
    elif distance_method == 'mac':
        # MAC-based distance (1 - MAC)
        for i in range(n):
            for j in range(i, n):
                if np.iscomplexobj(X):
                    numerator = np.abs(X[:, i].conj().T @ X[:, j])**2
                    denominator = np.real((X[:, i].conj().T @ X[:, i]) * 
                                        (X[:, j].conj().T @ X[:, j]))
                else:
                    numerator = (X[:, i].T @ X[:, j])**2
                    denominator = (X[:, i].T @ X[:, i]) * (X[:, j].T @ X[:, j])
                
                mac_value = numerator / denominator if denominator > 0 else 0
                distances[i, j] = distances[j, i] = 1 - mac_value
    else:
        raise ValueError(f"Unknown distance method: {distance_method}")
    
    return distances


def spclus_knn_similarity_matrix(
    X: npt.NDArray,
    k: int = 10,
    sigma: Optional[float] = None
) -> npt.NDArray[np.floating]:
    """
    Create k-NN similarity matrix for spectral clustering.
    
    Args:
        X: Feature matrix, shape (d, n)
        k: Number of nearest neighbors
        sigma: Gaussian kernel width (auto-estimated if None)
        
    Returns:
        k-NN similarity matrix, shape (n, n)
    """
    return clus_similarity_matrix(X, 'euclidean', 'knn', {'k': k, 'sigma': sigma})


def spclus_spectral(
    similarity_matrix: npt.NDArray[np.floating],
    n_clusters: int,
    method: str = 'normalized'
) -> npt.NDArray[np.int32]:
    """
    Spectral clustering using eigendecomposition of similarity matrix.
    
    Args:
        similarity_matrix: Symmetric similarity matrix, shape (n, n)
        n_clusters: Number of clusters
        method: Spectral clustering variant ('normalized', 'unnormalized')
        
    Returns:
        Cluster labels, shape (n,)
    """
    from sklearn.cluster import SpectralClustering
    
    # Use sklearn's implementation
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        n_init=10
    )
    
    labels = spectral.fit_predict(similarity_matrix)
    
    # Convert to 1-based indexing to match MATLAB
    return labels + 1


def clus_krnn_enhance(
    X: npt.NDArray,
    similarity_matrix: npt.NDArray[np.floating],
    k: int = 5,
    enhancement_factor: float = 2.0
) -> npt.NDArray[np.floating]:
    """
    Enhance similarity matrix using k-reciprocal nearest neighbors.
    
    This function enhances the similarity matrix by identifying reciprocal
    nearest neighbors and boosting their similarity values.
    
    Args:
        X: Feature matrix, shape (d, n)
        similarity_matrix: Input similarity matrix, shape (n, n)
        k: Number of nearest neighbors to consider
        enhancement_factor: Factor by which to enhance reciprocal similarities
        
    Returns:
        Enhanced similarity matrix, shape (n, n)
    """
    n = similarity_matrix.shape[0]
    enhanced_matrix = similarity_matrix.copy()
    
    # Compute distance matrix for neighbor finding
    distances = 1 - similarity_matrix
    
    # Find reciprocal nearest neighbors
    for i in range(n):
        # Find k nearest neighbors of point i
        neighbors_i = np.argsort(distances[i, :])[:k+1]
        neighbors_i = neighbors_i[neighbors_i != i][:k]
        
        for j in neighbors_i:
            # Check if i is also among k nearest neighbors of j
            neighbors_j = np.argsort(distances[j, :])[:k+1]
            neighbors_j = neighbors_j[neighbors_j != j][:k]
            
            if i in neighbors_j:
                # i and j are reciprocal neighbors - enhance similarity
                enhanced_matrix[i, j] *= enhancement_factor
                enhanced_matrix[j, i] *= enhancement_factor
    
    return enhanced_matrix 