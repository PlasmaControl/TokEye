import numpy as np
from scipy.stats import entropy

def cdf_threshold(data, adjust=0., multiplier=100):
    H, W = data.shape
    median = np.median(data)
    data_2 = data.copy()
    data_2[data < median] = median

    sorted_data = np.sort(data_2.flatten())
    data_min = sorted_data.min()
    minmax = sorted_data.max() - sorted_data.min()
    sorted_data = (sorted_data - sorted_data.min()) / minmax * multiplier
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_values = (cdf_values - cdf_values.min()) / (cdf_values.max() - cdf_values.min()) * multiplier * 2

    x_cdf = np.linspace(sorted_data.min(), sorted_data.max(), multiplier)
    cdf = np.interp(x_cdf, sorted_data, cdf_values)

    start_x, end_x = x_cdf[0], x_cdf[-1]
    start_y, end_y = cdf[0], cdf[-1]

    triangle_slope = (end_y - start_y) / (end_x - start_x)
    triangle_intercept = start_y - triangle_slope * start_x

    a, b, c = -triangle_slope, 1, -triangle_intercept

    distances = np.abs(a * x_cdf + b * cdf + c) / np.sqrt(a**2 + b**2)
    threshold_idx = np.argmax(distances)
    binary = x_cdf[threshold_idx]
    binary = binary / multiplier * minmax + data_min
    binary = binary + adjust * (data_2.max() - data_2.min())

    return binary

def spectral_entropy(data: np.ndarray) -> float:
    """Calculate spectral entropy of 2D data using FFT power spectrum."""
    fft = np.fft.fft2(data)
    power_spectrum = np.abs(fft) ** 2
    power_spectrum = power_spectrum.flatten()
    power_spectrum = power_spectrum / power_spectrum.sum()
    return entropy(power_spectrum[power_spectrum > 0])


def shannon_entropy(data: np.ndarray) -> float:
    """Calculate Shannon entropy using optimal thresholding (Otsu if threshold=None)."""
    threshold = get_threshold(data)
    binary = data > threshold
    vals, counts = np.unique(binary, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs)

def variance(data: np.ndarray) -> float:
    """Calculate variance of 2D data."""
    return float(np.var(data))

def total_variation(data: np.ndarray) -> float:
    """Calculate total variation (sum of gradient magnitudes) of 2D data."""
    dy = np.diff(data, axis=0)
    dx = np.diff(data, axis=1)
    return float(np.sum(np.abs(dy[:, :-1])) + np.sum(np.abs(dx[:-1, :])))