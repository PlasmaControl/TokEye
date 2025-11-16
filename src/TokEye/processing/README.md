# TokEye Processing Utilities

Comprehensive signal processing, model inference, and visualization utilities for the TokEye plasma disruption detection system.

## Overview

This module provides a complete pipeline for processing plasma signals, from raw time-domain data to visualized predictions:

1. **Signal Transformations** - Preemphasis, STFT, Wavelet decomposition
2. **Tiling/Stitching** - Split spectrograms for UNet processing
3. **Model Inference** - Batch processing with PyTorch models
4. **Post-processing** - Thresholding, object filtering, visualization
5. **Caching** - LRU-based caching system for performance

## Module Structure

```
processing/
├── __init__.py          # Main exports
├── transforms.py        # Signal processing transformations
├── tiling.py           # UNet tiling and stitching
├── inference.py        # Model loading and batch inference
├── postprocess.py      # Visualization and post-processing
├── cache.py            # Caching system with LRU eviction
└── README.md           # This file
```

## Quick Start

### Basic Signal Processing Pipeline

```python
import numpy as np
from TokEye.processing import (
    apply_preemphasis,
    compute_stft,
    compute_wavelet,
)

# Load or generate signal
signal = np.random.randn(100000)  # Example signal

# Apply preemphasis filter
emphasized = apply_preemphasis(signal, alpha=0.97)

# Compute STFT spectrogram
spectrogram = compute_stft(
    emphasized,
    n_fft=1024,
    hop_length=128,
    window='hann',
    clip_dc=True,
)

# Compute wavelet decomposition
wavelet_coeffs = compute_wavelet(
    signal,
    wavelet='db8',
    level=9,
    mode='sym',
    order='freq',
)

print(f"Spectrogram shape: {spectrogram.shape}")
print(f"Wavelet coeffs shape: {wavelet_coeffs.shape}")
```

### Complete Inference Pipeline

```python
import numpy as np
from TokEye.processing import (
    compute_stft,
    tile_spectrogram,
    load_model,
    batch_inference,
    stitch_predictions,
    apply_threshold,
    remove_small_objects,
    create_overlay,
)

# 1. Create spectrogram
signal = np.random.randn(100000)
spectrogram = compute_stft(signal, n_fft=1024, hop_length=128)

# 2. Tile for UNet processing
tile_size = 256
tiles, metadata = tile_spectrogram(spectrogram, tile_size=tile_size)
print(f"Created {len(tiles)} tiles")

# 3. Load model and run inference
model = load_model('model.pt', device='auto')
predictions = batch_inference(model, tiles, batch_size=32)

# 4. Stitch predictions back together
full_prediction = stitch_predictions(predictions, metadata)

# 5. Post-process
binary_mask = apply_threshold(full_prediction, threshold=0.5)
cleaned_mask, num_objects = remove_small_objects(binary_mask, min_size=50)

# 6. Create visualization
overlay = create_overlay(
    spectrogram,
    cleaned_mask,
    mode='white',
    alpha=0.5,
)

print(f"Detected {num_objects} objects")
```

### Using the Cache System

```python
from TokEye.processing import CacheManager, generate_cache_key
import numpy as np

# Initialize cache manager
cache = CacheManager(
    cache_dir='.cache',
    max_size_mb=1000,
    max_entries=1000,
)

# Generate cache key
signal = np.random.randn(10000)
params = {'n_fft': 1024, 'hop_length': 128}
key = generate_cache_key(signal, params, prefix='stft')

# Check if cached
if cache.exists(key, 'spectrogram'):
    spectrogram = cache.load(key, 'spectrogram')
    print("Loaded from cache")
else:
    # Compute and cache
    from TokEye.processing import compute_stft
    spectrogram = compute_stft(signal, **params)
    cache.save(key, spectrogram, cache_type='spectrogram')
    print("Computed and cached")

# Get cache statistics
stats = cache.get_statistics()
print(f"Cache size: {stats['total_size_mb']:.2f} MB")
print(f"Cache entries: {stats['num_entries']}")
```

## API Reference

### transforms.py

#### `apply_preemphasis(signal, alpha=0.97)`
Apply preemphasis filter to enhance high frequencies.

**Parameters:**
- `signal`: Input signal (1D or 2D array)
- `alpha`: Preemphasis coefficient (0-1)

**Returns:** Preemphasized signal

---

#### `compute_stft(signal, n_fft=1024, hop_length=128, window='hann', clip_dc=True, fs=1.0)`
Compute normalized STFT spectrogram.

**Parameters:**
- `signal`: Input time-domain signal (1D)
- `n_fft`: FFT size
- `hop_length`: Hop length between frames
- `window`: Window function name
- `clip_dc`: Remove DC component
- `fs`: Sampling frequency

**Returns:** Normalized STFT magnitude spectrogram (freq_bins, time_frames)

**Processing Pipeline:**
1. Compute STFT
2. Take magnitude
3. Apply log1p compression
4. Optional DC clipping
5. Mean-std normalization

---

#### `compute_wavelet(signal, wavelet='db8', level=9, mode='sym', order='freq')`
Compute wavelet packet decomposition.

**Parameters:**
- `signal`: Input signal (1D)
- `wavelet`: Wavelet name ('db8', 'db4', 'haar', etc.)
- `level`: Decomposition level
- `mode`: Signal extension mode
- `order`: Node ordering ('natural' or 'freq')

**Returns:** 2D array of wavelet coefficients (2^level, coeffs_per_node)

**Implementation:**
```python
wp = pywt.WaveletPacket(signal, wavelet, mode, maxlevel=level)
nodes = wp.get_level(level, order=order)
values = np.array([n.data for n in nodes], 'd')
values = np.log1p(np.abs(values))
```

---

### tiling.py

#### `tile_spectrogram(spectrogram, tile_size, overlap=0)`
Split spectrogram into square tiles.

**Parameters:**
- `spectrogram`: Input spectrogram (H, W) or (C, H, W)
- `tile_size`: Size of square tiles (must match height)
- `overlap`: Overlap between tiles in pixels

**Returns:** Tuple of (tiles_list, metadata_dict)

**Metadata Keys:**
- `original_width`, `original_height`: Original dimensions
- `tile_size`, `overlap`, `stride`: Tiling parameters
- `num_tiles`: Number of tiles created
- `padding`: Padding added to last tile
- `has_channels`, `num_channels`: Channel information

---

#### `stitch_predictions(tiles, metadata, blend_overlap=True)`
Reconstruct full prediction from tiles.

**Parameters:**
- `tiles`: List of prediction tiles
- `metadata`: Metadata from tile_spectrogram()
- `blend_overlap`: Average overlapping regions

**Returns:** Reconstructed full array with padding removed

---

### inference.py

#### `load_model(model_path, device='auto', map_location=None)`
Load TorchScript model for inference.

**Parameters:**
- `model_path`: Path to .pt model file
- `device`: Target device ('auto', 'cuda', 'cpu', 'cuda:0', etc.)
- `map_location`: Optional map location for loading

**Returns:** Loaded model in eval mode

---

#### `batch_inference(model, tiles, batch_size=32, device=None, show_progress=False)`
Run batch inference on tiles.

**Parameters:**
- `model`: PyTorch model
- `tiles`: List of input tiles (numpy arrays)
- `batch_size`: Batch size for processing
- `device`: Device for inference (None = use model device)
- `show_progress`: Print progress information

**Returns:** List of predictions (numpy arrays)

---

### postprocess.py

#### `apply_threshold(prediction, threshold=0.5, binary=True)`
Apply threshold to prediction.

**Parameters:**
- `prediction`: Input prediction array
- `threshold`: Threshold value
- `binary`: Return binary mask (0/1) vs thresholded values

**Returns:** Thresholded array

---

#### `remove_small_objects(mask, min_size=50, connectivity=8)`
Remove small connected components.

**Parameters:**
- `mask`: Binary mask
- `min_size`: Minimum object size in pixels
- `connectivity`: Connectivity (4 or 8)

**Returns:** Tuple of (cleaned_mask, num_objects)

**Uses:** OpenCV's `connectedComponentsWithStats`

---

#### `create_overlay(spectrogram, mask, mode='white', alpha=0.5, coherent_color=(0,0,255), transient_color=(0,255,0))`
Create visualization overlay.

**Parameters:**
- `spectrogram`: Base spectrogram (2D)
- `mask`: Binary or labeled mask
- `mode`: Visualization mode
  - `'white'`: Simple white overlay
  - `'bicolor'`: Blue=coherent, green=transient
  - `'hsv'`: Unique colors per component
- `alpha`: Overlay transparency (0-1)
- `coherent_color`, `transient_color`: BGR colors for bicolor mode

**Returns:** RGB image (H, W, 3) uint8

---

#### `compute_statistics(mask, min_size=0)`
Compute object statistics.

**Parameters:**
- `mask`: Binary mask
- `min_size`: Minimum object size to include

**Returns:** Dictionary with statistics:
- `num_objects`: Total object count
- `total_area`, `mean_area`, `median_area`, `min_area`, `max_area`
- `coverage`: Fraction of image covered

---

### cache.py

#### `generate_cache_key(data, params, prefix='')`
Generate unique cache key.

**Parameters:**
- `data`: Input numpy array
- `params`: Parameters dictionary
- `prefix`: Optional key prefix

**Returns:** Unique cache key string

**Method:** SHA256 hash of data + parameters

---

#### `CacheManager(cache_dir='.cache', max_size_mb=1000, max_entries=1000, enable_compression=True)`
Cache manager with LRU eviction.

**Methods:**

- `save(key, data, cache_type='general')`: Save data to cache
- `load(key, cache_type='general')`: Load data from cache
- `exists(key, cache_type='general')`: Check if entry exists
- `delete(key, cache_type='general')`: Delete cache entry
- `clear(cache_type=None)`: Clear cache (all or by type)
- `get_statistics()`: Get cache statistics

**Cache Types:**
- `'spectrogram'`: STFT/wavelet spectrograms
- `'inference'`: Model predictions
- `'wavelet'`: Wavelet decompositions
- `'general'`: General purpose cache

---

## Implementation Notes

### Signal Processing

1. **Preemphasis Filter**
   - Implements: y[n] = x[n] - α * x[n-1]
   - Typical α: 0.95-0.97
   - Enhances high-frequency components

2. **STFT Pipeline**
   - Uses scipy.signal.stft for computation
   - Returns: magnitude → log1p → DC clip → normalize
   - Normalization: (x - mean) / std

3. **Wavelet Transform**
   - Strict implementation per TokEye spec
   - Uses PyWavelets WaveletPacket
   - Returns log-compressed coefficients

### Tiling Strategy

- **Square Tiles**: Width is divided into height-sized squares
- **Padding**: Last tile padded if width not evenly divisible
- **Overlap**: Optional overlap with averaging during stitching
- **Metadata**: Comprehensive metadata for exact reconstruction

### Inference Optimization

- **Batch Processing**: Tiles processed in configurable batches
- **Device Management**: Auto-detect CUDA availability
- **Memory Efficiency**: Uses torch.no_grad() context
- **Model Warmup**: Optional warmup for CUDA kernel compilation

### Post-processing

- **Connected Components**: Uses OpenCV for analysis
- **Object Filtering**: Size-based filtering with statistics
- **Visualization Modes**:
  - White: Simple binary overlay
  - Bicolor: Classify by size heuristic
  - HSV: Unique color per component

### Caching System

- **LRU Eviction**: Least recently used entries removed first
- **Size Limits**: Both total size (MB) and entry count
- **Type Separation**: Different caches for different data types
- **Persistence**: Metadata saved to disk for recovery
- **Compression**: Optional pickle compression

## Error Handling

All functions include:
- Input validation with informative error messages
- Type checking for arrays and parameters
- Graceful handling of edge cases
- Warnings for potential issues
- Import guards for optional dependencies

## Performance Considerations

1. **STFT Computation**: O(N log N) for FFT
2. **Wavelet Transform**: O(N) for WaveletPacket
3. **Tiling**: O(N) memory allocation
4. **Inference**: GPU batch processing for speed
5. **Caching**: Constant-time lookup with LRU overhead

## Dependencies

- **Required**: numpy, pywavelets, torch
- **Optional**: scipy (STFT), opencv-python (post-processing)

Install all dependencies:
```bash
uv pip install scipy opencv-python
```

## Testing

Each module includes extensive docstrings with examples. Recommended testing:

1. **Unit Tests**: Test each function independently
2. **Integration Tests**: Test complete pipeline
3. **Edge Cases**: Empty arrays, single-element arrays, extreme parameters
4. **Performance Tests**: Benchmark critical operations
5. **Round-trip Tests**: Validate tiling/stitching reconstruction

## Future Enhancements

- [ ] GPU-accelerated STFT using PyTorch
- [ ] Parallel processing for tiling
- [ ] Advanced visualization modes
- [ ] Configurable cache eviction policies
- [ ] Distributed caching support
- [ ] Model ensemble inference
- [ ] Real-time streaming support
