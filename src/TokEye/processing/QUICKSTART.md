# TokEye Processing - Quick Start Guide

## Installation

```bash
# Install dependencies
cd c:/Users/twoga/Documents/GitHub/PlasmaControlGroup/TokEye
uv pip install scipy opencv-python
```

## Basic Usage

### 1. Signal to Spectrogram

```python
import numpy as np
from TokEye.processing import apply_preemphasis, compute_stft

# Load your signal
signal = np.load('plasma_signal.npy')  # shape: (N,)

# Apply preemphasis
emphasized = apply_preemphasis(signal, alpha=0.97)

# Compute STFT spectrogram
spectrogram = compute_stft(
    emphasized,
    n_fft=1024,
    hop_length=128,
    window='hann',
    clip_dc=True,
    fs=100000,  # 100 kHz sampling rate
)
# Output shape: (freq_bins, time_frames)
```

### 2. Wavelet Decomposition

```python
from TokEye.processing import compute_wavelet

# Compute wavelet decomposition
coeffs = compute_wavelet(
    signal,
    wavelet='db8',
    level=9,
    mode='sym',
    order='freq',
)
# Output shape: (512, coeffs_per_node) where 512 = 2^9
```

### 3. Model Inference Pipeline

```python
from TokEye.processing import (
    tile_spectrogram,
    load_model,
    batch_inference,
    stitch_predictions,
)

# Pad spectrogram to tile size (256x256)
import numpy as np
if spectrogram.shape[0] != 256:
    pad_height = 256 - spectrogram.shape[0]
    spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), mode='constant')

# Tile the spectrogram
tiles, metadata = tile_spectrogram(spectrogram, tile_size=256, overlap=32)

# Load model
model = load_model('path/to/model.pt', device='auto')

# Run inference
predictions = batch_inference(model, tiles, batch_size=32, show_progress=True)

# Stitch back together
full_prediction = stitch_predictions(predictions, metadata, blend_overlap=True)
```

### 4. Post-Processing

```python
from TokEye.processing import (
    apply_threshold,
    remove_small_objects,
    compute_statistics,
    create_overlay,
)

# Threshold predictions
binary_mask = apply_threshold(full_prediction, threshold=0.5)

# Remove small objects
cleaned_mask, num_objects = remove_small_objects(binary_mask, min_size=50)

# Get statistics
stats = compute_statistics(cleaned_mask)
print(f"Found {stats['num_objects']} objects")
print(f"Mean area: {stats['mean_area']:.1f} pixels")
print(f"Coverage: {stats['coverage']*100:.2f}%")

# Create visualization
overlay = create_overlay(
    spectrogram,
    cleaned_mask,
    mode='hsv',  # or 'white', 'bicolor'
    alpha=0.6,
)
# overlay is RGB image, shape: (H, W, 3)
```

### 5. Caching for Performance

```python
from TokEye.processing import CacheManager, generate_cache_key

# Initialize cache
cache = CacheManager(cache_dir='.cache', max_size_mb=1000)

# Generate key
params = {'n_fft': 1024, 'hop_length': 128}
key = generate_cache_key(signal, params, prefix='stft')

# Check cache
if cache.exists(key, 'spectrogram'):
    spectrogram = cache.load(key, 'spectrogram')
    print("Loaded from cache")
else:
    spectrogram = compute_stft(signal, **params)
    cache.save(key, spectrogram, cache_type='spectrogram')
    print("Computed and cached")

# View cache stats
stats = cache.get_statistics()
print(f"Cache: {stats['num_entries']} entries, {stats['total_size_mb']:.2f} MB")
```

## Complete Pipeline Example

```python
import numpy as np
from TokEye.processing import *

# 1. Load signal
signal = np.load('plasma_signal.npy')

# 2. Preprocess
emphasized = apply_preemphasis(signal, alpha=0.97)
spectrogram = compute_stft(emphasized, n_fft=1024, hop_length=128)

# 3. Pad to tile size
tile_size = 256
if spectrogram.shape[0] != tile_size:
    pad_height = tile_size - spectrogram.shape[0]
    spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), mode='constant')

# 4. Tile
tiles, metadata = tile_spectrogram(spectrogram, tile_size=tile_size)

# 5. Infer
model = load_model('model.pt', device='auto')
predictions = batch_inference(model, tiles, batch_size=32)
full_prediction = stitch_predictions(predictions, metadata)

# 6. Post-process
binary_mask = apply_threshold(full_prediction, threshold=0.5)
cleaned_mask, num_objects = remove_small_objects(binary_mask, min_size=50)

# 7. Visualize
overlay = create_overlay(spectrogram, cleaned_mask, mode='hsv', alpha=0.6)

# 8. Get results
stats = compute_statistics(cleaned_mask)
print(f"Detected {num_objects} plasma structures")
print(f"Coverage: {stats['coverage']*100:.2f}%")

# Save results
import matplotlib.pyplot as plt
plt.imsave('detection_overlay.png', overlay)
```

## Function Quick Reference

### Transforms
- `apply_preemphasis(signal, alpha=0.97)` - Enhance high frequencies
- `compute_stft(signal, n_fft=1024, hop_length=128, ...)` - STFT spectrogram
- `compute_wavelet(signal, wavelet='db8', level=9, ...)` - Wavelet decomposition

### Tiling
- `tile_spectrogram(spec, tile_size, overlap=0)` - Split into tiles
- `stitch_predictions(tiles, metadata, blend_overlap=True)` - Reconstruct

### Inference
- `load_model(path, device='auto')` - Load TorchScript model
- `batch_inference(model, tiles, batch_size=32)` - Batch process tiles

### Post-processing
- `apply_threshold(pred, threshold=0.5)` - Binary threshold
- `remove_small_objects(mask, min_size=50)` - Filter by size
- `create_overlay(spec, mask, mode='white', alpha=0.5)` - Visualize
- `compute_statistics(mask, min_size=0)` - Get detection stats

### Caching
- `generate_cache_key(data, params, prefix='')` - Create cache key
- `CacheManager(cache_dir, max_size_mb=1000)` - Create cache
  - `.save(key, data, cache_type)` - Save to cache
  - `.load(key, cache_type)` - Load from cache
  - `.exists(key, cache_type)` - Check existence
  - `.clear(cache_type=None)` - Clear cache

## Common Patterns

### Pattern 1: Process Multiple Signals

```python
import glob
from pathlib import Path

for signal_path in glob.glob('data/*.npy'):
    signal = np.load(signal_path)

    # Process...
    spectrogram = compute_stft(apply_preemphasis(signal))

    # Save
    output_path = Path(signal_path).stem + '_spec.npy'
    np.save(output_path, spectrogram)
```

### Pattern 2: Cached Processing

```python
def process_with_cache(signal, cache):
    key = generate_cache_key(signal, {}, prefix='processed')

    if cache.exists(key, 'inference'):
        return cache.load(key, 'inference')

    # Process pipeline...
    result = full_pipeline(signal)

    cache.save(key, result, cache_type='inference')
    return result
```

### Pattern 3: GPU Batch Processing

```python
# Process multiple spectrograms
all_tiles = []
all_metadata = []

for spec in spectrograms:
    tiles, meta = tile_spectrogram(spec, tile_size=256)
    all_tiles.extend(tiles)
    all_metadata.append(meta)

# Single batch inference
model = load_model('model.pt', device='cuda')
all_predictions = batch_inference(model, all_tiles, batch_size=64)

# Stitch each spectrogram separately
start_idx = 0
for meta in all_metadata:
    end_idx = start_idx + meta['num_tiles']
    predictions = all_predictions[start_idx:end_idx]
    result = stitch_predictions(predictions, meta)
    start_idx = end_idx
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch_size in batch_inference()

### Issue: Spectrogram height doesn't match tile_size
**Solution**: Pad before tiling:
```python
if spec.shape[0] != tile_size:
    pad = tile_size - spec.shape[0]
    spec = np.pad(spec, ((0, pad), (0, 0)), mode='constant')
```

### Issue: Import error for cv2
**Solution**: Install OpenCV:
```bash
uv pip install opencv-python
```

### Issue: Import error for scipy
**Solution**: Install scipy:
```bash
uv pip install scipy
```

## Performance Tips

1. **Use GPU**: Set `device='cuda'` in load_model()
2. **Batch Processing**: Use larger batch_size if GPU memory allows
3. **Enable Caching**: Cache expensive computations (STFT, wavelet)
4. **Overlap**: Use overlap=0 if blending not needed
5. **Warmup**: Use warmup_model() before batch processing

## For More Information

- Full API docs: `src/TokEye/processing/README.md`
- Demo script: `examples/processing_demo.py`
- Implementation details: `PROCESSING_SUMMARY.md`
