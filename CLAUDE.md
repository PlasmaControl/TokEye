# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TokEye is a plasma signal segmentation system for tokamak fusion physics research. It provides a complete pipeline for analyzing time-series signals from plasma diagnostics, detecting structures through deep learning, and visualizing results.

**Core Capabilities:**
- Signal preprocessing (preemphasis filtering)
- Time-frequency transforms (STFT, wavelet decomposition)
- Deep learning inference with UNet-based segmentation models
- Tiling/stitching for processing large spectrograms
- Post-processing and visualization overlays
- LRU-based caching for computational efficiency

## Development Environment

**Package Management:** This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

**Python Version:** 3.13+

### Setup Commands

```bash
# Install project in development mode
uv pip install -e .

# Install optional processing dependencies (required for signal processing)
uv pip install scipy opencv-python
```

## Running the Application

The primary interface is a Gradio-based web application with three tabs: Analyze, Annotate, and Utilities.

```bash
# Launch Gradio UI (default port 7860)
python -m TokEye.gradio

# Launch with specific port
python -m TokEye.gradio --port 8080

# Launch with public sharing enabled
python -m TokEye.gradio --share

# Launch and auto-open browser
python -m TokEye.gradio --open
```

**Port Handling:** The app automatically tries the next 10 ports if the default is unavailable.

**Required Directories:** The app auto-creates these on launch: `cache/`, `outputs/`, `annotations/`, `model/`, `data/`

## Architecture

### Processing Pipeline (`src/TokEye/processing/`)

The processing module provides the core signal analysis pipeline. All utilities are exposed via `__init__.py` for clean imports.

**Module Organization:**
- `transforms.py` - Signal transformations (preemphasis, STFT, wavelet)
- `tiling.py` - Spectrogram tiling for UNet processing and stitching results
- `inference.py` - PyTorch model loading and batch inference
- `postprocess.py` - Thresholding, object filtering, overlay visualization
- `cache.py` - LRU cache manager with size limits and type separation

**Standard Pipeline Flow:**
1. Load signal from `.npy` file
2. Apply preemphasis filter (typically α=0.97)
3. Compute STFT or wavelet spectrogram
4. Pad spectrogram to tile size (256x256)
5. Tile spectrogram with optional overlap
6. Run batch inference through PyTorch model
7. Stitch predictions back to full size
8. Apply threshold and remove small objects
9. Create visualization overlay

**Import Pattern:**
```python
from TokEye.processing import (
    apply_preemphasis, compute_stft, compute_wavelet,
    tile_spectrogram, stitch_predictions,
    load_model, batch_inference,
    apply_threshold, remove_small_objects, create_overlay,
    CacheManager, generate_cache_key
)
```

**Key Implementation Details:**
- STFT uses magnitude → log1p compression → normalization pipeline
- Wavelet uses PyWavelets WaveletPacket with strict freq ordering
- Tiling creates square tiles (height × height) along width dimension
- Last tile is zero-padded if width not evenly divisible
- Stitching blends overlapping regions by averaging
- Models can be TorchScript (.pt) or torch.export (.pt2)

### Gradio Application (`src/TokEye/gradio/`)

**Entry Point:** `__main__.py` defines `main()` function and handles app initialization

**Tab Structure:**
- `tabs/analyze.py` - Core analysis pipeline (signal → spectrogram → inference → visualization)
- `tabs/annotate.py` - Manual annotation tools for ground truth creation
- `tabs/utilities.py` - Additional utilities and tools

**UI Features:**
- Logo display from `assets/logo.png` if available
- Soft theme with custom CSS (full-width container, hidden footer)
- Auto-port fallback with logging
- Cache manager initialized globally per tab

### File Organization

**Source Code:**
```
src/TokEye/
├── processing/          # Core signal processing utilities
│   ├── __init__.py     # Public API exports
│   ├── transforms.py
│   ├── tiling.py
│   ├── inference.py
│   ├── postprocess.py
│   ├── cache.py
│   ├── README.md       # Full API reference
│   └── QUICKSTART.md   # Quick start guide
└── gradio/             # Web application
    ├── __main__.py     # App entry point
    └── tabs/           # UI tab definitions
        ├── analyze.py
        ├── annotate.py
        └── utilities.py
```

**Data Directories (all gitignored):**
- `cache/` - Cached spectrograms and inference results
- `data/` - Input signal files (.npy format)
- `model/` - PyTorch model files (.pt or .pt2)
- `outputs/` - Generated visualizations and results

**Other:**
- `assets/` - Static files (e.g., logo.png)
- `annotations/` - Manual annotations (not gitignored)

## Processing Module Details

### Signal Transforms

**Preemphasis:** Enhances high frequencies with filter: y[n] = x[n] - α·x[n-1]

**STFT:** Computed via scipy with normalization:
- Default: n_fft=1024, hop_length=128, window='hann'
- Pipeline: STFT → magnitude → log1p → clip DC → normalize (mean/std)
- Returns: (freq_bins, time_frames)

**Wavelet:** PyWavelets WaveletPacket decomposition:
- Default: wavelet='db8', level=9, mode='sym', order='freq'
- Returns: (2^level, coeffs_per_node) log-compressed coefficients

### Tiling System

The tiling system splits spectrograms into square tiles for UNet processing:

**Requirements:**
- Spectrogram height must equal tile_size (pad beforehand if needed)
- Width divided into height-sized squares
- Last tile zero-padded if width % tile_size != 0

**Metadata Tracking:**
- Original dimensions, tile parameters, padding amount
- Used by stitching to exactly reconstruct original size

**Overlap:** Optional overlap with blend averaging during stitching

### Model Inference

**Supported Formats:**
- TorchScript: `.pt` files loaded with `torch.jit.load()`
- torch.export: `.pt2` files loaded with `torch.export.load()`

**Device Selection:**
- `device='auto'` - Auto-detect CUDA availability
- `device='cuda'` or `device='cpu'` - Explicit selection
- `device='cuda:0'` - Specific GPU selection

**Batch Processing:**
- Processes tiles in configurable batch sizes
- Uses `torch.no_grad()` for memory efficiency
- Progress tracking optional

### Post-Processing

**Thresholding:** Converts model predictions to binary masks (default threshold=0.5)

**Object Filtering:** Uses OpenCV connected components analysis:
- Removes objects smaller than min_size pixels
- Returns cleaned mask and object count
- Supports 4 or 8-connectivity

**Visualization Modes:**
- `'white'` - Simple white overlay on grayscale spectrogram
- `'bicolor'` - Blue/green classification (coherent/transient)
- `'hsv'` - Unique color per connected component

### Caching System

**LRU Cache Manager:**
- Size limits: max_size_mb and max_entries
- Type separation: 'spectrogram', 'inference', 'wavelet', 'general'
- Disk persistence with metadata tracking
- Optional compression

**Cache Key Generation:** SHA256 hash of data + parameters

## Common Development Patterns

### Processing Single Signal

```python
import numpy as np
from TokEye.processing import *

# Load and process
signal = np.load('data/plasma_signal.npy')
emphasized = apply_preemphasis(signal, alpha=0.97)
spec = compute_stft(emphasized, n_fft=1024, hop_length=128)

# Pad to tile size
tile_size = 256
if spec.shape[0] != tile_size:
    spec = np.pad(spec, ((0, tile_size - spec.shape[0]), (0, 0)), mode='constant')

# Tile, infer, stitch
tiles, meta = tile_spectrogram(spec, tile_size=tile_size)
model = load_model('model/model.pt', device='auto')
preds = batch_inference(model, tiles, batch_size=32)
full_pred = stitch_predictions(preds, meta)

# Post-process
mask = apply_threshold(full_pred, threshold=0.5)
clean_mask, n_objects = remove_small_objects(mask, min_size=50)
overlay = create_overlay(spec, clean_mask, mode='hsv', alpha=0.6)
```

### Using Cache for Performance

```python
from TokEye.processing import CacheManager, generate_cache_key, compute_stft

cache = CacheManager(cache_dir='cache', max_size_mb=1000)

# Generate cache key from signal + parameters
params = {'n_fft': 1024, 'hop_length': 128}
key = generate_cache_key(signal, params, prefix='stft')

# Check cache before computing
if cache.exists(key, 'spectrogram'):
    spec = cache.load(key, 'spectrogram')
else:
    spec = compute_stft(signal, **params)
    cache.save(key, spec, cache_type='spectrogram')
```

### Batch Processing Multiple Signals

```python
# Collect all tiles from multiple spectrograms
all_tiles = []
all_metadata = []

for spec in spectrograms:
    tiles, meta = tile_spectrogram(spec, tile_size=256)
    all_tiles.extend(tiles)
    all_metadata.append(meta)

# Single batch inference call
model = load_model('model.pt', device='cuda')
all_preds = batch_inference(model, all_tiles, batch_size=64)

# Stitch each spectrogram separately
idx = 0
for meta in all_metadata:
    preds = all_preds[idx:idx + meta['num_tiles']]
    result = stitch_predictions(preds, meta)
    idx += meta['num_tiles']
```

## Important Notes

- **Height Constraint:** Spectrograms must be padded to tile_size before tiling
- **Model Location:** Place models in `model/` directory (gitignored)
- **Signal Format:** Input signals must be 1D numpy arrays saved as `.npy`
- **Device Memory:** Reduce batch_size if CUDA OOM occurs
- **Cache Management:** Cache automatically evicts LRU entries when limits exceeded

## Documentation References

- Full API reference: [src/TokEye/processing/README.md](src/TokEye/processing/README.md)
- Quick start guide: [src/TokEye/processing/QUICKSTART.md](src/TokEye/processing/QUICKSTART.md)
