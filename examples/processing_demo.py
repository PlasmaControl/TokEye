"""
TokEye Processing Pipeline Demo

This script demonstrates the complete signal processing pipeline
for the TokEye plasma disruption detection system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import TokEye processing modules
from TokEye.processing import (
    # Transforms
    apply_preemphasis,
    compute_stft,
    compute_wavelet,
    # Tiling
    tile_spectrogram,
    stitch_predictions,
    # Inference (commented out - requires model)
    # load_model,
    # batch_inference,
    # Post-processing
    apply_threshold,
    remove_small_objects,
    create_overlay,
    compute_statistics,
    # Caching
    CacheManager,
    generate_cache_key,
)


def demo_signal_transforms():
    """Demonstrate signal transformation functions."""
    print("=" * 60)
    print("SIGNAL TRANSFORMATION DEMO")
    print("=" * 60)

    # Generate synthetic plasma signal
    # Simulate: background + coherent mode + transient events
    fs = 100000  # 100 kHz sampling rate
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(fs * duration))

    # Background noise
    signal = np.random.randn(len(t)) * 0.1

    # Add coherent mode (low frequency oscillation)
    signal += 0.5 * np.sin(2 * np.pi * 1000 * t)

    # Add transient event
    pulse_start = int(0.5 * fs)
    pulse_duration = int(0.01 * fs)
    signal[pulse_start:pulse_start + pulse_duration] += 2.0

    print(f"\n1. Signal Statistics:")
    print(f"   Length: {len(signal)} samples")
    print(f"   Duration: {duration} seconds")
    print(f"   Sampling rate: {fs} Hz")
    print(f"   Mean: {np.mean(signal):.4f}")
    print(f"   Std: {np.std(signal):.4f}")

    # Apply preemphasis
    print(f"\n2. Applying Preemphasis Filter...")
    emphasized = apply_preemphasis(signal, alpha=0.97)
    print(f"   Output shape: {emphasized.shape}")
    print(f"   Mean: {np.mean(emphasized):.4f}")
    print(f"   Std: {np.std(emphasized):.4f}")

    # Compute STFT
    print(f"\n3. Computing STFT...")
    spectrogram = compute_stft(
        emphasized,
        n_fft=1024,
        hop_length=128,
        window='hann',
        clip_dc=True,
        fs=fs,
    )
    print(f"   Spectrogram shape: {spectrogram.shape}")
    print(f"   Frequency bins: {spectrogram.shape[0]}")
    print(f"   Time frames: {spectrogram.shape[1]}")

    # Compute wavelet decomposition
    print(f"\n4. Computing Wavelet Decomposition...")
    wavelet_coeffs = compute_wavelet(
        signal,
        wavelet='db8',
        level=9,
        mode='sym',
        order='freq',
    )
    print(f"   Wavelet coeffs shape: {wavelet_coeffs.shape}")
    print(f"   Number of nodes: {wavelet_coeffs.shape[0]} (2^9 = 512)")
    print(f"   Coefficients per node: {wavelet_coeffs.shape[1]}")

    return signal, emphasized, spectrogram, wavelet_coeffs


def demo_tiling_pipeline(spectrogram):
    """Demonstrate tiling and stitching pipeline."""
    print("\n" + "=" * 60)
    print("TILING & STITCHING DEMO")
    print("=" * 60)

    tile_size = 256
    overlap = 32

    print(f"\n1. Tiling Spectrogram:")
    print(f"   Input shape: {spectrogram.shape}")
    print(f"   Tile size: {tile_size}x{tile_size}")
    print(f"   Overlap: {overlap} pixels")

    # Pad spectrogram to match tile size
    if spectrogram.shape[0] != tile_size:
        pad_height = tile_size - spectrogram.shape[0]
        spectrogram_padded = np.pad(
            spectrogram,
            ((0, pad_height), (0, 0)),
            mode='constant',
        )
        print(f"   Padded to: {spectrogram_padded.shape}")
    else:
        spectrogram_padded = spectrogram

    # Tile the spectrogram
    tiles, metadata = tile_spectrogram(
        spectrogram_padded,
        tile_size=tile_size,
        overlap=overlap,
    )

    print(f"\n2. Tiling Results:")
    print(f"   Number of tiles: {len(tiles)}")
    print(f"   Tile shape: {tiles[0].shape}")
    print(f"   Total padding: {metadata['padding']} pixels")
    print(f"   Stride: {metadata['stride']} pixels")

    # Simulate inference (just use tiles as-is for demo)
    print(f"\n3. Simulating Inference...")
    predictions = tiles.copy()  # In reality, run through model
    print(f"   Processed {len(predictions)} tiles")

    # Stitch predictions
    print(f"\n4. Stitching Predictions:")
    reconstructed = stitch_predictions(
        predictions,
        metadata,
        blend_overlap=True,
    )
    print(f"   Reconstructed shape: {reconstructed.shape}")

    # Verify reconstruction
    max_diff = np.max(np.abs(spectrogram_padded - reconstructed))
    print(f"   Reconstruction error: {max_diff:.2e}")

    if max_diff < 1e-6:
        print(f"   ✓ Perfect reconstruction!")
    else:
        print(f"   ✗ Reconstruction has errors")

    return tiles, metadata, reconstructed


def demo_postprocessing(spectrogram):
    """Demonstrate post-processing pipeline."""
    print("\n" + "=" * 60)
    print("POST-PROCESSING DEMO")
    print("=" * 60)

    # Create synthetic prediction (simulate model output)
    # Add some "detected" regions
    prediction = np.random.rand(*spectrogram.shape) * 0.3

    # Add high-confidence regions (simulate detections)
    h, w = spectrogram.shape

    # Large coherent structure
    prediction[h//4:h//2, w//4:w//2] = 0.9

    # Small transient events
    prediction[h//3:h//3+20, w//3:w//3+20] = 0.85
    prediction[2*h//3:2*h//3+10, 2*w//3:2*w//3+10] = 0.75

    print(f"\n1. Prediction Statistics:")
    print(f"   Shape: {prediction.shape}")
    print(f"   Min: {prediction.min():.3f}")
    print(f"   Max: {prediction.max():.3f}")
    print(f"   Mean: {prediction.mean():.3f}")

    # Apply threshold
    print(f"\n2. Applying Threshold (0.5):")
    binary_mask = apply_threshold(prediction, threshold=0.5)
    num_positive = np.sum(binary_mask)
    print(f"   Positive pixels: {num_positive}")
    print(f"   Coverage: {num_positive / binary_mask.size * 100:.2f}%")

    # Remove small objects
    print(f"\n3. Removing Small Objects (<50 pixels):")
    cleaned_mask, num_objects = remove_small_objects(
        binary_mask,
        min_size=50,
    )
    num_cleaned = np.sum(cleaned_mask)
    removed = num_positive - num_cleaned
    print(f"   Objects found: {num_objects}")
    print(f"   Pixels removed: {removed}")
    print(f"   Final coverage: {num_cleaned / cleaned_mask.size * 100:.2f}%")

    # Compute statistics
    print(f"\n4. Object Statistics:")
    stats = compute_statistics(cleaned_mask, min_size=0)
    print(f"   Number of objects: {stats['num_objects']}")
    print(f"   Total area: {stats['total_area']} pixels")
    print(f"   Mean area: {stats['mean_area']:.1f} pixels")
    print(f"   Median area: {stats['median_area']:.1f} pixels")
    print(f"   Min area: {stats['min_area']} pixels")
    print(f"   Max area: {stats['max_area']} pixels")

    # Create overlays
    print(f"\n5. Creating Visualization Overlays:")

    try:
        overlay_white = create_overlay(
            spectrogram,
            cleaned_mask,
            mode='white',
            alpha=0.5,
        )
        print(f"   White overlay: {overlay_white.shape}")

        overlay_hsv = create_overlay(
            spectrogram,
            cleaned_mask,
            mode='hsv',
            alpha=0.6,
        )
        print(f"   HSV overlay: {overlay_hsv.shape}")

        return prediction, binary_mask, cleaned_mask, overlay_white, overlay_hsv
    except ImportError as e:
        print(f"   ✗ Overlay creation requires OpenCV: {e}")
        return prediction, binary_mask, cleaned_mask, None, None


def demo_caching():
    """Demonstrate caching system."""
    print("\n" + "=" * 60)
    print("CACHING SYSTEM DEMO")
    print("=" * 60)

    # Initialize cache manager
    cache_dir = Path('.cache_demo')
    cache = CacheManager(
        cache_dir=str(cache_dir),
        max_size_mb=100,
        max_entries=50,
    )

    print(f"\n1. Cache Configuration:")
    print(f"   Cache directory: {cache_dir}")
    print(f"   Max size: 100 MB")
    print(f"   Max entries: 50")

    # Generate test data
    signal = np.random.randn(10000)
    params = {
        'n_fft': 1024,
        'hop_length': 128,
        'window': 'hann',
    }

    # Generate cache key
    print(f"\n2. Generating Cache Key:")
    key = generate_cache_key(signal, params, prefix='stft')
    print(f"   Key: {key}")

    # Test cache miss
    print(f"\n3. Testing Cache Miss:")
    if cache.exists(key, 'spectrogram'):
        print(f"   ✗ Entry exists (unexpected)")
    else:
        print(f"   ✓ Entry does not exist (expected)")

    # Save to cache
    print(f"\n4. Saving to Cache:")
    test_data = np.random.randn(256, 256)
    cache.save(key, test_data, cache_type='spectrogram')
    print(f"   ✓ Data saved")

    # Test cache hit
    print(f"\n5. Testing Cache Hit:")
    if cache.exists(key, 'spectrogram'):
        print(f"   ✓ Entry exists (expected)")
        loaded_data = cache.load(key, 'spectrogram')
        if np.allclose(test_data, loaded_data):
            print(f"   ✓ Data matches (perfect)")
        else:
            print(f"   ✗ Data mismatch")
    else:
        print(f"   ✗ Entry does not exist (unexpected)")

    # Cache statistics
    print(f"\n6. Cache Statistics:")
    stats = cache.get_statistics()
    print(f"   Entries: {stats['num_entries']}")
    print(f"   Size: {stats['total_size_mb']:.3f} MB")
    print(f"   Entries by type: {stats['entries_by_type']}")

    # Clean up
    print(f"\n7. Cleaning Up:")
    cache.clear()
    print(f"   ✓ Cache cleared")

    # Remove cache directory
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"   ✓ Cache directory removed")


def demo_complete_pipeline():
    """Demonstrate complete end-to-end pipeline."""
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE DEMO")
    print("=" * 60)

    # 1. Generate signal
    print("\nStep 1: Signal Generation")
    fs = 100000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.random.randn(len(t)) * 0.1
    signal += 0.5 * np.sin(2 * np.pi * 2000 * t)
    print(f"  ✓ Generated signal: {len(signal)} samples")

    # 2. Preemphasis
    print("\nStep 2: Preemphasis")
    emphasized = apply_preemphasis(signal, alpha=0.97)
    print(f"  ✓ Applied preemphasis filter")

    # 3. STFT
    print("\nStep 3: STFT Computation")
    spectrogram = compute_stft(
        emphasized,
        n_fft=512,
        hop_length=64,
        fs=fs,
    )
    print(f"  ✓ Computed STFT: {spectrogram.shape}")

    # 4. Pad and tile
    print("\nStep 4: Tiling")
    tile_size = 256

    # Pad to tile size
    if spectrogram.shape[0] != tile_size:
        pad_height = tile_size - spectrogram.shape[0]
        spectrogram_padded = np.pad(
            spectrogram,
            ((0, pad_height), (0, 0)),
            mode='constant',
        )
    else:
        spectrogram_padded = spectrogram

    tiles, metadata = tile_spectrogram(spectrogram_padded, tile_size=tile_size)
    print(f"  ✓ Created {len(tiles)} tiles")

    # 5. Inference (simulated)
    print("\nStep 5: Inference (Simulated)")
    # In real usage: predictions = batch_inference(model, tiles)
    predictions = [t * 0.5 for t in tiles]  # Dummy predictions
    print(f"  ✓ Processed {len(predictions)} tiles")

    # 6. Stitch
    print("\nStep 6: Stitching")
    full_prediction = stitch_predictions(predictions, metadata)
    print(f"  ✓ Stitched to {full_prediction.shape}")

    # 7. Post-processing
    print("\nStep 7: Post-processing")
    binary_mask = apply_threshold(full_prediction, threshold=0.3)
    cleaned_mask, num_objects = remove_small_objects(binary_mask, min_size=30)
    print(f"  ✓ Detected {num_objects} objects")

    # 8. Statistics
    print("\nStep 8: Statistics")
    stats = compute_statistics(cleaned_mask)
    print(f"  ✓ Mean object area: {stats['mean_area']:.1f} pixels")
    print(f"  ✓ Coverage: {stats['coverage']*100:.2f}%")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    print("\nTokEye Processing Pipeline Demonstration\n")

    # Run individual demos
    signal, emphasized, spectrogram, wavelet_coeffs = demo_signal_transforms()
    tiles, metadata, reconstructed = demo_tiling_pipeline(spectrogram)

    try:
        results = demo_postprocessing(spectrogram)
        if results is not None:
            prediction, binary_mask, cleaned_mask, overlay_white, overlay_hsv = results
    except Exception as e:
        print(f"\nPost-processing demo failed: {e}")

    demo_caching()

    # Run complete pipeline
    demo_complete_pipeline()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\nTo run inference with a real model:")
    print("  1. Train or load a TorchScript model (.pt file)")
    print("  2. Use load_model() to load the model")
    print("  3. Use batch_inference() to process tiles")
    print("\nExample:")
    print("  model = load_model('model.pt', device='auto')")
    print("  predictions = batch_inference(model, tiles, batch_size=32)")
    print()
