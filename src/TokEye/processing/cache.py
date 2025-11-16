"""
Caching System for TokEye Processing

This module provides a flexible caching system with LRU eviction policy
for storing spectrograms and inference results.
"""

import numpy as np
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from collections import OrderedDict
import json
import warnings
import time


def generate_cache_key(
    data: np.ndarray,
    params: Dict,
    prefix: str = '',
) -> str:
    """
    Generate a unique cache key based on data and parameters.

    Uses hash-based key generation combining data checksum and parameters.

    Args:
        data: Input numpy array
        params: Dictionary of parameters that affect the computation
        prefix: Optional prefix for the key

    Returns:
        Unique cache key string

    Example:
        >>> data = np.random.randn(1000)
        >>> params = {'n_fft': 1024, 'hop_length': 128}
        >>> key = generate_cache_key(data, params, prefix='stft')
        >>> print(key)  # 'stft_a1b2c3d4_e5f6g7h8'
    """
    # Hash the data
    data_bytes = data.tobytes()
    data_hash = hashlib.sha256(data_bytes).hexdigest()[:16]

    # Hash the parameters
    # Sort keys for consistent hashing
    params_str = json.dumps(params, sort_keys=True, default=str)
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]

    # Combine hashes
    if prefix:
        key = f"{prefix}_{data_hash}_{params_hash}"
    else:
        key = f"{data_hash}_{params_hash}"

    return key


class CacheManager:
    """
    Cache manager with LRU eviction policy.

    Manages separate caches for different data types (e.g., 'spectrogram', 'inference')
    with configurable size limits and LRU eviction.

    Example:
        >>> cache = CacheManager(cache_dir='./cache', max_size_mb=1000)
        >>> key = cache.save(data, 'spectrogram')
        >>> loaded = cache.load(key, 'spectrogram')
    """

    def __init__(
        self,
        cache_dir: str = '.cache',
        max_size_mb: float = 1000,
        max_entries: int = 1000,
        enable_compression: bool = True,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum total cache size in megabytes
            max_entries: Maximum number of cache entries
            enable_compression: Enable compression for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.enable_compression = enable_compression

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Separate subdirectories for different cache types
        self.cache_types = ['spectrogram', 'inference', 'wavelet', 'general']
        for cache_type in self.cache_types:
            (self.cache_dir / cache_type).mkdir(exist_ok=True)

        # LRU tracking: OrderedDict maintains insertion order
        # Maps key -> (cache_type, file_path, size_bytes, timestamp)
        self.lru_tracker: OrderedDict[str, Tuple[str, Path, int, float]] = OrderedDict()

        # Load existing cache metadata
        self._load_metadata()

    def _get_cache_path(self, key: str, cache_type: str) -> Path:
        """Get file path for cache entry."""
        if cache_type not in self.cache_types:
            warnings.warn(
                f"Unknown cache_type '{cache_type}', using 'general'",
                RuntimeWarning
            )
            cache_type = 'general'

        return self.cache_dir / cache_type / f"{key}.pkl"

    def _get_metadata_path(self) -> Path:
        """Get path to metadata file."""
        return self.cache_dir / 'metadata.json'

    def _load_metadata(self):
        """Load cache metadata from disk."""
        metadata_path = self._get_metadata_path()

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Reconstruct LRU tracker
                for key, value in metadata.get('lru_tracker', {}).items():
                    cache_type, file_path_str, size_bytes, timestamp = value
                    file_path = Path(file_path_str)

                    # Verify file still exists
                    if file_path.exists():
                        self.lru_tracker[key] = (cache_type, file_path, size_bytes, timestamp)
            except Exception as e:
                warnings.warn(
                    f"Failed to load cache metadata: {e}. Starting with empty cache.",
                    RuntimeWarning
                )

    def _save_metadata(self):
        """Save cache metadata to disk."""
        metadata_path = self._get_metadata_path()

        try:
            # Convert LRU tracker to JSON-serializable format
            lru_data = {
                key: (cache_type, str(file_path), size_bytes, timestamp)
                for key, (cache_type, file_path, size_bytes, timestamp) in self.lru_tracker.items()
            }

            metadata = {
                'lru_tracker': lru_data,
                'max_size_bytes': self.max_size_bytes,
                'max_entries': self.max_entries,
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save cache metadata: {e}", RuntimeWarning)

    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(size for _, _, size, _ in self.lru_tracker.values())

    def _evict_lru(self):
        """Evict least recently used entries until size/count constraints are met."""
        while self.lru_tracker:
            # Check if we need to evict
            total_size = self._get_total_size()
            num_entries = len(self.lru_tracker)

            if total_size <= self.max_size_bytes and num_entries <= self.max_entries:
                break

            # Remove oldest entry (first item in OrderedDict)
            key, (cache_type, file_path, size_bytes, timestamp) = self.lru_tracker.popitem(last=False)

            # Delete file
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                warnings.warn(f"Failed to delete cache file {file_path}: {e}", RuntimeWarning)

    def save(
        self,
        key: str,
        data: Any,
        cache_type: str = 'general',
    ) -> str:
        """
        Save data to cache.

        Args:
            key: Cache key
            data: Data to cache (must be picklable)
            cache_type: Type of cache ('spectrogram', 'inference', etc.)

        Returns:
            Cache key (same as input)

        Example:
            >>> cache = CacheManager()
            >>> data = np.random.randn(256, 256)
            >>> key = cache.save('my_key', data, cache_type='spectrogram')
        """
        file_path = self._get_cache_path(key, cache_type)

        # Serialize data
        try:
            with open(file_path, 'wb') as f:
                if self.enable_compression:
                    # Use highest protocol for better compression
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(data, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save cache entry: {e}")

        # Get file size
        size_bytes = file_path.stat().st_size

        # Update LRU tracker (move to end = most recent)
        if key in self.lru_tracker:
            # Remove old entry
            self.lru_tracker.pop(key)

        # Add new entry
        self.lru_tracker[key] = (cache_type, file_path, size_bytes, time.time())

        # Evict if necessary
        self._evict_lru()

        # Save metadata
        self._save_metadata()

        return key

    def load(
        self,
        key: str,
        cache_type: str = 'general',
    ) -> Optional[Any]:
        """
        Load data from cache.

        Args:
            key: Cache key
            cache_type: Type of cache

        Returns:
            Cached data, or None if not found

        Example:
            >>> cache = CacheManager()
            >>> data = cache.load('my_key', cache_type='spectrogram')
        """
        file_path = self._get_cache_path(key, cache_type)

        if not file_path.exists():
            return None

        # Load data
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load cache entry: {e}", RuntimeWarning)
            return None

        # Update LRU tracker (move to end = most recent)
        if key in self.lru_tracker:
            cache_type, file_path, size_bytes, _ = self.lru_tracker.pop(key)
            self.lru_tracker[key] = (cache_type, file_path, size_bytes, time.time())
            self._save_metadata()

        return data

    def exists(
        self,
        key: str,
        cache_type: str = 'general',
    ) -> bool:
        """
        Check if cache entry exists.

        Args:
            key: Cache key
            cache_type: Type of cache

        Returns:
            True if entry exists, False otherwise

        Example:
            >>> cache = CacheManager()
            >>> if cache.exists('my_key', 'spectrogram'):
            ...     data = cache.load('my_key', 'spectrogram')
        """
        file_path = self._get_cache_path(key, cache_type)
        return file_path.exists()

    def delete(
        self,
        key: str,
        cache_type: str = 'general',
    ) -> bool:
        """
        Delete cache entry.

        Args:
            key: Cache key
            cache_type: Type of cache

        Returns:
            True if deleted, False if not found

        Example:
            >>> cache = CacheManager()
            >>> cache.delete('my_key', 'spectrogram')
        """
        file_path = self._get_cache_path(key, cache_type)

        if not file_path.exists():
            return False

        # Delete file
        try:
            file_path.unlink()
        except Exception as e:
            warnings.warn(f"Failed to delete cache file: {e}", RuntimeWarning)
            return False

        # Remove from LRU tracker
        if key in self.lru_tracker:
            self.lru_tracker.pop(key)
            self._save_metadata()

        return True

    def clear(
        self,
        cache_type: Optional[str] = None,
    ):
        """
        Clear all cache entries, or entries of specific type.

        Args:
            cache_type: If specified, only clear entries of this type.
                       If None, clear all entries.

        Example:
            >>> cache = CacheManager()
            >>> cache.clear('spectrogram')  # Clear only spectrogram cache
            >>> cache.clear()  # Clear entire cache
        """
        if cache_type is None:
            # Clear all entries
            for key, (ct, file_path, _, _) in list(self.lru_tracker.items()):
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    warnings.warn(f"Failed to delete {file_path}: {e}", RuntimeWarning)

            self.lru_tracker.clear()
        else:
            # Clear entries of specific type
            for key, (ct, file_path, _, _) in list(self.lru_tracker.items()):
                if ct == cache_type:
                    try:
                        if file_path.exists():
                            file_path.unlink()
                    except Exception as e:
                        warnings.warn(f"Failed to delete {file_path}: {e}", RuntimeWarning)

                    self.lru_tracker.pop(key)

        self._save_metadata()

    def get_statistics(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - 'num_entries': Total number of entries
            - 'total_size_mb': Total cache size in MB
            - 'entries_by_type': Breakdown by cache type
            - 'oldest_entry': Timestamp of oldest entry
            - 'newest_entry': Timestamp of newest entry

        Example:
            >>> cache = CacheManager()
            >>> stats = cache.get_statistics()
            >>> print(f"Cache size: {stats['total_size_mb']:.2f} MB")
        """
        total_size = self._get_total_size()

        # Count entries by type
        entries_by_type = {}
        for cache_type, _, _, _ in self.lru_tracker.values():
            entries_by_type[cache_type] = entries_by_type.get(cache_type, 0) + 1

        # Get oldest and newest timestamps
        timestamps = [timestamp for _, _, _, timestamp in self.lru_tracker.values()]
        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None

        return {
            'num_entries': len(self.lru_tracker),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'max_entries': self.max_entries,
            'entries_by_type': entries_by_type,
            'oldest_entry': oldest,
            'newest_entry': newest,
        }
