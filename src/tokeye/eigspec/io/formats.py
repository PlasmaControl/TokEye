"""
File format detection and validation for eigspec I/O.

This module provides utilities for detecting file formats, validating 
file contents, and managing supported format information.
"""

from typing import List, Dict, Optional, Set, Any
from pathlib import Path
import numpy as np
import numpy.typing as npt


# Supported file format definitions
SUPPORTED_FORMATS = {
    'hdf5': {
        'extensions': ['.h5', '.hdf5'],
        'description': 'Hierarchical Data Format 5',
        'features': ['compression', 'metadata', 'complex_data', 'large_files'],
        'dependencies': ['h5py']
    },
    'netcdf': {
        'extensions': ['.nc', '.netcdf'],
        'description': 'Network Common Data Form',
        'features': ['metadata', 'self_describing', 'portable'],
        'dependencies': ['netCDF4']
    },
    'matlab': {
        'extensions': ['.mat'],
        'description': 'MATLAB MAT-file',
        'features': ['matlab_compatible', 'structured_data'],
        'dependencies': ['scipy']
    },
    'ascii': {
        'extensions': ['.txt', '.dat', '.asc'],
        'description': 'ASCII text format',
        'features': ['human_readable', 'portable', 'simple'],
        'dependencies': []
    },
    'csv': {
        'extensions': ['.csv'],
        'description': 'Comma Separated Values',
        'features': ['human_readable', 'tabular', 'excel_compatible'],
        'dependencies': []
    },
    'binary': {
        'extensions': ['.bin', '.raw'],
        'description': 'Binary data format',
        'features': ['compact', 'fast_io'],
        'dependencies': []
    },
    'pickle': {
        'extensions': ['.pkl', '.pickle'],
        'description': 'Python pickle format',
        'features': ['python_objects', 'complete_serialization'],
        'dependencies': []
    },
    'json': {
        'extensions': ['.json'],
        'description': 'JavaScript Object Notation',
        'features': ['human_readable', 'web_compatible', 'portable'],
        'dependencies': []
    }
}

# File type signatures for format detection
FILE_SIGNATURES = {
    'hdf5': [
        b'\x89HDF\r\n\x1a\n',  # HDF5 signature
    ],
    'netcdf': [
        b'CDF\x01',  # NetCDF classic
        b'CDF\x02',  # NetCDF 64-bit offset
        b'\x89HDF\r\n\x1a\n\x00\x00\x00\x08\x00\x08\x00\x00',  # NetCDF-4
    ],
    'matlab': [
        b'MATLAB',  # MATLAB v4 format
        b'\x00\x01IM',  # Some MATLAB formats
    ],
    'gzip': [
        b'\x1f\x8b',  # Gzip compressed
    ]
}


def detect_format(file_path: Path) -> str:
    """Detect file format from path and content.
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected format string
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # First try extension-based detection
    format_from_ext = _detect_from_extension(file_path)
    if format_from_ext:
        return format_from_ext
    
    # Then try content-based detection
    format_from_content = _detect_from_content(file_path)
    if format_from_content:
        return format_from_content
    
    # Default fallback
    return 'ascii'


def supported_formats() -> Dict[str, Dict[str, Any]]:
    """Get information about supported file formats.
    
    Returns:
        Dictionary with format information
    """
    return SUPPORTED_FORMATS.copy()


def get_format_info(format_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific format.
    
    Args:
        format_name: Name of the format
        
    Returns:
        Format information dictionary or None if not found
    """
    return SUPPORTED_FORMATS.get(format_name)


def validate_format_support(format_name: str) -> tuple[bool, List[str]]:
    """Check if a format is supported and what dependencies are missing.
    
    Args:
        format_name: Name of the format to check
        
    Returns:
        Tuple of (is_supported, missing_dependencies)
    """
    if format_name not in SUPPORTED_FORMATS:
        return False, [f"Unknown format: {format_name}"]
    
    format_info = SUPPORTED_FORMATS[format_name]
    missing_deps = []
    
    for dep in format_info.get('dependencies', []):
        try:
            if dep == 'h5py':
                import h5py
            elif dep == 'netCDF4':
                import netCDF4
            elif dep == 'scipy':
                import scipy.io
            elif dep == 'pandas':
                import pandas
        except ImportError:
            missing_deps.append(dep)
    
    return len(missing_deps) == 0, missing_deps


def get_recommended_format(
    data_type: str,
    features_needed: Optional[List[str]] = None
) -> str:
    """Get recommended format for a given data type and features.
    
    Args:
        data_type: Type of data ('time_series', 'analysis_results', 'config')
        features_needed: List of required features
        
    Returns:
        Recommended format name
    """
    if features_needed is None:
        features_needed = []
    
    # Default recommendations by data type
    defaults = {
        'time_series': 'hdf5',
        'analysis_results': 'hdf5', 
        'config': 'json',
        'shot_data': 'hdf5',
        'processed_data': 'hdf5'
    }
    
    if data_type in defaults:
        recommended = defaults[data_type]
        
        # Check if recommended format supports needed features
        format_info = SUPPORTED_FORMATS.get(recommended, {})
        format_features = format_info.get('features', [])
        
        if all(feature in format_features for feature in features_needed):
            is_supported, _ = validate_format_support(recommended)
            if is_supported:
                return recommended
    
    # Fallback selection based on features
    for format_name, format_info in SUPPORTED_FORMATS.items():
        format_features = format_info.get('features', [])
        if all(feature in format_features for feature in features_needed):
            is_supported, _ = validate_format_support(format_name)
            if is_supported:
                return format_name
    
    # Final fallback
    return 'ascii'


def validate_file_structure(
    file_path: Path,
    expected_format: str,
    required_fields: Optional[List[str]] = None
) -> tuple[bool, List[str]]:
    """Validate file structure and content.
    
    Args:
        file_path: Path to file to validate
        expected_format: Expected file format
        required_fields: List of required data fields
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]
    
    errors = []
    
    # Check format detection matches expectation
    detected_format = detect_format(file_path)
    if detected_format != expected_format:
        errors.append(f"Format mismatch: expected {expected_format}, detected {detected_format}")
    
    # Format-specific validation
    if expected_format == 'hdf5':
        errors.extend(_validate_hdf5_structure(file_path, required_fields))
    elif expected_format == 'netcdf':
        errors.extend(_validate_netcdf_structure(file_path, required_fields))
    elif expected_format == 'matlab':
        errors.extend(_validate_matlab_structure(file_path, required_fields))
    elif expected_format in ['ascii', 'csv']:
        errors.extend(_validate_text_structure(file_path, required_fields))
    
    return len(errors) == 0, errors


def get_format_extensions() -> Dict[str, List[str]]:
    """Get file extensions for each supported format.
    
    Returns:
        Dictionary mapping format names to extension lists
    """
    return {name: info['extensions'] for name, info in SUPPORTED_FORMATS.items()}


def extension_to_format(extension: str) -> Optional[str]:
    """Map file extension to format name.
    
    Args:
        extension: File extension (with or without leading dot)
        
    Returns:
        Format name or None if not found
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    extension = extension.lower()
    
    for format_name, format_info in SUPPORTED_FORMATS.items():
        if extension in format_info['extensions']:
            return format_name
    
    return None


def _detect_from_extension(file_path: Path) -> Optional[str]:
    """Detect format from file extension."""
    suffix = file_path.suffix.lower()
    return extension_to_format(suffix)


def _detect_from_content(file_path: Path) -> Optional[str]:
    """Detect format from file content."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)  # Read more bytes for better detection
        
        for format_name, signatures in FILE_SIGNATURES.items():
            for signature in signatures:
                if header.startswith(signature):
                    return format_name
        
        # Try to detect text vs binary
        try:
            header.decode('utf-8')
            return 'ascii'  # Likely text file
        except UnicodeDecodeError:
            return 'binary'  # Binary file
            
    except Exception:
        return None


def _validate_hdf5_structure(
    file_path: Path,
    required_fields: Optional[List[str]] = None
) -> List[str]:
    """Validate HDF5 file structure."""
    errors = []
    
    try:
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            if required_fields:
                for field in required_fields:
                    if field not in f and field not in f.attrs:
                        errors.append(f"Required field '{field}' not found in HDF5 file")
    
    except ImportError:
        errors.append("h5py not available for HDF5 validation")
    except Exception as e:
        errors.append(f"Error reading HDF5 file: {e}")
    
    return errors


def _validate_netcdf_structure(
    file_path: Path,
    required_fields: Optional[List[str]] = None
) -> List[str]:
    """Validate NetCDF file structure."""
    errors = []
    
    try:
        import netCDF4
        
        with netCDF4.Dataset(file_path, 'r') as nc:
            if required_fields:
                available_vars = set(nc.variables.keys())
                available_attrs = set(nc.ncattrs())
                
                for field in required_fields:
                    if field not in available_vars and field not in available_attrs:
                        errors.append(f"Required field '{field}' not found in NetCDF file")
    
    except ImportError:
        errors.append("netCDF4 not available for NetCDF validation")
    except Exception as e:
        errors.append(f"Error reading NetCDF file: {e}")
    
    return errors


def _validate_matlab_structure(
    file_path: Path,
    required_fields: Optional[List[str]] = None
) -> List[str]:
    """Validate MATLAB file structure."""
    errors = []
    
    try:
        import scipy.io as sio
        
        data = sio.loadmat(str(file_path))
        
        if required_fields:
            available_fields = set(data.keys())
            
            for field in required_fields:
                if field not in available_fields:
                    errors.append(f"Required field '{field}' not found in MATLAB file")
    
    except ImportError:
        errors.append("scipy not available for MATLAB validation")
    except Exception as e:
        errors.append(f"Error reading MATLAB file: {e}")
    
    return errors


def _validate_text_structure(
    file_path: Path,
    required_fields: Optional[List[str]] = None
) -> List[str]:
    """Validate text file structure."""
    errors = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            errors.append("Text file is empty")
            return errors
        
        # Check if first line looks like a header
        first_line = lines[0].strip()
        if first_line.startswith('#'):
            # Header line, extract field names
            header_fields = first_line[1:].strip().split()
            
            if required_fields:
                for field in required_fields:
                    if field not in header_fields:
                        errors.append(f"Required field '{field}' not found in text file header")
        
        # Check data consistency
        if len(lines) > 1:
            try:
                # Try to parse a data line
                data_line = lines[1 if first_line.startswith('#') else 0]
                data_cols = data_line.strip().split()
                
                # Check if all lines have consistent number of columns
                for i, line in enumerate(lines[1:], start=2):
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    cols = line.strip().split()
                    if len(cols) != len(data_cols):
                        errors.append(f"Inconsistent number of columns at line {i}")
                        break
            
            except Exception as e:
                errors.append(f"Error parsing text file data: {e}")
    
    except Exception as e:
        errors.append(f"Error reading text file: {e}")
    
    return errors


def format_compatibility_matrix() -> Dict[str, Dict[str, bool]]:
    """Get compatibility matrix between formats.
    
    Returns:
        Dictionary showing which formats can be converted to which others
    """
    # This is a simplified compatibility matrix
    # In practice, compatibility depends on the specific data structure
    return {
        'hdf5': {'netcdf': True, 'matlab': True, 'ascii': True, 'csv': True, 'pickle': True},
        'netcdf': {'hdf5': True, 'matlab': True, 'ascii': True, 'csv': True},
        'matlab': {'hdf5': True, 'ascii': True, 'csv': True, 'pickle': True},
        'ascii': {'hdf5': True, 'netcdf': True, 'matlab': True, 'csv': True},
        'csv': {'hdf5': True, 'netcdf': True, 'matlab': True, 'ascii': True},
        'pickle': {'hdf5': True, 'matlab': True, 'ascii': True},
        'binary': {'ascii': False, 'csv': False},  # Limited conversion options
        'json': {'hdf5': True, 'matlab': True, 'ascii': True, 'pickle': True}
    } 