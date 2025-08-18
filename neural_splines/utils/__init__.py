"""
Neural Splines Utilities Module

Utility functions for tensor operations, geometric validation,
spline mathematics, and other supporting functionality.
"""

from .tensor_ops import (
    tensor_to_numpy, numpy_to_tensor, spline_to_tensor, tensor_to_spline,
    normalize_tensor, denormalize_tensor, calculate_tensor_statistics,
    memory_efficient_operation, safe_tensor_operation
)

from .geometric_validation import (
    validate_manifold_structure, validate_spline_reconstruction,
    validate_control_points, validate_harmonic_decomposition
)

from .spline_interpolation import (
    bicubic_interpolate, calculate_spline_coefficients, evaluate_bspline,
    adaptive_knot_placement, spline_smoothing, optimize_control_points
)

__all__ = [
    # Tensor operations
    'tensor_to_numpy',
    'numpy_to_tensor', 
    'spline_to_tensor',
    'tensor_to_spline',
    'normalize_tensor',
    'denormalize_tensor',
    'calculate_tensor_statistics',
    'memory_efficient_operation',
    'safe_tensor_operation',
    
    # Geometric validation
    'validate_manifold_structure',
    'validate_spline_reconstruction', 
    'validate_control_points',
    'validate_harmonic_decomposition',
    
    # Spline interpolation
    'bicubic_interpolate',
    'calculate_spline_coefficients',
    'evaluate_bspline',
    'adaptive_knot_placement', 
    'spline_smoothing',
    'optimize_control_points'
]