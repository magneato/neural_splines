"""
Geometric Validation Utilities

Provides validation functions for Neural Splines geometric structures,
ensuring mathematical correctness and quality of spline representations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from scipy.stats import normaltest
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)

def validate_manifold_structure(parameter_tensor: torch.Tensor, 
                              manifold_data: Dict[str, Any],
                              tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate the geometric structure of a parameter manifold
    
    Args:
        parameter_tensor: Original parameter tensor
        manifold_data: Manifold analysis results
        tolerance: Tolerance for validation checks
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid_manifold': True,
        'smoothness_score': 0.0,
        'curvature_consistency': 0.0,
        'geometric_errors': [],
        'recommendations': []
    }
    
    try:
        # Check tensor properties
        if torch.any(torch.isnan(parameter_tensor)) or torch.any(torch.isinf(parameter_tensor)):
            validation_results['is_valid_manifold'] = False
            validation_results['geometric_errors'].append("Parameter tensor contains NaN or infinite values")
            return validation_results
        
        # Validate smoothness
        smoothness_score = _calculate_smoothness_score(parameter_tensor)
        validation_results['smoothness_score'] = smoothness_score
        
        if smoothness_score < tolerance:
            validation_results['geometric_errors'].append(f"Low smoothness score: {smoothness_score:.3f}")
            validation_results['recommendations'].append("Consider increasing spline order or control points")
        
        # Validate curvature consistency
        curvature_score = _validate_curvature_consistency(parameter_tensor, manifold_data)
        validation_results['curvature_consistency'] = curvature_score
        
        if curvature_score < tolerance:
            validation_results['geometric_errors'].append(f"Inconsistent curvature: {curvature_score:.3f}")
            validation_results['recommendations'].append("Review control point placement")
        
        # Check manifold dimension consistency
        estimated_dim = manifold_data.get('manifold_dimension', 1)
        expected_dim = _estimate_intrinsic_dimension(parameter_tensor)
        
        if abs(estimated_dim - expected_dim) > 2:
            validation_results['geometric_errors'].append(
                f"Dimension mismatch: estimated {estimated_dim}, expected ~{expected_dim}"
            )
            validation_results['recommendations'].append("Recalculate manifold dimension")
        
        # Overall validation
        if validation_results['geometric_errors']:
            validation_results['is_valid_manifold'] = False
        
        logger.debug(f"Manifold validation complete: {len(validation_results['geometric_errors'])} errors found")
        
    except Exception as e:
        logger.error(f"Manifold validation failed: {e}")
        validation_results['is_valid_manifold'] = False
        validation_results['geometric_errors'].append(f"Validation error: {str(e)}")
    
    return validation_results

def validate_spline_reconstruction(original_tensor: torch.Tensor,
                                 spline_components: Dict[str, Any],
                                 reconstruction_tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Validate the quality of spline reconstruction
    
    Args:
        original_tensor: Original parameter tensor
        spline_components: Spline representation components
        reconstruction_tolerance: Maximum acceptable reconstruction error
        
    Returns:
        Reconstruction validation results
    """
    validation_results = {
        'reconstruction_error': float('inf'),
        'relative_error': float('inf'),
        'is_valid_reconstruction': False,
        'error_distribution': {},
        'quality_metrics': {}
    }
    
    try:
        # Extract control points
        control_points = spline_components.get('control_points')
        if control_points is None:
            validation_results['quality_metrics']['error'] = "No control points found"
            return validation_results
        
        # Simple reconstruction for validation
        reconstructed = _simple_spline_reconstruction(control_points, original_tensor.shape)
        
        # Calculate reconstruction error
        reconstruction_error = torch.norm(original_tensor - reconstructed) / torch.norm(original_tensor)
        validation_results['reconstruction_error'] = reconstruction_error.item()
        
        # Calculate relative error
        relative_error = torch.mean(torch.abs(original_tensor - reconstructed) / (torch.abs(original_tensor) + 1e-8))
        validation_results['relative_error'] = relative_error.item()
        
        # Check if reconstruction is acceptable
        validation_results['is_valid_reconstruction'] = reconstruction_error.item() < reconstruction_tolerance
        
        # Analyze error distribution
        error_tensor = torch.abs(original_tensor - reconstructed)
        validation_results['error_distribution'] = {
            'mean_error': torch.mean(error_tensor).item(),
            'max_error': torch.max(error_tensor).item(),
            'std_error': torch.std(error_tensor).item(),
            'percentile_95': torch.quantile(error_tensor, 0.95).item()
        }
        
        # Quality metrics
        validation_results['quality_metrics'] = {
            'signal_to_noise_ratio': _calculate_snr(original_tensor, reconstructed),
            'correlation': _calculate_correlation(original_tensor, reconstructed),
            'structural_similarity': _calculate_structural_similarity(original_tensor, reconstructed)
        }
        
        logger.debug(f"Reconstruction validation: error={reconstruction_error:.4f}, valid={validation_results['is_valid_reconstruction']}")
        
    except Exception as e:
        logger.error(f"Reconstruction validation failed: {e}")
        validation_results['quality_metrics']['error'] = str(e)
    
    return validation_results

def validate_control_points(control_points: torch.Tensor,
                          expected_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate control points for geometric and numerical properties
    
    Args:
        control_points: Control points tensor
        expected_properties: Expected properties (range, distribution, etc.)
        
    Returns:
        Control point validation results
    """
    validation_results = {
        'is_valid': True,
        'numerical_issues': [],
        'geometric_issues': [],
        'statistical_properties': {},
        'recommendations': []
    }
    
    try:
        # Check for numerical issues
        if torch.any(torch.isnan(control_points)):
            validation_results['is_valid'] = False
            validation_results['numerical_issues'].append("Contains NaN values")
        
        if torch.any(torch.isinf(control_points)):
            validation_results['is_valid'] = False
            validation_results['numerical_issues'].append("Contains infinite values")
        
        # Check value ranges
        min_val = torch.min(control_points).item()
        max_val = torch.max(control_points).item()
        
        if abs(max_val) > 100 or abs(min_val) > 100:
            validation_results['geometric_issues'].append(f"Large value range: [{min_val:.2f}, {max_val:.2f}]")
            validation_results['recommendations'].append("Consider normalizing control points")
        
        # Statistical analysis
        control_points_flat = control_points.flatten()
        validation_results['statistical_properties'] = {
            'mean': torch.mean(control_points_flat).item(),
            'std': torch.std(control_points_flat).item(),
            'skewness': _calculate_skewness(control_points_flat),
            'kurtosis': _calculate_kurtosis(control_points_flat),
            'num_points': len(control_points_flat)
        }
        
        # Check for smoothness in control point sequence
        if len(control_points) > 2:
            smoothness = _calculate_control_point_smoothness(control_points)
            validation_results['statistical_properties']['smoothness'] = smoothness
            
            if smoothness < 0.1:
                validation_results['geometric_issues'].append(f"Low smoothness: {smoothness:.3f}")
                validation_results['recommendations'].append("Consider smoothing control points")
        
        # Validate against expected properties
        if expected_properties:
            _validate_expected_properties(control_points, expected_properties, validation_results)
        
        # Overall validation
        if validation_results['numerical_issues'] or validation_results['geometric_issues']:
            validation_results['is_valid'] = False
        
    except Exception as e:
        logger.error(f"Control point validation failed: {e}")
        validation_results['is_valid'] = False
        validation_results['numerical_issues'].append(f"Validation error: {str(e)}")
    
    return validation_results

def validate_harmonic_decomposition(frequency_components: Dict[str, torch.Tensor],
                                  original_signal: torch.Tensor) -> Dict[str, Any]:
    """
    Validate harmonic decomposition quality
    
    Args:
        frequency_components: Harmonic decomposition results
        original_signal: Original signal tensor
        
    Returns:
        Harmonic validation results
    """
    validation_results = {
        'is_valid_decomposition': True,
        'frequency_coverage': 0.0,
        'energy_preservation': 0.0,
        'spectral_errors': [],
        'reconstruction_quality': 0.0
    }
    
    try:
        frequencies = frequency_components.get('frequencies')
        amplitudes = frequency_components.get('amplitudes')
        phases = frequency_components.get('phases')
        
        if frequencies is None or amplitudes is None or phases is None:
            validation_results['is_valid_decomposition'] = False
            validation_results['spectral_errors'].append("Missing frequency components")
            return validation_results
        
        # Check frequency coverage
        freq_range = torch.max(torch.abs(frequencies)) - torch.min(torch.abs(frequencies))
        nyquist_freq = 0.5  # Normalized frequency
        coverage = min(1.0, freq_range.item() / nyquist_freq)
        validation_results['frequency_coverage'] = coverage
        
        # Check energy preservation
        original_energy = torch.sum(original_signal ** 2)
        harmonic_energy = torch.sum(amplitudes ** 2)
        energy_ratio = harmonic_energy / original_energy
        validation_results['energy_preservation'] = energy_ratio.item()
        
        if energy_ratio < 0.8:
            validation_results['spectral_errors'].append(f"Low energy preservation: {energy_ratio:.3f}")
        
        # Validate frequency ordering
        if not torch.all(frequencies[1:] >= frequencies[:-1]):
            validation_results['spectral_errors'].append("Frequencies not properly ordered")
        
        # Check for frequency aliasing
        if torch.any(torch.abs(frequencies) > nyquist_freq):
            validation_results['spectral_errors'].append("Frequencies exceed Nyquist limit")
        
        # Reconstruction quality check
        reconstruction_quality = _validate_harmonic_reconstruction(
            frequencies, amplitudes, phases, original_signal
        )
        validation_results['reconstruction_quality'] = reconstruction_quality
        
        if reconstruction_quality < 0.8:
            validation_results['spectral_errors'].append(f"Poor reconstruction quality: {reconstruction_quality:.3f}")
        
        # Overall validation
        if validation_results['spectral_errors']:
            validation_results['is_valid_decomposition'] = False
        
    except Exception as e:
        logger.error(f"Harmonic validation failed: {e}")
        validation_results['is_valid_decomposition'] = False
        validation_results['spectral_errors'].append(f"Validation error: {str(e)}")
    
    return validation_results

# Helper Functions

def _calculate_smoothness_score(tensor: torch.Tensor) -> float:
    """Calculate smoothness score for a tensor"""
    if len(tensor.shape) == 1:
        # 1D case: use second derivative
        if len(tensor) > 2:
            second_deriv = torch.diff(tensor, n=2)
            smoothness = 1.0 / (1.0 + torch.mean(torch.abs(second_deriv)).item())
        else:
            smoothness = 1.0
    elif len(tensor.shape) == 2:
        # 2D case: use Laplacian
        laplacian = _calculate_2d_laplacian(tensor)
        smoothness = 1.0 / (1.0 + torch.mean(torch.abs(laplacian)).item())
    else:
        # Higher dimensions: use total variation
        flattened = tensor.flatten()
        if len(flattened) > 1:
            gradient = torch.diff(flattened)
            smoothness = 1.0 / (1.0 + torch.std(gradient).item())
        else:
            smoothness = 1.0
    
    return min(1.0, max(0.0, smoothness))

def _validate_curvature_consistency(tensor: torch.Tensor, manifold_data: Dict[str, Any]) -> float:
    """Validate consistency of curvature across the manifold"""
    try:
        curvature_tensor = manifold_data.get('curvature_tensor')
        if curvature_tensor is None:
            return 0.5  # Default score when no curvature data
        
        # Check for consistent curvature patterns
        curvature_std = torch.std(curvature_tensor)
        curvature_mean = torch.mean(torch.abs(curvature_tensor))
        
        # Consistency score: lower variation relative to mean = higher consistency
        if curvature_mean > 0:
            consistency = 1.0 / (1.0 + curvature_std / curvature_mean)
        else:
            consistency = 1.0
        
        return min(1.0, max(0.0, consistency.item()))
        
    except Exception as e:
        logger.warning(f"Curvature validation error: {e}")
        return 0.5

def _estimate_intrinsic_dimension(tensor: torch.Tensor) -> int:
    """Estimate intrinsic dimension of parameter tensor"""
    try:
        # Simple heuristic based on tensor shape and variance structure
        if len(tensor.shape) == 1:
            return 1
        elif len(tensor.shape) == 2:
            # Use SVD to estimate effective rank
            U, S, V = torch.svd(tensor)
            # Count significant singular values
            threshold = 0.01 * torch.max(S)
            effective_rank = torch.sum(S > threshold).item()
            return min(effective_rank, min(tensor.shape))
        else:
            # For higher dimensions, use a conservative estimate
            return min(8, tensor.numel() // 100 + 1)
    except Exception:
        return 3  # Default fallback

def _simple_spline_reconstruction(control_points: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Simple spline reconstruction for validation purposes"""
    try:
        # Linear interpolation between control points
        n_points = len(control_points.flatten())
        target_size = torch.prod(torch.tensor(target_shape)).item()
        
        if n_points >= target_size:
            # Downsample control points
            indices = torch.linspace(0, n_points - 1, target_size).long()
            reconstructed = control_points.flatten()[indices]
        else:
            # Upsample using linear interpolation
            reconstructed = F.interpolate(
                control_points.flatten().unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        return reconstructed.reshape(target_shape)
        
    except Exception as e:
        logger.warning(f"Reconstruction failed: {e}")
        return torch.zeros(target_shape)

def _calculate_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate signal-to-noise ratio"""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    
    if noise_power > 0:
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    else:
        return float('inf')

def _calculate_correlation(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate correlation coefficient"""
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    
    correlation_matrix = torch.corrcoef(torch.stack([original_flat, reconstructed_flat]))
    return correlation_matrix[0, 1].item()

def _calculate_structural_similarity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate structural similarity index"""
    # Simplified SSIM calculation
    mu1 = torch.mean(original)
    mu2 = torch.mean(reconstructed)
    
    sigma1_sq = torch.var(original)
    sigma2_sq = torch.var(reconstructed)
    sigma12 = torch.mean((original - mu1) * (reconstructed - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return torch.clamp(ssim, 0, 1).item()

def _calculate_skewness(tensor: torch.Tensor) -> float:
    """Calculate skewness of tensor values"""
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    
    if std > 0:
        skewness = torch.mean(((tensor - mean) / std) ** 3)
        return skewness.item()
    else:
        return 0.0

def _calculate_kurtosis(tensor: torch.Tensor) -> float:
    """Calculate kurtosis of tensor values"""
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    
    if std > 0:
        kurtosis = torch.mean(((tensor - mean) / std) ** 4) - 3
        return kurtosis.item()
    else:
        return 0.0

def _calculate_control_point_smoothness(control_points: torch.Tensor) -> float:
    """Calculate smoothness of control point sequence"""
    if len(control_points) < 3:
        return 1.0
    
    # Calculate second differences
    first_diff = torch.diff(control_points.flatten())
    second_diff = torch.diff(first_diff)
    
    # Smoothness is inverse of variation in second differences
    smoothness = 1.0 / (1.0 + torch.std(second_diff).item())
    
    return smoothness

def _calculate_2d_laplacian(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate 2D Laplacian for smoothness measurement"""
    # Simple finite difference Laplacian
    if tensor.shape[0] < 3 or tensor.shape[1] < 3:
        return torch.zeros_like(tensor)
    
    # Pad tensor for boundary conditions
    padded = F.pad(tensor, (1, 1, 1, 1), mode='replicate')
    
    # Calculate Laplacian using finite differences
    laplacian = (
        padded[2:, 1:-1] + padded[:-2, 1:-1] +  # vertical neighbors
        padded[1:-1, 2:] + padded[1:-1, :-2] -  # horizontal neighbors
        4 * padded[1:-1, 1:-1]                   # center
    )
    
    return laplacian

def _validate_expected_properties(control_points: torch.Tensor, 
                                expected_properties: Dict[str, Any],
                                validation_results: Dict[str, Any]):
    """Validate control points against expected properties"""
    
    # Check value range
    if 'value_range' in expected_properties:
        expected_min, expected_max = expected_properties['value_range']
        actual_min = torch.min(control_points).item()
        actual_max = torch.max(control_points).item()
        
        if actual_min < expected_min or actual_max > expected_max:
            validation_results['geometric_issues'].append(
                f"Value range [{actual_min:.3f}, {actual_max:.3f}] outside expected [{expected_min}, {expected_max}]"
            )
    
    # Check number of control points
    if 'num_points_range' in expected_properties:
        min_points, max_points = expected_properties['num_points_range']
        actual_points = control_points.numel()
        
        if actual_points < min_points or actual_points > max_points:
            validation_results['geometric_issues'].append(
                f"Number of control points {actual_points} outside expected [{min_points}, {max_points}]"
            )

def _validate_harmonic_reconstruction(frequencies: torch.Tensor, 
                                    amplitudes: torch.Tensor,
                                    phases: torch.Tensor,
                                    original_signal: torch.Tensor) -> float:
    """Validate harmonic reconstruction quality"""
    try:
        # Simple reconstruction from harmonic components
        n_samples = len(original_signal.flatten())
        t = torch.linspace(0, 2*np.pi, n_samples)
        
        reconstructed = torch.zeros(n_samples)
        
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            reconstructed += amp * torch.cos(freq * t + phase)
        
        # Calculate reconstruction quality
        correlation = _calculate_correlation(original_signal.flatten(), reconstructed)
        
        return max(0.0, correlation)
        
    except Exception as e:
        logger.warning(f"Harmonic reconstruction validation failed: {e}")
        return 0.0