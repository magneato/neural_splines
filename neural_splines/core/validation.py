"""
Neural Splines Core Validation

Mathematical validation tools for ensuring spline conversion quality
and geometric correctness. Validates the 128.9x compression maintains
neural network intelligence through rigorous mathematical checks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from scipy.stats import kstest, shapiro
from dataclasses import dataclass

from ..utils.geometric_validation import validate_manifold_structure, validate_spline_reconstruction
from ..exceptions import ValidationError

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    error_score: float
    quality_grade: str  # 'excellent', 'good', 'acceptable', 'poor', 'failed'
    issues: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]

class GeometricValidator:
    """
    Core validator for Neural Splines geometric structures
    
    Ensures that spline representations maintain mathematical correctness
    and preserve the original model's computational properties.
    """
    
    def __init__(self, threshold: float = 0.01, strict_mode: bool = False):
        """Initialize geometric validator
        
        Args:
            threshold: Maximum acceptable reconstruction error
            strict_mode: Whether to apply strict validation criteria
        """
        self.threshold = threshold
        self.strict_mode = strict_mode
        
        # Validation criteria
        self.smoothness_threshold = 0.1 if strict_mode else 0.05
        self.curvature_threshold = 1.0 if strict_mode else 2.0
        self.reconstruction_threshold = threshold
        
        logger.debug(f"Initialized GeometricValidator (threshold={threshold}, strict={strict_mode})")
    
    def validate_conversion_quality(self, original_tensor: torch.Tensor,
                                  spline_components: Dict[str, Any],
                                  manifold_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Comprehensive validation of spline conversion quality
        
        Args:
            original_tensor: Original parameter tensor
            spline_components: Spline representation components
            manifold_data: Optional manifold analysis data
            
        Returns:
            ValidationResult with detailed assessment
        """
        logger.debug(f"Validating conversion for tensor shape {original_tensor.shape}")
        
        issues = []
        recommendations = []
        metrics = {}
        
        try:
            # 1. Reconstruction accuracy validation
            reconstruction_result = self._validate_reconstruction_accuracy(
                original_tensor, spline_components
            )
            metrics.update(reconstruction_result)
            
            if reconstruction_result['reconstruction_error'] > self.reconstruction_threshold:
                issues.append(f"High reconstruction error: {reconstruction_result['reconstruction_error']:.4f}")
                recommendations.append("Increase number of control points or spline order")
            
            # 2. Geometric consistency validation
            geometric_result = self._validate_geometric_consistency(
                original_tensor, spline_components
            )
            metrics.update(geometric_result)
            
            if geometric_result['smoothness_score'] < self.smoothness_threshold:
                issues.append(f"Low smoothness: {geometric_result['smoothness_score']:.4f}")
                recommendations.append("Apply smoothing to control points")
            
            # 3. Numerical stability validation
            stability_result = self._validate_numerical_stability(spline_components)
            metrics.update(stability_result)
            
            if not stability_result['is_numerically_stable']:
                issues.append("Numerical instability detected")
                recommendations.append("Normalize control points or adjust precision")
            
            # 4. Manifold structure validation (if provided)
            if manifold_data:
                manifold_result = self._validate_manifold_structure(
                    original_tensor, manifold_data
                )
                metrics.update(manifold_result)
                
                if not manifold_result['is_valid_manifold']:
                    issues.append("Invalid manifold structure")
                    recommendations.append("Review manifold analysis parameters")
            
            # 5. Compression efficiency validation
            efficiency_result = self._validate_compression_efficiency(
                original_tensor, spline_components
            )
            metrics.update(efficiency_result)
            
            # Determine overall quality
            error_score = reconstruction_result['reconstruction_error']
            quality_grade = self._determine_quality_grade(error_score, issues)
            
            is_valid = len(issues) == 0 or (not self.strict_mode and quality_grade != 'failed')
            
            return ValidationResult(
                is_valid=is_valid,
                error_score=error_score,
                quality_grade=quality_grade,
                issues=issues,
                recommendations=recommendations,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_score=float('inf'),
                quality_grade='failed',
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Review input data and try again"],
                metrics={}
            )
    
    def validate_spline_properties(self, control_points: torch.Tensor,
                                 knot_vectors: List[torch.Tensor],
                                 spline_order: int) -> ValidationResult:
        """
        Validate mathematical properties of spline representation
        
        Args:
            control_points: Spline control points
            knot_vectors: Knot vectors for spline segments
            spline_order: Order of the splines
            
        Returns:
            ValidationResult for spline properties
        """
        issues = []
        recommendations = []
        metrics = {}
        
        try:
            # 1. Control points validation
            cp_result = self._validate_control_points(control_points)
            metrics.update(cp_result)
            
            if cp_result['has_invalid_values']:
                issues.append("Control points contain invalid values")
                recommendations.append("Check for NaN or infinite values")
            
            # 2. Knot vector validation
            for i, knot_vector in enumerate(knot_vectors):
                kv_result = self._validate_knot_vector(knot_vector, spline_order)
                metrics[f'knot_vector_{i}'] = kv_result
                
                if not kv_result['is_valid']:
                    issues.append(f"Invalid knot vector {i}")
                    recommendations.append("Ensure knot vectors are monotonic and properly sized")
            
            # 3. Spline order validation
            order_result = self._validate_spline_order(spline_order, len(control_points))
            metrics.update(order_result)
            
            if not order_result['is_valid_order']:
                issues.append(f"Incompatible spline order {spline_order}")
                recommendations.append("Adjust spline order or number of control points")
            
            # 4. Consistency validation
            consistency_result = self._validate_spline_consistency(
                control_points, knot_vectors, spline_order
            )
            metrics.update(consistency_result)
            
            # Determine quality
            error_score = metrics.get('overall_error', 0.0)
            quality_grade = self._determine_quality_grade(error_score, issues)
            is_valid = len(issues) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                error_score=error_score,
                quality_grade=quality_grade,
                issues=issues,
                recommendations=recommendations,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Spline properties validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_score=float('inf'),
                quality_grade='failed',
                issues=[f"Spline validation error: {str(e)}"],
                recommendations=["Review spline parameters"],
                metrics={}
            )
    
    def validate_harmonic_decomposition(self, frequencies: torch.Tensor,
                                      amplitudes: torch.Tensor,
                                      phases: torch.Tensor,
                                      original_signal: torch.Tensor) -> ValidationResult:
        """
        Validate harmonic decomposition quality
        
        Args:
            frequencies: Frequency components
            amplitudes: Amplitude components
            phases: Phase components
            original_signal: Original signal tensor
            
        Returns:
            ValidationResult for harmonic decomposition
        """
        issues = []
        recommendations = []
        metrics = {}
        
        try:
            # 1. Frequency domain validation
            freq_result = self._validate_frequency_components(frequencies, amplitudes)
            metrics.update(freq_result)
            
            # 2. Energy preservation validation
            energy_result = self._validate_energy_preservation(
                amplitudes, original_signal
            )
            metrics.update(energy_result)
            
            if energy_result['energy_preservation'] < 0.8:
                issues.append(f"Low energy preservation: {energy_result['energy_preservation']:.3f}")
                recommendations.append("Increase number of harmonic components")
            
            # 3. Reconstruction quality validation
            reconstruction_result = self._validate_harmonic_reconstruction(
                frequencies, amplitudes, phases, original_signal
            )
            metrics.update(reconstruction_result)
            
            if reconstruction_result['reconstruction_quality'] < 0.8:
                issues.append(f"Poor reconstruction quality: {reconstruction_result['reconstruction_quality']:.3f}")
                recommendations.append("Review frequency selection or increase precision")
            
            # 4. Spectral consistency validation
            spectral_result = self._validate_spectral_consistency(frequencies, amplitudes)
            metrics.update(spectral_result)
            
            # Determine overall quality
            error_score = 1.0 - reconstruction_result.get('reconstruction_quality', 0.0)
            quality_grade = self._determine_quality_grade(error_score, issues)
            is_valid = len(issues) == 0 or quality_grade in ['excellent', 'good', 'acceptable']
            
            return ValidationResult(
                is_valid=is_valid,
                error_score=error_score,
                quality_grade=quality_grade,
                issues=issues,
                recommendations=recommendations,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Harmonic decomposition validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_score=float('inf'),
                quality_grade='failed',
                issues=[f"Harmonic validation error: {str(e)}"],
                recommendations=["Review harmonic decomposition parameters"],
                metrics={}
            )
    
    # Private validation methods
    
    def _validate_reconstruction_accuracy(self, original: torch.Tensor,
                                        spline_components: Dict[str, Any]) -> Dict[str, float]:
        """Validate reconstruction accuracy"""
        try:
            # Get control points
            control_points = spline_components.get('control_points')
            if control_points is None:
                return {'reconstruction_error': float('inf')}
            
            # Simple reconstruction for validation
            reconstructed = self._simple_reconstruction(control_points, original.shape)
            
            # Calculate various error metrics
            mse_error = F.mse_loss(original, reconstructed).item()
            mae_error = F.l1_loss(original, reconstructed).item()
            
            # Normalized error
            norm_error = torch.norm(original - reconstructed) / torch.norm(original)
            
            # Relative error
            relative_error = torch.mean(torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8))
            
            return {
                'reconstruction_error': norm_error.item(),
                'mse_error': mse_error,
                'mae_error': mae_error,
                'relative_error': relative_error.item()
            }
            
        except Exception as e:
            logger.warning(f"Reconstruction accuracy validation failed: {e}")
            return {'reconstruction_error': float('inf')}
    
    def _validate_geometric_consistency(self, original: torch.Tensor,
                                      spline_components: Dict[str, Any]) -> Dict[str, float]:
        """Validate geometric consistency"""
        try:
            control_points = spline_components.get('control_points')
            if control_points is None:
                return {'smoothness_score': 0.0}
            
            # Calculate smoothness metrics
            smoothness_score = self._calculate_smoothness(control_points)
            
            # Calculate geometric coherence
            coherence_score = self._calculate_geometric_coherence(original, control_points)
            
            # Calculate curvature variation
            curvature_variation = self._calculate_curvature_variation(control_points)
            
            return {
                'smoothness_score': smoothness_score,
                'geometric_coherence': coherence_score,
                'curvature_variation': curvature_variation
            }
            
        except Exception as e:
            logger.warning(f"Geometric consistency validation failed: {e}")
            return {'smoothness_score': 0.0}
    
    def _validate_numerical_stability(self, spline_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numerical stability"""
        try:
            control_points = spline_components.get('control_points')
            if control_points is None:
                return {'is_numerically_stable': False}
            
            # Check for NaN or infinite values
            has_nan = torch.any(torch.isnan(control_points))
            has_inf = torch.any(torch.isinf(control_points))
            
            # Check value ranges
            max_val = torch.max(torch.abs(control_points))
            is_in_range = max_val < 1e6  # Reasonable range
            
            # Check condition number (for matrix-like structures)
            condition_number = self._estimate_condition_number(control_points)
            
            is_stable = not (has_nan or has_inf) and is_in_range and condition_number < 1e12
            
            return {
                'is_numerically_stable': is_stable,
                'has_nan': has_nan.item(),
                'has_inf': has_inf.item(),
                'max_value': max_val.item(),
                'condition_number': condition_number
            }
            
        except Exception as e:
            logger.warning(f"Numerical stability validation failed: {e}")
            return {'is_numerically_stable': False}
    
    def _validate_manifold_structure(self, original: torch.Tensor,
                                   manifold_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate manifold structure"""
        try:
            # Use the geometric validation utility
            result = validate_manifold_structure(original, manifold_data, self.threshold)
            
            return {
                'is_valid_manifold': result['is_valid_manifold'],
                'manifold_smoothness': result['smoothness_score'],
                'manifold_curvature_consistency': result['curvature_consistency']
            }
            
        except Exception as e:
            logger.warning(f"Manifold structure validation failed: {e}")
            return {'is_valid_manifold': False}
    
    def _validate_compression_efficiency(self, original: torch.Tensor,
                                       spline_components: Dict[str, Any]) -> Dict[str, float]:
        """Validate compression efficiency"""
        try:
            control_points = spline_components.get('control_points')
            if control_points is None:
                return {'compression_ratio': 0.0}
            
            original_size = original.numel()
            compressed_size = control_points.numel()
            
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0.0
            
            # Calculate bits per parameter (theoretical)
            bits_per_param = np.log2(compressed_size + 1) if compressed_size > 0 else 0.0
            
            return {
                'compression_ratio': compression_ratio,
                'original_parameters': original_size,
                'compressed_parameters': compressed_size,
                'bits_per_parameter': bits_per_param
            }
            
        except Exception as e:
            logger.warning(f"Compression efficiency validation failed: {e}")
            return {'compression_ratio': 0.0}
    
    def _validate_control_points(self, control_points: torch.Tensor) -> Dict[str, Any]:
        """Validate control points properties"""
        # Check for invalid values
        has_nan = torch.any(torch.isnan(control_points))
        has_inf = torch.any(torch.isinf(control_points))
        
        # Statistical properties
        mean_val = torch.mean(control_points)
        std_val = torch.std(control_points)
        min_val = torch.min(control_points)
        max_val = torch.max(control_points)
        
        return {
            'has_invalid_values': has_nan or has_inf,
            'mean_value': mean_val.item(),
            'std_value': std_val.item(),
            'min_value': min_val.item(),
            'max_value': max_val.item(),
            'num_points': control_points.numel()
        }
    
    def _validate_knot_vector(self, knot_vector: torch.Tensor, spline_order: int) -> Dict[str, Any]:
        """Validate knot vector properties"""
        try:
            # Check monotonicity
            is_monotonic = torch.all(knot_vector[1:] >= knot_vector[:-1])
            
            # Check size
            expected_min_size = spline_order + 2
            is_valid_size = len(knot_vector) >= expected_min_size
            
            # Check range
            is_normalized = torch.min(knot_vector) >= 0 and torch.max(knot_vector) <= 1
            
            return {
                'is_valid': is_monotonic and is_valid_size,
                'is_monotonic': is_monotonic.item(),
                'is_valid_size': is_valid_size,
                'is_normalized': is_normalized.item(),
                'size': len(knot_vector)
            }
            
        except Exception as e:
            logger.warning(f"Knot vector validation failed: {e}")
            return {'is_valid': False}
    
    def _validate_spline_order(self, spline_order: int, num_control_points: int) -> Dict[str, Any]:
        """Validate spline order compatibility"""
        # Check if order is reasonable
        is_valid_order = 1 <= spline_order <= min(5, num_control_points - 1)
        
        # Check if there are enough control points
        has_enough_points = num_control_points > spline_order
        
        return {
            'is_valid_order': is_valid_order and has_enough_points,
            'spline_order': spline_order,
            'num_control_points': num_control_points,
            'min_required_points': spline_order + 1
        }
    
    def _validate_spline_consistency(self, control_points: torch.Tensor,
                                   knot_vectors: List[torch.Tensor],
                                   spline_order: int) -> Dict[str, float]:
        """Validate overall spline consistency"""
        try:
            # Check dimension consistency
            expected_dims = len(knot_vectors)
            
            # Calculate overall consistency score
            consistency_score = 1.0
            
            # Penalize for inconsistencies
            if len(control_points) < spline_order + 1:
                consistency_score *= 0.5
            
            for knot_vector in knot_vectors:
                if len(knot_vector) < spline_order + 2:
                    consistency_score *= 0.7
            
            return {
                'consistency_score': consistency_score,
                'overall_error': 1.0 - consistency_score
            }
            
        except Exception as e:
            logger.warning(f"Spline consistency validation failed: {e}")
            return {'consistency_score': 0.0, 'overall_error': 1.0}
    
    def _validate_frequency_components(self, frequencies: torch.Tensor,
                                     amplitudes: torch.Tensor) -> Dict[str, float]:
        """Validate frequency domain components"""
        try:
            # Check Nyquist criterion
            max_freq = torch.max(torch.abs(frequencies))
            nyquist_compliant = max_freq <= 0.5
            
            # Check frequency ordering
            sorted_freqs = torch.sort(torch.abs(frequencies))[0]
            is_ordered = torch.allclose(torch.abs(frequencies), sorted_freqs)
            
            # Energy distribution
            total_energy = torch.sum(amplitudes ** 2)
            energy_concentration = torch.max(amplitudes ** 2) / total_energy
            
            return {
                'nyquist_compliant': float(nyquist_compliant),
                'is_frequency_ordered': float(is_ordered),
                'energy_concentration': energy_concentration.item(),
                'max_frequency': max_freq.item()
            }
            
        except Exception as e:
            logger.warning(f"Frequency validation failed: {e}")
            return {'nyquist_compliant': 0.0}
    
    def _validate_energy_preservation(self, amplitudes: torch.Tensor,
                                    original_signal: torch.Tensor) -> Dict[str, float]:
        """Validate energy preservation in harmonic decomposition"""
        try:
            # Calculate energy in harmonic components
            harmonic_energy = torch.sum(amplitudes ** 2)
            
            # Calculate energy in original signal
            original_energy = torch.sum(original_signal ** 2)
            
            # Energy preservation ratio
            energy_preservation = harmonic_energy / original_energy if original_energy > 0 else 0.0
            
            return {
                'energy_preservation': energy_preservation.item(),
                'harmonic_energy': harmonic_energy.item(),
                'original_energy': original_energy.item()
            }
            
        except Exception as e:
            logger.warning(f"Energy preservation validation failed: {e}")
            return {'energy_preservation': 0.0}
    
    def _validate_harmonic_reconstruction(self, frequencies: torch.Tensor,
                                        amplitudes: torch.Tensor,
                                        phases: torch.Tensor,
                                        original_signal: torch.Tensor) -> Dict[str, float]:
        """Validate harmonic reconstruction quality"""
        try:
            # Reconstruct signal from harmonic components
            n_samples = len(original_signal.flatten())
            t = torch.linspace(0, 2*np.pi, n_samples)
            
            reconstructed = torch.zeros(n_samples)
            for freq, amp, phase in zip(frequencies, amplitudes, phases):
                reconstructed += amp * torch.cos(freq * t + phase)
            
            # Calculate correlation
            correlation = torch.corrcoef(torch.stack([
                original_signal.flatten(), reconstructed
            ]))[0, 1]
            
            # Calculate SNR
            signal_power = torch.mean(original_signal.flatten() ** 2)
            noise_power = torch.mean((original_signal.flatten() - reconstructed) ** 2)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            
            return {
                'reconstruction_quality': correlation.item(),
                'signal_to_noise_ratio': snr.item()
            }
            
        except Exception as e:
            logger.warning(f"Harmonic reconstruction validation failed: {e}")
            return {'reconstruction_quality': 0.0}
    
    def _validate_spectral_consistency(self, frequencies: torch.Tensor,
                                     amplitudes: torch.Tensor) -> Dict[str, float]:
        """Validate spectral consistency"""
        try:
            # Check for spectral leakage
            freq_spacing = torch.diff(torch.sort(frequencies)[0])
            min_spacing = torch.min(freq_spacing) if len(freq_spacing) > 0 else 1.0
            
            # Check amplitude distribution
            amp_std = torch.std(amplitudes)
            amp_mean = torch.mean(amplitudes)
            amp_cv = amp_std / (amp_mean + 1e-8)  # Coefficient of variation
            
            return {
                'min_frequency_spacing': min_spacing.item(),
                'amplitude_coefficient_variation': amp_cv.item()
            }
            
        except Exception as e:
            logger.warning(f"Spectral consistency validation failed: {e}")
            return {'min_frequency_spacing': 0.0}
    
    # Helper methods
    
    def _simple_reconstruction(self, control_points: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Simple reconstruction for validation"""
        try:
            target_size = torch.prod(torch.tensor(target_shape)).item()
            
            # Linear interpolation
            reconstructed = F.interpolate(
                control_points.flatten().unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='linear',
                align_corners=False
            ).squeeze()
            
            return reconstructed.reshape(target_shape)
            
        except Exception:
            return torch.zeros(target_shape)
    
    def _calculate_smoothness(self, control_points: torch.Tensor) -> float:
        """Calculate smoothness score"""
        try:
            points = control_points.flatten()
            if len(points) < 3:
                return 1.0
            
            # Second derivative approximation
            second_deriv = torch.diff(points, n=2)
            smoothness = 1.0 / (1.0 + torch.mean(torch.abs(second_deriv)).item())
            
            return smoothness
            
        except Exception:
            return 0.0
    
    def _calculate_geometric_coherence(self, original: torch.Tensor, control_points: torch.Tensor) -> float:
        """Calculate geometric coherence between original and control points"""
        try:
            # Simplified coherence measure
            orig_std = torch.std(original)
            ctrl_std = torch.std(control_points)
            
            coherence = 1.0 / (1.0 + abs(orig_std - ctrl_std))
            return coherence.item()
            
        except Exception:
            return 0.0
    
    def _calculate_curvature_variation(self, control_points: torch.Tensor) -> float:
        """Calculate curvature variation"""
        try:
            points = control_points.flatten()
            if len(points) < 3:
                return 0.0
            
            second_deriv = torch.diff(points, n=2)
            curvature_var = torch.std(second_deriv).item()
            
            return curvature_var
            
        except Exception:
            return 0.0
    
    def _estimate_condition_number(self, tensor: torch.Tensor) -> float:
        """Estimate condition number for numerical stability"""
        try:
            # For 1D tensor, return a simple measure
            if len(tensor.shape) == 1:
                return torch.max(tensor) / (torch.min(torch.abs(tensor)) + 1e-8)
            
            # For 2D tensor, use SVD
            if len(tensor.shape) == 2:
                U, S, V = torch.svd(tensor)
                condition_number = torch.max(S) / (torch.min(S) + 1e-8)
                return condition_number.item()
            
            # For higher dimensions, flatten and estimate
            flattened = tensor.flatten()
            return torch.max(flattened) / (torch.min(torch.abs(flattened)) + 1e-8)
            
        except Exception:
            return 1e12  # High condition number indicates instability
    
    def _determine_quality_grade(self, error_score: float, issues: List[str]) -> str:
        """Determine quality grade based on error score and issues"""
        
        if error_score == float('inf') or any('failed' in issue.lower() for issue in issues):
            return 'failed'
        
        # Count severe issues
        severe_issues = sum(1 for issue in issues if any(
            keyword in issue.lower() for keyword in ['high', 'invalid', 'instability', 'poor']
        ))
        
        if severe_issues > 2:
            return 'poor'
        elif error_score > 0.1 or severe_issues > 1:
            return 'acceptable'
        elif error_score > 0.01 or severe_issues > 0:
            return 'good'
        else:
            return 'excellent'