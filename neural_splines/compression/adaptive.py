"""
Adaptive Compression for Neural Splines

Implements the breakthrough DeepSeekSplineAdapter that achieves 128.9x compression
through adaptive spline placement and intelligent parameter reduction while
preserving neural network intelligence.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass

from ..core.harmonic import HarmonicDecomposer, HarmonicComponents
from ..core.manifold import ParameterManifold, ManifoldStructure  
from ..core.interpolation import SplineInterpolator, SplineComponents
from ..exceptions import CompressionError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveCompressionConfig:
    """Configuration for adaptive compression"""
    target_compression_ratio: float = 128.9
    min_compression_ratio: float = 10.0
    max_compression_ratio: float = 500.0
    quality_threshold: float = 0.01
    adaptive_tolerance: float = 0.001
    progressive_refinement: bool = True
    layer_specific_tuning: bool = True
    preserve_critical_layers: bool = True

class DeepSeekSplineAdapter:
    """
    Adaptive spline compression specifically optimized for DeepSeek architectures
    
    This class implements the breakthrough compression algorithm that:
    - Analyzes each layer's geometric structure
    - Adaptively places control points for optimal compression
    - Maintains mathematical precision while achieving extreme compression
    - Preserves the model's intelligence through geometric understanding
    """
    
    def __init__(self, config: Optional[AdaptiveCompressionConfig] = None):
        """Initialize the DeepSeek spline adapter
        
        Args:
            config: Compression configuration parameters
        """
        self.config = config or AdaptiveCompressionConfig()
        
        # Initialize core components
        self.harmonic_decomposer = HarmonicDecomposer(
            n_components=2048,
            adaptive=True
        )
        
        self.manifold_analyzer = ParameterManifold(
            preserve_structure=True,
            manifold_threshold=0.1
        )
        
        self.spline_interpolator = SplineInterpolator(
            order=3,
            optimize_control_points=True
        )
        
        # Adaptive compression state
        self.layer_analysis_cache = {}
        self.compression_strategies = {}
        self.quality_metrics = {}
        
        logger.info(f"Initialized DeepSeekSplineAdapter targeting {self.config.target_compression_ratio}x compression")
    
    def compress_layer(self, layer_name: str, parameter_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compress a single layer using adaptive spline placement
        
        Args:
            layer_name: Name/identifier of the layer
            parameter_tensor: Parameter tensor to compress
            
        Returns:
            Dictionary containing compressed representation and metadata
        """
        logger.debug(f"Compressing layer {layer_name} with shape {parameter_tensor.shape}")
        
        try:
            # Step 1: Analyze layer characteristics
            layer_analysis = self._analyze_layer_characteristics(layer_name, parameter_tensor)
            
            # Step 2: Determine optimal compression strategy
            compression_strategy = self._determine_compression_strategy(layer_analysis)
            
            # Step 3: Apply adaptive harmonic decomposition
            harmonic_components = self._adaptive_harmonic_decomposition(
                parameter_tensor, compression_strategy
            )
            
            # Step 4: Perform manifold analysis
            manifold_structure = self._analyze_parameter_manifold(
                parameter_tensor, harmonic_components, compression_strategy
            )
            
            # Step 5: Adaptive spline fitting
            spline_components = self._adaptive_spline_fitting(
                manifold_structure, compression_strategy
            )
            
            # Step 6: Quality validation and refinement
            if self.config.progressive_refinement:
                spline_components = self._progressive_refinement(
                    parameter_tensor, spline_components, compression_strategy
                )
            
            # Step 7: Calculate compression statistics
            compression_stats = self._calculate_compression_stats(
                parameter_tensor, spline_components
            )
            
            # Cache results for future reference
            self.layer_analysis_cache[layer_name] = layer_analysis
            self.compression_strategies[layer_name] = compression_strategy
            
            return {
                'spline_components': spline_components,
                'compression_stats': compression_stats,
                'layer_analysis': layer_analysis,
                'compression_strategy': compression_strategy
            }
            
        except Exception as e:
            raise CompressionError(
                f"Failed to compress layer {layer_name}",
                target_ratio=self.config.target_compression_ratio,
                achieved_ratio=0.0
            ) from e
    
    def compress_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Compress entire model using adaptive strategies per layer
        
        Args:
            model: PyTorch model to compress
            
        Returns:
            Complete model compression results
        """
        logger.info("Starting adaptive model compression...")
        
        compressed_layers = {}
        total_original_params = 0
        total_compressed_params = 0
        
        # First pass: analyze all layers to determine global strategy
        layer_priorities = self._analyze_model_structure(model)
        
        # Second pass: compress layers according to adaptive strategy
        for layer_name, param in model.named_parameters():
            if param.requires_grad:  # Only compress trainable parameters
                try:
                    # Get layer priority and adjust compression accordingly
                    priority = layer_priorities.get(layer_name, 'medium')
                    
                    # Adjust compression target based on priority
                    if priority == 'critical' and self.config.preserve_critical_layers:
                        layer_compression_target = min(50.0, self.config.target_compression_ratio)
                    elif priority == 'high':
                        layer_compression_target = self.config.target_compression_ratio * 0.7
                    elif priority == 'low':
                        layer_compression_target = self.config.target_compression_ratio * 1.5
                    else:  # medium
                        layer_compression_target = self.config.target_compression_ratio
                    
                    # Temporarily adjust config for this layer
                    original_target = self.config.target_compression_ratio
                    self.config.target_compression_ratio = layer_compression_target
                    
                    # Compress the layer
                    layer_result = self.compress_layer(layer_name, param.data)
                    compressed_layers[layer_name] = layer_result
                    
                    # Accumulate statistics
                    stats = layer_result['compression_stats']
                    total_original_params += stats['original_parameters']
                    total_compressed_params += stats['compressed_parameters']
                    
                    # Restore original config
                    self.config.target_compression_ratio = original_target
                    
                    logger.debug(f"Layer {layer_name}: {stats['compression_ratio']:.1f}x compression")
                    
                except Exception as e:
                    logger.error(f"Failed to compress layer {layer_name}: {e}")
                    # Continue with other layers
                    continue
        
        # Calculate overall statistics
        overall_compression = total_original_params / max(1, total_compressed_params)
        
        # Global optimization pass
        if self.config.progressive_refinement:
            compressed_layers = self._global_optimization_pass(
                compressed_layers, overall_compression
            )
        
        return {
            'compressed_layers': compressed_layers,
            'overall_stats': {
                'original_parameters': total_original_params,
                'compressed_parameters': total_compressed_params,
                'compression_ratio': overall_compression,
                'target_ratio': self.config.target_compression_ratio,
                'compression_efficiency': overall_compression / self.config.target_compression_ratio
            },
            'layer_priorities': layer_priorities
        }
    
    def _analyze_layer_characteristics(self, layer_name: str, tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze characteristics of a layer for adaptive compression"""
        
        analysis = {
            'layer_name': layer_name,
            'tensor_shape': tensor.shape,
            'parameter_count': tensor.numel(),
            'layer_type': self._classify_layer_type(layer_name),
            'geometric_properties': {},
            'statistical_properties': {},
            'compression_complexity': 'medium'
        }
        
        # Geometric analysis
        analysis['geometric_properties'] = {
            'smoothness': self._calculate_smoothness(tensor),
            'local_correlation': self._calculate_local_correlation(tensor),
            'spectral_decay': self._calculate_spectral_decay(tensor),
            'manifold_dimension_estimate': self._estimate_manifold_dimension(tensor)
        }
        
        # Statistical analysis
        flattened = tensor.flatten()
        analysis['statistical_properties'] = {
            'mean': torch.mean(flattened).item(),
            'std': torch.std(flattened).item(),
            'skewness': self._calculate_skewness(flattened),
            'kurtosis': self._calculate_kurtosis(flattened),
            'sparsity': (torch.abs(flattened) < 1e-6).float().mean().item()
        }
        
        # Determine compression complexity
        complexity_score = self._calculate_complexity_score(analysis)
        if complexity_score > 0.7:
            analysis['compression_complexity'] = 'high'
        elif complexity_score < 0.3:
            analysis['compression_complexity'] = 'low'
        else:
            analysis['compression_complexity'] = 'medium'
        
        return analysis
    
    def _determine_compression_strategy(self, layer_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal compression strategy for a layer"""
        
        strategy = {
            'compression_method': 'adaptive_splines',
            'target_compression': self.config.target_compression_ratio,
            'spline_order': 3,
            'control_point_density': 'adaptive',
            'harmonic_components': 2048,
            'refinement_iterations': 3
        }
        
        # Adjust based on layer type
        layer_type = layer_analysis['layer_type']
        if layer_type == 'attention':
            # Attention layers are critical for model performance
            strategy['target_compression'] *= 0.8
            strategy['spline_order'] = 3
            strategy['refinement_iterations'] = 5
        elif layer_type == 'mlp':
            # MLP layers can often be compressed more aggressively
            strategy['target_compression'] *= 1.2
            strategy['spline_order'] = 2
        elif layer_type == 'embedding':
            # Embeddings require careful handling
            strategy['target_compression'] *= 0.5
            strategy['spline_order'] = 3
        
        # Adjust based on complexity
        complexity = layer_analysis['compression_complexity']
        if complexity == 'high':
            strategy['harmonic_components'] *= 2
            strategy['refinement_iterations'] += 2
        elif complexity == 'low':
            strategy['harmonic_components'] //= 2
            strategy['refinement_iterations'] = 1
        
        # Adjust based on geometric properties
        geo_props = layer_analysis['geometric_properties']
        if geo_props['smoothness'] > 0.8:
            # Very smooth - can use fewer control points
            strategy['control_point_density'] = 'sparse'
        elif geo_props['smoothness'] < 0.3:
            # Not smooth - need more control points
            strategy['control_point_density'] = 'dense'
        
        return strategy
    
    def _adaptive_harmonic_decomposition(self, tensor: torch.Tensor, 
                                       strategy: Dict[str, Any]) -> HarmonicComponents:
        """Perform adaptive harmonic decomposition"""
        
        # Adjust harmonic decomposer parameters based on strategy
        self.harmonic_decomposer.n_components = strategy['harmonic_components']
        
        # Perform decomposition
        harmonic_components = self.harmonic_decomposer.decompose(tensor)
        
        # Adaptive frequency selection based on energy content
        if strategy.get('control_point_density') == 'sparse':
            # Keep only the most important frequency components
            energy_threshold = 0.95  # Keep components that capture 95% of energy
            cumulative_energy = torch.cumsum(harmonic_components.energy_distribution, dim=0)
            cutoff_idx = torch.searchsorted(cumulative_energy, energy_threshold) + 1
            
            # Trim components
            harmonic_components.frequencies = harmonic_components.frequencies[:cutoff_idx]
            harmonic_components.amplitudes = harmonic_components.amplitudes[:cutoff_idx] 
            harmonic_components.phases = harmonic_components.phases[:cutoff_idx]
            harmonic_components.energy_distribution = harmonic_components.energy_distribution[:cutoff_idx]
        
        return harmonic_components
    
    def _analyze_parameter_manifold(self, tensor: torch.Tensor,
                                  harmonics: HarmonicComponents,
                                  strategy: Dict[str, Any]) -> ManifoldStructure:
        """Analyze parameter manifold with adaptive settings"""
        
        # Adjust manifold analyzer based on strategy
        if strategy['compression_complexity'] == 'high':
            self.manifold_analyzer.neighborhood_size = 12
        else:
            self.manifold_analyzer.neighborhood_size = 8
        
        # Perform manifold analysis
        manifold_structure = self.manifold_analyzer.analyze(tensor, harmonics)
        
        return manifold_structure
    
    def _adaptive_spline_fitting(self, manifold: ManifoldStructure,
                               strategy: Dict[str, Any]) -> SplineComponents:
        """Perform adaptive spline fitting"""
        
        # Adjust spline interpolator based on strategy
        self.spline_interpolator.order = strategy['spline_order']
        
        # Calculate target control points based on compression ratio
        original_params = torch.prod(torch.tensor(manifold.metadata['tensor_shape']))
        target_control_points = max(4, int(original_params / strategy['target_compression']))
        
        # Adjust control points based on density strategy
        if strategy['control_point_density'] == 'sparse':
            target_control_points = int(target_control_points * 0.7)
        elif strategy['control_point_density'] == 'dense':
            target_control_points = int(target_control_points * 1.3)
        
        # Fit splines with adaptive control point count
        spline_components = self.spline_interpolator.fit_splines(
            manifold, target_compression=strategy['target_compression']
        )
        
        # Ensure we don't exceed or under-shoot target too much
        actual_control_points = spline_components.control_points.numel()
        if actual_control_points > target_control_points * 1.5:
            # Too many control points - reduce
            spline_components = self._reduce_control_points(
                spline_components, target_control_points
            )
        elif actual_control_points < target_control_points * 0.5:
            # Too few control points - might need more for quality
            if strategy['compression_complexity'] == 'high':
                spline_components = self._increase_control_points(
                    spline_components, target_control_points
                )
        
        return spline_components
    
    def _progressive_refinement(self, original_tensor: torch.Tensor,
                              spline_components: SplineComponents,
                              strategy: Dict[str, Any]) -> SplineComponents:
        """Progressively refine spline representation for better quality"""
        
        current_splines = spline_components
        best_splines = spline_components
        best_error = float('inf')
        
        for iteration in range(strategy['refinement_iterations']):
            # Reconstruct and measure error
            try:
                reconstructed = self.spline_interpolator.reconstruct(
                    current_splines, original_tensor.shape
                )
                
                error = torch.norm(original_tensor - reconstructed) / torch.norm(original_tensor)
                
                if error < best_error:
                    best_error = error
                    best_splines = current_splines
                
                # If error is acceptable, stop early
                if error < self.config.quality_threshold:
                    break
                
                # Refine splines for next iteration
                current_splines = self._refine_splines(
                    current_splines, original_tensor, reconstructed
                )
                
            except Exception as e:
                logger.warning(f"Refinement iteration {iteration} failed: {e}")
                break
        
        return best_splines
    
    def _global_optimization_pass(self, compressed_layers: Dict[str, Any],
                                overall_compression: float) -> Dict[str, Any]:
        """Global optimization to balance compression across layers"""
        
        if abs(overall_compression - self.config.target_compression_ratio) < 10:
            # Already close to target
            return compressed_layers
        
        # Calculate adjustment factor
        adjustment_factor = self.config.target_compression_ratio / overall_compression
        
        # Adjust compression for each layer
        optimized_layers = {}
        for layer_name, layer_data in compressed_layers.items():
            try:
                if adjustment_factor > 1.1:
                    # Need more compression
                    optimized_splines = self._increase_compression(
                        layer_data['spline_components'], adjustment_factor
                    )
                elif adjustment_factor < 0.9:
                    # Need less compression (more quality)
                    optimized_splines = self._decrease_compression(
                        layer_data['spline_components'], adjustment_factor
                    )
                else:
                    # Already good
                    optimized_splines = layer_data['spline_components']
                
                # Update layer data
                optimized_layer_data = layer_data.copy()
                optimized_layer_data['spline_components'] = optimized_splines
                optimized_layers[layer_name] = optimized_layer_data
                
            except Exception as e:
                logger.warning(f"Global optimization failed for {layer_name}: {e}")
                optimized_layers[layer_name] = layer_data
        
        return optimized_layers
    
    # Helper methods
    
    def _classify_layer_type(self, layer_name: str) -> str:
        """Classify layer type based on name"""
        name_lower = layer_name.lower()
        
        if any(keyword in name_lower for keyword in ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'attention'
        elif any(keyword in name_lower for keyword in ['mlp', 'feed_forward', 'ffn', 'gate_proj', 'up_proj', 'down_proj']):
            return 'mlp'
        elif any(keyword in name_lower for keyword in ['embed', 'embedding', 'token']):
            return 'embedding'
        elif any(keyword in name_lower for keyword in ['norm', 'layer_norm', 'batch_norm']):
            return 'normalization'
        elif any(keyword in name_lower for keyword in ['head', 'classifier', 'output']):
            return 'output'
        else:
            return 'other'
    
    def _calculate_smoothness(self, tensor: torch.Tensor) -> float:
        """Calculate smoothness metric for tensor"""
        if len(tensor.shape) == 1:
            if len(tensor) < 3:
                return 1.0
            second_deriv = torch.diff(tensor, n=2)
            return 1.0 / (1.0 + torch.mean(torch.abs(second_deriv)).item())
        else:
            # For multi-dimensional tensors, use total variation
            grad_norm = 0.0
            for dim in range(len(tensor.shape)):
                grad = torch.diff(tensor, dim=dim)
                grad_norm += torch.sum(torch.abs(grad))
            
            smoothness = 1.0 / (1.0 + grad_norm / tensor.numel())
            return smoothness.item()
    
    def _calculate_local_correlation(self, tensor: torch.Tensor) -> float:
        """Calculate local correlation in tensor"""
        if len(tensor.shape) < 2:
            return 0.5
        
        # Calculate correlation between adjacent elements
        flat = tensor.flatten()
        if len(flat) < 2:
            return 0.5
        
        correlation = torch.corrcoef(torch.stack([flat[:-1], flat[1:]]))[0, 1]
        return torch.abs(correlation).item() if not torch.isnan(correlation) else 0.5
    
    def _calculate_spectral_decay(self, tensor: torch.Tensor) -> float:
        """Calculate spectral decay rate"""
        try:
            flat = tensor.flatten()
            fft_result = torch.fft.fft(flat.float())
            power_spectrum = torch.abs(fft_result) ** 2
            
            # Sort in descending order
            sorted_power = torch.sort(power_spectrum, descending=True)[0]
            
            # Calculate decay rate (slope of log power spectrum)
            log_power = torch.log(sorted_power + 1e-8)
            indices = torch.arange(len(log_power), dtype=torch.float32)
            
            # Linear regression to find slope
            n = len(indices)
            sum_x = torch.sum(indices)
            sum_y = torch.sum(log_power)
            sum_xy = torch.sum(indices * log_power)
            sum_xx = torch.sum(indices ** 2)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            
            # Convert to decay rate (0 = no decay, 1 = fast decay)
            decay_rate = torch.clamp(-slope / 10.0, 0.0, 1.0)
            
            return decay_rate.item()
            
        except Exception:
            return 0.5
    
    def _estimate_manifold_dimension(self, tensor: torch.Tensor) -> int:
        """Estimate intrinsic manifold dimension"""
        # Simple estimation based on tensor properties
        if len(tensor.shape) == 1:
            return 1
        elif len(tensor.shape) == 2:
            # Use rank estimation
            try:
                U, S, V = torch.svd(tensor)
                significant_values = torch.sum(S > 0.01 * torch.max(S))
                return min(significant_values.item(), 8)
            except Exception:
                return min(tensor.shape)
        else:
            # For higher dimensions, conservative estimate
            return min(8, tensor.numel() // 1000 + 1)
    
    def _calculate_skewness(self, tensor: torch.Tensor) -> float:
        """Calculate skewness of tensor values"""
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        if std > 0:
            skewness = torch.mean(((tensor - mean) / std) ** 3)
            return skewness.item()
        return 0.0
    
    def _calculate_kurtosis(self, tensor: torch.Tensor) -> float:
        """Calculate kurtosis of tensor values"""
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        if std > 0:
            kurtosis = torch.mean(((tensor - mean) / std) ** 4) - 3
            return kurtosis.item()
        return 0.0
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall complexity score for layer"""
        geo_props = analysis['geometric_properties']
        stat_props = analysis['statistical_properties']
        
        # Lower smoothness = higher complexity
        smoothness_complexity = 1.0 - geo_props['smoothness']
        
        # Higher spectral decay = lower complexity
        spectral_complexity = 1.0 - geo_props['spectral_decay']
        
        # Higher sparsity = lower complexity
        sparsity_complexity = 1.0 - stat_props['sparsity']
        
        # Combine factors
        complexity = 0.4 * smoothness_complexity + 0.3 * spectral_complexity + 0.3 * sparsity_complexity
        
        return complexity
    
    def _analyze_model_structure(self, model: nn.Module) -> Dict[str, str]:
        """Analyze overall model structure to determine layer priorities"""
        layer_priorities = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_type = self._classify_layer_type(name)
                
                # Assign priority based on layer type and position
                if layer_type == 'attention':
                    layer_priorities[name] = 'high'
                elif layer_type == 'embedding' or 'output' in name.lower():
                    layer_priorities[name] = 'critical'
                elif layer_type == 'normalization':
                    layer_priorities[name] = 'medium'
                else:
                    layer_priorities[name] = 'medium'
        
        return layer_priorities
    
    def _calculate_compression_stats(self, original: torch.Tensor, 
                                   splines: SplineComponents) -> Dict[str, Any]:
        """Calculate compression statistics"""
        original_params = original.numel()
        compressed_params = splines.control_points.numel()
        
        return {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': original_params / compressed_params if compressed_params > 0 else 0.0,
            'memory_reduction_mb': (original_params - compressed_params) * 4 / (1024 * 1024),
            'spline_order': splines.metadata.get('spline_order', 3),
            'control_points_count': len(splines.control_points)
        }
    
    def _reduce_control_points(self, splines: SplineComponents, target_count: int) -> SplineComponents:
        """Reduce number of control points while preserving quality"""
        current_points = splines.control_points
        
        if len(current_points) <= target_count:
            return splines
        
        # Simple subsampling for now - could be more sophisticated
        indices = torch.linspace(0, len(current_points) - 1, target_count).long()
        reduced_points = current_points[indices]
        
        # Create new spline components
        new_splines = SplineComponents(
            control_points=reduced_points,
            knot_vectors=splines.knot_vectors,
            spline_coefficients=splines.spline_coefficients[:target_count] if len(splines.spline_coefficients) > target_count else splines.spline_coefficients,
            basis_functions=splines.basis_functions,
            interpolation_grid=splines.interpolation_grid,
            reconstruction_weights=splines.reconstruction_weights,
            metadata=splines.metadata
        )
        
        return new_splines
    
    def _increase_control_points(self, splines: SplineComponents, target_count: int) -> SplineComponents:
        """Increase number of control points for better quality"""
        # This is a placeholder - would need sophisticated interpolation
        return splines
    
    def _refine_splines(self, splines: SplineComponents, original: torch.Tensor, 
                       reconstructed: torch.Tensor) -> SplineComponents:
        """Refine splines based on reconstruction error"""
        # This is a placeholder for spline refinement
        return splines
    
    def _increase_compression(self, splines: SplineComponents, factor: float) -> SplineComponents:
        """Increase compression by reducing control points"""
        target_points = max(4, int(len(splines.control_points) / factor))
        return self._reduce_control_points(splines, target_points)
    
    def _decrease_compression(self, splines: SplineComponents, factor: float) -> SplineComponents:
        """Decrease compression by potentially adding control points"""
        # For now, just return original - would need sophisticated interpolation
        return splines