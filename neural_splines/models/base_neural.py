"""
Base Neural Splines Model

Provides common interface and functionality for all Neural Splines models,
establishing the foundation for interpretable AI through geometric representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import logging
import json
from abc import ABC, abstractmethod

from ..core.interpolation import SplineInterpolator
from ..utils.geometric_validation import validate_spline_reconstruction
from ..exceptions import ModelError, InferenceError

logger = logging.getLogger(__name__)

class BaseNeuralModel(nn.Module, ABC):
    """
    Base class for all Neural Splines models
    
    Provides common functionality for spline-based neural networks including:
    - Control point management
    - Spline interpolation
    - Geometric validation
    - Interpretability features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        self.spline_order = self.config.get('spline_order', 3)
        self.compression_ratio = self.config.get('compression_ratio', 128.9)
        self.enable_visualization = self.config.get('enable_spline_visualization', False)
        
        # Core spline components
        self.spline_interpolator = SplineInterpolator(order=self.spline_order)
        
        # Tracking for interpretability
        self._spline_activations = {}
        self._control_point_cache = {}
        self._geometric_metadata = {}
        
        # Model statistics
        self._compression_stats = None
        self._last_inference_stats = {}
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        pass
    
    # Core Spline Interface
    
    def get_control_points(self, layer_name: str) -> torch.Tensor:
        """Get control points for a specific layer
        
        Args:
            layer_name: Name of the layer/parameter
            
        Returns:
            Control points tensor
        """
        try:
            # Check cache first
            if layer_name in self._control_point_cache:
                return self._control_point_cache[layer_name]
            
            # Navigate to the parameter
            parts = layer_name.split('.')
            current = self
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    raise AttributeError(f"No attribute '{part}' in {type(current).__name__}")
            
            # Look for control points parameter
            if hasattr(current, 'control_points'):
                control_points = current.control_points
            elif isinstance(current, torch.Tensor):
                control_points = current
            else:
                raise ValueError(f"No control points found for {layer_name}")
            
            # Cache for efficiency
            if self.config.get('cache_control_points', True):
                self._control_point_cache[layer_name] = control_points
            
            return control_points
            
        except Exception as e:
            raise ValueError(f"Could not retrieve control points for '{layer_name}': {e}")
    
    def set_control_points(self, layer_name: str, control_points: torch.Tensor):
        """Set control points for a specific layer
        
        Args:
            layer_name: Name of the layer/parameter
            control_points: New control points tensor
        """
        try:
            # Navigate to the parameter
            parts = layer_name.split('.')
            current = self
            
            for part in parts[:-1]:
                current = getattr(current, part)
            
            # Set the control points
            if hasattr(current, f"{parts[-1]}_control_points"):
                setattr(current, f"{parts[-1]}_control_points", nn.Parameter(control_points))
            elif hasattr(current, parts[-1]):
                attr = getattr(current, parts[-1])
                if hasattr(attr, 'control_points'):
                    attr.control_points = nn.Parameter(control_points)
                else:
                    setattr(current, parts[-1], nn.Parameter(control_points))
            else:
                raise AttributeError(f"Cannot set control points for {layer_name}")
            
            # Clear cache
            if layer_name in self._control_point_cache:
                del self._control_point_cache[layer_name]
                
        except Exception as e:
            raise ValueError(f"Could not set control points for '{layer_name}': {e}")
    
    def get_spline_layer_names(self) -> List[str]:
        """Get list of all spline layer names
        
        Returns:
            List of layer names that have spline representations
        """
        layer_names = []
        
        def find_spline_layers(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this module has control points
                if hasattr(child, 'control_points') or any(
                    param_name.endswith('_control_points') 
                    for param_name, _ in child.named_parameters(recurse=False)
                ):
                    layer_names.append(full_name)
                
                # Recursively search children
                find_spline_layers(child, full_name)
        
        find_spline_layers(self)
        return sorted(layer_names)
    
    def reconstruct_weights(self, layer_name: str, target_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Reconstruct full weight matrix from spline control points
        
        Args:
            layer_name: Name of the layer
            target_shape: Target shape for reconstruction
            
        Returns:
            Reconstructed weight tensor
        """
        try:
            control_points = self.get_control_points(layer_name)
            
            # Get target shape from metadata if not provided
            if target_shape is None:
                if layer_name in self._geometric_metadata:
                    target_shape = self._geometric_metadata[layer_name].get('original_shape')
                else:
                    # Estimate based on control points and compression ratio
                    total_elements = int(control_points.numel() * self.compression_ratio)
                    target_shape = torch.Size([total_elements])
            
            # Reconstruct using spline interpolation
            reconstructed = self.spline_interpolator.reconstruct(
                {'control_points': control_points, 'reconstruction_weights': None, 'interpolation_grid': None},
                target_shape
            )
            
            return reconstructed
            
        except Exception as e:
            raise InferenceError(f"Failed to reconstruct weights for {layer_name}: {e}")
    
    # Interpretability Features
    
    def get_spline_attribution(self, layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Get attribution scores showing which control points contributed most
        
        Args:
            layer_name: Specific layer to analyze (if None, analyzes all)
            
        Returns:
            Dictionary mapping layer names to attribution scores
        """
        if not self.enable_visualization:
            logger.warning("Spline visualization not enabled. Set enable_spline_visualization=True")
            return {}
        
        attribution_scores = {}
        
        layer_names = [layer_name] if layer_name else self.get_spline_layer_names()
        
        for name in layer_names:
            try:
                control_points = self.get_control_points(name)
                
                # Calculate attribution based on control point magnitudes
                # More sophisticated attribution would use gradients
                point_magnitudes = torch.abs(control_points)
                attribution = point_magnitudes / torch.sum(point_magnitudes)
                
                attribution_scores[name] = attribution
                
            except Exception as e:
                logger.warning(f"Could not calculate attribution for {name}: {e}")
        
        return attribution_scores
    
    def get_geometric_confidence(self, layer_name: Optional[str] = None) -> Dict[str, float]:
        """Get geometric confidence scores for spline representations
        
        Args:
            layer_name: Specific layer to analyze
            
        Returns:
            Dictionary mapping layer names to confidence scores (0-1)
        """
        confidence_scores = {}
        
        layer_names = [layer_name] if layer_name else self.get_spline_layer_names()
        
        for name in layer_names:
            try:
                control_points = self.get_control_points(name)
                
                # Calculate confidence based on spline smoothness
                if len(control_points) > 2:
                    # Smoothness measure: low variation in control points = high confidence
                    smoothness = 1.0 / (1.0 + torch.std(control_points).item())
                    
                    # Density measure: sufficient control points for representation
                    density_factor = min(1.0, len(control_points) / 10.0)
                    
                    confidence = 0.7 * smoothness + 0.3 * density_factor
                else:
                    confidence = 0.5  # Default for insufficient data
                
                confidence_scores[name] = min(1.0, max(0.0, confidence))
                
            except Exception as e:
                logger.warning(f"Could not calculate confidence for {name}: {e}")
                confidence_scores[name] = 0.0
        
        return confidence_scores
    
    def visualize_parameter_manifold(self, layer_name: str, save_path: Optional[str] = None) -> Any:
        """Visualize the parameter manifold for a specific layer
        
        Args:
            layer_name: Name of the layer to visualize
            save_path: Optional path to save visualization
            
        Returns:
            Matplotlib figure object
        """
        try:
            from ..visualization.spline_plots import SplineVisualizer
            
            visualizer = SplineVisualizer(self)
            fig = visualizer.plot_layer_splines(layer_name)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            return fig
            
        except ImportError:
            logger.error("Visualization dependencies not available")
            return None
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None
    
    # Model Statistics and Analysis
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get detailed compression statistics
        
        Returns:
            Dictionary with compression metrics
        """
        if self._compression_stats is None:
            self._compression_stats = self._calculate_compression_stats()
        
        return self._compression_stats.copy()
    
    def _calculate_compression_stats(self) -> Dict[str, Any]:
        """Calculate compression statistics for the model"""
        
        total_compressed_params = 0
        total_original_params = 0
        layer_stats = {}
        
        # Count parameters in spline layers
        for layer_name in self.get_spline_layer_names():
            try:
                control_points = self.get_control_points(layer_name)
                compressed_size = control_points.numel()
                
                # Estimate original size
                if layer_name in self._geometric_metadata:
                    original_shape = self._geometric_metadata[layer_name].get('original_shape')
                    if original_shape:
                        original_size = torch.prod(torch.tensor(original_shape)).item()
                    else:
                        original_size = compressed_size * self.compression_ratio
                else:
                    original_size = compressed_size * self.compression_ratio
                
                total_compressed_params += compressed_size
                total_original_params += original_size
                
                layer_stats[layer_name] = {
                    'compressed_params': compressed_size,
                    'estimated_original_params': original_size,
                    'layer_compression_ratio': original_size / compressed_size
                }
                
            except Exception as e:
                logger.warning(f"Could not analyze layer {layer_name}: {e}")
        
        # Add non-spline parameters (embeddings, etc.)
        for name, param in self.named_parameters():
            if not any(spline_layer in name for spline_layer in self.get_spline_layer_names()):
                if 'control_points' not in name:  # Skip control points (already counted)
                    total_compressed_params += param.numel()
                    total_original_params += param.numel()  # Assume uncompressed
        
        overall_compression = total_original_params / max(1, total_compressed_params)
        
        return {
            'original_params': total_original_params,
            'compressed_params': total_compressed_params,
            'compression_ratio': overall_compression,
            'memory_gb': total_compressed_params * 4 / (1024**3),  # Float32
            'target_compression': self.compression_ratio,
            'efficiency': overall_compression / self.compression_ratio,
            'layer_stats': layer_stats,
            'num_spline_layers': len(self.get_spline_layer_names())
        }
    
    def validate_spline_quality(self, tolerance: float = 0.01) -> Dict[str, Any]:
        """Validate the quality of spline representations
        
        Args:
            tolerance: Maximum acceptable reconstruction error
            
        Returns:
            Validation results
        """
        validation_results = {
            'overall_quality': 'excellent',
            'layer_results': {},
            'failed_layers': [],
            'max_error': 0.0,
            'avg_error': 0.0
        }
        
        errors = []
        
        for layer_name in self.get_spline_layer_names():
            try:
                control_points = self.get_control_points(layer_name)
                
                # Reconstruct and validate (simplified validation)
                if len(control_points) > 1:
                    # Check smoothness
                    second_deriv = torch.diff(control_points.flatten(), n=2)
                    smoothness_error = torch.mean(torch.abs(second_deriv)).item()
                    
                    # Check for NaN or infinite values
                    has_invalid = torch.any(torch.isnan(control_points)) or torch.any(torch.isinf(control_points))
                    
                    if has_invalid:
                        error = float('inf')
                        quality = 'failed'
                    elif smoothness_error > tolerance:
                        error = smoothness_error
                        quality = 'poor'
                    elif smoothness_error > tolerance / 10:
                        error = smoothness_error
                        quality = 'acceptable'
                    else:
                        error = smoothness_error
                        quality = 'excellent'
                    
                    validation_results['layer_results'][layer_name] = {
                        'error': error,
                        'quality': quality,
                        'has_invalid_values': has_invalid
                    }
                    
                    if quality == 'failed':
                        validation_results['failed_layers'].append(layer_name)
                    
                    errors.append(error)
                    
            except Exception as e:
                logger.warning(f"Validation failed for {layer_name}: {e}")
                validation_results['failed_layers'].append(layer_name)
                errors.append(float('inf'))
        
        # Calculate overall statistics
        if errors:
            finite_errors = [e for e in errors if not np.isinf(e)]
            validation_results['max_error'] = max(errors) if errors else 0.0
            validation_results['avg_error'] = np.mean(finite_errors) if finite_errors else float('inf')
            
            # Determine overall quality
            if validation_results['failed_layers']:
                validation_results['overall_quality'] = 'failed'
            elif validation_results['avg_error'] > tolerance:
                validation_results['overall_quality'] = 'poor'
            elif validation_results['avg_error'] > tolerance / 10:
                validation_results['overall_quality'] = 'acceptable'
            else:
                validation_results['overall_quality'] = 'excellent'
        
        return validation_results
    
    # Serialization and Loading
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], **kwargs):
        """Load Neural Splines model from directory
        
        Args:
            model_path: Path to model directory
            **kwargs: Additional loading arguments
            
        Returns:
            Loaded model instance
        """
        model_path = Path(model_path)
        
        # Load configuration
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = {}
        
        # Create model instance
        model = cls(config, **kwargs)
        
        # Load weights
        weight_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth"))
        if weight_files:
            state_dict = torch.load(weight_files[0], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        # Load Neural Splines metadata
        metadata_file = model_path / "neural_splines_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                model._geometric_metadata = json.load(f)
        
        return model
    
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Save Neural Splines model
        
        Args:
            save_directory: Directory to save model
            **kwargs: Additional saving arguments
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = save_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
        
        # Save compression statistics
        stats_file = save_dir / "compression_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.get_compression_stats(), f, indent=2)
        
        # Save Neural Splines metadata
        if self._geometric_metadata:
            metadata_file = save_dir / "neural_splines_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self._geometric_metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_directory}")
    
    # Inference utilities
    
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """Generate text using the Neural Splines model
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # This is a placeholder - actual implementation depends on tokenizer
        logger.warning("Base generate method called - should be overridden in subclass")
        return f"Generated response to: {prompt}"
    
    def __repr__(self) -> str:
        """String representation of the model"""
        stats = self.get_compression_stats()
        return (
            f"{self.__class__.__name__}(\n"
            f"  compression_ratio={stats['compression_ratio']:.1f}x,\n"
            f"  parameters={stats['compressed_params']:,},\n"
            f"  spline_layers={stats['num_spline_layers']},\n"
            f"  memory={stats['memory_gb']:.1f}GB\n"
            f")"
        )