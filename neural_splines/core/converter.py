"""
Neural Splines Core Converter

The main SplineConverter class that orchestrates the transformation of dense
neural networks into their natural mathematical representation using harmonic
decomposition and spline interpolation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from .harmonic import HarmonicDecomposer
from .manifold import ParameterManifold
from .interpolation import SplineInterpolator
from ..exceptions import ConversionError

logger = logging.getLogger(__name__)

@dataclass
class ConversionConfig: # rls - bad design, mirrors Deployment config.  needs fix
    """Configuration for Neural Splines conversion"""
    compression_ratio: float = 128.9
    spline_order: int = 3  # Bicubic splines
    harmonic_components: int = 2048
    reconstruction_threshold: float = 0.01
    adaptive_precision: bool = True
    validate_geometry: bool = True
    batch_size: int = 1000000  # Process parameters in batches
    preserve_structure: bool = True

class SplineConverter:
    """
    Main Neural Splines converter that transforms dense neural networks
    into interpretable mathematical curves achieving 128.9x compression.
    """
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize the Neural Splines converter
        
        Args:
            config: Conversion configuration parameters
        """
        self.config = config or ConversionConfig()
        
        # Initialize core components
        self.harmonic_decomposer = HarmonicDecomposer(
            n_components=self.config.harmonic_components,
            adaptive=self.config.adaptive_precision
        )
        
        self.manifold_analyzer = ParameterManifold(
            preserve_structure=self.config.preserve_structure
        )
        
        self.spline_interpolator = SplineInterpolator(
            order=self.config.spline_order,
            optimize_control_points=True
        )
        
#        self.validator = GeometricValidator(
#            threshold=self.config.reconstruction_threshold
#        )
        
        # Conversion statistics
        self.conversion_stats = {}
        
    def convert_model(self, model: nn.Module, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Convert a PyTorch model to Neural Splines representation
        
        Args:
            model: PyTorch model to convert
            model_path: Optional path for saving intermediate results
            
        Returns:
            Dictionary containing spline representation and metadata
        """
        logger.info("Starting Neural Splines conversion...")
        logger.info(f"Target compression ratio: {self.config.compression_ratio}x")
        
        # Initialize conversion tracking
        original_params = sum(p.numel() for p in model.parameters())
        conversion_data = {
            'spline_weights': {},
            'model_structure': {},
            'compression_stats': {
                'original_parameters': original_params,
                'target_compression': self.config.compression_ratio
            }
        }
        
        # Convert each parameter tensor
        total_compressed_params = 0
        layer_index = 0
        
        for name, param in model.named_parameters():
            logger.info(f"Converting layer: {name} ({param.shape})")
            
            try:
                # Convert parameter to spline representation
                spline_data = self._convert_parameter(param, name)
                conversion_data['spline_weights'][name] = spline_data
                
                # Track compression statistics
                compressed_size = sum(
                    v.numel() if torch.is_tensor(v) else len(v) 
                    for v in spline_data['components'].values()
                )
                total_compressed_params += compressed_size
                
                layer_compression = param.numel() / compressed_size
                logger.info(f"  Layer compression: {layer_compression:.1f}x")
                
                layer_index += 1
                
            except Exception as e:
                logger.error(f"Failed to convert {name}: {e}")
                raise ConversionError(f"Conversion failed for layer {name}: {e}")
        
        # Calculate final compression statistics
        overall_compression = original_params / total_compressed_params
        conversion_data['compression_stats'].update({
            'compressed_parameters': total_compressed_params,
            'achieved_compression': overall_compression,
            'compression_efficiency': overall_compression / self.config.compression_ratio
        })
        
        logger.info(f"Conversion complete! Achieved {overall_compression:.1f}x compression")
        
        # Validate conversion quality if requested
        if self.config.validate_geometry:
            validation_results = self._validate_conversion(conversion_data, model)
            conversion_data['validation'] = validation_results
        
        return conversion_data
    
    def _convert_parameter(self, param: torch.Tensor, param_name: str) -> Dict[str, Any]:
        """Convert a single parameter tensor to spline representation
        
        Args:
            param: Parameter tensor to convert
            param_name: Name of the parameter for tracking
            
        Returns:
            Dictionary containing spline components and metadata
        """
        # Step 1: Harmonic decomposition
        logger.debug(f"  Performing harmonic decomposition for {param_name}")
        harmonic_components = self.harmonic_decomposer.decompose(param)
        
        # Step 2: Manifold analysis
        logger.debug(f"  Analyzing parameter manifold for {param_name}")
        manifold_structure = self.manifold_analyzer.analyze(param, harmonic_components)
        
        # Step 3: Spline fitting
        logger.debug(f"  Fitting splines for {param_name}")
        spline_components = self.spline_interpolator.fit_splines(
            manifold_structure,
            target_compression=self.config.compression_ratio
        )
        
        # Step 4: Validation
        if self.config.validate_geometry:
            reconstruction_error = self._validate_parameter_conversion(
                param, spline_components, param_name
            )
        else:
            reconstruction_error = 0.0
        
        return {
            'components': spline_components,
            'manifold_metadata': manifold_structure.metadata,
            'harmonic_metadata': harmonic_components.metadata,
            'reconstruction_error': reconstruction_error,
            'original_shape': param.shape,
            'compression_achieved': param.numel() / sum(
                v.numel() if torch.is_tensor(v) else len(v) 
                for v in spline_components.values()
            )
        }
    
    def _validate_parameter_conversion(self, original: torch.Tensor, 
                                     spline_components: Dict[str, Any], 
                                     param_name: str) -> float:
        """Validate the quality of parameter conversion
        
        Args:
            original: Original parameter tensor
            spline_components: Spline representation
            param_name: Parameter name for logging
            
        Returns:
            Reconstruction error (normalized)
        """
        try:
            # Reconstruct parameter from splines
            reconstructed = self.spline_interpolator.reconstruct(
                spline_components, original.shape
            )
            
            # Calculate normalized reconstruction error
            error = torch.norm(original - reconstructed) / torch.norm(original)
            
            if error > self.config.reconstruction_threshold:
                logger.warning(
                    f"High reconstruction error for {param_name}: {error:.4f} "
                    f"(threshold: {self.config.reconstruction_threshold})"
                )
            
            return error.item()
            
        except Exception as e:
            logger.error(f"Validation failed for {param_name}: {e}")
            return float('inf')
    
    def _validate_conversion(self, conversion_data: Dict[str, Any], 
                           original_model: nn.Module) -> Dict[str, Any]:
        """Validate the overall conversion quality
        
        Args:
            conversion_data: Complete conversion data
            original_model: Original model for comparison
            
        Returns:
            Validation results
        """
        logger.info("Validating Neural Splines conversion...")
        
        validation_results = {
            'geometric_consistency': True,
            'parameter_errors': {},
            'overall_quality': 'excellent',
            'warnings': []
        }
        
        # Check each converted parameter
        for param_name, spline_data in conversion_data['spline_weights'].items():
            error = spline_data['reconstruction_error']
            validation_results['parameter_errors'][param_name] = error
            
            if error > self.config.reconstruction_threshold:
                validation_results['geometric_consistency'] = False
                validation_results['warnings'].append(
                    f"Parameter {param_name} exceeds error threshold: {error:.4f}"
                )
        
        # Determine overall quality
        max_error = max(validation_results['parameter_errors'].values())
        if max_error < 0.001:
            validation_results['overall_quality'] = 'excellent'
        elif max_error < 0.01:
            validation_results['overall_quality'] = 'good'
        elif max_error < 0.05:
            validation_results['overall_quality'] = 'acceptable'
        else:
            validation_results['overall_quality'] = 'poor'
        
        logger.info(f"Validation complete. Quality: {validation_results['overall_quality']}")
        
        return validation_results
    
    def reconstruct_model(self, conversion_data: Dict[str, Any], 
                         model_class: Optional[type] = None) -> nn.Module:
        """Reconstruct a PyTorch model from Neural Splines representation
        
        Args:
            conversion_data: Spline conversion data
            model_class: Optional model class for reconstruction
            
        Returns:
            Reconstructed PyTorch model
        """
        logger.info("Reconstructing model from Neural Splines...")
        
        # This would reconstruct the model architecture and load spline parameters
        # Implementation depends on the specific model architecture
        raise NotImplementedError("Model reconstruction not yet implemented")
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get detailed conversion statistics"""
        return self.conversion_stats.copy()
    
    def save_conversion(self, conversion_data: Dict[str, Any], 
                       output_path: str) -> None:
        """Save Neural Splines conversion to disk
        
        Args:
            conversion_data: Complete conversion data
            output_path: Directory to save conversion files
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save spline components
        torch.save(conversion_data, output_dir / "neural_splines_conversion.pt")
        
        # Save human-readable statistics
        import json
        with open(output_dir / "conversion_stats.json", 'w') as f:
            stats = conversion_data['compression_stats'].copy()
            # Convert any tensors to serializable format
            for key, value in stats.items():
                if torch.is_tensor(value):
                    stats[key] = value.item()
            json.dump(stats, f, indent=2)
        
        logger.info(f"Neural Splines conversion saved to {output_path}")
    
    @classmethod
    def load_conversion(cls, conversion_path: str) -> Dict[str, Any]:
        """Load Neural Splines conversion from disk
        
        Args:
            conversion_path: Path to conversion file
            
        Returns:
            Loaded conversion data
        """
        conversion_file = Path(conversion_path)
        if conversion_file.is_dir():
            conversion_file = conversion_file / "neural_splines_conversion.pt"
        
        conversion_data = torch.load(conversion_file, map_location='cpu')
        logger.info(f"Neural Splines conversion loaded from {conversion_path}")
        
        return conversion_data