"""
Neural Splines Compression Optimizer

Advanced optimization algorithms for achieving optimal compression ratios
while maintaining model quality. Implements gradient-based and evolutionary
optimization for spline control point placement.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.interpolation import SplineComponents
from ..exceptions import CompressionError

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for compression optimization"""
    max_iterations: int = 100
    learning_rate: float = 0.01
    tolerance: float = 1e-6
    quality_weight: float = 0.7
    compression_weight: float = 0.3
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    use_adaptive_lr: bool = True

class CompressionObjective(ABC):
    """Abstract base class for compression objectives"""
    
    @abstractmethod
    def compute_loss(self, original: torch.Tensor, 
                    reconstructed: torch.Tensor,
                    spline_components: SplineComponents) -> torch.Tensor:
        """Compute optimization objective"""
        pass

class QualityCompressionObjective(CompressionObjective):
    """Objective that balances reconstruction quality and compression ratio"""
    
    def __init__(self, target_compression: float = 128.9, 
                 quality_weight: float = 0.7):
        self.target_compression = target_compression
        self.quality_weight = quality_weight
        self.compression_weight = 1.0 - quality_weight
    
    def compute_loss(self, original: torch.Tensor,
                    reconstructed: torch.Tensor, 
                    spline_components: SplineComponents) -> torch.Tensor:
        """Compute combined quality-compression loss"""
        
        # Reconstruction quality loss (MSE)
        quality_loss = torch.nn.functional.mse_loss(original, reconstructed)
        
        # Compression efficiency loss
        original_params = original.numel()
        compressed_params = spline_components.control_points.numel()
        current_compression = original_params / compressed_params
        
        # Penalize deviation from target compression
        compression_loss = torch.abs(current_compression - self.target_compression) / self.target_compression
        
        # Combined loss
        total_loss = (self.quality_weight * quality_loss + 
                     self.compression_weight * compression_loss)
        
        return total_loss

class PerceptualCompressionObjective(CompressionObjective):
    """Objective that considers perceptual quality metrics"""
    
    def __init__(self, target_compression: float = 128.9):
        self.target_compression = target_compression
    
    def compute_loss(self, original: torch.Tensor,
                    reconstructed: torch.Tensor,
                    spline_components: SplineComponents) -> torch.Tensor:
        """Compute perceptual quality loss"""
        
        # Structural similarity loss
        ssim_loss = 1.0 - self._compute_ssim(original, reconstructed)
        
        # Frequency domain loss
        freq_loss = self._compute_frequency_loss(original, reconstructed)
        
        # Smoothness preservation loss
        smoothness_loss = self._compute_smoothness_loss(original, reconstructed)
        
        # Compression efficiency
        original_params = original.numel()
        compressed_params = spline_components.control_points.numel()
        current_compression = original_params / compressed_params
        compression_loss = torch.abs(current_compression - self.target_compression) / self.target_compression
        
        # Weighted combination
        total_loss = (0.4 * ssim_loss + 0.3 * freq_loss + 
                     0.2 * smoothness_loss + 0.1 * compression_loss)
        
        return total_loss
    
    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Structural Similarity Index"""
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        
        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        
        return torch.clamp(ssim, 0, 1)
    
    def _compute_frequency_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain loss"""
        try:
            x_fft = torch.fft.fft(x.flatten().float())
            y_fft = torch.fft.fft(y.flatten().float())
            
            # Compare magnitude spectra
            freq_loss = torch.nn.functional.mse_loss(torch.abs(x_fft), torch.abs(y_fft))
            
            return freq_loss
        except Exception:
            return torch.tensor(0.0)
    
    def _compute_smoothness_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute smoothness preservation loss"""
        if len(x.shape) == 1:
            if len(x) > 2:
                x_smooth = torch.var(torch.diff(x, n=2))
                y_smooth = torch.var(torch.diff(y, n=2))
                return torch.abs(x_smooth - y_smooth)
        
        return torch.tensor(0.0)

class CompressionOptimizer:
    """
    Advanced optimizer for Neural Splines compression
    
    Uses gradient-based optimization to find optimal control point placements
    that maximize compression while preserving reconstruction quality.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None,
                 objective: Optional[CompressionObjective] = None):
        """Initialize compression optimizer
        
        Args:
            config: Optimization configuration
            objective: Compression objective function
        """
        self.config = config or OptimizationConfig()
        self.objective = objective or QualityCompressionObjective()
        
        # Optimization state
        self.optimization_history = []
        self.best_loss = float('inf')
        self.best_splines = None
        self.patience_counter = 0
        
        logger.debug(f"Initialized CompressionOptimizer with {type(self.objective).__name__}")
    
    def optimize_splines(self, original_tensor: torch.Tensor,
                        initial_splines: SplineComponents,
                        spline_reconstructor: Callable) -> SplineComponents:
        """
        Optimize spline control points for better compression-quality tradeoff
        
        Args:
            original_tensor: Original parameter tensor
            initial_splines: Initial spline representation
            spline_reconstructor: Function to reconstruct tensor from splines
            
        Returns:
            Optimized spline components
        """
        logger.info("Starting spline compression optimization...")
        
        try:
            # Initialize optimization
            optimized_splines = self._initialize_optimization(initial_splines)
            
            # Setup optimizer
            optimizer = self._create_optimizer(optimized_splines)
            scheduler = self._create_scheduler(optimizer) if self.config.use_adaptive_lr else None
            
            # Optimization loop
            for iteration in range(self.config.max_iterations):
                
                # Forward pass
                reconstructed = spline_reconstructor(optimized_splines, original_tensor.shape)
                loss = self.objective.compute_loss(original_tensor, reconstructed, optimized_splines)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [optimized_splines.control_points], 
                        self.config.gradient_clip_norm
                    )
                
                optimizer.step()
                
                # Learning rate scheduling
                if scheduler:
                    scheduler.step(loss)
                
                # Track progress
                self._update_optimization_history(iteration, loss.item(), optimized_splines)
                
                # Early stopping check
                if self._check_early_stopping(loss.item()):
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
                
                # Convergence check
                if loss.item() < self.config.tolerance:
                    logger.info(f"Converged at iteration {iteration}")
                    break
                
                if iteration % 20 == 0:
                    logger.debug(f"Iteration {iteration}: loss = {loss.item():.6f}")
            
            # Return best result
            result_splines = self.best_splines if self.best_splines is not None else optimized_splines
            
            logger.info(f"Optimization completed. Best loss: {self.best_loss:.6f}")
            
            return result_splines
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise CompressionError(f"Spline optimization failed: {e}")
    
    def optimize_global_compression(self, layers_data: Dict[str, Any],
                                  target_overall_compression: float) -> Dict[str, Any]:
        """
        Globally optimize compression across all layers
        
        Args:
            layers_data: Dictionary of layer compression data
            target_overall_compression: Target overall compression ratio
            
        Returns:
            Optimized layers data
        """
        logger.info("Starting global compression optimization...")
        
        # Calculate current overall compression
        total_original = sum(data['compression_stats']['original_parameters'] 
                           for data in layers_data.values())
        total_compressed = sum(data['compression_stats']['compressed_parameters']
                             for data in layers_data.values())
        current_compression = total_original / total_compressed
        
        if abs(current_compression - target_overall_compression) < 1.0:
            # Already close enough
            return layers_data
        
        # Determine adjustment strategy
        adjustment_factor = target_overall_compression / current_compression
        
        optimized_layers = {}
        
        for layer_name, layer_data in layers_data.items():
            try:
                # Calculate layer-specific adjustment
                layer_priority = self._get_layer_priority(layer_name)
                layer_adjustment = self._calculate_layer_adjustment(
                    adjustment_factor, layer_priority
                )
                
                # Apply adjustment
                if layer_adjustment > 1.0:
                    # Need more compression
                    optimized_splines = self._increase_layer_compression(
                        layer_data['spline_components'], layer_adjustment
                    )
                else:
                    # Need less compression (better quality)
                    optimized_splines = self._decrease_layer_compression(
                        layer_data['spline_components'], 1.0 / layer_adjustment
                    )
                
                # Update layer data
                optimized_layer_data = layer_data.copy()
                optimized_layer_data['spline_components'] = optimized_splines
                
                # Recalculate compression stats
                original_params = layer_data['compression_stats']['original_parameters']
                new_compressed_params = optimized_splines.control_points.numel()
                optimized_layer_data['compression_stats']['compressed_parameters'] = new_compressed_params
                optimized_layer_data['compression_stats']['compression_ratio'] = original_params / new_compressed_params
                
                optimized_layers[layer_name] = optimized_layer_data
                
            except Exception as e:
                logger.warning(f"Failed to optimize layer {layer_name}: {e}")
                optimized_layers[layer_name] = layer_data
        
        return optimized_layers
    
    def evolutionary_optimization(self, original_tensor: torch.Tensor,
                                initial_splines: SplineComponents,
                                spline_reconstructor: Callable,
                                population_size: int = 20,
                                generations: int = 50) -> SplineComponents:
        """
        Evolutionary optimization for control point placement
        
        Args:
            original_tensor: Original parameter tensor
            initial_splines: Initial spline representation
            spline_reconstructor: Function to reconstruct tensor from splines
            population_size: Size of evolutionary population
            generations: Number of generations
            
        Returns:
            Evolutionarily optimized spline components
        """
        logger.info(f"Starting evolutionary optimization with {population_size} individuals, {generations} generations")
        
        try:
            # Initialize population
            population = self._initialize_population(initial_splines, population_size)
            
            for generation in range(generations):
                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    reconstructed = spline_reconstructor(individual, original_tensor.shape)
                    loss = self.objective.compute_loss(original_tensor, reconstructed, individual)
                    fitness_scores.append(1.0 / (1.0 + loss.item()))  # Convert loss to fitness
                
                # Selection and reproduction
                population = self._evolve_population(population, fitness_scores)
                
                # Track best individual
                best_idx = np.argmax(fitness_scores)
                if fitness_scores[best_idx] > 1.0 / (1.0 + self.best_loss):
                    self.best_loss = 1.0 / fitness_scores[best_idx] - 1.0
                    self.best_splines = population[best_idx]
                
                if generation % 10 == 0:
                    logger.debug(f"Generation {generation}: best fitness = {max(fitness_scores):.4f}")
            
            return self.best_splines if self.best_splines is not None else population[0]
            
        except Exception as e:
            logger.error(f"Evolutionary optimization failed: {e}")
            return initial_splines
    
    # Private methods
    
    def _initialize_optimization(self, initial_splines: SplineComponents) -> SplineComponents:
        """Initialize optimization with trainable parameters"""
        # Create a copy with trainable control points
        control_points = initial_splines.control_points.clone().detach().requires_grad_(True)
        
        # Create new spline components with trainable parameters
        optimized_splines = SplineComponents(
            control_points=control_points,
            knot_vectors=initial_splines.knot_vectors,
            spline_coefficients=initial_splines.spline_coefficients,
            basis_functions=initial_splines.basis_functions,
            interpolation_grid=initial_splines.interpolation_grid,
            reconstruction_weights=initial_splines.reconstruction_weights,
            metadata=initial_splines.metadata
        )
        
        return optimized_splines
    
    def _create_optimizer(self, splines: SplineComponents) -> optim.Optimizer:
        """Create PyTorch optimizer for spline parameters"""
        return optim.Adam([splines.control_points], lr=self.config.learning_rate)
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
    
    def _update_optimization_history(self, iteration: int, loss: float, 
                                   splines: SplineComponents):
        """Update optimization tracking"""
        self.optimization_history.append({
            'iteration': iteration,
            'loss': loss,
            'control_points_norm': torch.norm(splines.control_points).item(),
            'num_control_points': splines.control_points.numel()
        })
        
        # Track best result
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_splines = SplineComponents(
                control_points=splines.control_points.clone().detach(),
                knot_vectors=splines.knot_vectors,
                spline_coefficients=splines.spline_coefficients,
                basis_functions=splines.basis_functions,
                interpolation_grid=splines.interpolation_grid,
                reconstruction_weights=splines.reconstruction_weights,
                metadata=splines.metadata
            )
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def _check_early_stopping(self, current_loss: float) -> bool:
        """Check if early stopping criteria are met"""
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _get_layer_priority(self, layer_name: str) -> str:
        """Determine layer priority for global optimization"""
        name_lower = layer_name.lower()
        
        if any(keyword in name_lower for keyword in ['attention', 'attn']):
            return 'high'
        elif any(keyword in name_lower for keyword in ['embed', 'output', 'head']):
            return 'critical'
        elif any(keyword in name_lower for keyword in ['norm', 'bias']):
            return 'low'
        else:
            return 'medium'
    
    def _calculate_layer_adjustment(self, global_adjustment: float, priority: str) -> float:
        """Calculate layer-specific adjustment factor"""
        if priority == 'critical':
            return 1.0 + 0.3 * (global_adjustment - 1.0)  # Less aggressive
        elif priority == 'high':
            return 1.0 + 0.7 * (global_adjustment - 1.0)
        elif priority == 'low':
            return 1.0 + 1.5 * (global_adjustment - 1.0)  # More aggressive
        else:  # medium
            return global_adjustment
    
    def _increase_layer_compression(self, splines: SplineComponents, factor: float) -> SplineComponents:
        """Increase compression for a layer"""
        current_points = len(splines.control_points)
        target_points = max(4, int(current_points / factor))
        
        if target_points >= current_points:
            return splines
        
        # Subsample control points
        indices = torch.linspace(0, current_points - 1, target_points).long()
        new_control_points = splines.control_points[indices]
        
        return SplineComponents(
            control_points=new_control_points,
            knot_vectors=splines.knot_vectors,
            spline_coefficients=splines.spline_coefficients,
            basis_functions=splines.basis_functions,
            interpolation_grid=splines.interpolation_grid,
            reconstruction_weights=splines.reconstruction_weights,
            metadata=splines.metadata
        )
    
    def _decrease_layer_compression(self, splines: SplineComponents, factor: float) -> SplineComponents:
        """Decrease compression for a layer (improve quality)"""
        # For now, just return original - would need sophisticated interpolation
        return splines
    
    def _initialize_population(self, base_splines: SplineComponents, 
                             population_size: int) -> List[SplineComponents]:
        """Initialize evolutionary population"""
        population = []
        
        for _ in range(population_size):
            # Create variant by adding noise to control points
            noise_scale = 0.1 * torch.std(base_splines.control_points)
            noise = torch.randn_like(base_splines.control_points) * noise_scale
            
            variant_control_points = base_splines.control_points + noise
            
            variant = SplineComponents(
                control_points=variant_control_points,
                knot_vectors=base_splines.knot_vectors,
                spline_coefficients=base_splines.spline_coefficients,
                basis_functions=base_splines.basis_functions,
                interpolation_grid=base_splines.interpolation_grid,
                reconstruction_weights=base_splines.reconstruction_weights,
                metadata=base_splines.metadata
            )
            
            population.append(variant)
        
        return population
    
    def _evolve_population(self, population: List[SplineComponents], 
                          fitness_scores: List[float]) -> List[SplineComponents]:
        """Evolve population through selection, crossover, and mutation"""
        
        # Selection: choose top 50% as parents
        sorted_indices = np.argsort(fitness_scores)[::-1]
        num_parents = len(population) // 2
        parents = [population[i] for i in sorted_indices[:num_parents]]
        
        # Generate offspring
        new_population = parents.copy()  # Keep best parents
        
        while len(new_population) < len(population):
            # Select two random parents
            parent1 = parents[np.random.randint(len(parents))]
            parent2 = parents[np.random.randint(len(parents))]
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population[:len(population)]
    
    def _crossover(self, parent1: SplineComponents, parent2: SplineComponents) -> SplineComponents:
        """Perform crossover between two parents"""
        
        # Simple blend crossover for control points
        alpha = np.random.random()
        child_control_points = alpha * parent1.control_points + (1 - alpha) * parent2.control_points
        
        # Inherit other components from parent1
        child = SplineComponents(
            control_points=child_control_points,
            knot_vectors=parent1.knot_vectors,
            spline_coefficients=parent1.spline_coefficients,
            basis_functions=parent1.basis_functions,
            interpolation_grid=parent1.interpolation_grid,
            reconstruction_weights=parent1.reconstruction_weights,
            metadata=parent1.metadata
        )
        
        return child
    
    def _mutate(self, individual: SplineComponents, mutation_rate: float = 0.1) -> SplineComponents:
        """Apply mutation to an individual"""
        
        if np.random.random() < mutation_rate:
            # Add small random noise
            noise_scale = 0.05 * torch.std(individual.control_points)
            noise = torch.randn_like(individual.control_points) * noise_scale
            
            mutated_control_points = individual.control_points + noise
            
            mutated = SplineComponents(
                control_points=mutated_control_points,
                knot_vectors=individual.knot_vectors,
                spline_coefficients=individual.spline_coefficients,
                basis_functions=individual.basis_functions,
                interpolation_grid=individual.interpolation_grid,
                reconstruction_weights=individual.reconstruction_weights,
                metadata=individual.metadata
            )
            
            return mutated
        
        return individual
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization process"""
        if not self.optimization_history:
            return {}
        
        losses = [entry['loss'] for entry in self.optimization_history]
        
        return {
            'total_iterations': len(self.optimization_history),
            'initial_loss': losses[0] if losses else None,
            'final_loss': losses[-1] if losses else None,
            'best_loss': self.best_loss,
            'convergence_rate': (losses[0] - losses[-1]) / losses[0] if len(losses) > 1 and losses[0] > 0 else 0.0,
            'optimization_history': self.optimization_history
        }