"""
Neural Splines Visualization

Creates beautiful visualizations of spline structures, control points,
and parameter manifolds. Makes the geometric nature of Neural Splines
visible and interpretable.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..models.base_neural import BaseNeuralModel
from ..exceptions import VisualizationError

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SplineVisualizer:
    """
    Main class for creating Neural Splines visualizations
    
    Provides methods to visualize:
    - Control point distributions
    - Spline curves and manifolds
    - Compression statistics
    - Model architectures
    """
    
    def __init__(self, model: BaseNeuralModel):
        """Initialize visualizer with a Neural Splines model
        
        Args:
            model: Neural Splines model to visualize
        """
        self.model = model
        self.layer_names = model.get_spline_layer_names()
        
        # Visualization settings
        self.figsize = (12, 8)
        self.dpi = 300
        self.colormap = 'viridis'
        
        logger.debug(f"Initialized SplineVisualizer with {len(self.layer_names)} spline layers")
    
    def plot_model_overview(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create overview visualization of the entire model
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🌊 Neural Splines Model Overview', fontsize=16, fontweight='bold')
        
        try:
            # Plot 1: Compression statistics
            self._plot_compression_overview(axes[0, 0])
            
            # Plot 2: Layer-wise compression ratios
            self._plot_layer_compression(axes[0, 1])
            
            # Plot 3: Control point distribution
            self._plot_control_point_distribution(axes[1, 0])
            
            # Plot 4: Spline quality metrics
            self._plot_quality_metrics(axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Model overview saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create model overview: {e}")
    
    def plot_layer_splines(self, layer_name: str, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize splines for a specific layer
        
        Args:
            layer_name: Name of the layer to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Get control points for the layer
            control_points = self.model.get_control_points(layer_name)
            
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle(f'🎯 Layer: {layer_name}', fontsize=14, fontweight='bold')
            
            # Plot 1: Control points sequence
            self._plot_control_points_sequence(control_points, axes[0, 0])
            
            # Plot 2: Spline curve reconstruction
            self._plot_spline_curve(control_points, axes[0, 1])
            
            # Plot 3: Control point distribution histogram
            self._plot_control_point_histogram(control_points, axes[1, 0])
            
            # Plot 4: Curvature analysis
            self._plot_curvature_analysis(control_points, axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Layer visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to visualize layer {layer_name}: {e}", plot_type="layer_splines")
    
    def plot_parameter_manifold(self, layer_name: str, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the parameter manifold for a layer
        
        Args:
            layer_name: Name of the layer
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            control_points = self.model.get_control_points(layer_name)
            
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f'🌐 Parameter Manifold: {layer_name}', fontsize=16, fontweight='bold')
            
            # Create 3D subplot for manifold visualization
            ax1 = fig.add_subplot(221, projection='3d')
            self._plot_3d_manifold(control_points, ax1)
            
            # 2D projection
            ax2 = fig.add_subplot(222)
            self._plot_2d_manifold_projection(control_points, ax2)
            
            # Manifold properties
            ax3 = fig.add_subplot(223)
            self._plot_manifold_properties(control_points, ax3)
            
            # Geometric flow
            ax4 = fig.add_subplot(224)
            self._plot_geometric_flow(control_points, ax4)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Manifold visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to visualize manifold for {layer_name}: {e}", plot_type="manifold")
    
    def plot_compression_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create detailed compression analysis visualization
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            stats = self.model.get_compression_stats()
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('📊 Neural Splines Compression Analysis', fontsize=16, fontweight='bold')
            
            # Plot compression ratio comparison
            self._plot_compression_comparison(stats, axes[0, 0])
            
            # Plot memory usage breakdown
            self._plot_memory_breakdown(stats, axes[0, 1])
            
            # Plot efficiency metrics
            self._plot_efficiency_metrics(stats, axes[0, 2])
            
            # Plot parameter distribution
            self._plot_parameter_distribution(axes[1, 0])
            
            # Plot compression by layer type
            self._plot_compression_by_type(axes[1, 1])
            
            # Plot quality vs compression tradeoff
            self._plot_quality_compression_tradeoff(axes[1, 2])
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Compression analysis saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create compression analysis: {e}", plot_type="compression")
    
    def create_interactive_dashboard(self) -> Any:
        """Create interactive Plotly dashboard
        
        Returns:
            Plotly figure object or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive visualizations")
            return None
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Model Overview', 'Layer Analysis', 'Compression Stats', 'Interactive Exploration'],
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "pie"}, {"type": "scatter3d"}]]
            )
            
            # Add plots
            stats = self.model.get_compression_stats()
            
            # Compression comparison
            fig.add_trace(
                go.Bar(
                    x=['Original', 'Compressed'],
                    y=[stats['original_params'], stats['compressed_params']],
                    name='Parameters',
                    marker_color=['red', 'green']
                ),
                row=1, col=1
            )
            
            # Layer compression ratios
            layer_stats = stats.get('layer_stats', {})
            if layer_stats:
                layer_names = list(layer_stats.keys())[:10]  # Top 10 layers
                compression_ratios = [layer_stats[name]['layer_compression_ratio'] for name in layer_names]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(layer_names))),
                        y=compression_ratios,
                        mode='markers+lines',
                        name='Layer Compression'
                    ),
                    row=1, col=2
                )
            
            # Memory breakdown
            fig.add_trace(
                go.Pie(
                    labels=['Compressed Model', 'Saved Space'],
                    values=[stats['memory_gb'], stats['original_params']/1e9 - stats['memory_gb']],
                    name='Memory Usage'
                ),
                row=2, col=1
            )
            
            # 3D visualization of first layer control points
            if self.layer_names:
                try:
                    control_points = self.model.get_control_points(self.layer_names[0])
                    points_3d = self._prepare_3d_points(control_points)
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=points_3d[:, 0],
                            y=points_3d[:, 1],
                            z=points_3d[:, 2],
                            mode='markers',
                            marker=dict(size=5, color=points_3d[:, 2], colorscale='viridis'),
                            name='Control Points'
                        ),
                        row=2, col=2
                    )
                except Exception:
                    pass  # Skip 3D plot if it fails
            
            # Update layout
            fig.update_layout(
                title='🌊 Neural Splines Interactive Dashboard',
                showlegend=True,
                height=800
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create interactive dashboard: {e}", plot_type="interactive")
    
    # Helper plotting methods
    
    def _plot_compression_overview(self, ax):
        """Plot compression overview"""
        stats = self.model.get_compression_stats()
        
        # Create bar chart
        categories = ['Original\nParameters', 'Compressed\nParameters']
        values = [stats['original_params'] / 1e6, stats['compressed_params'] / 1e6]  # Convert to millions
        colors = ['#ff6b6b', '#4ecdc4']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Parameters (Millions)')
        ax.set_title(f'📈 Compression: {stats["compression_ratio"]:.1f}x')
        ax.grid(True, alpha=0.3)
    
    def _plot_layer_compression(self, ax):
        """Plot layer-wise compression ratios"""
        layer_stats = self.model.get_compression_stats().get('layer_stats', {})
        
        if not layer_stats:
            ax.text(0.5, 0.5, 'No layer statistics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Compression Ratios')
            return
        
        # Get top 15 layers by compression ratio
        sorted_layers = sorted(layer_stats.items(), 
                             key=lambda x: x[1]['layer_compression_ratio'], 
                             reverse=True)[:15]
        
        layer_names = [name.split('.')[-1] for name, _ in sorted_layers]  # Simplified names
        ratios = [stats['layer_compression_ratio'] for _, stats in sorted_layers]
        
        bars = ax.barh(range(len(layer_names)), ratios, color='skyblue', alpha=0.8)
        
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=8)
        ax.set_xlabel('Compression Ratio')
        ax.set_title('🔍 Layer Compression Ratios')
        ax.grid(True, alpha=0.3)
        
        # Add ratio labels
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                   f'{ratio:.1f}x', ha='left', va='center', fontsize=7)
    
    def _plot_control_point_distribution(self, ax):
        """Plot distribution of control points across layers"""
        try:
            point_counts = []
            layer_names = []
            
            for layer_name in self.layer_names[:20]:  # Top 20 layers
                try:
                    control_points = self.model.get_control_points(layer_name)
                    point_counts.append(control_points.numel())
                    layer_names.append(layer_name.split('.')[-1])  # Simplified name
                except Exception:
                    continue
            
            if point_counts:
                ax.hist(point_counts, bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Number of Control Points')
                ax.set_ylabel('Number of Layers')
                ax.set_title('📊 Control Point Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_points = np.mean(point_counts)
                ax.axvline(mean_points, color='red', linestyle='--', 
                          label=f'Mean: {mean_points:.0f}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No control points data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Control Point Distribution')
                
        except Exception as e:
            logger.warning(f"Failed to plot control point distribution: {e}")
            ax.text(0.5, 0.5, 'Distribution plot failed', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Control Point Distribution')
    
    def _plot_quality_metrics(self, ax):
        """Plot quality metrics"""
        try:
            # Get quality metrics from model validation
            validation_results = self.model.validate_spline_quality()
            
            if 'layer_results' in validation_results:
                layer_results = validation_results['layer_results']
                
                quality_counts = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}
                
                for layer_result in layer_results.values():
                    quality = layer_result.get('quality', 'unknown')
                    if quality in quality_counts:
                        quality_counts[quality] += 1
                
                # Create pie chart
                labels = [k for k, v in quality_counts.items() if v > 0]
                sizes = [v for v in quality_counts.values() if v > 0]
                colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6'][:len(labels)]
                
                if sizes:
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                                     autopct='%1.1f%%', startangle=90)
                    ax.set_title('🎯 Spline Quality Distribution')
                else:
                    ax.text(0.5, 0.5, 'No quality data', ha='center', va='center')
                    ax.set_title('Spline Quality')
            else:
                ax.text(0.5, 0.5, 'Quality metrics unavailable', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Spline Quality')
                
        except Exception as e:
            logger.warning(f"Failed to plot quality metrics: {e}")
            ax.text(0.5, 0.5, 'Quality plot failed', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Spline Quality')
    
    def _plot_control_points_sequence(self, control_points: torch.Tensor, ax):
        """Plot control points as a sequence"""
        points = control_points.flatten().detach().cpu().numpy()
        
        ax.plot(points, 'o-', color='blue', alpha=0.7, markersize=4)
        ax.set_xlabel('Control Point Index')
        ax.set_ylabel('Value')
        ax.set_title('Control Points Sequence')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(points)
        std_val = np.std(points)
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.axhline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'±σ: {std_val:.3f}')
        ax.axhline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
        ax.legend()
    
    def _plot_spline_curve(self, control_points: torch.Tensor, ax):
        """Plot reconstructed spline curve"""
        points = control_points.flatten().detach().cpu().numpy()
        
        # Create dense interpolation for smooth curve
        x_points = np.linspace(0, 1, len(points))
        x_dense = np.linspace(0, 1, len(points) * 10)
        
        # Simple interpolation (in full implementation, would use proper spline)
        y_dense = np.interp(x_dense, x_points, points)
        
        # Plot control points and spline curve
        ax.plot(x_points, points, 'ro', markersize=6, label='Control Points', alpha=0.8)
        ax.plot(x_dense, y_dense, 'b-', linewidth=2, label='Spline Curve', alpha=0.7)
        
        ax.set_xlabel('Parameter Position')
        ax.set_ylabel('Value')
        ax.set_title('Reconstructed Spline Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_control_point_histogram(self, control_points: torch.Tensor, ax):
        """Plot histogram of control point values"""
        points = control_points.flatten().detach().cpu().numpy()
        
        ax.hist(points, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Control Point Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Value Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add distribution statistics
        mean_val = np.mean(points)
        median_val = np.median(points)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.3f}')
        ax.legend()
    
    def _plot_curvature_analysis(self, control_points: torch.Tensor, ax):
        """Plot curvature analysis of control points"""
        points = control_points.flatten().detach().cpu().numpy()
        
        if len(points) > 2:
            # Calculate approximate curvature using second differences
            first_diff = np.diff(points)
            second_diff = np.diff(first_diff)
            
            # Plot curvature
            x_curvature = range(1, len(second_diff) + 1)
            ax.plot(x_curvature, np.abs(second_diff), 'g-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Position')
            ax.set_ylabel('|Curvature|')
            ax.set_title('Spline Curvature Analysis')
            ax.grid(True, alpha=0.3)
            
            # Add mean curvature line
            mean_curvature = np.mean(np.abs(second_diff))
            ax.axhline(mean_curvature, color='red', linestyle='--', 
                      label=f'Mean: {mean_curvature:.4f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient points for curvature analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Curvature Analysis')
    
    def _plot_3d_manifold(self, control_points: torch.Tensor, ax):
        """Plot 3D manifold representation"""
        points_3d = self._prepare_3d_points(control_points)
        
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                           c=points_3d[:, 2], cmap=self.colormap, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Parameter Manifold')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    
    def _plot_2d_manifold_projection(self, control_points: torch.Tensor, ax):
        """Plot 2D projection of manifold"""
        points_3d = self._prepare_3d_points(control_points)
        
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], 
                           c=points_3d[:, 2], cmap=self.colormap, alpha=0.7)
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('2D Manifold Projection')
        
        plt.colorbar(scatter, ax=ax)
    
    def _prepare_3d_points(self, control_points: torch.Tensor) -> np.ndarray:
        """Prepare control points for 3D visualization"""
        points = control_points.flatten().detach().cpu().numpy()
        
        # Create 3D coordinates
        n_points = len(points)
        
        if n_points >= 3:
            # Reshape into 3D if possible
            n_3d = n_points // 3
            points_3d = points[:n_3d * 3].reshape(-1, 3)
        else:
            # Create artificial 3D coordinates
            points_3d = np.column_stack([
                np.arange(len(points)),
                points,
                np.zeros(len(points))
            ])
        
        return points_3d
    
    def _plot_manifold_properties(self, control_points: torch.Tensor, ax):
        """Plot manifold geometric properties"""
        # Placeholder for manifold properties visualization
        ax.text(0.5, 0.5, 'Manifold Properties\n(Advanced geometric analysis)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Manifold Properties')
    
    def _plot_geometric_flow(self, control_points: torch.Tensor, ax):
        """Plot geometric information flow"""
        # Placeholder for geometric flow visualization
        ax.text(0.5, 0.5, 'Geometric Flow\n(Information propagation)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Geometric Flow')
    
    # Additional helper methods for compression analysis
    
    def _plot_compression_comparison(self, stats: Dict[str, Any], ax):
        """Plot compression comparison"""
        # Implementation for compression comparison plot
        pass
    
    def _plot_memory_breakdown(self, stats: Dict[str, Any], ax):
        """Plot memory usage breakdown"""
        # Implementation for memory breakdown plot
        pass
    
    def _plot_efficiency_metrics(self, stats: Dict[str, Any], ax):
        """Plot efficiency metrics"""
        # Implementation for efficiency metrics plot
        pass
    
    def _plot_parameter_distribution(self, ax):
        """Plot parameter distribution across layers"""
        # Implementation for parameter distribution plot
        pass
    
    def _plot_compression_by_type(self, ax):
        """Plot compression ratios by layer type"""
        # Implementation for compression by type plot
        pass
    
    def _plot_quality_compression_tradeoff(self, ax):
        """Plot quality vs compression tradeoff"""
        # Implementation for quality-compression tradeoff plot
        pass

def create_spline_visualization(model: BaseNeuralModel, 
                              layer_name: Optional[str] = None,
                              visualization_type: str = "overview",
                              **kwargs) -> plt.Figure:
    """
    High-level function to create Neural Splines visualizations
    
    Args:
        model: Neural Splines model
        layer_name: Specific layer to visualize (optional)
        visualization_type: Type of visualization ("overview", "layer", "manifold", "compression")
        **kwargs: Additional visualization parameters
        
    Returns:
        Matplotlib figure object
    """
    try:
        visualizer = SplineVisualizer(model)
        
        if visualization_type == "overview":
            return visualizer.plot_model_overview(**kwargs)
        elif visualization_type == "layer" and layer_name:
            return visualizer.plot_layer_splines(layer_name, **kwargs)
        elif visualization_type == "manifold" and layer_name:
            return visualizer.plot_parameter_manifold(layer_name, **kwargs)
        elif visualization_type == "compression":
            return visualizer.plot_compression_analysis(**kwargs)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
            
    except Exception as e:
        raise VisualizationError(f"Failed to create {visualization_type} visualization: {e}")