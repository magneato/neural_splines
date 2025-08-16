#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: dense_to_neural_spline.py
# Author: Robert Sitton
# Contact: rsitton@quholo.com
#
# This file is part of a project licensed under the GNU Affero General Public License.
#
# GNU AFFERO GENERAL PUBLIC LICENSE
# Version 3, 19 November 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

"""
dense_to_neural_spline.py
========================

Demonstration of converting dense neural network models to Neural Splines
using the HarmonicCollapseConverter. This script shows how to:

1. Load a pre-trained dense model
2. Convert it to Neural Splines representation
3. Save the compressed model
4. Compare performance and compression ratios
5. Handle various edge cases

The harmonic collapse algorithm discovers the underlying spline structure
within trained networks, achieving dramatic compression while preserving
accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import argparse
from typing import Dict, Any

# Import from neural_spline.py
from neural_spline import (
    SplineLinear, SplineMLP, HarmonicCollapseConverter,
    DenseMLP
)


def create_spline_model_from_dense(dense_model: nn.Module, 
                                  control_ratio: float = 0.1) -> nn.Module:
    """
    Create a complete spline model from a dense model.
    
    This function handles the conversion of various model architectures,
    replacing Linear layers with SplineLinear equivalents.
    """
    # Create a new model with the same structure but spline layers
    spline_model = nn.ModuleDict()
    converter = HarmonicCollapseConverter()
    
    # Convert the model
    spline_data = converter.convert_network(dense_model, control_ratio)
    
    # Reconstruct model with spline layers
    for name, module in dense_model.named_modules():
        if name in spline_data:
            layer_data = spline_data[name]
            
            if isinstance(module, nn.Linear):
                # Create SplineLinear from the converted data
                out_features, in_features = layer_data['weight_shape']
                cp_m, cp_n = layer_data['control_grid']
                
                spline_layer = SplineLinear(
                    in_features=in_features,
                    out_features=out_features,
                    cp_h=cp_m,
                    cp_w=cp_n
                )
                
                # Set the control points
                spline_layer.weight_control_points.data = layer_data['control_points']
                
                # Handle bias
                if 'bias' in layer_data:
                    # Interpolate bias down to control points
                    bias = layer_data['bias']
                    cp_bias = spline_layer.cp_bias
                    if len(bias) >= cp_bias:
                        indices = torch.linspace(0, len(bias)-1, cp_bias).long()
                        spline_layer.bias_control_points.data = bias[indices]
                    else:
                        # Pad if bias is smaller than control points
                        spline_layer.bias_control_points.data[:len(bias)] = bias
                
                spline_model[name] = spline_layer
    
    return spline_model, spline_data


def save_spline_model(spline_model: Dict[str, Any], 
                     spline_data: Dict[str, Dict],
                     filepath: Path):
    """Save spline model to disk."""
    checkpoint = {
        'spline_data': spline_data,
        'model_config': {
            name: {
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'cp_h': layer.cp_h,
                'cp_w': layer.cp_w,
                'cp_bias': layer.cp_bias
            }
            for name, layer in spline_model.items()
            if isinstance(layer, SplineLinear)
        }
    }
    
    # Save state dict
    state_dict = {}
    for name, layer in spline_model.items():
        if isinstance(layer, SplineLinear):
            state_dict[f"{name}.weight_control_points"] = layer.weight_control_points
            state_dict[f"{name}.bias_control_points"] = layer.bias_control_points
    
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, filepath)
    print(f"Saved spline model to {filepath}")


def load_spline_model(filepath: Path) -> Dict[str, SplineLinear]:
    """Load spline model from disk."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    spline_model = nn.ModuleDict()
    config = checkpoint['model_config']
    state_dict = checkpoint['state_dict']
    
    # Recreate layers
    for name, layer_config in config.items():
        layer = SplineLinear(**layer_config)
        layer.weight_control_points = state_dict[f"{name}.weight_control_points"]
        layer.bias_control_points = state_dict[f"{name}.bias_control_points"]
        spline_model[name] = layer
    
    return spline_model


def compare_models(dense_model: nn.Module, spline_model: nn.ModuleDict, 
                  test_input: torch.Tensor):
    """Compare outputs of dense and spline models."""
    dense_model.eval()
    
    with torch.no_grad():
        # Get dense model output
        dense_output = dense_model(test_input)
        
        # Get spline model output (manual forward pass)
        x = test_input
        for name, module in dense_model.named_modules():
            if name in spline_model:
                # Apply spline layer
                x = spline_model[name](x.view(x.size(0), -1))
            elif isinstance(module, nn.ReLU):
                x = F.relu(x)
            elif isinstance(module, nn.Sigmoid):
                x = torch.sigmoid(x)
            elif isinstance(module, nn.Tanh):
                x = torch.tanh(x)
    
    # Compare outputs
    difference = torch.norm(dense_output - x) / torch.norm(dense_output)
    print(f"Relative output difference: {difference:.6f}")
    
    return difference


def analyze_compression(dense_model: nn.Module, spline_data: Dict[str, Dict]):
    """Analyze compression achieved by spline conversion."""
    total_original = 0
    total_compressed = 0
    
    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS")
    print("="*60)
    
    for name, module in dense_model.named_modules():
        if hasattr(module, 'weight'):
            original = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                original += module.bias.numel()
            
            if name in spline_data:
                compressed = spline_data[name]['control_points'].numel()
                if 'bias' in spline_data[name]:
                    # Account for bias control points
                    compressed += min(spline_data[name]['control_grid'][0], 
                                    spline_data[name]['bias'].numel())
                
                ratio = original / compressed
                print(f"{name:30} {original:10,} → {compressed:6,} ({ratio:6.1f}x)")
                
                total_original += original
                total_compressed += compressed
    
    if total_compressed > 0:
        total_ratio = total_original / total_compressed
        print("-"*60)
        print(f"{'TOTAL':30} {total_original:10,} → {total_compressed:6,} ({total_ratio:6.1f}x)")
    print("="*60)


def demo_edge_cases():
    """Demonstrate handling of various edge cases."""
    print("\n" + "="*60)
    print("EDGE CASE DEMONSTRATIONS")
    print("="*60)
    
    converter = HarmonicCollapseConverter()
    
    # Edge case 1: Very small matrix
    print("\n1. Very small matrix (3x3):")
    tiny_layer = nn.Linear(3, 3)
    tiny_result = converter.convert_layer(tiny_layer, control_ratio=0.5)
    print(f"   Original: {tiny_layer.weight.shape}")
    print(f"   Control points: {tiny_result['control_points'].shape}")
    print(f"   Compression: {tiny_result['compression_ratio']:.1f}x")
    
    # Edge case 2: Single dimension
    print("\n2. Single output dimension:")
    single_layer = nn.Linear(100, 1)
    single_result = converter.convert_layer(single_layer)
    print(f"   Original: {single_layer.weight.shape}")
    print(f"   Control points: {single_result['control_points'].shape}")
    
    # Edge case 3: No bias
    print("\n3. Layer without bias:")
    no_bias_layer = nn.Linear(50, 50, bias=False)
    no_bias_result = converter.convert_layer(no_bias_layer)
    print(f"   Has bias in result: {'bias' in no_bias_result}")
    
    # Edge case 4: Conv2D layer
    print("\n4. Convolutional layer:")
    conv_layer = nn.Conv2d(3, 64, kernel_size=3)
    conv_result = converter.convert_layer(conv_layer)
    print(f"   Original shape: {conv_layer.weight.shape}")
    print(f"   Control points: {conv_result['control_points'].shape}")
    print(f"   Compression: {conv_result['compression_ratio']:.1f}x")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Convert dense models to Neural Splines')
    parser.add_argument('--model', type=str, default='demo', 
                       help='Model to convert (demo/path to .pth file)')
    parser.add_argument('--control-ratio', type=float, default=0.05,
                       help='Ratio of control points to original parameters (default: 0.05)')
    parser.add_argument('--output', type=str, default='spline_model.pth',
                       help='Output filename for spline model')
    parser.add_argument('--show-edge-cases', action='store_true',
                       help='Demonstrate edge case handling')
    
    args = parser.parse_args()
    
    print("🌊 NEURAL SPLINE HARMONIC COLLAPSE CONVERTER 🌊")
    print("="*60)
    
    if args.show_edge_cases:
        demo_edge_cases()
        return
    
    # Create or load model
    if args.model == 'demo':
        print("Creating demo MLP model...")
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Initialize with some reasonable weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    else:
        print(f"Loading model from {args.model}...")
        model = torch.load(args.model, map_location='cpu')
    
    # Convert to splines
    print(f"\nConverting with control ratio: {args.control_ratio}")
    spline_model, spline_data = create_spline_model_from_dense(model, args.control_ratio)
    
    # Analyze compression
    analyze_compression(model, spline_data)
    
    # Test conversion accuracy
    print("\nTesting conversion accuracy...")
    test_input = torch.randn(32, 784)  # Batch of 32 MNIST-like inputs
    difference = compare_models(model, spline_model, test_input)
    
    if difference < 0.01:
        print("✅ Excellent conversion quality!")
    elif difference < 0.05:
        print("⚠️  Good conversion quality, minor differences")
    else:
        print("❌ Significant differences, consider increasing control points")
    
    # Save the spline model
    save_spline_model(spline_model, spline_data, Path(args.output))
    
    # Demonstrate loading
    print(f"\nLoading saved model from {args.output}...")
    loaded_model = load_spline_model(Path(args.output))
    print(f"Successfully loaded {len(loaded_model)} spline layers")
    
    print("\n🍪 The neural network never existed. Only curves in space. 🍪")


if __name__ == "__main__":
    main()