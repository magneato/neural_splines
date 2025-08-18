"""
Neural Splines Core Module

Core algorithms for Neural Splines including harmonic decomposition,
manifold analysis, spline interpolation, and conversion utilities.
"""

from .converter import SplineConverter, ConversionConfig
from .harmonic import HarmonicDecomposer, HarmonicComponents
from .manifold import ParameterManifold, ManifoldStructure
from .interpolation import SplineInterpolator, SplineComponents

__all__ = [
    'SplineConverter',
    'ConversionConfig',
    'HarmonicDecomposer', 
    'HarmonicComponents',
    'ParameterManifold',
    'ManifoldStructure',
    'SplineInterpolator',
    'SplineComponents',
]