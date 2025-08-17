"""Simplified working layout for Neural Splines."""
__version__="1.0.0"
try:
    from .core.neural_spline import SplineLinear, SplineMLP, DenseMLP, HarmonicCollapseConverter as SplineConverter
except Exception:
    SplineLinear=SplineMLP=DenseMLP=SplineConverter=None
from .models.base_neural import BaseNeuralModel
from .models.deepseek_neural import DeepSeekNeuralModel
from .compression.adaptive import DeepSeekSplineAdapter
from .compression.optimizer import CompressionOptimizer
from .utils.geometric_validation import geometric_validation
from .utils.spline_interpolation import spline_interpolation
