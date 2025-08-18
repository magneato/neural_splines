"""Simplified working layout for Neural Splines."""
__version__="1.0.0"

try:
    # Export core spline classes and converter. The SplineConverter alias
    # exposes the harmonic collapse converter used throughout the project.
    from .core.neural_spline import (
        SplineLinear,
        SplineMLP,
        DenseMLP,
        HarmonicCollapseConverter as SplineConverter,
    )
except Exception:
    # If import fails (e.g. during partial installation), gracefully degrade
    SplineLinear = SplineMLP = DenseMLP = SplineConverter = None

# Expose the harmonic decomposer and parameter manifold analysis at the
# top-level package so users can import them directly.
try:
    from .core.harmonic import HarmonicDecomposer
except Exception:
    HarmonicDecomposer = None

try:
    from .core.manifold import ParameterManifold
except Exception:
    ParameterManifold = None

from .models.base_neural import BaseNeuralModel
from .models.deepseek_neural import DeepSeekNeuralModel
from .compression.adaptive import DeepSeekSplineAdapter
from .compression.optimizer import CompressionOptimizer
# from .utils.geometric_validation import geometric_validation
from .utils.spline_interpolation import spline_interpolation
