"""
Neural Splines Models Module

Model implementations for various architectures converted to Neural Splines,
including DeepSeek, Llama, and other transformer architectures.
"""

from .base_neural import BaseNeuralModel
from .deepseek_neural import DeepSeekNeuralModel, DeepSeekNeuralConfig

# Import Llama Neural model if present.  If this import fails due to a
# missing file (e.g. when running against an older version of
# neural_splines), we fall back silently.  This allows the module to
# function without requiring Llama support.
try:
    from .llama_neural import LlamaNeuralModel, LlamaNeuralConfig  # type: ignore
except Exception:
    LlamaNeuralModel = None  # type: ignore
    LlamaNeuralConfig = None  # type: ignore

__all__ = [
    'BaseNeuralModel',
    'DeepSeekNeuralModel', 
    'DeepSeekNeuralConfig'
    , 'LlamaNeuralModel', 'LlamaNeuralConfig'
]

# Model registry for easy access
MODEL_REGISTRY = {
    'deepseek': DeepSeekNeuralModel,
    'deepseek-v3': DeepSeekNeuralModel,
    'base': BaseNeuralModel,
    # Register Llama Neural model if available.  If the class could not
    # be imported, attempting to access this key will raise an error.
    'llama': LlamaNeuralModel
}

def get_model_class(model_type: str):
    """Get model class by type name
    
    Args:
        model_type: Type of model ('deepseek', 'llama', etc.)
        
    Returns:
        Model class
    """
    model_type = model_type.lower()
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type]
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

def list_available_models():
    """List all available model types
    
    Returns:
        List of available model type names
    """
    return list(MODEL_REGISTRY.keys())