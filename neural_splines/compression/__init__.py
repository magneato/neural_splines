"""
Neural Splines Compression Module

Advanced compression algorithms and optimization techniques for achieving
extreme compression ratios while preserving model quality.
"""

from .adaptive import DeepSeekSplineAdapter, AdaptiveCompressionConfig
from .optimizer import CompressionOptimizer, OptimizationConfig, QualityCompressionObjective, PerceptualCompressionObjective

__all__ = [
    'DeepSeekSplineAdapter',
    'AdaptiveCompressionConfig', 
    'CompressionOptimizer',
    'OptimizationConfig',
    'QualityCompressionObjective',
    'PerceptualCompressionObjective'
]

# Compression strategy registry
COMPRESSION_STRATEGIES = {
    'adaptive': DeepSeekSplineAdapter,
    'deepseek': DeepSeekSplineAdapter
}

def get_compression_adapter(strategy: str = 'adaptive', **kwargs):
    """Get compression adapter by strategy name
    
    Args:
        strategy: Compression strategy name
        **kwargs: Additional arguments for adapter
        
    Returns:
        Compression adapter instance
    """
    strategy = strategy.lower()
    if strategy in COMPRESSION_STRATEGIES:
        adapter_class = COMPRESSION_STRATEGIES[strategy]
        return adapter_class(**kwargs)
    else:
        raise ValueError(f"Unknown compression strategy: {strategy}. Available: {list(COMPRESSION_STRATEGIES.keys())}")

def list_compression_strategies():
    """List all available compression strategies
    
    Returns:
        List of available strategy names
    """
    return list(COMPRESSION_STRATEGIES.keys())