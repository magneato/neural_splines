"""
Neural Splines Custom Exceptions

Provides specific exception classes for different types of errors that can
occur during Neural Splines conversion, validation, and inference.
"""

class NeuralSplinesError(Exception):
    """Base exception class for all Neural Splines errors"""
    
    def __init__(self, message: str, error_code: str = "NS_GENERAL"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"

class ConversionError(NeuralSplinesError):
    """Raised when model conversion to Neural Splines fails"""
    
    def __init__(self, message: str, layer_name: str = None, original_error: Exception = None):
        self.layer_name = layer_name
        self.original_error = original_error
        
        if layer_name:
            full_message = f"Conversion failed for layer '{layer_name}': {message}"
        else:
            full_message = f"Model conversion failed: {message}"
        
        if original_error:
            full_message += f" (caused by: {type(original_error).__name__}: {original_error})"
        
        super().__init__(full_message, "NS_CONVERSION")

class ValidationError(NeuralSplinesError):
    """Raised when spline validation fails"""
    
    def __init__(self, message: str, validation_type: str = "general", 
                 error_details: dict = None):
        self.validation_type = validation_type
        self.error_details = error_details or {}
        
        full_message = f"{validation_type.title()} validation failed: {message}"
        
        if error_details:
            details_str = ", ".join(f"{k}={v}" for k, v in error_details.items())
            full_message += f" (details: {details_str})"
        
        super().__init__(full_message, "NS_VALIDATION")

class ModelLoadError(NeuralSplinesError):
    """Raised when loading a Neural Splines model fails"""
    
    def __init__(self, message: str, model_path: str = None, missing_files: list = None):
        self.model_path = model_path
        self.missing_files = missing_files or []
        
        full_message = f"Model loading failed: {message}"
        
        if model_path:
            full_message += f" (path: {model_path})"
        
        if missing_files:
            full_message += f" (missing files: {', '.join(missing_files)})"
        
        super().__init__(full_message, "NS_MODEL_LOAD")

class ModelError(NeuralSplinesError):
    """Raised for general model-related errors"""
    
    def __init__(self, message: str, model_type: str = None):
        self.model_type = model_type
        
        if model_type:
            full_message = f"Model error ({model_type}): {message}"
        else:
            full_message = f"Model error: {message}"
        
        super().__init__(full_message, "NS_MODEL")

class InferenceError(NeuralSplinesError):
    """Raised when model inference fails"""
    
    def __init__(self, message: str, input_shape: tuple = None, 
                 expected_shape: tuple = None):
        self.input_shape = input_shape
        self.expected_shape = expected_shape
        
        full_message = f"Inference failed: {message}"
        
        if input_shape and expected_shape:
            full_message += f" (input shape: {input_shape}, expected: {expected_shape})"
        elif input_shape:
            full_message += f" (input shape: {input_shape})"
        
        super().__init__(full_message, "NS_INFERENCE")

class SplineError(NeuralSplinesError):
    """Raised when spline operations fail"""
    
    def __init__(self, message: str, spline_order: int = None, 
                 num_control_points: int = None):
        self.spline_order = spline_order
        self.num_control_points = num_control_points
        
        full_message = f"Spline operation failed: {message}"
        
        details = []
        if spline_order is not None:
            details.append(f"order={spline_order}")
        if num_control_points is not None:
            details.append(f"control_points={num_control_points}")
        
        if details:
            full_message += f" ({', '.join(details)})"
        
        super().__init__(full_message, "NS_SPLINE")

class HarmonicError(NeuralSplinesError):
    """Raised when harmonic decomposition fails"""
    
    def __init__(self, message: str, num_components: int = None, 
                 signal_length: int = None):
        self.num_components = num_components
        self.signal_length = signal_length
        
        full_message = f"Harmonic decomposition failed: {message}"
        
        details = []
        if num_components is not None:
            details.append(f"components={num_components}")
        if signal_length is not None:
            details.append(f"signal_length={signal_length}")
        
        if details:
            full_message += f" ({', '.join(details)})"
        
        super().__init__(full_message, "NS_HARMONIC")

class ManifoldError(NeuralSplinesError):
    """Raised when manifold analysis fails"""
    
    def __init__(self, message: str, manifold_dimension: int = None, 
                 tensor_shape: tuple = None):
        self.manifold_dimension = manifold_dimension
        self.tensor_shape = tensor_shape
        
        full_message = f"Manifold analysis failed: {message}"
        
        details = []
        if manifold_dimension is not None:
            details.append(f"dimension={manifold_dimension}")
        if tensor_shape is not None:
            details.append(f"shape={tensor_shape}")
        
        if details:
            full_message += f" ({', '.join(details)})"
        
        super().__init__(full_message, "NS_MANIFOLD")

class CompressionError(NeuralSplinesError):
    """Raised when compression targets cannot be achieved"""
    
    def __init__(self, message: str, target_ratio: float = None, 
                 achieved_ratio: float = None):
        self.target_ratio = target_ratio
        self.achieved_ratio = achieved_ratio
        
        full_message = f"Compression failed: {message}"
        
        if target_ratio and achieved_ratio:
            full_message += f" (target: {target_ratio:.1f}x, achieved: {achieved_ratio:.1f}x)"
        elif target_ratio:
            full_message += f" (target: {target_ratio:.1f}x)"
        
        super().__init__(full_message, "NS_COMPRESSION")

class VisualizationError(NeuralSplinesError):
    """Raised when visualization operations fail"""
    
    def __init__(self, message: str, plot_type: str = None):
        self.plot_type = plot_type
        
        if plot_type:
            full_message = f"Visualization failed ({plot_type}): {message}"
        else:
            full_message = f"Visualization failed: {message}"
        
        super().__init__(full_message, "NS_VISUALIZATION")

class ConfigurationError(NeuralSplinesError):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: str = None, 
                 config_value: str = None):
        self.config_key = config_key
        self.config_value = config_value
        
        full_message = f"Configuration error: {message}"
        
        if config_key:
            full_message += f" (key: {config_key}"
            if config_value:
                full_message += f", value: {config_value}"
            full_message += ")"
        
        super().__init__(full_message, "NS_CONFIG")

class DependencyError(NeuralSplinesError):
    """Raised when required dependencies are missing"""
    
    def __init__(self, message: str, missing_package: str = None, 
                 required_version: str = None):
        self.missing_package = missing_package
        self.required_version = required_version
        
        full_message = f"Dependency error: {message}"
        
        if missing_package:
            full_message += f" (missing: {missing_package}"
            if required_version:
                full_message += f">={required_version}"
            full_message += ")"
        
        super().__init__(full_message, "NS_DEPENDENCY")

class MemoryError(NeuralSplinesError):
    """Raised when memory requirements exceed available resources"""
    
    def __init__(self, message: str, required_memory_gb: float = None, 
                 available_memory_gb: float = None):
        self.required_memory_gb = required_memory_gb
        self.available_memory_gb = available_memory_gb
        
        full_message = f"Memory error: {message}"
        
        if required_memory_gb and available_memory_gb:
            full_message += f" (required: {required_memory_gb:.1f}GB, available: {available_memory_gb:.1f}GB)"
        elif required_memory_gb:
            full_message += f" (required: {required_memory_gb:.1f}GB)"
        
        super().__init__(full_message, "NS_MEMORY")

# Helper functions for error handling

def handle_conversion_error(func):
    """Decorator to handle conversion errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConversionError:
            raise  # Re-raise conversion errors as-is
        except Exception as e:
            raise ConversionError(
                f"Unexpected error in {func.__name__}",
                original_error=e
            )
    return wrapper

def handle_spline_error(func):
    """Decorator to handle spline operation errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SplineError:
            raise  # Re-raise spline errors as-is
        except Exception as e:
            raise SplineError(
                f"Unexpected error in spline operation {func.__name__}: {str(e)}"
            )
    return wrapper

def validate_tensor_input(tensor, name="tensor"):
    """Validate tensor input and raise appropriate errors"""
    if tensor is None:
        raise ValidationError(f"{name} cannot be None")
    
    if not hasattr(tensor, 'shape'):
        raise ValidationError(f"{name} must be a tensor-like object")
    
    if torch.any(torch.isnan(tensor)):
        raise ValidationError(f"{name} contains NaN values")
    
    if torch.any(torch.isinf(tensor)):
        raise ValidationError(f"{name} contains infinite values")

def validate_compression_ratio(ratio):
    """Validate compression ratio parameter"""
    if ratio <= 1.0:
        raise ConfigurationError(
            f"Compression ratio must be > 1.0, got {ratio}",
            config_key="compression_ratio",
            config_value=str(ratio)
        )
    
    if ratio > 1000.0:
        raise ConfigurationError(
            f"Compression ratio {ratio}x may be too aggressive",
            config_key="compression_ratio", 
            config_value=str(ratio)
        )

def validate_spline_order(order):
    """Validate spline order parameter"""
    if not isinstance(order, int):
        raise ConfigurationError(
            f"Spline order must be an integer, got {type(order).__name__}",
            config_key="spline_order"
        )
    
    if order < 1 or order > 5:
        raise ConfigurationError(
            f"Spline order must be between 1 and 5, got {order}",
            config_key="spline_order",
            config_value=str(order)
        )

def check_memory_requirements(required_gb, operation_name="operation"):
    """Check if sufficient memory is available"""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if required_gb > available_gb:
            raise MemoryError(
                f"Insufficient memory for {operation_name}",
                required_memory_gb=required_gb,
                available_memory_gb=available_gb
            )
    except ImportError:
        # psutil not available, skip check
        pass

def check_gpu_memory(required_gb, operation_name="operation"):
    """Check if sufficient GPU memory is available"""
    try:
        import torch
        if torch.cuda.is_available():
            available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if required_gb > available_gb:
                raise MemoryError(
                    f"Insufficient GPU memory for {operation_name}",
                    required_memory_gb=required_gb,
                    available_memory_gb=available_gb
                )
    except Exception:
        # GPU check failed, continue without error
        pass

# Context managers for error handling

class ErrorContext:
    """Context manager for handling Neural Splines operations"""
    
    def __init__(self, operation_name: str, layer_name: str = None):
        self.operation_name = operation_name
        self.layer_name = layer_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return False  # No exception occurred
        
        # Convert generic exceptions to Neural Splines specific exceptions
        if isinstance(exc_value, NeuralSplinesError):
            return False  # Re-raise Neural Splines exceptions as-is
        
        # Convert common exceptions
        if exc_type == ValueError:
            raise ValidationError(str(exc_value))
        elif exc_type == RuntimeError:
            raise InferenceError(str(exc_value))
        elif exc_type == MemoryError:
            raise MemoryError(f"Out of memory during {self.operation_name}")
        else:
            # Wrap other exceptions
            if self.layer_name:
                raise ConversionError(
                    f"Error in {self.operation_name}",
                    layer_name=self.layer_name,
                    original_error=exc_value
                )
            else:
                raise NeuralSplinesError(
                    f"Error in {self.operation_name}: {exc_value}",
                    error_code="NS_UNEXPECTED"
                )

# Export commonly used exceptions for easy import
__all__ = [
    'NeuralSplinesError',
    'ConversionError', 
    'ValidationError',
    'ModelLoadError',
    'ModelError',
    'InferenceError',
    'SplineError',
    'HarmonicError',
    'ManifoldError',
    'CompressionError',
    'VisualizationError',
    'ConfigurationError',
    'DependencyError',
    'MemoryError',
    'ErrorContext',
    'handle_conversion_error',
    'handle_spline_error',
    'validate_tensor_input',
    'validate_compression_ratio',
    'validate_spline_order',
    'check_memory_requirements',
    'check_gpu_memory'
]