"""
Tensor Operations Utilities

Utilities for converting between tensor formats, spline representations,
and handling tensor manipulations specific to Neural Splines.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array safely
    
    Args:
        tensor: Input PyTorch tensor
        
    Returns:
        NumPy array
    """
    return tensor.detach().cpu().numpy()

def numpy_to_tensor(array: np.ndarray, device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor
    
    Args:
        array: Input NumPy array
        device: Target device
        dtype: Target data type
        
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array)
    
    if dtype is not None:
        tensor = tensor.to(dtype)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor

def spline_to_tensor(spline_components: Dict[str, Any], 
                    target_shape: torch.Size) -> torch.Tensor:
    """Convert spline components back to tensor representation
    
    Args:
        spline_components: Dictionary containing spline data
        target_shape: Desired output tensor shape
        
    Returns:
        Reconstructed tensor
    """
    try:
        control_points = spline_components.get('control_points')
        if control_points is None:
            return torch.zeros(target_shape)
        
        # Simple reconstruction using interpolation
        total_elements = torch.prod(torch.tensor(target_shape)).item()
        
        # Ensure control_points is a tensor
        if not isinstance(control_points, torch.Tensor):
            control_points = torch.tensor(control_points)
        
        flattened_control = control_points.flatten()
        
        if len(flattened_control) >= total_elements:
            # Downsample if we have too many control points
            indices = torch.linspace(0, len(flattened_control) - 1, total_elements).long()
            reconstructed = flattened_control[indices]
        else:
            # Upsample using interpolation
            reconstructed = F.interpolate(
                flattened_control.unsqueeze(0).unsqueeze(0).float(),
                size=total_elements,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        return reconstructed.reshape(target_shape)
        
    except Exception as e:
        logger.error(f"Failed to convert spline to tensor: {e}")
        return torch.zeros(target_shape)

def tensor_to_spline(tensor: torch.Tensor, num_control_points: Optional[int] = None) -> Dict[str, Any]:
    """Convert tensor to spline representation
    
    Args:
        tensor: Input tensor
        num_control_points: Number of control points to use
        
    Returns:
        Dictionary containing spline components
    """
    try:
        flattened = tensor.flatten()
        
        if num_control_points is None:
            # Default: use square root of tensor size
            num_control_points = max(4, int(np.sqrt(len(flattened))))
        
        # Subsample to get control points
        if len(flattened) <= num_control_points:
            control_points = flattened
        else:
            indices = torch.linspace(0, len(flattened) - 1, num_control_points).long()
            control_points = flattened[indices]
        
        return {
            'control_points': control_points,
            'original_shape': tensor.shape,
            'compression_ratio': len(flattened) / len(control_points)
        }
        
    except Exception as e:
        logger.error(f"Failed to convert tensor to spline: {e}")
        return {'control_points': torch.zeros(4), 'original_shape': tensor.shape}

def normalize_tensor(tensor: torch.Tensor, method: str = 'standard') -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Normalize tensor for better spline fitting
    
    Args:
        tensor: Input tensor
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (normalized_tensor, normalization_stats)
    """
    if method == 'standard':
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        normalized = (tensor - mean) / (std + 1e-8)
        stats = {'method': 'standard', 'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        range_val = max_val - min_val
        normalized = (tensor - min_val) / (range_val + 1e-8)
        stats = {'method': 'minmax', 'min': min_val, 'max': max_val, 'range': range_val}
        
    elif method == 'robust':
        median = torch.median(tensor)
        mad = torch.median(torch.abs(tensor - median))  # Median Absolute Deviation
        normalized = (tensor - median) / (mad + 1e-8)
        stats = {'method': 'robust', 'median': median, 'mad': mad}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats

def denormalize_tensor(normalized_tensor: torch.Tensor, 
                      normalization_stats: Dict[str, Any]) -> torch.Tensor:
    """Denormalize tensor using stored statistics
    
    Args:
        normalized_tensor: Normalized tensor
        normalization_stats: Statistics from normalization
        
    Returns:
        Denormalized tensor
    """
    method = normalization_stats['method']
    
    if method == 'standard':
        mean = normalization_stats['mean']
        std = normalization_stats['std']
        return normalized_tensor * std + mean
        
    elif method == 'minmax':
        min_val = normalization_stats['min']
        range_val = normalization_stats['range']
        return normalized_tensor * range_val + min_val
        
    elif method == 'robust':
        median = normalization_stats['median']
        mad = normalization_stats['mad']
        return normalized_tensor * mad + median
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def reshape_for_splines(tensor: torch.Tensor, target_dimensions: int = 1) -> Tuple[torch.Tensor, torch.Size]:
    """Reshape tensor for optimal spline fitting
    
    Args:
        tensor: Input tensor
        target_dimensions: Target number of dimensions
        
    Returns:
        Tuple of (reshaped_tensor, original_shape)
    """
    original_shape = tensor.shape
    
    if target_dimensions == 1:
        # Flatten to 1D
        reshaped = tensor.flatten()
    elif target_dimensions == 2:
        # Reshape to 2D for surface fitting
        total_elements = tensor.numel()
        side_length = int(np.sqrt(total_elements))
        
        if side_length * side_length == total_elements:
            reshaped = tensor.view(side_length, side_length)
        else:
            # Pad or truncate to make square
            needed_elements = side_length * side_length
            if total_elements < needed_elements:
                # Pad with zeros
                flattened = tensor.flatten()
                padded = F.pad(flattened, (0, needed_elements - total_elements))
                reshaped = padded.view(side_length, side_length)
            else:
                # Truncate
                flattened = tensor.flatten()[:needed_elements]
                reshaped = flattened.view(side_length, side_length)
    else:
        # Keep original shape
        reshaped = tensor
    
    return reshaped, original_shape

def calculate_tensor_statistics(tensor: torch.Tensor) -> Dict[str, float]:
    """Calculate comprehensive statistics for a tensor
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary of statistics
    """
    flattened = tensor.flatten()
    
    stats = {
        'mean': torch.mean(flattened).item(),
        'std': torch.std(flattened).item(),
        'min': torch.min(flattened).item(),
        'max': torch.max(flattened).item(),
        'median': torch.median(flattened).item(),
        'l1_norm': torch.norm(flattened, p=1).item(),
        'l2_norm': torch.norm(flattened, p=2).item(),
        'l_inf_norm': torch.norm(flattened, p=float('inf')).item(),
        'sparsity': (torch.abs(flattened) < 1e-6).float().mean().item(),
        'entropy': calculate_entropy(flattened),
        'effective_rank': calculate_effective_rank(tensor)
    }
    
    return stats

def calculate_entropy(tensor: torch.Tensor, bins: int = 256) -> float:
    """Calculate entropy of tensor values
    
    Args:
        tensor: Input tensor
        bins: Number of bins for histogram
        
    Returns:
        Entropy value
    """
    try:
        # Convert to numpy for histogram calculation
        values = tensor_to_numpy(tensor.flatten())
        
        # Calculate histogram
        hist, _ = np.histogram(values, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        hist = hist + 1e-10
        
        # Normalize to get probabilities
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
        
    except Exception as e:
        logger.warning(f"Failed to calculate entropy: {e}")
        return 0.0

def calculate_effective_rank(tensor: torch.Tensor) -> float:
    """Calculate effective rank of tensor
    
    Args:
        tensor: Input tensor
        
    Returns:
        Effective rank
    """
    try:
        if len(tensor.shape) == 1:
            return 1.0
        elif len(tensor.shape) == 2:
            # Use SVD
            U, S, V = torch.svd(tensor)
            
            # Calculate effective rank using entropy of singular values
            S_normalized = S / torch.sum(S)
            entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-10))
            effective_rank = torch.exp(entropy)
            
            return effective_rank.item()
        else:
            # For higher dimensions, flatten to 2D
            reshaped = tensor.view(tensor.shape[0], -1)
            return calculate_effective_rank(reshaped)
            
    except Exception as e:
        logger.warning(f"Failed to calculate effective rank: {e}")
        return float(min(tensor.shape)) if len(tensor.shape) > 1 else 1.0

def batch_process_tensors(tensors: List[torch.Tensor], 
                         operation: callable,
                         batch_size: int = 32) -> List[Any]:
    """Process tensors in batches for memory efficiency
    
    Args:
        tensors: List of tensors to process
        operation: Function to apply to each tensor
        batch_size: Number of tensors to process at once
        
    Returns:
        List of operation results
    """
    results = []
    
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size]
        
        try:
            batch_results = [operation(tensor) for tensor in batch]
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Batch processing failed for batch {i//batch_size}: {e}")
            # Add placeholder results for failed batch
            results.extend([None] * len(batch))
    
    return results

def memory_efficient_operation(tensor: torch.Tensor,
                              operation: callable,
                              chunk_size: Optional[int] = None) -> torch.Tensor:
    """Perform operation on tensor in chunks for memory efficiency
    
    Args:
        tensor: Input tensor
        operation: Operation to perform
        chunk_size: Size of chunks to process
        
    Returns:
        Result tensor
    """
    if chunk_size is None:
        # Default chunk size based on tensor size
        chunk_size = max(1000, tensor.numel() // 10)
    
    flattened = tensor.flatten()
    
    if len(flattened) <= chunk_size:
        # Small enough to process at once
        result = operation(tensor)
    else:
        # Process in chunks
        results = []
        
        for i in range(0, len(flattened), chunk_size):
            chunk = flattened[i:i + chunk_size]
            chunk_reshaped = chunk.view(-1, *tensor.shape[1:]) if len(tensor.shape) > 1 else chunk
            
            try:
                chunk_result = operation(chunk_reshaped)
                results.append(chunk_result.flatten())
            except Exception as e:
                logger.error(f"Chunk operation failed: {e}")
                results.append(torch.zeros(len(chunk)))
        
        # Concatenate results
        result_flattened = torch.cat(results)
        result = result_flattened.view(tensor.shape)
    
    return result

def safe_tensor_operation(operation: callable, *tensors: torch.Tensor,
                         fallback_value: Any = None) -> Any:
    """Safely perform tensor operation with error handling
    
    Args:
        operation: Operation to perform
        *tensors: Input tensors
        fallback_value: Value to return on error
        
    Returns:
        Operation result or fallback value
    """
    try:
        return operation(*tensors)
    except Exception as e:
        logger.warning(f"Tensor operation failed: {e}")
        
        if fallback_value is not None:
            return fallback_value
        else:
            # Return appropriate default based on first tensor
            if tensors:
                return torch.zeros_like(tensors[0])
            else:
                return torch.tensor(0.0)

def validate_tensor_compatibility(*tensors: torch.Tensor) -> bool:
    """Check if tensors are compatible for operations
    
    Args:
        *tensors: Tensors to check
        
    Returns:
        True if compatible, False otherwise
    """
    if not tensors:
        return True
    
    first_tensor = tensors[0]
    
    for tensor in tensors[1:]:
        # Check device compatibility
        if tensor.device != first_tensor.device:
            return False
        
        # Check dtype compatibility
        if tensor.dtype != first_tensor.dtype:
            return False
    
    return True

def ensure_tensor_compatibility(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """Ensure tensors are compatible by converting to common format
    
    Args:
        *tensors: Input tensors
        
    Returns:
        List of compatible tensors
    """
    if not tensors:
        return []
    
    # Use first tensor as reference
    reference = tensors[0]
    compatible_tensors = [reference]
    
    for tensor in tensors[1:]:
        # Convert to same device and dtype
        converted = tensor.to(device=reference.device, dtype=reference.dtype)
        compatible_tensors.append(converted)
    
    return compatible_tensors

def compress_tensor_representation(tensor: torch.Tensor, 
                                 compression_ratio: float = 10.0) -> Dict[str, Any]:
    """Create compressed representation of tensor
    
    Args:
        tensor: Input tensor
        compression_ratio: Target compression ratio
        
    Returns:
        Compressed representation
    """
    original_size = tensor.numel()
    target_size = max(1, int(original_size / compression_ratio))
    
    # Simple compression using subsampling
    flattened = tensor.flatten()
    
    if len(flattened) <= target_size:
        compressed_data = flattened
        indices = torch.arange(len(flattened))
    else:
        indices = torch.linspace(0, len(flattened) - 1, target_size).long()
        compressed_data = flattened[indices]
    
    return {
        'compressed_data': compressed_data,
        'indices': indices,
        'original_shape': tensor.shape,
        'original_size': original_size,
        'compressed_size': len(compressed_data),
        'compression_ratio': original_size / len(compressed_data)
    }

def decompress_tensor_representation(compressed_repr: Dict[str, Any]) -> torch.Tensor:
    """Decompress tensor from compressed representation
    
    Args:
        compressed_repr: Compressed representation
        
    Returns:
        Decompressed tensor
    """
    try:
        compressed_data = compressed_repr['compressed_data']
        original_shape = compressed_repr['original_shape']
        original_size = compressed_repr['original_size']
        
        # Reconstruct using interpolation
        if len(compressed_data) >= original_size:
            # Already full size or larger
            reconstructed = compressed_data[:original_size]
        else:
            # Interpolate to original size
            reconstructed = F.interpolate(
                compressed_data.unsqueeze(0).unsqueeze(0).float(),
                size=original_size,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        return reconstructed.view(original_shape)
        
    except Exception as e:
        logger.error(f"Failed to decompress tensor: {e}")
        original_shape = compressed_repr.get('original_shape', (1,))
        return torch.zeros(original_shape)