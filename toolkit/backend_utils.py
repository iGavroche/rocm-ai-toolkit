"""
Backend detection and utility functions for CUDA and ROCm support.
Provides backend-agnostic functions for device management and memory operations.
"""
import torch
from typing import Optional


def is_rocm_available() -> bool:
    """
    Check if ROCm backend is available.
    
    Returns:
        True if ROCm/HIP backend is available, False otherwise.
    """
    try:
        # ROCm PyTorch still uses torch.cuda API but with HIP backend
        # Check if HIP version is available (ROCm backend indicator)
        return hasattr(torch.version, 'hip') and torch.version.hip is not None
    except AttributeError:
        return False


def is_cuda_available() -> bool:
    """
    Check if CUDA backend is available (not ROCm).
    
    Returns:
        True if CUDA backend is available and it's not ROCm, False otherwise.
    """
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return False
        # If HIP is available, it's ROCm, not CUDA
        if is_rocm_available():
            return False
        return True
    except Exception:
        return False


def get_backend_type() -> str:
    """
    Get the current backend type.
    
    Returns:
        'rocm' if ROCm backend is available, 'cuda' if CUDA is available, 'cpu' otherwise.
    """
    if is_rocm_available():
        return 'rocm'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def is_gpu_available() -> bool:
    """
    Check if any GPU backend (CUDA or ROCm) is available.
    
    Returns:
        True if either CUDA or ROCm is available, False otherwise.
    """
    return torch.cuda.is_available()


def clear_gpu_cache():
    """
    Clear GPU cache for both CUDA and ROCm backends.
    This is a backend-agnostic replacement for torch.cuda.empty_cache().
    """
    if is_gpu_available():
        torch.cuda.empty_cache()


def synchronize_gpu():
    """
    Synchronize GPU operations to catch async errors.
    This helps with HIP errors that might be reported asynchronously.
    """
    if is_gpu_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            # If synchronization fails, it might be due to a previous error
            pass


def manual_seed(seed: int):
    """
    Set manual seed for GPU operations (backend-agnostic).
    
    Args:
        seed: The seed value to set.
    """
    if is_gpu_available():
        torch.cuda.manual_seed(seed)


def get_device_name(device_index: int = 0) -> Optional[str]:
    """
    Get the name of the GPU device (backend-agnostic).
    
    Args:
        device_index: Index of the GPU device.
        
    Returns:
        Device name string or None if not available.
    """
    if is_gpu_available():
        try:
            return torch.cuda.get_device_name(device_index)
        except Exception:
            return None
    return None


def normalize_device_string(device: str) -> str:
    """
    Normalize device string to ensure proper format.
    For ROCm, PyTorch still uses 'cuda' prefix, so we keep compatibility.
    
    Args:
        device: Device string (e.g., 'cuda', 'cuda:0', 'cpu')
        
    Returns:
        Normalized device string with device index if needed.
    """
    if "cuda" in device.lower() and ":" not in device:
        return f"{device}:0"
    return device

