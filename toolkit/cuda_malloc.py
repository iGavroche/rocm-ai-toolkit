# Backward compatibility wrapper for gpu_malloc.py
# This module now supports both CUDA and ROCm backends
# Import from gpu_malloc to maintain backward compatibility
from toolkit.gpu_malloc import (
    get_gpu_names,
    gpu_malloc_supported,
    cuda_malloc_supported,
    gpu_malloc,
    cuda_malloc
)

# Re-export for backward compatibility
__all__ = [
    'get_gpu_names',
    'cuda_malloc_supported',
    'cuda_malloc',
]
