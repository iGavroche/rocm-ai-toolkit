from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module
import os

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        # Check if we're on ROCm and need special configuration
        try:
            import torch
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        except:
            is_rocm = False
        
        # For ROCm, use device_placement=True (default) to allow Accelerate to place models on GPU
        # Matching musubi-tuner approach: models are loaded to accelerator.device and kept there
        if is_rocm:
            # Create Accelerator with device_placement=True (default) to allow GPU placement
            global_accelerator = Accelerator(device_placement=True)
            # Monkey patch prepare_model to allow device placement (but with error handling)
            _patch_accelerate_for_rocm(global_accelerator)
        else:
            global_accelerator = Accelerator()
        
        if is_rocm:
            # Log accelerator configuration for debugging
            print(f"Accelerate initialized on ROCm backend")
            print(f"  Device: {global_accelerator.device}")
            print(f"  Mixed precision: {global_accelerator.mixed_precision}")
            print(f"  Device placement: {global_accelerator.device_placement}")
    return global_accelerator


def _patch_accelerate_for_rocm(accelerator: Accelerator):
    """
    Monkey patch Accelerator.prepare_model to allow device transfers on ROCm with error handling.
    
    Matching musubi-tuner approach: allow device placement but catch HIP errors gracefully.
    """
    # Get the unbound method from the class
    original_prepare_model = Accelerator.prepare_model
    
    def patched_prepare_model(self, model, device_placement=None, evaluation_mode=False):
        """
        Patched version that allows device transfer for ROCm with error handling.
        Note: When bound as a method, 'self' is automatically passed as the first argument.
        """
        try:
            from toolkit.backend_utils import is_rocm_available
            is_rocm = is_rocm_available()
        except ImportError:
            is_rocm = False
        
        if is_rocm:
            # Allow device placement (default to True if not specified)
            # This matches musubi-tuner's approach of loading models to accelerator.device
            if device_placement is None:
                device_placement = True
        
        # Call original unbound method with self explicitly
        try:
            return original_prepare_model(self, model, device_placement=device_placement, evaluation_mode=evaluation_mode)
        except (RuntimeError, Exception) as e:
            # Catch HIP errors during prepare but allow them to propagate
            # The caller (safe_prepare) will handle them
            error_str = str(e)
            if "HIP" in error_str or "hipError" in error_str:
                # Re-raise to let safe_prepare handle it
                raise
            else:
                raise
    
    # Replace the method - bind it to the accelerator instance
    import types
    accelerator.prepare_model = types.MethodType(patched_prepare_model, accelerator)

def safe_prepare(accelerator: Accelerator, *args, device_placement=None):
    """
    Wrapper for accelerator.prepare() with ROCm error handling.
    
    Matching musubi-tuner approach: models are already on GPU from loading,
    and we allow Accelerate to place them on GPU with device_placement=True.
    
    For ROCm, this function:
    - Models should already be on GPU from model loading
    - Uses device_placement=True (or provided value) to allow GPU placement
    - Handles HIP errors gracefully if they occur during prepare()
    
    Args:
        accelerator: The Accelerator instance
        *args: Objects to prepare (models, optimizers, etc.)
        device_placement: Optional list of booleans for device placement control.
                         For ROCm, defaults to True (allow GPU placement).
    
    Returns:
        Prepared objects in the same order as args
    """
    try:
        from toolkit.backend_utils import is_rocm_available, synchronize_gpu
        is_rocm = is_rocm_available()
    except ImportError:
        is_rocm = False
    
    if is_rocm:
        # For ROCm, synchronize GPU before prepare
        synchronize_gpu()
        
        # Models should already be on GPU from model loading (matching musubi-tuner)
        # Don't move to CPU - let Accelerate handle device placement
        
        # For ROCm, use device_placement=True (default) to allow GPU placement
        # This matches musubi-tuner's approach
        if device_placement is None:
            # Create a list of True for each argument (allow GPU placement)
            device_placement = [True] * len(args)
        
        try:
            # Call prepare with device_placement=True to allow GPU placement
            # Wrap in try/except to catch HIP errors that occur DURING prepare()
            try:
                result = accelerator.prepare(*args, device_placement=device_placement)
            except (RuntimeError, Exception) as prepare_error:
                # On ROCm, catch HIP errors during prepare() but don't force CPU fallback
                # Let the error propagate so caller can handle it appropriately
                error_str = str(prepare_error)
                error_type = type(prepare_error).__name__
                
                is_hip_error = "HIP" in error_str or "hipError" in error_str or "hipErrorInvalidValue" in error_str
                
                if is_hip_error:
                    # Log the error but re-raise - models should stay on GPU
                    print(f"Warning: Accelerate prepare() encountered HIP error on ROCm")
                    print(f"  Error type: {error_type}")
                    print(f"  Error message: {error_str[:200]}")
                    print(f"  Models will remain on their current device")
                    # Re-raise to let caller handle it
                    raise
                else:
                    # Non-HIP errors should propagate
                    raise
            
            # Accelerate returns a tuple if multiple args, single value if one arg
            # Models should now be on GPU (matching musubi-tuner)
            if len(args) == 1:
                return result
            return result
            
        except (RuntimeError, Exception) as e:
            error_str = str(e)
            error_type = type(e).__name__
            # For HIP errors, re-raise - don't force CPU fallback
            # The models should stay on GPU
            if "HIP" in error_str or "hipError" in error_str:
                print(f"Warning: Accelerate prepare() failed with HIP error on ROCm")
                print(f"  Error: {error_str[:200]}")
                # Re-raise - models should stay on GPU, caller will handle
                raise
            else:
                raise
    else:
        # CUDA: normal behavior
        return accelerator.prepare(*args, device_placement=device_placement)


def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
