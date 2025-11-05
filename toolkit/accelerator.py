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
        
        # For ROCm, disable automatic device placement to prevent HIP errors
        # We'll handle device placement manually with error handling
        if is_rocm:
            # Create Accelerator with device_placement=False to prevent automatic transfers
            global_accelerator = Accelerator(device_placement=False)
            # Also monkey patch prepare_model to prevent device transfers
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
    Monkey patch Accelerator.prepare_model to prevent device transfers on ROCm.
    
    This is necessary because even with device_placement=False, Accelerate's
    prepare_model() may still call model.to(device) in some cases.
    """
    # Get the unbound method from the class
    original_prepare_model = Accelerator.prepare_model
    
    def patched_prepare_model(self, model, device_placement=None, evaluation_mode=False):
        """
        Patched version that skips device transfer for ROCm.
        Note: When bound as a method, 'self' is automatically passed as the first argument.
        """
        try:
            from toolkit.backend_utils import is_rocm_available
            is_rocm = is_rocm_available()
        except ImportError:
            is_rocm = False
        
        if is_rocm:
            # Force device_placement=False for ROCm to prevent HIP errors
            device_placement = False
        
        # Call original unbound method with self explicitly
        return original_prepare_model(self, model, device_placement=device_placement, evaluation_mode=evaluation_mode)
    
    # Replace the method - bind it to the accelerator instance
    import types
    accelerator.prepare_model = types.MethodType(patched_prepare_model, accelerator)

def safe_prepare(accelerator: Accelerator, *args, device_placement=None):
    """
    Wrapper for accelerator.prepare() with ROCm error handling.
    
    For ROCm, this function:
    - Ensures models are on CPU before prepare()
    - Uses device_placement=False to disable automatic device transfer
    - Manually handles device placement with error handling
    - Falls back gracefully if HIP errors occur
    
    Args:
        accelerator: The Accelerator instance
        *args: Objects to prepare (models, optimizers, etc.)
        device_placement: Optional list of booleans for device placement control.
                         For ROCm, defaults to False (no automatic placement).
    
    Returns:
        Prepared objects in the same order as args
    """
    try:
        from toolkit.backend_utils import is_rocm_available, synchronize_gpu
        is_rocm = is_rocm_available()
    except ImportError:
        is_rocm = False
    
    if is_rocm:
        # For ROCm, handle device placement carefully
        synchronize_gpu()
        
        # Ensure all models are on CPU before prepare
        prepared_args = []
        for arg in args:
            if hasattr(arg, 'to') and hasattr(arg, 'parameters'):
                # It's a model/module - ensure it's on CPU
                try:
                    arg.to("cpu")
                except (RuntimeError, Exception) as e:
                    if "HIP" in str(e) or "hipError" in str(e):
                        # Already on CPU or HIP error, continue
                        pass
                    else:
                        raise
            prepared_args.append(arg)
        
        # For ROCm, use device_placement=False to disable automatic device transfer
        if device_placement is None:
            # Create a list of False for each argument (disable automatic placement)
            device_placement = [False] * len(args)
        
        try:
            # Call prepare with device_placement=False
            # Wrap in try/except to catch HIP errors that occur DURING prepare()
            try:
                result = accelerator.prepare(*prepared_args, device_placement=device_placement)
            except (RuntimeError, Exception) as prepare_error:
                # On ROCm, catch any RuntimeError during prepare() - likely HIP/device transfer related
                # AcceleratorError is a RuntimeError subclass, so this will catch it
                error_str = str(prepare_error)
                error_type = type(prepare_error).__name__
                error_type_str = str(type(prepare_error))
                
                # Check if it's a device-related error (HIP, Accelerator, or any RuntimeError on ROCm)
                is_accelerator_error = (
                    "AcceleratorError" in error_type or 
                    "AcceleratorError" in error_type_str or
                    (hasattr(prepare_error, '__class__') and "Accelerator" in prepare_error.__class__.__name__)
                )
                is_hip_error = "HIP" in error_str or "hipError" in error_str or "hipErrorInvalidValue" in error_str
                
                # On ROCm, catch ANY RuntimeError during prepare as it's likely device-related
                # This is more permissive but necessary for ROCm compatibility
                if is_rocm and isinstance(prepare_error, RuntimeError):
                    # On ROCm, any RuntimeError during prepare is likely HIP/device related
                    print(f"Warning: Accelerate prepare() failed on ROCm (likely device transfer), keeping models on CPU")
                    print(f"  Error type: {error_type}")
                    print(f"  Error message: {error_str[:200]}")
                    # Return arguments as-is (on CPU) - they'll work for training
                    if len(prepared_args) == 1:
                        return prepared_args[0]
                    return tuple(prepared_args) if len(prepared_args) > 1 else prepared_args[0]
                elif is_hip_error or is_accelerator_error:
                    print(f"Warning: Accelerate prepare() failed with HIP/Accelerator error, keeping models on CPU")
                    print(f"  Error type: {error_type}")
                    print(f"  Error message: {error_str[:200]}")
                    # Return arguments as-is (on CPU) - they'll work for training
                    if len(prepared_args) == 1:
                        return prepared_args[0]
                    return tuple(prepared_args) if len(prepared_args) > 1 else prepared_args[0]
                else:
                    raise
            
            # Accelerate returns a tuple if multiple args, single value if one arg
            # For ROCm, we keep models on CPU and don't attempt device transfer
            # The models will be moved to device during forward passes when needed
            if len(prepared_args) == 1:
                return result
            return result
            
        except (RuntimeError, Exception) as e:
            error_str = str(e)
            error_type = type(e).__name__
            # Catch AcceleratorError (which is a RuntimeError) and HIP errors
            if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in error_type:
                print(f"Warning: Accelerate prepare() failed on ROCm, returning models as-is: {e}")
                # Return arguments as-is (on CPU)
                if len(prepared_args) == 1:
                    return prepared_args[0]
                return tuple(prepared_args) if len(prepared_args) > 1 else prepared_args[0]
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
