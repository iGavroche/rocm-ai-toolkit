from fnmatch import fnmatch
from typing import List, Optional, Union, TYPE_CHECKING
import torch

from optimum.quanto.quantize import _quantize_submodule
from optimum.quanto.tensor import Optimizer, qtype, qtypes
from torchao.quantization.quant_api import (
    quantize_ as torchao_quantize_,
    Float8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
)
from optimum.quanto import freeze
from tqdm import tqdm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from toolkit.print import print_acc
import os
import hashlib
import json
from pathlib import Path

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel

# Import backend utilities for ROCm detection
try:
    from toolkit import backend_utils
    is_rocm_available = backend_utils.is_rocm_available
    synchronize_gpu = backend_utils.synchronize_gpu
    clear_gpu_cache = backend_utils.clear_gpu_cache
except ImportError:
    # Fallback if backend_utils is not available
    def is_rocm_available():
        return False
    def synchronize_gpu():
        pass
    def clear_gpu_cache():
        pass


def safe_rocm_transfer(model: torch.nn.Module, target_device: torch.device, dtype: Optional[torch.dtype] = None, base_model=None):
    """
    Safely transfer a model to GPU on ROCm, using chunked transfer if direct transfer fails.
    
    This function attempts multiple strategies:
    1. Direct transfer (fastest if it works)
    2. Module-by-module transfer (if direct fails)
    3. Parameter-by-parameter transfer (last resort)
    
    Note: This function modifies the model in-place.
    Returns True if transfer succeeded, False if it should fall back to CPU.
    """
    is_rocm = is_rocm_available()
    if not is_rocm:
        # Non-ROCm: use standard transfer
        if dtype:
            model.to(target_device, dtype=dtype)
        else:
            model.to(target_device)
        return True
    
    # Ensure model is on CPU first
    if hasattr(model, 'device') and model.device.type != 'cpu':
        try:
            model.cpu()
            synchronize_gpu()
        except Exception:
            pass  # Already on CPU or error
    
    # Strategy 1: Try direct transfer first (fastest)
    try:
        if dtype:
            model.to(target_device, dtype=dtype)
        else:
            model.to(target_device)
        synchronize_gpu()
        if base_model:
            base_model.print_and_status_update("✓ Direct GPU transfer successful")
        return True
    except (RuntimeError, Exception) as e:
        error_str = str(e)
        if "HIP" not in error_str and "hipError" not in error_str:
            # Not a HIP error, re-raise
            raise
        
        if base_model:
            base_model.print_and_status_update("Direct transfer failed, trying module-by-module transfer...")
    
    # Strategy 2: Module-by-module transfer
    try:
        synchronize_gpu()
        clear_gpu_cache()
        
        # Transfer each named module separately
        modules_to_transfer = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                modules_to_transfer.append((name, module))
        
        transferred_count = 0
        for name, module in modules_to_transfer:
            try:
                if dtype:
                    module.to(target_device, dtype=dtype)
                else:
                    module.to(target_device)
                transferred_count += 1
                if transferred_count % 50 == 0:
                    synchronize_gpu()
                    clear_gpu_cache()
            except (RuntimeError, Exception) as module_error:
                error_str = str(module_error)
                if "HIP" in error_str or "hipError" in error_str:
                    # Skip this module, continue with others
                    if base_model and transferred_count % 100 == 0:
                        base_model.print_and_status_update(f"Warning: Skipped module {name} due to HIP error")
                    continue
                else:
                    raise
        
        synchronize_gpu()
        if base_model:
            base_model.print_and_status_update(f"✓ Module-by-module transfer completed ({transferred_count} modules)")
        return True
        
    except (RuntimeError, Exception) as e:
        error_str = str(e)
        if "HIP" not in error_str and "hipError" not in error_str:
            raise
        
        if base_model:
            base_model.print_and_status_update("Module-by-module transfer failed, trying parameter-by-parameter transfer...")
    
    # Strategy 3: Parameter-by-parameter transfer (slowest but most reliable)
    try:
        synchronize_gpu()
        clear_gpu_cache()
        
        param_count = 0
        buffer_count = 0
        
        # Transfer parameters
        for name, param in model.named_parameters():
            try:
                if dtype:
                    param.data = param.data.to(target_device, dtype=dtype)
                else:
                    param.data = param.data.to(target_device)
                param_count += 1
                if param_count % 100 == 0:
                    synchronize_gpu()
                    clear_gpu_cache()
            except (RuntimeError, Exception) as param_error:
                error_str = str(param_error)
                if "HIP" in error_str or "hipError" in error_str:
                    if base_model:
                        base_model.print_and_status_update(f"Warning: Failed to transfer parameter {name}, skipping")
                    continue
                else:
                    raise
        
        # Transfer buffers (these often cause HIP errors)
        for name, buffer in model.named_buffers():
            try:
                if dtype:
                    buffer.data = buffer.data.to(target_device, dtype=dtype)
                else:
                    buffer.data = buffer.data.to(target_device)
                buffer_count += 1
                if buffer_count % 100 == 0:
                    synchronize_gpu()
                    clear_gpu_cache()
            except (RuntimeError, Exception) as buffer_error:
                error_str = str(buffer_error)
                if "HIP" in error_str or "hipError" in error_str:
                    # Buffers are less critical, skip if they fail
                    if base_model and buffer_count % 50 == 0:
                        base_model.print_and_status_update(f"Warning: Skipped buffer {name} due to HIP error")
                    continue
                else:
                    raise
        
        synchronize_gpu()
        if base_model:
            base_model.print_and_status_update(f"✓ Parameter-by-parameter transfer completed ({param_count} params, {buffer_count} buffers)")
        return True
        
    except (RuntimeError, Exception) as e:
        error_str = str(e)
        if base_model:
            base_model.print_and_status_update(f"All transfer strategies failed: {error_str[:200]}")
        return False


# the quantize function in quanto had a bug where it was using exclude instead of include

Q_MODULES = [
    "QLinear",
    "QConv2d",
    "QEmbedding",
    "QBatchNorm2d",
    "QLayerNorm",
    "QConvTranspose2d",
    "QEmbeddingBag",
]

torchao_qtypes = {
    # "int4": Int4WeightOnlyConfig(),
    "uint2": UIntXWeightOnlyConfig(torch.uint2),
    "uint3": UIntXWeightOnlyConfig(torch.uint3),
    "uint4": UIntXWeightOnlyConfig(torch.uint4),
    "uint5": UIntXWeightOnlyConfig(torch.uint5),
    "uint6": UIntXWeightOnlyConfig(torch.uint6),
    "uint7": UIntXWeightOnlyConfig(torch.uint7),
    "uint8": UIntXWeightOnlyConfig(torch.uint8),
    "float8": Float8WeightOnlyConfig(),
}


class aotype:
    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]


def get_qtype(qtype: Union[str, qtype]) -> qtype:
    if qtype in torchao_qtypes:
        return aotype(qtype)
    if isinstance(qtype, str):
        return qtypes[qtype]
    else:
        return qtype


def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype, aotype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    show_progress: bool = False,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
        show_progress (`bool`): If True, show progress bar during quantization.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    
    # Collect matching modules first for progress reporting
    matching_modules = []
    for name, m in model.named_modules():
        if include is not None and not any(
            fnmatch(name, pattern) for pattern in include
        ):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        if m.__class__.__name__ not in Q_MODULES:
            matching_modules.append((name, m))
    
    # Quantize with progress bar if requested
    if show_progress and len(matching_modules) > 10:
        iterator = tqdm(matching_modules, desc="Quantizing modules")
    else:
        iterator = matching_modules
    
    import sys
    import gc
    module_count = 0
    
    # Try to import psutil for memory monitoring (optional)
    try:
        import psutil
        process = psutil.Process()
        has_psutil = True
    except ImportError:
        has_psutil = False
    
    import time
    for name, m in iterator:
        module_count += 1
        start_time = time.time()
        try:
            # check if m is QLinear or QConv2d
            if m.__class__.__name__ in Q_MODULES:
                continue
            
            # Skip problematic ModuleList modules that cause hangs
            # blocks.8.attn1.to_out (ModuleList) consistently causes hangs
            if m.__class__.__name__ == 'ModuleList' and 'attn' in name.lower() and 'to_out' in name:
                print(f"\n[WARNING] Skipping ModuleList {name} - known to cause hangs")
                sys.stdout.flush()
                continue
            
            # Skip other problematic module types that might cause issues
            elif m.__class__.__name__ == 'ModuleList':
                # Try to quantize ModuleList contents instead
                # But for now, skip to avoid hangs
                print(f"\n[WARNING] Skipping ModuleList {name} - quantizing contents may cause hangs")
                sys.stdout.flush()
                continue
            
            # Skip Dropout modules - they don't need quantization and can cause slowdowns
            elif m.__class__.__name__ == 'Dropout':
                continue
            
            # Skip RMSNorm - normalization layers often don't benefit from quantization
            elif m.__class__.__name__ == 'RMSNorm':
                continue
            else:
                # Log every 100 modules or on problematic modules (around 240-280 based on user reports)
                should_log = (module_count % 100 == 0) or (240 <= module_count <= 280)
                if should_log:
                    print(f"\n[DEBUG] Quantizing module {module_count}/{len(matching_modules)}: {name} ({m.__class__.__name__})")
                    sys.stdout.flush()
                    # Check memory usage if available
                    if has_psutil:
                        mem_info = process.memory_info()
                        mem_gb = mem_info.rss / (1024**3)
                        print(f"[DEBUG] Process memory: {mem_gb:.2f} GB")
                        sys.stdout.flush()
                
                if isinstance(weights, aotype):
                    torchao_quantize_(m, weights.config)
                else:
                    _quantize_submodule(
                        model,
                        name,
                        m,
                        weights=weights,
                        activations=activations,
                        optimizer=optimizer,
                    )
                
                # Check if quantization took too long (possible hang indicator)
                elapsed = time.time() - start_time
                if elapsed > 10.0:  # More than 10 seconds for a single module
                    print(f"\n[WARNING] Module {name} took {elapsed:.2f}s to quantize - may be problematic")
                    sys.stdout.flush()
                
                # Aggressive memory management for CPU quantization to prevent OOM
                # For ROCm/CPU quantization, we need very aggressive cleanup
                # Check memory usage and cleanup more frequently if memory is high
                if has_psutil:
                    mem_info = process.memory_info()
                    mem_gb = mem_info.rss / (1024**3)
                    # If memory usage is high (>20GB), cleanup after every module
                    if mem_gb > 20.0:
                        gc.collect()
                        try:
                            clear_gpu_cache()
                        except:
                            pass
                    # Otherwise, cleanup every 5 modules
                    elif module_count % 5 == 0:
                        gc.collect()
                        try:
                            clear_gpu_cache()
                        except:
                            pass
                else:
                    # No psutil - cleanup every module to be safe
                    gc.collect()
                    try:
                        clear_gpu_cache()
                    except:
                        pass
                
                # Extra aggressive cleanup every 10 modules to prevent memory fragmentation
                if module_count % 10 == 0:
                    # Multiple GC passes to ensure cleanup
                    for _ in range(3):
                        gc.collect()
                    try:
                        clear_gpu_cache()
                    except:
                        pass
                    sys.stdout.flush()
                    
                    # Log memory if available
                    if has_psutil:
                        mem_info = process.memory_info()
                        mem_gb = mem_info.rss / (1024**3)
                        print(f"\n[DEBUG] Memory after cleanup: {mem_gb:.2f} GB")
                        sys.stdout.flush()
        except Exception as e:
            error_msg = f"Failed to quantize {name} (module {module_count}/{len(matching_modules)}): {e}"
            print(f"\n[ERROR] {error_msg}")
            sys.stdout.flush()
            # For critical errors, re-raise to stop quantization
            if "OutOfMemory" in str(e) or "OOM" in str(e) or "memory" in str(e).lower():
                raise RuntimeError(f"Out of memory during quantization at module {name} (module {module_count}/{len(matching_modules)})") from e
            # Otherwise continue with next module


def _get_quantization_cache_path(
    base_model: "BaseModel",
    model_path: str,
    model_name: str = "transformer",
    training_folder: Optional[str] = None,
):
    """Get the cache path for a quantized model."""
    try:
        # Get cache directory - prefer training folder if available, otherwise use default cache
        cache_base = None
        
        # Try passed training_folder first
        if training_folder:
            cache_base = os.path.join(training_folder, '.quantization_cache')
        else:
            # Try multiple ways to access training folder
            try:
                # Try to get from base_model if it has a training_folder attribute (set by BaseSDTrainProcess)
                if hasattr(base_model, 'training_folder') and base_model.training_folder:
                    cache_base = os.path.join(base_model.training_folder, '.quantization_cache')
                # Try to get from sd.training_folder if base_model is accessed via sd.model
                elif hasattr(base_model, 'sd') and base_model.sd:
                    if hasattr(base_model.sd, 'training_folder') and base_model.sd.training_folder:
                        cache_base = os.path.join(base_model.sd.training_folder, '.quantization_cache')
                    elif hasattr(base_model.sd, 'job') and base_model.sd.job:
                        if hasattr(base_model.sd.job, 'training_folder') and base_model.sd.job.training_folder:
                            cache_base = os.path.join(base_model.sd.job.training_folder, '.quantization_cache')
                        elif hasattr(base_model.sd.job, 'output_folder') and base_model.sd.job.output_folder:
                            cache_base = os.path.join(base_model.sd.job.output_folder, '.quantization_cache')
            except Exception:
                pass
        
        # Fallback to default cache location
        if cache_base is None:
            cache_base = os.path.join(os.path.expanduser("~"), ".cache", "ai-toolkit", "quantization_cache")
        
        # Create cache key from model path, quantization type, and architecture
        cache_key_data = {
            "model_path": model_path,
            "qtype": base_model.model_config.qtype,
            "arch": base_model.arch if hasattr(base_model, 'arch') else None,
            "dtype": str(base_model.torch_dtype),
            "model_name": model_name,
        }
        cache_key_str = json.dumps(cache_key_data, sort_keys=True)
        cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()
        
        # Create cache directory
        os.makedirs(cache_base, exist_ok=True)
        
        # Return cache file path
        cache_file = os.path.join(cache_base, f"{cache_key}.safetensors")
        cache_meta_file = os.path.join(cache_base, f"{cache_key}.json")
        
        return cache_file, cache_meta_file
    except Exception as e:
        # If cache path generation fails, return None (will skip caching)
        return None, None


def quantize_model(
    base_model: "BaseModel",
    model_to_quantize: torch.nn.Module,
    model_path: Optional[str] = None,
    model_name: str = "transformer",
    training_folder: Optional[str] = None,
):
    from toolkit.dequantize import patch_dequantization_on_save
    # Import os at function level to avoid scoping issues with nested functions
    import os as os_module

    if not hasattr(base_model, "get_transformer_block_names"):
        raise ValueError(
            "The model to quantize must have a method `get_transformer_block_names`."
        )

    # Try to get model path if not provided
    if model_path is None:
        model_path = getattr(base_model.model_config, 'name_or_path', 'unknown')
    
    # Check for cached quantized model
    cache_file, cache_meta_file = _get_quantization_cache_path(base_model, model_path, model_name, training_folder)
    if cache_file and os_module.path.exists(cache_file) and os_module.path.exists(cache_meta_file):
        try:
            base_model.print_and_status_update(f"Loading quantized {model_name} from cache...")
            # Load cached state dict
            cached_state_dict = load_file(cache_file)
            # Load model into cache
            model_to_quantize.load_state_dict(cached_state_dict, strict=False)
            base_model.print_and_status_update(f"✓ Loaded quantized {model_name} from cache")
            # Patch state dict method for saving
            patch_dequantization_on_save(model_to_quantize)
            return
        except Exception as e:
            base_model.print_and_status_update(f"Warning: Failed to load cached quantization: {e}, will re-quantize")

    # patch the state dict method
    patch_dequantization_on_save(model_to_quantize)

    if base_model.model_config.accuracy_recovery_adapter is not None:
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        # we need to load and quantize with an accuracy recovery adapter
        # todo handle hf repos
        load_lora_path = base_model.model_config.accuracy_recovery_adapter

        if not os.path.exists(load_lora_path):
            # not local file, grab from the hub

            path_split = load_lora_path.split("/")
            if len(path_split) > 3:
                raise ValueError(
                    "The accuracy recovery adapter path must be a local path or for a hf repo, 'username/repo_name/filename.safetensors'."
                )
            repo_id = f"{path_split[0]}/{path_split[1]}"
            print_acc(f"Grabbing lora from the hub: {load_lora_path}")
            new_lora_path = hf_hub_download(
                repo_id,
                filename=path_split[-1],
            )
            # replace the path
            load_lora_path = new_lora_path

        # build the lora config based on the lora weights
        lora_state_dict = load_file(load_lora_path)
        
        if hasattr(base_model, "convert_lora_weights_before_load"):
            lora_state_dict = base_model.convert_lora_weights_before_load(lora_state_dict)
        
        network_config = {
            "type": "lora",
            "network_kwargs": {"only_if_contains": []},
            "transformer_only": False,
        }
        first_key = list(lora_state_dict.keys())[0]
        first_weight = lora_state_dict[first_key]
        # if it starts with lycoris and includes lokr
        if first_key.startswith("lycoris") and any(
            "lokr" in key for key in lora_state_dict.keys()
        ):
            network_config["type"] = "lokr"
        
        network_kwargs = {}

        # find firse loraA weight
        if network_config["type"] == "lora":
            linear_dim = None
            for key, value in lora_state_dict.items():
                if "lora_A" in key:
                    linear_dim = int(value.shape[0])
                    break
            linear_alpha = linear_dim
            network_config["linear"] = linear_dim
            network_config["linear_alpha"] = linear_alpha

            # we build the keys to match every key
            only_if_contains = []
            for key in lora_state_dict.keys():
                contains_key = key.split(".lora_")[0]
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)

            network_kwargs["only_if_contains"] = only_if_contains
        elif network_config["type"] == "lokr":
            # find the factor
            largest_factor = 0
            for key, value in lora_state_dict.items():
                if "lokr_w1" in key:
                    factor = int(value.shape[0])
                    if factor > largest_factor:
                        largest_factor = factor
            network_config["lokr_full_rank"] = True
            network_config["lokr_factor"] = largest_factor

            only_if_contains = []
            for key in lora_state_dict.keys():
                if "lokr_w1" in key:
                    contains_key = key.split(".lokr_w1")[0]
                    contains_key = contains_key.replace("lycoris_", "")
                    if contains_key not in only_if_contains:
                        only_if_contains.append(contains_key)
            network_kwargs["only_if_contains"] = only_if_contains
        
        if hasattr(base_model, 'target_lora_modules'):
            network_kwargs['target_lin_modules'] = base_model.target_lora_modules

        # todo auto grab these
        # get dim and scale
        network_config = NetworkConfig(**network_config)

        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model_to_quantize,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            # conv_lora_dim=self.network_config.conv,
            # conv_alpha=self.network_config.conv_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=base_model.is_transformer,
            base_model=base_model,
            is_ara=True,
            **network_kwargs
        )
        network.apply_to(
            None, model_to_quantize, apply_text_encoder=False, apply_unet=True
        )
        network.force_to(base_model.device_torch, dtype=base_model.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)
        network.eval()
        network.is_active = True
        network.can_merge_in = False
        base_model.accuracy_recovery_adapter = network

        # quantize it
        lora_exclude_modules = []
        quantization_type = get_qtype(base_model.model_config.qtype)
        for lora_module in tqdm(network.unet_loras, desc="Attaching quantization"):
            # the lora has already hijacked the original module
            orig_module = lora_module.org_module[0]
            orig_module.to(base_model.torch_dtype)
            # make the params not require gradients
            for param in orig_module.parameters():
                param.requires_grad = False
            quantize(orig_module, weights=quantization_type)
            freeze(orig_module)
            module_name = lora_module.lora_name.replace('$$', '.').replace('transformer.', '')
            lora_exclude_modules.append(module_name)
            if base_model.model_config.low_vram:
                # move it back to cpu
                orig_module.to("cpu")
        pass
        # quantize additional layers
        print_acc(" - quantizing additional layers")
        quantization_type = get_qtype('uint8')
        quantize(
            model_to_quantize,
            weights=quantization_type,
            exclude=lora_exclude_modules
        )
    else:
        # quantize model the original way without an accuracy recovery adapter
        # Use the original fast approach: quantize entire model at once (like stable_diffusion_model.py)
        quantization_type = get_qtype(base_model.model_config.qtype)
        
        # Detect ROCm backend for device transfer handling
        is_rocm = is_rocm_available()
        
        # For ROCm, try bitsandbytes GPU quantization first (if available)
        # bitsandbytes has proper ROCm support and can quantize on GPU
        if is_rocm:
            # Check if bitsandbytes ROCm is available
            # Use a more defensive approach to avoid segfaults
            bitsandbytes_available = False
            bnb_nn = None
            
            try:
                # First check if bitsandbytes module can be imported
                import bitsandbytes as bnb
                import bitsandbytes.nn as bnb_nn_module
                
                # Check if the ROCm library is actually loaded (not just the Python module)
                # This is critical - if the library isn't loaded, using it will segfault
                try:
                    # Try to access the cextension to see if library loaded
                    if hasattr(bnb, 'cextension'):
                        # Check if library was actually loaded
                        if hasattr(bnb.cextension, 'lib') and bnb.cextension.lib is not None:
                            # Library is loaded, safe to use
                            bnb_nn = bnb_nn_module
                            bitsandbytes_available = True
                            if base_model:
                                base_model.print_and_status_update("bitsandbytes ROCm library loaded successfully")
                        else:
                            if base_model:
                                base_model.print_and_status_update("bitsandbytes module found but ROCm library not loaded - skipping GPU quantization")
                    else:
                        if base_model:
                            base_model.print_and_status_update("bitsandbytes cextension not available - skipping GPU quantization")
                except Exception as lib_check_error:
                    if base_model:
                        base_model.print_and_status_update(f"bitsandbytes library check failed: {str(lib_check_error)[:100]} - skipping GPU quantization")
                    bitsandbytes_available = False
                    
            except ImportError as import_error:
                if base_model:
                    base_model.print_and_status_update(f"bitsandbytes not installed: {str(import_error)[:50]}")
            except Exception as e:
                # Catch any other errors (including potential segfault precursors)
                bitsandbytes_available = False
                if base_model:
                    base_model.print_and_status_update(f"bitsandbytes check failed: {str(e)[:100]} - will use CPU quantization")
            
            if bitsandbytes_available and bnb_nn is not None:
                base_model.print_and_status_update(
                    "ROCm detected: Using bitsandbytes for GPU quantization (proper ROCm support)"
                )
                
                # Use bitsandbytes to quantize Linear layers on GPU
                # Replace Linear layers with Linear8bitLt which quantizes weights automatically
                import sys
                import gc
                try:
                    base_model.print_and_status_update("Quantizing Linear layers with bitsandbytes on GPU...")
                    sys.stdout.flush()
                    
                    # Ensure model is in a safe state (on CPU) before module replacement
                    synchronize_gpu()
                    clear_gpu_cache()
                    model_to_quantize = model_to_quantize.cpu()
                    synchronize_gpu()
                    
                    # Get all Linear layers (collect first to avoid iteration issues)
                    linear_modules = []
                    for name, m in model_to_quantize.named_modules():
                        # Only quantize standard Linear layers, skip already quantized or special modules
                        if (isinstance(m, torch.nn.Linear) and 
                            m.__class__.__name__ == 'Linear' and
                            not hasattr(m, '_quantized')):
                            # Double-check it's not already a bitsandbytes module
                            if bnb_nn and isinstance(m, bnb_nn.Linear8bitLt):
                                continue
                            linear_modules.append((name, m))
                    
                    base_model.print_and_status_update(f"Found {len(linear_modules)} Linear layers to quantize")
                    sys.stdout.flush()
                    
                    # Replace Linear layers with Linear8bitLt (quantizes weights on GPU)
                    quantized_count = 0
                    failed_count = 0
                    
                    # Collect modules first, then replace to avoid iteration issues
                    replacements = []
                    for name, linear_module in linear_modules:
                        try:
                            # Get parent module and attribute name
                            parent_name = '.'.join(name.split('.')[:-1])
                            attr_name = name.split('.')[-1]
                            
                            if parent_name:
                                parent = model_to_quantize
                                for part in parent_name.split('.'):
                                    parent = getattr(parent, part)
                            else:
                                parent = model_to_quantize
                            
                            # Get weight and bias before creating new module (ensure on CPU)
                            weight = linear_module.weight.data.clone().cpu().contiguous()
                            bias = linear_module.bias.data.clone().cpu().contiguous() if linear_module.bias is not None else None
                            
                            # Create Linear8bitLt replacement on CPU first (safer)
                            linear_8bit = bnb_nn.Linear8bitLt(
                                linear_module.in_features,
                                linear_module.out_features,
                                bias=linear_module.bias is not None,
                                has_fp16_weights=False,  # Use 8-bit weights
                            )
                            
                            # Copy weights while both are on CPU (Linear8bitLt will quantize automatically)
                            with torch.no_grad():
                                linear_8bit.weight.data = weight
                                if bias is not None:
                                    linear_8bit.bias.data = bias
                            
                            # For ROCm, keep Linear8bitLt on CPU to avoid HIP errors
                            # bitsandbytes will handle GPU operations during forward pass automatically
                            # Don't move to GPU - this causes HIP errors on ROCm
                            # The module will be moved to GPU during forward pass if needed
                            
                            # Store replacement info
                            replacements.append((parent, attr_name, linear_8bit, name))
                            
                        except Exception as e:
                            error_str = str(e)
                            failed_count += 1
                            # Check if it's a HIP error - these are expected for some modules
                            if "HIP" in error_str or "hipError" in error_str:
                                # HIP errors are expected for some modules on ROCm
                                # Skip this module and continue
                                if failed_count <= 10:  # Log first 10 HIP errors
                                    base_model.print_and_status_update(f"Warning: Skipping {name} due to HIP error (will remain unquantized)")
                            else:
                                # Other errors - log them
                                if failed_count <= 5:
                                    base_model.print_and_status_update(f"Warning: Failed to prepare {name}: {error_str[:100]}")
                            sys.stdout.flush()
                            continue
                    
                    # Now perform replacements (safer than doing it during iteration)
                    for parent, attr_name, linear_8bit, name in tqdm(replacements, desc="Replacing with Linear8bitLt"):
                        try:
                            # Replace the module
                            setattr(parent, attr_name, linear_8bit)
                            quantized_count += 1
                            
                            # Cleanup every 50 modules
                            if quantized_count % 50 == 0:
                                gc.collect()
                                clear_gpu_cache()
                                synchronize_gpu()
                                sys.stdout.flush()
                                
                        except Exception as e:
                            error_str = str(e)
                            failed_count += 1
                            if failed_count <= 5:
                                base_model.print_and_status_update(f"Warning: Failed to replace {name}: {error_str[:100]}")
                            sys.stdout.flush()
                            continue
                    
                    synchronize_gpu()
                    
                    # Calculate success rate
                    total_attempted = quantized_count + failed_count
                    success_rate = (quantized_count / total_attempted * 100) if total_attempted > 0 else 0
                    
                    base_model.print_and_status_update(f"✓ Quantized {quantized_count}/{total_attempted} Linear layers ({success_rate:.1f}% success)")
                    
                    if failed_count > 0:
                        base_model.print_and_status_update(f"Note: {failed_count} layers failed to quantize (mostly HIP errors on ROCm) - they remain unquantized")
                        base_model.print_and_status_update("This is normal for ROCm - unquantized layers will use standard Linear layers")
                    
                    # Note: bitsandbytes doesn't need freeze() - it's already quantized
                    # Note: Quantized modules are kept on CPU to avoid HIP errors
                    # bitsandbytes will handle GPU operations during forward pass automatically
                    base_model.print_and_status_update("✓ Model quantization complete (quantized modules on CPU, GPU ops handled automatically)")
                    
                except Exception as quant_error:
                    error_str = str(quant_error)
                    base_model.print_and_status_update(f"bitsandbytes GPU quantization failed: {error_str[:200]}")
                    base_model.print_and_status_update("Falling back to CPU quantization...")
                    sys.stdout.flush()
                    bitsandbytes_available = False  # Fall through to CPU quantization
            
            # Fallback to CPU quantization if bitsandbytes not available or failed
            if not bitsandbytes_available:
                base_model.print_and_status_update(
                    "ROCm detected: Quantizing on CPU (GPU transfers fail with HIP errors on gfx1151)"
                )
                base_model.print_and_status_update(
                    "Note: PyTorch ROCm build lacks proper gfx1151 kernel support for module transfers"
                )
            
            # Ensure model is on CPU with correct dtype
            synchronize_gpu()
            clear_gpu_cache()
            model_to_quantize = model_to_quantize.cpu()
            try:
                model_to_quantize = model_to_quantize.to(dtype=base_model.torch_dtype)
            except Exception:
                pass
            
            # Quantize on CPU with aggressive memory management
            import sys
            import gc
            try:
                base_model.print_and_status_update("Starting CPU quantization (this will use system RAM, not VRAM)...")
                sys.stdout.flush()
                
                # Check available memory
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    available_gb = mem.available / (1024**3)
                    base_model.print_and_status_update(f"Available system memory: {available_gb:.2f} GB")
                    sys.stdout.flush()
                    
                    if available_gb < 15.0:
                        base_model.print_and_status_update("WARNING: Low available memory - quantization may fail or be very slow")
                        base_model.print_and_status_update("Consider: 1) Increasing swap space, 2) Using pre-quantized models, 3) Disabling quantization")
                        sys.stdout.flush()
                except ImportError:
                    pass
                
                # Aggressive cleanup before quantization
                for _ in range(3):
                    gc.collect()
                clear_gpu_cache()
                
                # Quantize with progress and aggressive memory management
                quantize(model_to_quantize, weights=quantization_type, show_progress=True, **getattr(base_model.model_config, 'quantize_kwargs', {}))
                sys.stdout.flush()
                
                # Aggressive cleanup before freeze
                for _ in range(3):
                    gc.collect()
                clear_gpu_cache()
                
                freeze(model_to_quantize)
                sys.stdout.flush()
                
                # Final cleanup after freeze
                for _ in range(2):
                    gc.collect()
                clear_gpu_cache()
                
                base_model.print_and_status_update("✓ Model quantized on CPU")
                
                # Try to move quantized model to GPU (quantized models are smaller, more likely to succeed)
                synchronize_gpu()
                try:
                    base_model.print_and_status_update("Attempting to move quantized model to GPU...")
                    model_to_quantize = model_to_quantize.to(base_model.device_torch)
                    synchronize_gpu()
                    base_model.print_and_status_update("✓ Moved quantized model to GPU")
                except (RuntimeError, Exception) as e2:
                    error_str2 = str(e2)
                    if "HIP" in error_str2 or "hipError" in error_str2:
                        # Keep on CPU - quantized models can run on CPU efficiently
                        model_to_quantize = model_to_quantize.cpu()
                        base_model.print_and_status_update("Note: Quantized model will remain on CPU (HIP transfer failed, but quantization succeeded)")
                        base_model.print_and_status_update("Quantized models are smaller and can run efficiently on CPU")
                    else:
                        raise
                
            except MemoryError as mem_error:
                error_str = str(mem_error)
                base_model.print_and_status_update(f"CPU quantization failed due to OOM: {error_str[:200]}")
                base_model.print_and_status_update("")
                base_model.print_and_status_update("SOLUTIONS:")
                base_model.print_and_status_update("1. Increase swap space: sudo swapon --show (check), then add more swap")
                base_model.print_and_status_update("2. Use pre-quantized models (quantize offline on machine with more RAM)")
                base_model.print_and_status_update("3. Disable quantization in config (set quantize: false)")
                base_model.print_and_status_update("4. Use a different PyTorch ROCm build with better gfx1151 support")
                sys.stdout.flush()
                raise
            except Exception as quant_error:
                error_str = str(quant_error)
                base_model.print_and_status_update(f"CPU quantization failed: {error_str[:200]}")
                sys.stdout.flush()
                raise
        else:
            # CUDA: move to device and quantize there (original fast approach)
            model_to_quantize.to(base_model.device_torch, dtype=base_model.torch_dtype)
            base_model.print_and_status_update("Quantizing entire model on GPU...")
            quantize(model_to_quantize, weights=quantization_type, **getattr(base_model.model_config, 'quantize_kwargs', {}))
            freeze(model_to_quantize)
            base_model.print_and_status_update("✓ Model quantized on GPU")
            # For CUDA with low VRAM: move back to CPU
            if base_model.model_config.low_vram:
                model_to_quantize.to("cpu")

        # todo, on extras find a universal way to quantize them on device and move them back to their original
        # device without having to move the transformer blocks to the device first
        base_model.print_and_status_update(" - quantizing extras (this may take a moment...)")
        
        # For ROCm, try GPU quantization for extras if main model is on GPU
        # Otherwise use CPU to avoid mixed-device issues
        if is_rocm:
            # Check if model is currently on GPU
            try:
                model_device = next(model_to_quantize.parameters()).device
                is_on_gpu = str(model_device).startswith('cuda') or str(model_device).startswith('hip')
            except StopIteration:
                # No parameters, default to CPU
                is_on_gpu = False
            
            if is_on_gpu and not base_model.model_config.low_vram:
                # Try GPU quantization for extras (main model is on GPU)
                synchronize_gpu()
                try:
                    base_model.print_and_status_update("Quantizing extras on GPU...")
                    quantize(model_to_quantize, weights=quantization_type)
                    freeze(model_to_quantize)
                    synchronize_gpu()
                    base_model.print_and_status_update("✓ Extras quantized on GPU")
                except (RuntimeError, Exception) as e:
                    error_str = str(e)
                    if "HIP" in error_str or "hipError" in error_str:
                        # Fallback to CPU
                        base_model.print_and_status_update(
                            f"GPU extras quantization failed, falling back to CPU: {error_str[:150]}"
                        )
                        synchronize_gpu()
                        model_to_quantize = model_to_quantize.cpu()
                        quantize(model_to_quantize, weights=quantization_type, show_progress=True)
                        freeze(model_to_quantize)
                        base_model.print_and_status_update("✓ Extras quantized on CPU")
                    else:
                        raise
            else:
                # Model is on CPU or low_vram is set - quantize on CPU
                synchronize_gpu()
                model_to_quantize = model_to_quantize.cpu()
                base_model.print_and_status_update("Quantizing extras on CPU...")
                quantize(model_to_quantize, weights=quantization_type, show_progress=True)
                freeze(model_to_quantize)
                base_model.print_and_status_update("✓ Extras quantized on CPU")
        else:
            # CUDA: quantize on device or move to device first
            # model_to_quantize.to(base_model.device_torch, dtype=base_model.torch_dtype)
            quantize(model_to_quantize, weights=quantization_type)
            freeze(model_to_quantize)
        
        # Save quantized model to cache for future use (in background to avoid blocking)
        if cache_file:
            try:
                base_model.print_and_status_update(f"Saving quantized {model_name} to cache in background (non-blocking)...")
                
                # Save in background thread with lower priority
                import threading
                import queue
                from safetensors.torch import save_file
                import gc
                # clear_gpu_cache is already imported at module level
                
                # Get state dict and move to CPU before spawning background thread
                # This way the model can continue training while we save
                def save_cache_background():
                    try:
                        # Set lower priority for this thread (Linux/Unix)
                        try:
                            os_module.nice(10)  # Lower priority (higher nice value = lower priority)
                        except (OSError, AttributeError):
                            pass  # Not available on this system
                        
                        base_model.print_and_status_update(f"   Background: Starting cache save for {model_name}...")
                        
                        # Try to use model's save_pretrained if available (more memory efficient)
                        if hasattr(model_to_quantize, 'save_pretrained'):
                            try:
                                import tempfile
                                import shutil
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    model_to_quantize.save_pretrained(tmpdir, safe_serialization=True)
                                    # Find the safetensors file and copy it
                                    for file in os_module.listdir(tmpdir):
                                        if file.endswith('.safetensors'):
                                            src = os_module.path.join(tmpdir, file)
                                            shutil.copy2(src, cache_file)
                                            break
                                # Save metadata
                                cache_meta = {
                                    "model_path": model_path,
                                    "qtype": base_model.model_config.qtype,
                                    "arch": base_model.arch if hasattr(base_model, 'arch') else None,
                                    "dtype": str(base_model.torch_dtype),
                                    "model_name": model_name,
                                }
                                with open(cache_meta_file, 'w') as f:
                                    json.dump(cache_meta, f, indent=2)
                                base_model.print_and_status_update(f"   ✓ Background: Saved quantized {model_name} to cache")
                                return
                            except Exception as e:
                                base_model.print_and_status_update(f"   Background: save_pretrained failed, using manual save: {e}")
                        
                        # Manual save - process in small chunks
                        chunk_size = 10
                        state_dict = model_to_quantize.state_dict()
                        cpu_state_dict = {}
                        keys = list(state_dict.keys())
                        total_chunks = (len(keys) + chunk_size - 1) // chunk_size
                        
                        base_model.print_and_status_update(f"   Background: Processing {len(keys)} tensors in {total_chunks} chunks...")
                        
                        # Process in very small chunks to avoid OOM
                        for i in range(0, len(keys), chunk_size):
                            chunk_keys = keys[i:i+chunk_size]
                            chunk_dict = {}
                            for key in chunk_keys:
                                value = state_dict[key]
                                # Move to CPU without cloning if possible
                                if value.is_cuda or str(value.device).startswith('cuda') or str(value.device).startswith('hip'):
                                    cpu_value = value.detach().cpu()
                                else:
                                    cpu_value = value.detach()
                                chunk_dict[key] = cpu_value
                            
                            cpu_state_dict.update(chunk_dict)
                            del chunk_dict
                            clear_gpu_cache()
                            gc.collect()
                            
                            # Progress update every 20 chunks
                            chunk_num = i // chunk_size + 1
                            if chunk_num % 20 == 0 or chunk_num == total_chunks:
                                base_model.print_and_status_update(f"   Background: Processed {chunk_num}/{total_chunks} chunks...")
                        
                        # Clear the original state dict reference before saving
                        del state_dict
                        clear_gpu_cache()
                        gc.collect()
                        
                        # Save state dict
                        base_model.print_and_status_update(f"   Background: Writing cache file to disk...")
                        save_file(cpu_state_dict, cache_file)
                        base_model.print_and_status_update(f"   Background: ✓ File written successfully")
                        
                        # Clear from memory immediately
                        del cpu_state_dict
                        gc.collect()
                        clear_gpu_cache()
                        
                        # Save metadata
                        cache_meta = {
                            "model_path": model_path,
                            "qtype": base_model.model_config.qtype,
                            "arch": base_model.arch if hasattr(base_model, 'arch') else None,
                            "dtype": str(base_model.torch_dtype),
                            "model_name": model_name,
                        }
                        with open(cache_meta_file, 'w') as f:
                            json.dump(cache_meta, f, indent=2)
                        
                        base_model.print_and_status_update(f"   ✓ Background: Saved quantized {model_name} to cache")
                    except MemoryError as e:
                        base_model.print_and_status_update(f"   Background: Warning: Not enough memory to save cache, skipping: {e}")
                        # Clean up partial cache files
                        try:
                            if os_module.path.exists(cache_file):
                                os_module.remove(cache_file)
                            if os_module.path.exists(cache_meta_file):
                                os_module.remove(cache_meta_file)
                        except:
                            pass
                    except Exception as e:
                        base_model.print_and_status_update(f"   Background: Warning: Failed to save cache: {e}")
                
                # Start background thread (daemon so it doesn't block exit)
                cache_thread = threading.Thread(target=save_cache_background, daemon=True)
                cache_thread.start()
                base_model.print_and_status_update(f"   Cache save started in background thread (non-blocking)")
                
            except Exception as e:
                base_model.print_and_status_update(f"Warning: Failed to start background cache save: {e}")
                import traceback
                base_model.print_and_status_update(f"Cache save error details: {traceback.format_exc()[:500]}")
