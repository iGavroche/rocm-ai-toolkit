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
    from toolkit.backend_utils import is_rocm_available, synchronize_gpu
except ImportError:
    # Fallback if backend_utils is not available
    def is_rocm_available():
        return False
    def synchronize_gpu():
        pass


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
    
    for name, m in iterator:
        try:
            # check if m is QLinear or QConv2d
            if m.__class__.__name__ in Q_MODULES:
                continue
            else:
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
        except Exception as e:
            print(f"Failed to quantize {name}: {e}")
            # raise e


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
        
        # For ROCm, skip GPU quantization attempt (it always fails with HIP errors)
        # Go straight to CPU quantization (original fast approach)
        if is_rocm:
            base_model.print_and_status_update(
                "ROCm detected: quantizing entire model on CPU (GPU transfer not supported)"
            )
            # Ensure model is on CPU
            model_to_quantize = model_to_quantize.cpu()
            try:
                model_to_quantize = model_to_quantize.to(dtype=base_model.torch_dtype)
            except Exception:
                pass
            
            # Quantize entire model on CPU (original fast approach - no include patterns)
            base_model.print_and_status_update("Quantizing entire model on CPU...")
            quantize(model_to_quantize, weights=quantization_type, **getattr(base_model.model_config, 'quantize_kwargs', {}))
            freeze(model_to_quantize)
            base_model.print_and_status_update("✓ Model quantized on CPU")
            
            # Try to move quantized model to GPU after CPU quantization
            synchronize_gpu()
            try:
                model_to_quantize = model_to_quantize.to(base_model.device_torch)
                synchronize_gpu()
                base_model.print_and_status_update("✓ Moved quantized model to GPU")
            except (RuntimeError, Exception) as e2:
                error_str2 = str(e2)
                if "HIP" in error_str2 or "hipError" in error_str2:
                    # Keep on CPU - that's fine for ROCm
                    model_to_quantize = model_to_quantize.cpu()
                    base_model.print_and_status_update("Note: Quantized model will remain on CPU")
                else:
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
        
        # For ROCm, always quantize extras on CPU to avoid mixed-device issues
        # (blocks may be on different devices after block quantization)
        if is_rocm:
            # Ensure model is on CPU first (some blocks might be on GPU from block quantization)
            synchronize_gpu()
            try:
                # Move entire model to CPU first to avoid mixed-device issues
                model_to_quantize = model_to_quantize.cpu()
                # Quantize on CPU (safer for ROCm)
                quantize(model_to_quantize, weights=quantization_type)
                freeze(model_to_quantize)
                
                # After quantization, try to move extras to GPU if possible
                # But don't fail if it doesn't work - CPU is fine
                synchronize_gpu()
                try:
                    model_to_quantize = model_to_quantize.to(base_model.device_torch)
                    synchronize_gpu()
                except (RuntimeError, Exception) as e2:
                    error_str2 = str(e2)
                    if "HIP" in error_str2 or "hipError" in error_str2:
                        # Keep on CPU - that's fine
                        base_model.print_and_status_update(
                            f"Note: Extras will remain on CPU (GPU transfer not needed for quantization)"
                        )
                        model_to_quantize = model_to_quantize.cpu()
                    else:
                        raise
            except (RuntimeError, Exception) as e:
                error_str = str(e)
                if "HIP" in error_str or "hipError" in error_str:
                    # Final fallback - ensure on CPU and quantize
                    base_model.print_and_status_update(
                        f"Warning: Extras quantization issue, using CPU: {error_str[:150]}"
                    )
                    model_to_quantize = model_to_quantize.cpu()
                    quantize(model_to_quantize, weights=quantization_type)
                    freeze(model_to_quantize)
                else:
                    raise
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
                from toolkit.backend_utils import clear_gpu_cache
                import gc
                
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
