import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# Configure PyTorch memory allocator BEFORE importing torch
# For ROCm/HIP on Strix Halo (gfx1151), we need to force dedicated GPU memory usage, not shared/unified memory
# Strix Halo has unified memory architecture that can use system RAM - we must prevent this

# CRITICAL: Disable unified memory/page migration on Strix Halo
# HSA_XNACK=0 prevents using system RAM as GPU memory
if os.environ.get('HSA_XNACK') is None:
    os.environ['HSA_XNACK'] = '0'
elif os.environ.get('HSA_XNACK') != '0':
    os.environ['HSA_XNACK'] = '0'  # Force disable

# Disable SDMA (System DMA) to prevent shared memory transfers
if os.environ.get('HSA_ENABLE_SDMA') is None:
    os.environ['HSA_ENABLE_SDMA'] = '0'

# Force device kernel argument allocation (prevents using system memory for kernel args)
if os.environ.get('HIP_FORCE_DEV_KERNARG') is None:
    os.environ['HIP_FORCE_DEV_KERNARG'] = '1'

# Override GFX version for gfx1151 (Strix Halo)
# Some users report 11.0.0 works better than 11.5.1 for runtime detection
# Try 11.5.1 first, but allow override via environment variable
if os.environ.get('HSA_OVERRIDE_GFX_VERSION') is None:
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.5.1'
elif os.environ.get('HSA_OVERRIDE_GFX_VERSION') == '':
    # Empty string means try 11.0.0 as fallback
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

# Force memory pool to use only device memory, not system memory
if os.environ.get('HIP_MEM_POOL_SUPPORT') is None:
    os.environ['HIP_MEM_POOL_SUPPORT'] = '1'

# Limit allocation to device memory only (prevent fallback to shared memory)
if os.environ.get('GPU_SINGLE_ALLOC_PERCENT') is None:
    os.environ['GPU_SINGLE_ALLOC_PERCENT'] = '100'  # Use 100% of device memory

# Initialize device memory pool immediately
if os.environ.get('HIP_INITIAL_DM_SIZE') is None:
    os.environ['HIP_INITIAL_DM_SIZE'] = '0'  # 0 = use all available device memory

# Disable peer-to-peer memory access (which can trigger shared memory)
if os.environ.get('HIP_ENABLE_PEER_ACCESS') is None:
    os.environ['HIP_ENABLE_PEER_ACCESS'] = '0'

# PyTorch memory allocator configuration (optimized for ROCm/gfx1151 based on rocm-ninodes)
# Reference: https://github.com/iGavroche/rocm-ninodes
if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None:
    # Set expandable_segments for ROCm to reduce fragmentation
    # roundup_power2_divisions helps with memory alignment
    # max_split_size_mb:256 is default (reduced from 512 for severe fragmentation)
    # rocm-ninodes recommends 512, but we're seeing severe fragmentation issues
    # For even more severe fragmentation, try 128 or 64
    # Allow override via environment variable for experimentation
    max_split = os.environ.get('PYTORCH_MAX_SPLIT_SIZE_MB', '256')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'expandable_segments:True,roundup_power2_divisions:2,max_split_size_mb:{max_split}'
else:
    # Add missing settings if not already present
    existing = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    needs_expandable = 'expandable_segments' not in existing
    needs_roundup = 'roundup_power2_divisions' not in existing
    needs_max_split = 'max_split_size_mb' not in existing
    
    additions = []
    if needs_expandable:
        additions.append('expandable_segments:True')
    if needs_roundup:
        additions.append('roundup_power2_divisions:2')
    if needs_max_split:
        max_split = os.environ.get('PYTORCH_MAX_SPLIT_SIZE_MB', '256')
        additions.append(f'max_split_size_mb:{max_split}')  # Default 256, allow override
    
    if additions:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'{existing},{",".join(additions)}'

# ROCm-specific HIP allocator configuration (separate from CUDA allocator)
# rocm-ninodes sets this separately for better ROCm compatibility
if os.environ.get('PYTORCH_HIP_ALLOC_CONF') is None:
    os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

# Memory pool type configuration (rocm-ninodes recommendation)
if os.environ.get('PYTORCH_CUDA_MEMORY_POOL_TYPE') is None:
    os.environ['PYTORCH_CUDA_MEMORY_POOL_TYPE'] = 'expandable_segments'

# CRITICAL FIX for unified memory architecture (gfx1151/Strix Halo):
# Disable PyTorch's caching allocator which doesn't properly account for unified memory
# The caching allocator reserves memory blocks that it thinks are free but aren't actually
# available due to unified memory architecture confusion
# This forces PyTorch to allocate/deallocate memory directly without caching
# 
# WARNING: Setting this to '1' causes ROCm driver crashes (0xC0000005) with quantized models
# If you experience crashes, leave this unset (enables caching, may cause fragmentation)
# if os.environ.get('PYTORCH_NO_HIP_MEMORY_CACHING') is None:
#     os.environ['PYTORCH_NO_HIP_MEMORY_CACHING'] = '1'

# import toolkit.cuda_malloc  # Commented out - we configure manually for ROCm compatibility

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# CRITICAL: Patch torch._ops.c10d_functional FIRST, before anything else imports torch
# This must happen before torchao/transformers can access it
try:
    import torch
    # ALSO patch torch.distributed immediately to add missing attributes
    # Some modules import from torch.distributed early, so we need this before other imports
    try:
        import torch.distributed
        if not hasattr(torch.distributed, 'group'):
            # group is used as a module with a WORLD attribute
            class ProcessGroupStub:
                def __init__(self, *args, **kwargs):
                    pass
                WORLD = None  # Common default process group
            torch.distributed.group = ProcessGroupStub
        if not hasattr(torch.distributed, 'ReduceOp'):
            from enum import Enum
            class ReduceOpStub(Enum):
                SUM = 0
                AVG = 1
                PRODUCT = 2
                MIN = 3
                MAX = 4
                BAND = 5
                BOR = 6
                BXOR = 7
                PREMUL_SUM = 8
            torch.distributed.ReduceOp = ReduceOpStub
        # Add is_initialized function (used by accelerate during cleanup)
        if not hasattr(torch.distributed, 'is_initialized'):
            def is_initialized_stub():
                return False  # Distributed training is not initialized in ROCm single-GPU scenarios
            torch.distributed.is_initialized = is_initialized_stub
    except (ImportError, AttributeError):
        pass
    # Patch torch._ops immediately to stub missing c10d_functional operations
    try:
        class StubOperation:
            """Stub for a PyTorch operation that has a .default attribute"""
            def __init__(self):
                def stub_op(*args, **kwargs):
                    if args and isinstance(args[0], torch.Tensor):
                        return args[0]
                    return None
                self.default = stub_op
        
        # Wrap torch._ops to intercept c10d_functional access
        original_ops = torch._ops
        class OpsWrapper:
            """Wrapper for torch._ops that provides stubs for missing c10d_functional operations"""
            def __init__(self, original):
                self._original = original
                self._c10d_stub = None
            
            def __getattr__(self, name):
                if name == 'c10d_functional':
                    # Return a wrapper for c10d_functional that stubs missing operations
                    if self._c10d_stub is None:
                        original_c10d = getattr(self._original, 'c10d_functional', None)
                        
                        class C10DFunctionalWrapper:
                            """Wrapper that provides stubs for missing operations"""
                            def __init__(self, original_ns):
                                self._original = original_ns
                                self.all_gather_into_tensor = StubOperation()
                            
                            def __getattr__(self, name):
                                if name == 'all_gather_into_tensor':
                                    return StubOperation()
                                # Try to get from original, but if it fails, return a stub
                                try:
                                    if self._original:
                                        return getattr(self._original, name)
                                except AttributeError:
                                    pass
                                return StubOperation()
                        
                        self._c10d_stub = C10DFunctionalWrapper(original_c10d)
                    return self._c10d_stub
                return getattr(self._original, name)
        
        torch._ops = OpsWrapper(original_ops)
        
        # ALSO patch _OpNamespace class to handle missing operations
        # This is needed because torchao accesses operations through the namespace object
        try:
            from torch._ops import _OpNamespace
            original_opns_getattr = _OpNamespace.__getattr__
            
            def patched_opns_getattr(self, name):
                # Check if this is a namespace we need to stub - try multiple possible attribute names
                ns_name = None
                for attr_name in ['_name', 'name', '__name__']:
                    try:
                        ns_name = object.__getattribute__(self, attr_name)
                        break
                    except AttributeError:
                        continue
                
                # Stub missing operations for namespaces that don't exist in ROCm builds
                if ns_name:
                    # Handle c10d_functional operations
                    if (ns_name == 'c10d_functional' or ns_name == '_c10d_functional') and name == 'all_gather_into_tensor':
                        return StubOperation()
                    # Handle _dtensor operations
                    if ns_name == '_dtensor' and name == 'shard_dim_alltoall':
                        return StubOperation()
                
                # Otherwise try original
                try:
                    return original_opns_getattr(self, name)
                except AttributeError:
                    # If original fails and this is a namespace we're stubbing, return stub
                    if ns_name:
                        if ns_name == 'c10d_functional' or ns_name == '_c10d_functional':
                            return StubOperation()
                        if ns_name == '_dtensor':
                            return StubOperation()
                    raise
            
            _OpNamespace.__getattr__ = patched_opns_getattr
        except (ImportError, AttributeError):
            pass
    except (AttributeError, ImportError, TypeError):
        pass
except ImportError:
    pass

# Workaround for ROCm nightlies: Patch torch.distributed before transformers imports it
# This prevents transformers from trying to import distributed modules that don't exist in ROCm builds
try:
    if 'torch' not in sys.modules:
        import torch
    import types
    
    # Create a wrapper that catches import errors for distributed modules
    class DistributedStub:
        """Stub for torch.distributed that prevents import errors"""
        def __getattr__(self, name):
            if name == 'tensor':
                # Return a stub module for tensor with all expected submodules
                tensor_stub = types.ModuleType('tensor')
                # Create _dtensor_spec stub
                dtensor_spec_stub = types.ModuleType('_dtensor_spec')
                dtensor_spec_stub.DTensorSpec = type('DTensorSpec', (), {})
                dtensor_spec_stub.TensorMeta = type('TensorMeta', (), {})
                tensor_stub._dtensor_spec = dtensor_spec_stub
                # Create _ops stub
                ops_stub = types.ModuleType('_ops')
                ops_stub._conv_ops = types.ModuleType('_conv_ops')
                tensor_stub._ops = ops_stub
                # Create placement_types stub
                placement_types_stub = types.ModuleType('placement_types')
                tensor_stub.placement_types = placement_types_stub
                return tensor_stub
            elif name == '_functional_collectives':
                funcol_stub = types.ModuleType('_functional_collectives')
                return funcol_stub
            elif name == 'distributed_c10d':
                # Create a stub for distributed_c10d that transformers expects
                c10d_stub = types.ModuleType('distributed_c10d')
                return c10d_stub
            return types.ModuleType(name)
    
    # Patch sys.modules to intercept distributed imports before they fail
    # This way when transformers/torchao tries to import, it gets our stubs
    # List of submodules that torch.distributed.tensor might need (based on torch.distributed._tensor.__init__.py)
    tensor_submodules = [
        '_dtensor_spec', '_ops', 'placement_types', '_shards_wrapper', 
        '_device_mesh', '_op_schema', '_dispatch_op', '_view', '_redistribute'
    ]
    
    if 'torch.distributed.tensor' not in sys.modules:
        # Create all submodules first and register them in sys.modules
        # This ensures Python recognizes torch.distributed.tensor as a package
        for submodule in tensor_submodules:
            submodule_name = f'torch.distributed.tensor.{submodule}'
            if submodule_name not in sys.modules:
                submodule_stub = types.ModuleType(submodule_name)
                submodule_stub.__package__ = 'torch.distributed.tensor'
                sys.modules[submodule_name] = submodule_stub
        
        # Now create the main tensor package module
        tensor_stub = types.ModuleType('torch.distributed.tensor')
        tensor_stub.__package__ = 'torch.distributed.tensor'
        # Set __path__ to make it a proper package (use a dummy path)
        tensor_stub.__path__ = ['__torch_distributed_tensor_stub__']
        
        # Attach all submodules as attributes
        for submodule in tensor_submodules:
            attr_name = submodule.split('.')[-1]
            submodule_name = f'torch.distributed.tensor.{submodule}'
            setattr(tensor_stub, attr_name, sys.modules[submodule_name])
        
        # Add specific attributes that might be needed
        dtensor_spec_stub = sys.modules.get('torch.distributed.tensor._dtensor_spec')
        if dtensor_spec_stub:
            dtensor_spec_stub.DTensorSpec = type('DTensorSpec', (), {})
            dtensor_spec_stub.TensorMeta = type('TensorMeta', (), {})
        
        # Add attributes that torch.distributed.fsdp and other modules expect
        # These are imported from torch.distributed.tensor by various PyTorch modules
        # Shard and Replicate must accept arguments (e.g., Shard(-1) in transformers)
        class ShardStub:
            """Stub for Shard placement that accepts arguments"""
            def __init__(self, *args, **kwargs):
                pass
        class ReplicateStub:
            """Stub for Replicate placement that accepts arguments"""
            def __init__(self, *args, **kwargs):
                pass
        class PlacementStub:
            """Stub for Placement base class"""
            def __init__(self, *args, **kwargs):
                pass
        
        tensor_stub.DeviceMesh = type('DeviceMesh', (), {})
        tensor_stub.DTensor = type('DTensor', (), {})
        tensor_stub.Replicate = ReplicateStub
        tensor_stub.Shard = ShardStub
        tensor_stub.Placement = PlacementStub  # Used by transformers.integrations.tensor_parallel
        
        # Function stubs
        def init_device_mesh(*args, **kwargs):
            # Stub function - returns None or a dummy DeviceMesh
            return tensor_stub.DeviceMesh()
        tensor_stub.init_device_mesh = init_device_mesh
        
        # Create _ops submodules
        ops_stub = sys.modules.get('torch.distributed.tensor._ops')
        if ops_stub:
            conv_ops_name = 'torch.distributed.tensor._ops._conv_ops'
            if conv_ops_name not in sys.modules:
                conv_ops_stub = types.ModuleType(conv_ops_name)
                conv_ops_stub.__package__ = 'torch.distributed.tensor._ops'
                sys.modules[conv_ops_name] = conv_ops_stub
            ops_stub._conv_ops = sys.modules[conv_ops_name]
        
        # Register the main tensor module last
        sys.modules['torch.distributed.tensor'] = tensor_stub
    
    # Also stub torch.distributed._tensor (used by torchao)
    if 'torch.distributed._tensor' not in sys.modules:
        dtensor_stub = types.ModuleType('torch.distributed._tensor')
        dtensor_stub.__package__ = 'torch.distributed._tensor'
        dtensor_stub.__path__ = ['__torch_distributed_tensor_stub__']  # Make it a package
        dtensor_stub.DTensor = type('DTensor', (), {})
        sys.modules['torch.distributed._tensor'] = dtensor_stub
    
    if 'torch.distributed._functional_collectives' not in sys.modules:
        funcol_stub = types.ModuleType('torch.distributed._functional_collectives')
        
        # Create AsyncCollectiveTensor class that torchao expects
        class AsyncCollectiveTensorStub:
            """Stub for AsyncCollectiveTensor - used by torchao"""
            def __init__(self, *args, **kwargs):
                pass
            def wait(self):
                pass
        funcol_stub.AsyncCollectiveTensor = AsyncCollectiveTensorStub
        
        # Create stub functions for collective operations that torchao uses
        # These are stubs - they won't actually perform collective operations
        # but they'll prevent import errors
        def all_reduce_stub(tensor, *args, **kwargs):
            # Return the tensor unchanged (stub behavior)
            return tensor
        funcol_stub.all_reduce = all_reduce_stub
        
        def all_gather_stub(tensor, *args, **kwargs):
            return tensor
        funcol_stub.all_gather = all_gather_stub
        
        def reduce_scatter_stub(tensor, *args, **kwargs):
            return tensor
        funcol_stub.reduce_scatter = reduce_scatter_stub
        
        def broadcast_stub(tensor, *args, **kwargs):
            return tensor
        funcol_stub.broadcast = broadcast_stub
        
        sys.modules['torch.distributed._functional_collectives'] = funcol_stub
    
    # Also patch the actual torch.distributed attribute to use our stub
    try:
        import torch.distributed.tensor
    except (ImportError, ModuleNotFoundError, AttributeError):
        # If import fails, ensure torch.distributed uses our stub
        if hasattr(torch, 'distributed'):
            try:
                import torch.distributed
                original_distributed = torch.distributed
                
                # Add missing attributes that ROCm builds don't have
                if not hasattr(original_distributed, 'group'):
                    # Create a stub for group (used by torch.distributed.nn.functional)
                    class ProcessGroupStub:
                        def __init__(self, *args, **kwargs):
                            pass
                    original_distributed.group = ProcessGroupStub
                
                if not hasattr(original_distributed, 'ReduceOp'):
                    # Create a stub for ReduceOp (used by distributed operations)
                    from enum import Enum
                    class ReduceOpStub(Enum):
                        SUM = 0
                        AVG = 1
                        PRODUCT = 2
                        MIN = 3
                        MAX = 4
                        BAND = 5
                        BOR = 6
                        BXOR = 7
                        PREMUL_SUM = 8
                    original_distributed.ReduceOp = ReduceOpStub
                
                class DistributedWrapper:
                    def __init__(self, original):
                        self._original = original
                        self._stub = DistributedStub()
                    def __getattr__(self, name):
                        try:
                            return getattr(self._original, name)
                        except (AttributeError, ImportError, ModuleNotFoundError):
                            return getattr(self._stub, name)
                torch.distributed = DistributedWrapper(original_distributed)
            except:
                torch.distributed = DistributedStub()
except Exception:
    # If patching fails, continue - the error handler in train_tools will catch it
    pass

# Verify GPU device and memory configuration for ROCm/HIP
# This helps diagnose issues where shared memory is used instead of dedicated GPU memory
try:
    import torch
    if torch.cuda.is_available():
        print(f"CUDA/HIP is available. Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / 1024**3
            print(f"    Total memory: {total_mem_gb:.2f} GB")
            # Print current memory allocation
            allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    Allocated: {allocated_gb:.2f} GB")
            print(f"    Reserved: {reserved_gb:.2f} GB")
            print(f"    Free (calc): {total_mem_gb - reserved_gb:.2f} GB")
        
        # Check ROCm-specific environment variables for Strix Halo
        rocm_vars = {
            'HSA_XNACK': os.environ.get('HSA_XNACK', 'not set'),
            'HSA_ENABLE_SDMA': os.environ.get('HSA_ENABLE_SDMA', 'not set'),
            'HIP_FORCE_DEV_KERNARG': os.environ.get('HIP_FORCE_DEV_KERNARG', 'not set'),
            'HSA_OVERRIDE_GFX_VERSION': os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set'),
            'HIP_MEM_POOL_SUPPORT': os.environ.get('HIP_MEM_POOL_SUPPORT', 'not set'),
            'GPU_SINGLE_ALLOC_PERCENT': os.environ.get('GPU_SINGLE_ALLOC_PERCENT', 'not set'),
            'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', 'not set'),
            'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set'),
            'HIP_INITIAL_DM_SIZE': os.environ.get('HIP_INITIAL_DM_SIZE', 'not set'),
            'HIP_ENABLE_PEER_ACCESS': os.environ.get('HIP_ENABLE_PEER_ACCESS', 'not set'),
        }
        print("ROCm/HIP environment variables (Strix Halo configuration):")
        for key, value in rocm_vars.items():
            print(f"  {key}: {value}")
        
        # Try to set memory fraction to force GPU memory usage
        # This forces PyTorch to use dedicated GPU memory, not shared memory
        try:
            torch.cuda.set_per_process_memory_fraction(1.0)  # Use 100% of GPU memory
            print("  Set memory fraction to 1.0 (100% dedicated GPU memory)")
        except Exception as e:
            print(f"  Could not set memory fraction: {e}")
        
        # Force memory allocation to device immediately
        # Pre-allocate and free memory to "warm up" the allocator and reduce fragmentation
        # This helps the allocator establish its memory pool structure early
        try:
            import torch._C as _C
            if hasattr(_C, '_cuda_getDevice'):
                print("  PyTorch CUDA backend initialized")
            # Force empty cache and reset memory allocator state
            torch.cuda.empty_cache()
            
            # Memory warm-up strategy: Pre-allocate large blocks then free them
            # This helps the allocator establish memory pools and reduces fragmentation
            # during actual model loading. Based on kohya_ss and rocm-ninodes patterns.
            print("  Warming up GPU memory allocator...")
            try:
                # Allocate several large blocks (1GB each) to establish memory pools
                # Then free them to create a "clean slate" with established pools
                warmup_blocks = []
                block_size = 1024 * 1024 * 256  # 256MB blocks
                for i in range(4):  # Allocate 4 blocks = 1GB total
                    try:
                        block = torch.zeros(block_size, device='cuda', dtype=torch.float32)
                        warmup_blocks.append(block)
                    except RuntimeError:
                        # If we can't allocate, that's okay - we'll work with what we have
                        break
                
                # Free all blocks immediately to establish memory pool structure
                for block in warmup_blocks:
                    del block
                del warmup_blocks
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("  Memory allocator warmed up (pre-allocated and freed memory pools)")
            except Exception as e:
                print(f"  Memory warm-up failed (non-critical): {e}")
            
            # Try to allocate and free a small tensor to force device memory initialization
            try:
                test_tensor = torch.zeros(1024, device='cuda')
                del test_tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("  Forced device memory initialization")
            except Exception as e2:
                print(f"  Could not force device memory init: {e2}")
        except Exception as e:
            print(f"  Could not perform additional memory setup: {e}")
    else:
        print("WARNING: CUDA/HIP is not available!")
except Exception as e:
    print(f"Warning: Could not verify GPU configuration: {e}")

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        job = None
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            # Only try to call on_error if job was successfully created
            if job is not None and hasattr(job, 'process') and len(job.process) > 0:
                try:
                    job.process[0].on_error(e)
                except Exception as e2:
                    print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            # Only try to call on_error if job was successfully created
            if job is not None and hasattr(job, 'process') and len(job.process) > 0:
                try:
                    job.process[0].on_error(e)
                except Exception as e2:
                    print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e


if __name__ == '__main__':
    main()
