# ref comfy ui
# Updated to support both CUDA and ROCm backends
import os
import importlib.util


# Can't use pytorch to get the GPU names because the cuda malloc has to be set before the first import.
def get_gpu_names():
    if os.name == 'nt':
        import ctypes

        # Define necessary C structures and types
        class DISPLAY_DEVICEA(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('DeviceName', ctypes.c_char * 32),
                ('DeviceString', ctypes.c_char * 128),
                ('StateFlags', ctypes.c_ulong),
                ('DeviceID', ctypes.c_char * 128),
                ('DeviceKey', ctypes.c_char * 128)
            ]

        # Load user32.dll
        user32 = ctypes.windll.user32

        # Call EnumDisplayDevicesA
        def enum_display_devices():
            device_info = DISPLAY_DEVICEA()
            device_info.cb = ctypes.sizeof(device_info)
            device_index = 0
            gpu_names = set()

            while user32.EnumDisplayDevicesA(None, device_index, ctypes.byref(device_info), 0):
                device_index += 1
                gpu_names.add(device_info.DeviceString.decode('utf-8'))
            return gpu_names

        return enum_display_devices()
    else:
        # On Linux, try to detect GPUs via system commands
        # For ROCm, we can check via rocm-smi or other methods
        gpu_names = set()
        try:
            # Try to get GPU info via rocm-smi for AMD GPUs
            import subprocess
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Product Name' in line or 'Card series' in line:
                        # Extract GPU name from rocm-smi output
                        parts = line.split(':')
                        if len(parts) > 1:
                            gpu_name = parts[1].strip()
                            if gpu_name:
                                gpu_names.add(gpu_name)
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return gpu_names


# NVIDIA GPU blacklist for CUDA malloc async
nvidia_blacklist = {"GeForce GTX TITAN X", "GeForce GTX 980", "GeForce GTX 970", "GeForce GTX 960", "GeForce GTX 950",
             "GeForce 945M",
             "GeForce 940M", "GeForce 930M", "GeForce 920M", "GeForce 910M", "GeForce GTX 750", "GeForce GTX 745",
             "Quadro K620",
             "Quadro K1200", "Quadro K2200", "Quadro M500", "Quadro M520", "Quadro M600", "Quadro M620", "Quadro M1000",
             "Quadro M1200", "Quadro M2000", "Quadro M2200", "Quadro M3000", "Quadro M4000", "Quadro M5000",
             "Quadro M5500", "Quadro M6000",
             "GeForce MX110", "GeForce MX130", "GeForce 830M", "GeForce 840M", "GeForce GTX 850M", "GeForce GTX 860M",
             "GeForce GTX 1650", "GeForce GTX 1630"
             }


def gpu_malloc_supported():
    """
    Check if GPU malloc async is supported.
    For CUDA: checks against NVIDIA blacklist
    For ROCm: generally supported, but may need specific configuration
    """
    try:
        names = get_gpu_names()
    except:
        names = set()
    
    # Check for NVIDIA GPUs in blacklist
    for x in names:
        if "NVIDIA" in x:
            for b in nvidia_blacklist:
                if b in x:
                    return False
    
    # For ROCm/AMD GPUs, malloc async is generally supported
    # Check if we're using ROCm backend
    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            for folder in torch_spec.submodule_search_locations:
                ver_file = os.path.join(folder, "version.py")
                if os.path.isfile(ver_file):
                    spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # Check if HIP version exists (ROCm backend)
                    if hasattr(module, 'hip') and module.hip is not None:
                        # ROCm supports malloc async via HIP
                        return True
    except:
        pass
    
    # Default: if no blacklisted NVIDIA GPU found, assume supported
    return True


# Backward compatibility
def cuda_malloc_supported():
    """Backward compatibility alias for gpu_malloc_supported"""
    return gpu_malloc_supported()


gpu_malloc = False
cuda_malloc = False  # Backward compatibility

if not gpu_malloc:
    try:
        version = ""
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            for folder in torch_spec.submodule_search_locations:
                ver_file = os.path.join(folder, "version.py")
                if os.path.isfile(ver_file):
                    spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    version = module.__version__
            if int(version[0]) >= 2:  # enable by default for torch version 2.0 and up
                gpu_malloc = gpu_malloc_supported()
                cuda_malloc = gpu_malloc  # Backward compatibility
    except:
        pass

if gpu_malloc:
    # For CUDA, use PYTORCH_CUDA_ALLOC_CONF
    # For ROCm, PyTorch may use different environment variables
    # but PYTORCH_CUDA_ALLOC_CONF should work for both (ROCm PyTorch uses CUDA API)
    env_var = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', None)
    if env_var is None:
        env_var = "backend:cudaMallocAsync"
    else:
        env_var += ",backend:cudaMallocAsync"

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = env_var
    # Try to detect backend type
    backend_type = "CUDA"
    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            for folder in torch_spec.submodule_search_locations:
                ver_file = os.path.join(folder, "version.py")
                if os.path.isfile(ver_file):
                    spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'hip') and module.hip is not None:
                        backend_type = "ROCm"
                        break
    except:
        pass
    print(f"GPU Malloc Async Enabled ({backend_type})")

