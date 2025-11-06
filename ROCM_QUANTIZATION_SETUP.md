# ROCm GPU Quantization Setup Guide

## Overview
This guide documents the setup and troubleshooting for GPU quantization on ROCm (specifically gfx1151/Strix Halo).

## System Requirements

### Verified Setup
- **ROCm Version**: 7.8.0 (rocm-smi 4.0.0)
- **PyTorch Version**: 2.10.0a0+rocm7.10.0a20251015
- **GPU**: Radeon 8060S Graphics (gfx1151)
- **Python**: 3.12.11
- **Quantization Libraries**:
  - `optimum-quanto`: 0.2.4
  - `torchao`: 0.10.0

## Environment Variables

The `start_toolkit.sh` script automatically sets the following ROCm environment variables:

### Essential Variables
- `PYTORCH_ROCM_ARCH=gfx1151` - Specifies the GPU architecture
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` - Overrides GPU version detection for gfx1151 compatibility
- `ROCBLAS_USE_HIPBLASLT=1` - Enables HIPBLASLT for better performance

### Memory Management
- `PYTORCH_ROCM_ALLOC_CONF=max_split_size_mb:256,garbage_collect=1` - Configures VRAM allocation to reduce fragmentation

### Debugging/Error Reporting
- `AMD_SERIALIZE_KERNEL=3` - Enables kernel serialization for better error reporting
- `TORCH_USE_HIP_DSA=1` - Enables device-side assertions for debugging

### Library Paths
- `LD_LIBRARY_PATH` - Points to ROCm libraries (`$ROCM_PATH/lib`)
- `DEVICE_LIB_PATH` - Points to ROCm device libraries (`$ROCM_PATH/llvm/amdgcn/bitcode`)
- `HIP_DEVICE_LIB_PATH` - Points to HIP device libraries

## GPU Quantization Implementation

### Safe ROCm Transfer Function

The `safe_rocm_transfer()` function in `toolkit/util/quantize.py` implements a multi-strategy approach to transfer models to GPU:

1. **Direct Transfer** (fastest): Attempts standard `model.to(device)` transfer
2. **Module-by-Module Transfer**: If direct fails, transfers each leaf module separately
3. **Parameter-by-Parameter Transfer**: Last resort - transfers parameters and buffers individually

This approach handles HIP errors gracefully by:
- Skipping problematic modules/parameters that cause HIP errors
- Continuing with remaining modules
- Providing detailed progress feedback
- Falling back to CPU quantization if all strategies fail

### Usage

The quantization code automatically uses `safe_rocm_transfer()` when ROCm is detected:

```python
transfer_success = safe_rocm_transfer(
    model_to_quantize,
    base_model.device_torch,
    dtype=base_model.torch_dtype,
    base_model=base_model
)
```

## Known Issues and Solutions

### HIP Error: Invalid Argument

**Problem**: GPU quantization fails with "HIP error: invalid argument" during model transfer.

**Root Cause**: PyTorch ROCm builds may not fully support large model transfers for gfx1151, especially for complex diffusers models with nested modules and buffers.

**Solution**: The `safe_rocm_transfer()` function implements fallback strategies:
- If direct transfer fails, it tries module-by-module transfer
- If that fails, it tries parameter-by-parameter transfer
- If all strategies fail, it falls back to CPU quantization

**Status**: GPU quantization should work with the chunked transfer approach. If it still fails, CPU quantization is used as fallback.

### CPU Quantization OOM

**Problem**: CPU quantization causes OOM kills due to insufficient RAM.

**Solution**: 
- Ensure GPU quantization is attempted first (it should use VRAM)
- If GPU quantization fails, consider:
  - Increasing system swap space
  - Using `low_cpu_mem_usage=True` when loading models
  - Quantizing transformers separately

## Testing GPU Quantization

### Verify Setup
```bash
# Check ROCm installation
rocm-smi --version

# Check PyTorch ROCm support
python -c "import torch; print('ROCm:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# Test basic GPU operations
python -c "import torch; x = torch.randn(1000, 1000, device='cuda'); print('GPU ops work')"
```

### Test Quantization
Run training with quantization enabled:
```bash
./start_toolkit.sh train config/examples/train_lora_wan22_14b_i2v_24gb.yaml
```

Watch for these messages:
- `"ROCm detected: attempting GPU quantization (will fallback to CPU if needed)"`
- `"Moving model to GPU (using safe ROCm transfer)..."`
- `"✓ Direct GPU transfer successful"` or fallback messages
- `"✓ Model quantized on GPU"`

## Troubleshooting

### Check Logs
Look for HIP errors in the training logs:
```bash
tail -f output/*/log.txt | grep -i "hip\|error\|quantization"
```

### Verify Environment Variables
```bash
env | grep -E "ROCM|HIP|PYTORCH_ROCM|HSA"
```

### Monitor GPU Usage
```bash
watch -n 1 rocm-smi
```

### Check VRAM Usage
During quantization, VRAM should be used instead of RAM. Monitor with:
```bash
# System RAM
free -h

# GPU VRAM (if rocm-smi supports it)
rocm-smi --showmeminfo
```

## Library Versions

### Current Versions (Verified Working)
- PyTorch: `2.10.0a0+rocm7.10.0a20251015`
- optimum-quanto: `0.2.4`
- torchao: `0.10.0`
- bitsandbytes: `0.48.2` (built from source with ROCm support)

### Installation

#### PyTorch and Standard Quantization Libraries
```bash
# PyTorch with ROCm (gfx1151-specific nightly builds)
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade

# Quantization libraries
uv pip install optimum-quanto torchao
```

#### bitsandbytes ROCm (Required for GPU Quantization)

**Why build from source?** The standard bitsandbytes package doesn't include ROCm binaries. You must build it from source to enable GPU quantization on ROCm.

**Step-by-step build instructions:**

1. **Clone the repository:**
   ```bash
   cd ~
   git clone --recurse https://github.com/ROCm/bitsandbytes.git bitsandbytes-rocm
   cd bitsandbytes-rocm
   ```

2. **Install build dependencies:**
   ```bash
   # Activate your ai-toolkit virtual environment
   source /path/to/ai-toolkit/.venv/bin/activate
   
   # Install build requirements
   pip install -r requirements-dev.txt
   ```

3. **Configure CMake for ROCm:**
   ```bash
   # Replace gfx1151 with your GPU architecture if different
   # Find your architecture: rocm-smi --version
   cmake -B build \
     -DCOMPUTE_BACKEND=hip \
     -DBNB_ROCM_ARCH=gfx1151 \
     -DCMAKE_HIP_ARCHITECTURES=gfx1151
   ```

4. **Build bitsandbytes:**
   ```bash
   cmake --build build
   ```

5. **Copy the ROCm library to your venv:**
   ```bash
   # Determine your ROCm version
   ROCM_VERSION=$(rocm-smi --version | grep "ROCM-SMI-LIB version" | awk '{print $NF}')
   ROCM_ID=$(echo $ROCM_VERSION | cut -d. -f1,2 | tr -d '.')
   
   # Copy library (adjust path if needed)
   cp bitsandbytes/libbitsandbytes_rocm${ROCM_ID}.so \
      /path/to/ai-toolkit/.venv/lib/python3.*/site-packages/bitsandbytes/
   ```

   Or use this Python script to auto-detect and copy:
   ```bash
   python3 << 'EOF'
   import bitsandbytes
   import os
   import shutil
   import subprocess
   
   # Get bitsandbytes install directory
   bnb_dir = os.path.dirname(bitsandbytes.__file__)
   
   # Detect ROCm version
   result = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True)
   if result.returncode == 0:
       version_line = [l for l in result.stdout.split('\n') if 'ROCM-SMI-LIB version' in l]
       if version_line:
           rocm_version = version_line[0].split()[-1]
           major_minor = rocm_version.split('.')[:2]
           rocm_id = ''.join(major_minor)
           lib_name = f"libbitsandbytes_rocm{rocm_id}.so"
           
           # Copy from build directory
           src = f"bitsandbytes/{lib_name}"
           if os.path.exists(src):
               dst = os.path.join(bnb_dir, lib_name)
               shutil.copy2(src, dst)
               print(f"✓ Copied {src} to {dst}")
           else:
               print(f"Error: {src} not found. Check build directory.")
   EOF
   ```

6. **Verify installation:**
   ```bash
   python3 -c "import bitsandbytes as bnb; print('bitsandbytes ROCm:', bnb.__version__)"
   ```

   You should see the version number without any errors. If you see "Configured ROCm binary not found", the library wasn't copied correctly.

**Troubleshooting bitsandbytes build:**

- **Build fails with HIP errors**: Ensure ROCm is properly installed and `$ROCM_PATH` is set
- **Library not found**: Check that the library name matches your ROCm version (rocm71, rocm78, etc.)
- **Import errors**: Verify the library was copied to the correct bitsandbytes package directory
- **Architecture mismatch**: Ensure `BNB_ROCM_ARCH` matches your GPU (use `rocm-smi` to check)

**Benefits of bitsandbytes ROCm:**
- Enables GPU quantization (uses VRAM instead of system RAM)
- Avoids HIP transfer errors that occur with optimum-quanto
- Faster quantization process
- Better memory efficiency for large models

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [Strix Halo HomeLab Guide](https://strixhalo-homelab.d7.wtf/AI/AI-Capabilities-Overview)
- See `QUANTIZATION_TROUBLESHOOTING.md` for detailed troubleshooting history


