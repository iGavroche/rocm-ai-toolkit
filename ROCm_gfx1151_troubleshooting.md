# ROCm gfx1151 (Strix Halo) Memory Troubleshooting Guide

## Current Status

As of 2025, gfx1151 support in ROCm is evolving. ROCm 7.9.0 includes official support, but memory issues persist for some users.

### Latest Update (2025-01-XX)
- ✅ **FIXED**: Text encoder forward pass hang - now runs on CPU for quantized models when `PYTORCH_NO_HIP_MEMORY_CACHING=1`
- ❌ **CRITICAL ISSUE**: ROCm driver crash (0xC0000005) during transformer forward pass on GPU
  - **Crash location**: `rocblas_create_handle()` / `miopenCreate()` in ROCm libraries
  - **Root cause**: Quantized models + ROCm + `PYTORCH_NO_HIP_MEMORY_CACHING=1` causes driver access violation
  - **Impact**: Process crashes completely (not just hangs)
  - **Workarounds**:
    1. **Remove `PYTORCH_NO_HIP_MEMORY_CACHING=1`** - May cause memory fragmentation but avoids crash
    2. **Use unquantized models** - Avoids quantized model driver bug
    3. **Update ROCm drivers** - Newer versions may fix this
    4. **Run transformer on CPU** - Very slow but works (not implemented yet)

## Key Findings from Research

### 1. ROCm Version
- **ROCm 7.9.0**: Official gfx1151 support
- **ROCm 6.4.4**: Earlier support on Windows
- Consider upgrading to latest ROCm if experiencing issues

### 2. MIOpen Library
- Some MIOpen versions have compatibility issues with gfx1151
- Upgrading to MIOpen 3.5.1+ has resolved batch normalization errors for some users

### 3. BIOS Settings
- **Dedicated Graphics Memory**: Increase in BIOS settings can help
- On AMD Strix Halo development boards, adjust "Dedicated Graphics Memory" setting

### 4. Environment Variables

#### GFX Version Override
Some users report `HSA_OVERRIDE_GFX_VERSION=11.0.0` works better than `11.5.1`:
```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"  # Try this if 11.5.1 doesn't work
```

#### Memory Allocator Settings
If experiencing severe fragmentation, try smaller `max_split_size_mb`:
```powershell
# Default (rocm-ninodes recommendation)
$env:PYTORCH_MAX_SPLIT_SIZE_MB = "512"

# For severe fragmentation, try:
$env:PYTORCH_MAX_SPLIT_SIZE_MB = "256"  # Smaller blocks
$env:PYTORCH_MAX_SPLIT_SIZE_MB = "128"  # Even smaller (more fragmentation-resistant)
```

### 5. Experimental Features
The codebase now includes:
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` - Enables experimental AOTriton support
- Memory warm-up strategy - Pre-allocates memory pools at startup
- Memory defragmentation - Attempts to consolidate memory before critical allocations

## Troubleshooting Steps

### Step 1: Check ROCm Version
```powershell
rocm-smi --version
python -c "import torch; print(torch.version.hip)"
```

### Step 2: Try Alternative GFX Version
If using `HSA_OVERRIDE_GFX_VERSION=11.5.1` and having issues:
```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
```

### Step 3: Reduce Memory Block Size
If fragmentation persists, reduce `max_split_size_mb`:
```powershell
$env:PYTORCH_MAX_SPLIT_SIZE_MB = "256"
# Or even:
$env:PYTORCH_MAX_SPLIT_SIZE_MB = "128"
```

### Step 4: Check BIOS Settings
- Enter BIOS/UEFI settings
- Look for "Dedicated Graphics Memory" or similar setting
- Increase if possible (varies by motherboard)

### Step 5: Monitor Memory Usage
The codebase now includes extensive memory diagnostics. Check logs for:
- Memory warm-up status
- Available vs. allocated memory
- Fragmentation warnings

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [rocm-ninodes Repository](https://github.com/iGavroche/rocm-ninodes) - Optimizations for gfx1151
- [kohya_ss Repository](https://github.com/bmaltais/kohya_ss) - Training framework with ROCm support

## Current Optimizations Applied

1. **Memory Warm-up**: Pre-allocates 1GB in 256MB blocks at startup to establish memory pools
2. **Memory Defragmentation**: Attempts to consolidate memory before critical allocations
3. **Incremental LoRA Loading**: Moves LoRA modules one at a time with cache clearing
4. **Dynamic CPU Offloading**: Automatically offloads large components when memory is tight
5. **Retry Logic**: Aggressive retry with cache clearing on OOM errors
6. **VAE CPU Offload Before Prepare**: **CRITICAL FIX** - Moves VAE to CPU before `accelerator.prepare()` to force release of reserved memory blocks. This avoids parameter-by-parameter fragmentation during model device placement.

## Key Finding: PyTorch Caching Allocator on Unified Memory Architecture

The core issue is **PyTorch's caching allocator doesn't properly account for unified memory architecture**. On Strix Halo (gfx1151), the GPU shares system RAM, but PyTorch's caching allocator reserves memory blocks that it thinks are free but aren't actually available due to unified memory confusion.

**Root Cause**: The caching allocator maintains a cache of "free" memory blocks, but on unified memory systems, these blocks may not actually be available when needed, leading to "reserved but unallocated memory" errors.

**CRITICAL SOLUTION**: Disable PyTorch's caching allocator with `PYTORCH_NO_HIP_MEMORY_CACHING=1`. This forces PyTorch to allocate/deallocate memory directly without caching, which properly accounts for unified memory architecture.

**Why This Works**: Without caching, PyTorch checks actual memory availability at allocation time rather than relying on cached "free" blocks that may not exist in unified memory systems.

## Next Steps if Issues Persist

1. **Upgrade ROCm**: Ensure you're on ROCm 7.9.0 or later
2. **Try fp32**: Switch from bf16 to fp32 in training config (rocm-ninodes recommendation)
3. **Reduce Batch Size**: Lower batch size to reduce memory pressure
4. **Enable Low VRAM Mode**: Set `low_vram: true` in model config
5. **Check Windows Paging File**: Ensure sufficient virtual memory (16-32GB recommended)

