# Quantization Troubleshooting - WAN 2.2 14B I2V on ROCm

## Problem Summary
CPU quantization consistently stops/hangs at module 245-247 (`blocks.8.attn1` modules) during Transformer 2 quantization. The process slows dramatically (from ~1.5s/it to ~5.8s/it) and then appears to hang.

## Repeated Issues (Last 3 Days)

### 1. GPU Quantization Fails with HIP Errors
**Problem**: GPU quantization fails with "HIP error: invalid argument" when trying to move model to GPU.
**What We Tried**:
- ✅ Installed PyTorch from gfx1151-specific nightly builds (`uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade`)
- ✅ Added chunked parameter-by-parameter transfer (removed - was too slow/hung)
- ✅ Simplified to direct GPU transfer with immediate CPU fallback on HIP error
**Status**: GPU quantization still fails, falls back to CPU

### 2. CPU Quantization Hangs at Module 245-247
**Problem**: CPU quantization consistently stops at `blocks.8.attn1.to_out` (ModuleList) around module 247/1138.
**What We Tried**:
- ✅ Added progress reporting (`show_progress=True`)
- ✅ Added status updates and `sys.stdout.flush()` calls
- ✅ Added error handling with try/except blocks
- ✅ Added periodic garbage collection (every 5, 25, 50 modules)
- ✅ Made garbage collection more aggressive (after every module, every 10 modules)
- ✅ Added memory monitoring with psutil
- ✅ Added debug logging for modules 240-260
- ✅ Added GPU cache clearing during CPU quantization

**Observations**:
- Memory stays stable (~22-25 GB)
- Speed degrades dramatically at module 245-247 (from ~1.5s/it to ~5.8s/it)
- Progress bar shows it's working but gets progressively slower
- Specific modules that cause slowdown: `blocks.8.attn1.to_k`, `blocks.8.attn1.to_v`, `blocks.8.attn1.to_out` (ModuleList)

**Root Cause Hypothesis** (Updated):
1. ✅ CONFIRMED: `ModuleList` modules (especially `attn.to_out`) cause hangs - now skipped
2. ⚠️ ONGOING: After module 260, quantization slows down dramatically even for regular Linear modules
3. Hypothesis: The slowdown after 260 suggests:
   - Memory fragmentation from cumulative quantization operations
   - The quanto library itself slows down as more modules are quantized (internal state buildup)
   - Later modules in the model are larger/more complex
   - CPU cache thrashing from memory pressure (24GB used of 30GB total)
4. The quantization is completing but extremely slowly - taking hours instead of minutes

## Files Modified
- `toolkit/util/quantize.py` - All quantization logic, memory management, and `safe_rocm_transfer()` function
- `extensions_built_in/diffusion_models/wan22/wan22_14b_i2v_model.py` - Memory clearing between transformers
- `start_toolkit.sh` - ROCm environment variables for quantization (PYTORCH_ROCM_ALLOC_CONF, AMD_SERIALIZE_KERNEL, TORCH_USE_HIP_DSA)
- `ROCM_QUANTIZATION_SETUP.md` - New comprehensive setup guide

## Current Status (Updated 2025-11-06)
- GPU quantization: ✅ **IMPLEMENTED** - Multi-strategy safe transfer (direct → module-by-module → parameter-by-parameter)
- CPU quantization: ⚠️ Works but slows down dramatically after module ~260 (even after skipping ModuleList)
- Progress reporting: ✅ Working (shows progress bar and debug messages)
- Memory management: ✅ Balanced (gc every 10/25/50 modules)
- Module skipping: ✅ Skips ModuleList modules, Dropout, RMSNorm
- ROCm environment: ✅ Configured with optimal settings for gfx1151

## What HAS Been Tried (Updated)
1. ✅ **Skip ModuleList modules**: Added logic to skip all ModuleList, especially `attn.to_out` ones
2. ✅ **Skip Dropout/RMSNorm**: Added skipping of these modules to speed up
3. ✅ **Memory management**: Tried various frequencies (every module, every 5/10/25/50)
4. ✅ **Safe ROCm GPU transfer**: Implemented `safe_rocm_transfer()` with 3 fallback strategies:
   - Direct transfer (fastest)
   - Module-by-module transfer (if direct fails)
   - Parameter-by-parameter transfer (last resort)
5. ✅ **ROCm environment configuration**: Added PYTORCH_ROCM_ALLOC_CONF, AMD_SERIALIZE_KERNEL, TORCH_USE_HIP_DSA
6. ❌ **Different quantization strategy**: NOT tried - still doing module-by-module
7. ❌ **Alternative quantization libraries**: NOT tried
8. ❌ **Increase swap space**: NOT tried
9. ❌ **Quantize in batches**: NOT tried

## NEW Observation (2025-11-06)
- Even after skipping ModuleList modules, slowdown continues after module 260
- Speed degrades from ~5.76 it/s at module 260 to ~2.95s/it at module 275
- Suggests the issue isn't just ModuleList - could be:
  - Linear layers getting progressively slower to quantize
  - Memory fragmentation despite GC
  - The quantization library itself slowing down as more modules quantized
  - Large modules that appear later in the model

## Next Steps to Try
1. ✅ **Add module filtering**: DONE - Skip ModuleList modules
2. ✅ **Add timeout mechanism**: DONE - Added timing to detect slow modules (>10s)
3. ❌ **Investigate quanto's ModuleList handling**: NOT done - would need to check quanto source
4. ❌ **Try excluding large Linear modules**: NOT tried - Could skip very large Linear layers
5. ❌ **Batch quantization**: NOT tried - Quantize multiple small modules at once
6. ❌ **Check quantization order**: NOT tried - Maybe quantize in reverse order to see if later modules are the issue
7. ❌ **Limit quantization scope**: NOT tried - Only quantize certain module types (e.g., only Linear, skip others)
8. ❌ **Use quantization caching**: Check if we can cache partial quantization results

## Key Learnings
- GPU quantization needs PyTorch build with proper gfx1151 kernels (current build may not have them)
- CPU quantization works but is slow and hangs at specific modules
- Memory management helps but doesn't solve the hang issue
- The hang is consistent at the same modules (blocks.8.attn1), suggesting a module-specific issue
- ModuleList quantization may be problematic with quanto library

## Commands Used
```bash
# Install gfx1151-specific PyTorch
cd /home/nino/projects/ai-toolkit
source .venv/bin/activate
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade

# Monitor training
tail -f output/*/log.txt
```

## Error Patterns
- HIP errors: Always "invalid argument" / "hipErrorInvalidValue"
- CPU quantization: Always stops at blocks.8.attn1 modules (245-247/1138)
- Time per iteration: Increases dramatically from ~1.5s to ~5.8s before hanging

