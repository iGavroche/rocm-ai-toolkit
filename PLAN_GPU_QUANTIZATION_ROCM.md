# Plan: Enable GPU Quantization for ROCm (gfx1151)

## Objective
Enable GPU quantization for ROCm to prevent OOM kills when quantizing large models (WAN 2.2 14B transformers). User has sufficient VRAM (~96GB) and GPU operations work correctly.

## Current Status

### Verified Working
- ✅ PyTorch ROCm 7.1.25413 properly installed
- ✅ gfx1151 GPU detected and working (Radeon 8060S Graphics)
- ✅ GPU tensor operations successful (creation, computation, matmul, device transfers)
- ✅ Accelerate backend detection working
- ✅ ROCm 7.7.0 installed (rocm-smi)

### Changes Completed

#### 1. GPU Quantization for ROCm (`toolkit/util/quantize.py`)
**Lines 412-476**: Modified ROCm quantization path to attempt GPU quantization first

- **Before**: Always quantized on CPU for ROCm, then tried to move to GPU (often failed)
- **After**: 
  - Attempts GPU quantization first (moves model to GPU, quantizes there)
  - Falls back to CPU quantization if HIP errors occur
  - Respects `low_vram` setting (moves to CPU after GPU quantization if needed)
  - Graceful error handling with clear messages

**Key implementation**:
```python
if is_rocm:
    try:
        # Move to GPU and quantize
        model_to_quantize = model_to_quantize.to(base_model.device_torch, dtype=base_model.torch_dtype)
        quantize(model_to_quantize, weights=quantization_type, ...)
        # Handle low_vram if needed
    except (RuntimeError, Exception) as e:
        if "HIP" in str(e) or "hipError" in str(e):
            # Fallback to CPU quantization
```

#### 2. Extras Quantization (`toolkit/util/quantize.py`)
**Lines 492-533**: Updated extras quantization to try GPU if main model is on GPU

- **Before**: Always quantized extras on CPU for ROCm
- **After**:
  - Checks if main model is on GPU
  - Attempts GPU quantization for extras if main model is on GPU
  - Falls back to CPU if GPU fails or if `low_vram` is set

#### 3. Accelerate Configuration (`toolkit/accelerator.py`)
**Lines 8-31**: Updated Accelerate initialization for ROCm

- **Before**: Default Accelerator() initialization
- **After**:
  - Explicitly sets `device_placement=True` for ROCm (since GPU operations work)
  - Adds logging for debugging Accelerate configuration
  - Maintains CUDA compatibility

#### 4. ROCm Environment Variables (`start_toolkit.sh`)
**Lines 77-117**: Re-enabled essential ROCm environment variables for gfx1151

- **Before**: Minimal ROCm setup (only paths)
- **After**: Sets essential variables:
  - `PYTORCH_ROCM_ARCH=gfx1151` - GPU architecture
  - `HSA_OVERRIDE_GFX_VERSION=11.0.0` - Required for gfx1151 compatibility (from Reddit reports)
  - `ROCBLAS_USE_HIPBLASLT=1` - Optimized BLAS backend
  - Library paths (LD_LIBRARY_PATH, DEVICE_LIB_PATH, etc.)

## Testing Plan

### Test 1: GPU Quantization Success
**Goal**: Verify GPU quantization works without HIP errors

**Steps**:
1. Run training: `./start_toolkit.sh train config/examples/train_lora_wan22_14b_i2v_24gb.yaml`
2. Monitor output for:
   - "ROCm detected: attempting GPU quantization (will fallback to CPU if needed)"
   - "Model moved to GPU, starting quantization..."
   - "✓ Model quantized on GPU"
3. **Expected**: Quantization completes on GPU without errors
4. **Success criteria**: No HIP errors, quantization completes successfully

### Test 2: CPU Fallback on HIP Error
**Goal**: Verify graceful fallback if GPU quantization fails

**Steps**:
1. If HIP errors occur, monitor for:
   - "GPU quantization failed with HIP error, falling back to CPU: ..."
   - "Quantizing entire model on CPU..."
   - "✓ Model quantized on CPU"
2. **Expected**: Falls back to CPU quantization automatically
3. **Success criteria**: Training continues without crashing

### Test 3: Low VRAM Handling
**Goal**: Verify `low_vram: true` moves quantized model to CPU after GPU quantization

**Steps**:
1. Verify config has `low_vram: true`
2. Monitor for:
   - "✓ Model quantized on GPU"
   - "✓ Moved quantized model to CPU (low_vram)"
3. **Expected**: Model quantized on GPU, then moved to CPU
4. **Success criteria**: Memory usage reduced after quantization

### Test 4: Extras Quantization
**Goal**: Verify extras quantization follows main model device

**Steps**:
1. Monitor for extras quantization messages:
   - If main model on GPU: "Quantizing extras on GPU..."
   - If main model on CPU: "Quantizing extras on CPU..."
2. **Expected**: Extras quantization uses same device as main model
3. **Success criteria**: No mixed-device errors

### Test 5: Memory Usage
**Goal**: Verify GPU quantization uses VRAM instead of RAM

**Steps**:
1. Monitor system memory during quantization:
   - Check RAM usage (should not spike)
   - Check VRAM usage (should increase during quantization)
2. **Expected**: VRAM usage increases, RAM stays stable
3. **Success criteria**: No OOM kills, quantization completes

## Troubleshooting

### If HIP Errors Occur During GPU Quantization

**Possible causes**:
1. ROCm version incompatibility with kernel 6.18-rc4
2. Missing ROCm libraries or drivers
3. GPU memory fragmentation

**Solutions**:
1. Check ROCm version: `rocm-smi --version`
2. Verify GPU is accessible: `rocminfo`
3. Check for HIP errors in logs
4. If persistent, consider updating ROCm (user mentioned willingness to update)

### If Quantization Still Causes OOM

**Possible causes**:
1. Model too large even for GPU
2. Temporary buffers during quantization
3. Multiple models in memory simultaneously

**Solutions**:
1. Check VRAM usage: `rocm-smi --showmeminfo vram`
2. Verify quantization is happening on GPU (check logs)
3. Consider quantizing transformers separately
4. Add more aggressive memory clearing between transformers

## Success Metrics

- ✅ GPU quantization completes without HIP errors
- ✅ No OOM kills during quantization
- ✅ Training proceeds after quantization
- ✅ Memory usage appropriate (VRAM used, not RAM)
- ✅ Graceful fallback to CPU if GPU fails

## Next Steps After Testing

1. **If GPU quantization works**: 
   - Document the successful configuration
   - Remove debug logging if excessive
   - Update documentation

2. **If HIP errors persist**:
   - Check ROCm version compatibility
   - Review error messages for specific HIP error codes
   - Consider ROCm update if needed
   - Document workarounds

3. **If OOM still occurs**:
   - Verify quantization is actually happening on GPU
   - Check for memory leaks in quantization code
   - Consider more aggressive memory management
   - Possibly revert to CPU quantization with better memory management

## Files Modified

1. `toolkit/util/quantize.py` - GPU quantization implementation
2. `toolkit/accelerator.py` - Accelerate configuration for ROCm
3. `start_toolkit.sh` - ROCm environment variables for gfx1151
4. `extensions_built_in/diffusion_models/wan22/wan22_14b_i2v_model.py` - Memory clearing between transformers

## Environment Requirements

- ROCm 7.7.0+ (currently installed)
- PyTorch with ROCm support (verified: 2.10.0a0+rocm7.10.0a20251015)
- gfx1151 GPU (Radeon 8060S Graphics)
- Sufficient VRAM (~96GB available)
- Kernel 6.18-rc4 (Manjaro)

## References

- Reddit reports on gfx1151 success with `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- ROCm documentation for gfx1151 architecture
- PyTorch ROCm compatibility matrix


