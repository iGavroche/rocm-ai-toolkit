# ROCm Configuration Summary

This document summarizes the ROCm (ROCm) configuration for WAN 2.2 LoRA training on Strix Halo (gfx1151).

## Environment Variables Configured

The `start_toolkit.sh` script automatically sets the following environment variables for ROCm:

### Core ROCm Settings
- **ROCM_PATH**: Path to ROCm installation (default: `/opt/rocm`)
- **PYTORCH_ROCM_ARCH**: GPU architecture (set to `gfx1151` for Strix Halo)
- **HSA_OVERRIDE_GFX_VERSION**: Set to `11.0.0` for gfx1151 compatibility
- **ROCBLAS_USE_HIPBLASLT**: Set to `1` to enable hipBLASLt backend (optimized performance)

### Library Paths
- **LD_LIBRARY_PATH**: Includes `$ROCM_PATH/lib`
- **DEVICE_LIB_PATH**: Set to `$ROCM_PATH/llvm/amdgcn/bitcode`
- **HIP_DEVICE_LIB_PATH**: Set to `$ROCM_PATH/llvm/amdgcn/bitcode`
- **PATH**: Includes `$ROCM_PATH/bin` for ROCm tools

### Debugging and Error Reporting
- **AMD_SERIALIZE_KERNEL**: Set to `3` for better error reporting (serializes kernel launches)
- **TORCH_USE_HIP_DSA**: Set to `1` to enable device-side assertions
- **HIP_LAUNCH_BLOCKING**: Defaults to `0` (set to `1` for synchronous debugging)

### Optional Settings
- **HIP_VISIBLE_DEVICES**: Can be set to select specific GPUs (similar to CUDA_VISIBLE_DEVICES)

## Accelerate Library Configuration

The Accelerate library is configured automatically:
- Uses default `Accelerator()` initialization
- Automatically detects ROCm backend
- Handles device placement during `prepare()` calls
- Logs configuration when ROCm is detected

## Known Issues and Workarounds

### HIP "invalid argument" Errors

HIP errors during device transfers are common on ROCm, especially with large models. We've implemented several workarounds:

1. **Step-by-step device transfers**: Models are moved CPU → device → dtype separately
2. **CPU fallback**: If GPU transfer fails, models stay on CPU and Accelerate handles placement
3. **Error handling**: HIP errors are caught and handled gracefully
4. **Synchronization**: `synchronize_gpu()` is called before device operations

### Quantization on ROCm

- Quantization may fall back to CPU if GPU transfers fail
- This is expected and acceptable - quantization will still work correctly
- Warnings will appear but can be ignored

### bitsandbytes Support

- bitsandbytes doesn't fully support ROCm
- The toolkit automatically falls back to custom 8-bit optimizers
- Error messages are suppressed

## Debugging HIP Errors

If you encounter persistent HIP errors:

1. **Enable synchronous debugging**:
   ```bash
   export HIP_LAUNCH_BLOCKING=1
   ./start_toolkit.sh train config.yaml
   ```
   This makes kernel launches synchronous, providing better error reporting.

2. **Check ROCm installation**:
   ```bash
   rocminfo
   rocm-smi
   ```

3. **Verify PyTorch ROCm backend**:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"HIP version: {torch.version.hip}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device name: {torch.cuda.get_device_name(0)}")
   ```

4. **Check environment variables**:
   The `start_toolkit.sh` script prints a configuration summary when ROCm is detected.

## Performance Notes

- **HIP_LAUNCH_BLOCKING=1**: Slows down training but provides better error reporting
- **CPU fallback**: Quantization on CPU is slower but functional
- **rocBLAS/Tensile**: Automatically configured via `ROCBLAS_USE_HIPBLASLT=1`

## Verification Checklist

Before training, verify:

- [ ] ROCm is installed and accessible (`rocminfo` works)
- [ ] PyTorch detects ROCm backend (`torch.version.hip` is not None)
- [ ] GPU is detected (`torch.cuda.is_available()` returns True)
- [ ] Environment variables are set (check `start_toolkit.sh` output)
- [ ] Accelerate initializes correctly (check startup logs)

## Additional Resources

- ROCm Documentation: https://rocm.docs.amd.com
- PyTorch ROCm Notes: https://pytorch.org/docs/stable/notes/hip.html
- Accelerate Documentation: https://huggingface.co/docs/accelerate

