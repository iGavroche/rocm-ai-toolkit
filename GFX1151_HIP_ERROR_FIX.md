# Fixing "HIP error: invalid device function" on gfx1151

## Problem
You're hitting [PyTorch Issue #164346](https://github.com/pytorch/pytorch/issues/164346) - "HIP error: invalid device function" when using gfx1151 (Radeon 8060S Graphics) with PyTorch ROCm.

## Root Cause
Some PyTorch ROCm builds don't include all gfx1151 kernels compiled, causing "invalid device function" errors during specific operations (like quantization or certain tensor operations).

## Solution Options

### Option 1: Use AMD's gfx1151-Specific Nightly Builds (Recommended)

AMD provides PyTorch builds specifically compiled for gfx1151:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch from AMD's gfx1151-specific nightly builds
pip install --pre torch torchvision torchaudio \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

**Note**: This uses AMD's TheRock repository which has builds specifically compiled for gfx1151 with all kernels included.

### Option 2: Rebuild PyTorch from Source with gfx1151 Support

If the nightly builds don't work, you may need to rebuild PyTorch from source:

```bash
# Set environment variables for gfx1151
export PYTORCH_ROCM_ARCH="gfx1151"
export HSA_OVERRIDE_GFX_VERSION="11.0.0"

# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Build with gfx1151 support
python setup.py install --cmake
```

### Option 3: Use Workaround - Force CPU for Problematic Operations

If you can't rebuild or use nightly builds, add a workaround to catch and handle the error:

```python
# In quantization code, catch HIP errors and fallback to CPU
try:
    model.to(device)
    # quantization operation
except RuntimeError as e:
    if "invalid device function" in str(e) or "hipErrorInvalidDeviceFunction" in str(e):
        # Fallback to CPU quantization
        model.cpu()
        # continue with CPU quantization
```

## Current Status

✅ **Basic GPU operations work** - Your current PyTorch build (`2.10.0a0+rocm7.10.0a20251015`) can create tensors and do basic computations on gfx1151.

❓ **Specific operations may fail** - The error likely occurs during:
- Quantization operations
- Certain kernel launches
- Complex tensor operations

## Environment Variables Already Set

Your `start_toolkit.sh` already sets:
- ✅ `PYTORCH_ROCM_ARCH=gfx1151`
- ✅ `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- ✅ `ROCBLAS_USE_HIPBLASLT=1`

## Recommended Action

1. **Try AMD's gfx1151 nightly builds first** (Option 1) - easiest solution
2. **Monitor the GitHub issue** - [Issue #164346](https://github.com/pytorch/pytorch/issues/164346) for official fixes
3. **Use CPU fallback** - Our quantization code already has CPU fallback if GPU fails

## Related PRs/Issues to Watch

- [PyTorch Issue #164346](https://github.com/pytorch/pytorch/issues/164346) - The issue you're hitting
- Check for PRs linked to this issue for official fixes
- Monitor PyTorch ROCm releases for gfx1151 support improvements

## Testing After Fix

After installing gfx1151-specific builds, test with:

```python
import torch

# Test basic operations
x = torch.randn(100, 100).cuda()
y = torch.matmul(x, x)
print(f"✓ Basic operations work: {y.device}")

# Test quantization-like operations
z = torch.nn.functional.linear(x, torch.randn(100, 100).cuda())
print(f"✓ Linear operations work: {z.device}")
```

## References

- [AMD TheRock Repository](https://rocm.nightlies.amd.com/) - AMD's nightly builds
- [PyTorch ROCm Documentation](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)
- [GitHub Issue #164346](https://github.com/pytorch/pytorch/issues/164346)


