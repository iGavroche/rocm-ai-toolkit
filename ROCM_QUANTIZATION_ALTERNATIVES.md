# ROCm Quantization Alternatives

## Current Problem

The current implementation (`optimum-quanto` + `torchao`) is:
1. Failing GPU transfers due to HIP errors
2. Skipping modules during transfer (ModuleList, Dropout, RMSNorm)
3. Potentially not quantizing anything if all transfers fail

## Alternative Approaches for ROCm

### Option 1: CPU-Only Quantization (Simplest)
**Pros**: Reliable, no GPU transfer issues
**Cons**: Uses RAM instead of VRAM, slower

**Implementation**: Skip GPU transfer entirely, quantize on CPU, then optionally move to GPU.

### Option 2: GPTQModel (ROCm-Compatible)
**Pros**: Tested on ROCm 6.2+, designed for ROCm
**Cons**: Requires different API, may need model conversion

**Installation**:
```bash
pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/rocm6.2/
# Or build from source:
git clone https://github.com/ModelCloud/GPTQModel && cd GPTQModel
ROCM_VERSION=6.2 python -m build --no-isolation --wheel .
```

### Option 3: Manual Quantization with PyTorch Native
**Pros**: Full control, works on ROCm
**Cons**: More code, need to implement quantization logic

**Implementation**: Use `torch.quantization` or manual weight quantization.

### Option 4: Pre-Quantize Before Loading
**Pros**: Avoids runtime quantization issues
**Cons**: Requires separate quantization step

**Implementation**: Quantize model weights offline, save quantized checkpoint, load directly.

## Recommended Solution

**For immediate fix**: Use CPU-only quantization path, skip GPU transfer attempts entirely.

**For long-term**: Consider GPTQModel or pre-quantized models.


