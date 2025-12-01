# GPU Quantization Status for ROCm (gfx1151)

## Current Status

✅ **Fixed**: Scoping issue with `clear_gpu_cache` - quantization now proceeds without `UnboundLocalError`

❌ **Issue**: GPU quantization fails with "HIP error: invalid argument" during device transfer
- Direct model transfer to GPU fails
- Chunked parameter-by-parameter transfer also fails
- Falls back to CPU quantization (which works but uses RAM, not VRAM)

## Problem

The current PyTorch build (`2.10.0a0+rocm7.10.0a20251015`) doesn't fully support gfx1151 device transfer operations needed for GPU quantization. The error occurs when trying to move large models (WAN 2.2 14B transformers) to GPU.

## Solution

Install PyTorch from AMD's gfx1151-specific nightly builds:

```bash
cd /home/nino/projects/ai-toolkit
source .venv/bin/activate
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade
```

After installation, restart training. GPU quantization should work without HIP errors.

## Current Behavior

1. Attempts GPU quantization first
2. If direct transfer fails → tries chunked parameter transfer
3. If chunked transfer fails → falls back to CPU quantization
4. CPU quantization works but uses system RAM (not VRAM)

## Next Steps

1. Install gfx1151-specific PyTorch build (see command above)
2. Re-run training - GPU quantization should succeed
3. If issues persist, check `GFX1151_HIP_ERROR_FIX.md` for troubleshooting

